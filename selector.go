package llmcost

import (
	"regexp"
	"strings"
	"sync"
)

// Provider is the service that actually served a request — the biller. The
// same vendor model bills differently per provider (and per Azure data
// zone: azure/us/gpt-5.4 carries a ~10% premium over azure/gpt-5.4), so the
// provider is part of model identity here. The constants cover Anthropic
// and OpenAI models everywhere LiteLLM prices them: direct, Bedrock,
// Vertex, and Azure.
type Provider string

const (
	// ProviderOpenAI is api.openai.com: keys are the model id verbatim
	// ("gpt-5.4"). Regional data-residency hosts bill via
	// [OpenAIUsage.DataResidency], not the key — Region must be empty.
	ProviderOpenAI Provider = "openai"
	// ProviderAnthropic is api.anthropic.com: keys are the model id verbatim
	// ("claude-sonnet-4-5"). Regional premiums are asserted by the response
	// ([ClaudeUsage.InferenceGeo]), not the key — Region must be empty.
	ProviderAnthropic Provider = "anthropic"
	// ProviderAzure is Azure OpenAI Service: keys are azure/{model}, or
	// azure/{region}/{model} for the data-zone deployments ("us", "eu")
	// whose premium is baked into the region key's rates.
	ProviderAzure Provider = "azure"
	// ProviderAzureAI is Azure AI Foundry — Anthropic (and other non-OpenAI)
	// models on Azure: keys are azure_ai/{model}, or
	// azure_ai/{region}/{model} where upstream prices a region.
	ProviderAzureAI Provider = "azure_ai"
	// ProviderBedrock is AWS Bedrock: the AWS model id IS the key, verbatim —
	// including any cross-region inference-profile prefix
	// ("us.anthropic.claude-sonnet-4-5-20250929-v1:0"); Region must be empty.
	ProviderBedrock Provider = "bedrock"
	// ProviderVertexAI is Google Vertex AI: keys are vertex_ai/{model} with
	// the caller's Vertex model id verbatim — "@" version suffixes and
	// publisher paths included ("claude-sonnet-4-5@20250929").
	ProviderVertexAI Provider = "vertex_ai"
)

// ModelSelector identifies a model by how it was served, in the caller's own
// vocabulary: the provider that billed the request, a model name, and — for
// providers that price by region — the region.
//
// Model accepts TWO spellings, resolved in this order:
//
//  1. The provider's NATIVE id, verbatim: "gpt-5.4", "claude-sonnet-4-5",
//     "anthropic.claude-sonnet-4-5-20250929-v1:0", "claude-sonnet-4-5@20250929".
//  2. The VENDOR's canonical model name, with the cloud's renaming handled
//     bespoke per provider: Bedrock's vendor prefix, "-vN:M" artifact
//     suffix (date-gated — "claude-v2" is a model version, not an
//     artifact), "@date" form, geo-profile prefixes and aws-region key
//     forms; Vertex's "@date" and "@default"; Azure's "gpt-35" spelling.
//     So {ProviderBedrock, "claude-sonnet-4-5-20250929"} selects
//     "anthropic.claude-sonnet-4-5-20250929-v1:0", and
//     {ProviderVertexAI, "claude-opus-4-6"} selects
//     "vertex_ai/claude-opus-4-6@default". An undated canonical name
//     resolves to the provider's dated variant when exactly one exists and
//     FAILS when several do — never guessing which date the caller meant.
//
// [ModelSelector.Key] is DETERMINISTIC and verified; it never fuzzy-matches
// and never falls back to a differently-priced key. This is the module's
// whole alias policy: the key grammar and the clouds' renaming schemes
// (coupled to the vendored data, maintained with it, and gated by
// TestSelectorCanonicalCoverage on every sync) live here, while WHICH
// selectors exist stays with the caller — who should test that every
// selector they bill resolves, so a vendored-data sync that drops a key
// fails their build instead of production.
type ModelSelector struct {
	Provider Provider
	// Model is the provider's native model id or the vendor's canonical
	// model name (see above).
	Model string
	// Region selects region-priced keys, matched case-insensitively:
	// Azure data zones ("us", "eu"), Bedrock cross-region inference
	// profiles ("us", "eu", "apac", …) or aws-region-keyed entries
	// ("us-gov-west-1") when Model is a canonical name. It must be empty
	// for providers whose regional pricing lives elsewhere: OpenAI
	// residency is [OpenAIUsage.DataResidency] and Anthropic geo is
	// [ClaudeUsage.InferenceGeo].
	Region string
}

// Key returns the verified LiteLLM pricing key for the selector: the native
// spelling first, then the canonical vendor name through the provider's
// renaming scheme, then an undated canonical against a unique dated
// variant. ok is false when the selector is malformed — empty model, Region
// on a provider that does not price by region, unknown provider — when no
// key resolves to a priceable model, when the resolved entry belongs to a
// DIFFERENT provider (by the entry's own litellm_provider — so a prefixed
// key smuggled into Model or a cross-vendor id fails rather than silently
// billing another provider's rates), or when an undated name is ambiguous
// across several dated variants. There is no fallback in any direction: an
// Azure data-zone deployment whose region key is missing upstream fails
// rather than billing the ~10%-cheaper global key.
func (s ModelSelector) Key() (string, bool) {
	if key, ok := s.nativeKey(); ok {
		return key, true
	}
	idx := canonicalIndexes()[s.Provider]
	region := strings.ToLower(s.Region)
	if key, ok := idx[canonicalRef{s.Model, region}]; ok {
		return key, true
	}
	// Undated canonical name: resolve against the dated variants the
	// provider actually serves — unique or nothing.
	var match string
	for ref, key := range idx {
		if ref.region == region && isDatedVariantOf(ref.name, s.Model) {
			if match != "" {
				return "", false // several dates — never guess which the caller meant
			}
			match = key
		}
	}
	return match, match != ""
}

// nativeKey resolves the provider's native spelling verbatim, with the
// provider-ownership check.
func (s ModelSelector) nativeKey() (string, bool) {
	key, ok := s.key()
	if !ok {
		return "", false
	}
	r, ok := table()[key][TierStandard] // standard anchors membership: present for every table entry
	if !ok || !s.Provider.owns(r.litellmProvider) {
		return "", false
	}
	return key, true
}

// isDatedVariantOf reports whether name is base plus an 8-digit date suffix
// ("claude-sonnet-4-5-20250929" is a dated variant of "claude-sonnet-4-5").
func isDatedVariantOf(name, base string) bool {
	date, ok := strings.CutPrefix(name, base+"-")
	if !ok || len(date) != 8 {
		return false
	}
	return strings.IndexFunc(date, func(r rune) bool { return r < '0' || r > '9' }) < 0
}

// owns reports whether a table entry with the given upstream
// litellm_provider is billed by this provider. LiteLLM subdivides some
// providers, so each case lists its variants; an empty value (entry without
// litellm_provider) is unowned and never matches.
func (p Provider) owns(litellmProvider string) bool {
	switch p {
	case ProviderOpenAI:
		return litellmProvider == "openai" || litellmProvider == "text-completion-openai"
	case ProviderAnthropic:
		return litellmProvider == "anthropic"
	case ProviderAzure:
		return litellmProvider == "azure" || litellmProvider == "azure_text"
	case ProviderAzureAI:
		return litellmProvider == "azure_ai"
	case ProviderBedrock:
		return litellmProvider == "bedrock" || litellmProvider == "bedrock_converse"
	case ProviderVertexAI:
		// vertex_ai, vertex_ai-anthropic_models, vertex_ai-openai_models, …
		return strings.HasPrefix(litellmProvider, "vertex_ai")
	}
	return false
}

// key is the pure grammar of NATIVE ids: provider-prefixed deterministic
// construction, no table lookup. Canonical vendor names resolve separately
// through canonicalIndexes.
func (s ModelSelector) key() (string, bool) {
	if s.Model == "" {
		return "", false
	}
	region := strings.ToLower(s.Region)
	switch s.Provider {
	case ProviderOpenAI, ProviderAnthropic, ProviderBedrock:
		if region != "" {
			return "", false
		}
		return s.Model, true
	case ProviderAzure, ProviderAzureAI, ProviderVertexAI:
		if region == "" {
			return string(s.Provider) + "/" + s.Model, true
		}
		return string(s.Provider) + "/" + region + "/" + s.Model, true
	}
	return "", false
}

// canonicalRef is a canonical-name lookup key: the VENDOR's model name plus
// the region ("" = the provider's default/global entry).
type canonicalRef struct {
	name, region string
}

// bedrockArtifactVersion matches AWS artifact-version suffixes: "-vN" or
// "-vN:M".
var bedrockArtifactVersion = regexp.MustCompile(`-v\d+(:\d+)?$`)

// bedrockBareArtifact matches OpenAI-on-Bedrock artifact suffixes with no
// "v": "openai.gpt-oss-120b-1:0" → "-1:0". The ":\d+" requirement keeps it
// off ordinary dash-digit names.
var bedrockBareArtifact = regexp.MustCompile(`-\d+:\d+$`)

// bedrockID splits an AWS model id into its optional geo-profile prefix
// ("us.", "eu.", "apac.", "us-gov.", …), vendor segment ("anthropic.",
// "openai.", "mistral.", …), and the model rest.
var bedrockID = regexp.MustCompile(`^(?:([a-z][a-z-]{1,8})\.)?(?:([a-z][a-z0-9]*)\.)(.+)$`)

// stripBedrockArtifact removes the AWS artifact-version suffix from a
// Bedrock model rest — "-v1:0"/"-v1" (claude-sonnet-4-5-20250929-v1:0,
// claude-opus-4-6-v1) and versionless "-1:0" (gpt-oss-120b-1:0) — EXCEPT on
// the legacy Claude 1/2 family, where vN is the MODEL version:
// "claude-v2:1", "claude-v1", and "claude-instant-v1" are different models,
// not artifact revisions of one, and keep their suffix. Bespoke by
// necessity: AWS overloaded the same syntax for both meanings.
func stripBedrockArtifact(name string) string {
	if s := bedrockArtifactVersion.ReplaceAllString(name, ""); s != name && s != "claude" && s != "claude-instant" {
		return s
	}
	return bedrockBareArtifact.ReplaceAllString(name, "")
}

// canonicalization is one pricing key's place in the canonical-name scheme:
// the provider it belongs to, the vendor-canonical (name, region) that
// selects it, and its priority when several keys claim the same name (lower
// wins; ties fail closed).
type canonicalization struct {
	provider Provider
	ref      canonicalRef
	prio     int
}

// canonicalize inverts the serving cloud's renaming scheme for one pricing
// key — the single source of truth for how each cloud renames vendor
// models, used by both the index builder and the sync-gate tests:
//
//   - Bedrock: strip aws-region key prefixes (bedrock/us-gov-west-1/… →
//     region "us-gov-west-1"), geo-profile prefixes (us.anthropic.… →
//     region "us"), the vendor segment, and artifact versions (-v1:0, -v1,
//     -1:0 — except the legacy claude/claude-instant family, see
//     stripBedrockArtifact); "@date" → "-date". The standard
//     artifact-versioned AWS id outranks "@date" twins and vendorless
//     oddities.
//   - Vertex: "@default" → the undated vendor name (an explicit undated key
//     outranks it); "@date" → "-date"; publisher MaaS paths
//     ("openai/gpt-oss-120b-maas") → the bare vendor name.
//   - Azure: "gpt-35…" → the vendor's "gpt-3.5…" spelling; data-zone
//     segments → region.
//   - Azure AI: names are already vendor-canonical; region segments → region.
//
// ok is false for keys outside the scheme: entries of other providers,
// deeper-nested non-chat keys, malformed forms.
func canonicalize(key, litellmProvider string) (canonicalization, bool) {
	switch lp := litellmProvider; {
	case ProviderBedrock.owns(lp):
		id, region := key, ""
		if rest, ok := strings.CutPrefix(id, "bedrock/"); ok {
			// bedrock/us-gov-west-1/{id} (aws-region key form) or a plain
			// bedrock/{id} spelling of the AWS id.
			if seg, r2, ok := strings.Cut(rest, "/"); ok {
				region, id = seg, r2
			} else {
				id = rest
			}
		}
		m := bedrockID.FindStringSubmatch(id)
		name, prio := id, 2 // vendorless oddities lose to real AWS ids
		if m != nil {
			name, prio = m[3], 1
			if m[1] != "" {
				if region != "" {
					return canonicalization{}, false // geo profile AND aws region — no current form; refuse to guess
				}
				region = m[1]
			}
			if stripBedrockArtifact(name) != name {
				prio = 0 // the standard artifact-versioned id is THE id
			}
		}
		name = strings.ReplaceAll(stripBedrockArtifact(name), "@", "-")
		return canonicalization{ProviderBedrock, canonicalRef{name, region}, prio}, true
	case ProviderVertexAI.owns(lp):
		name := strings.TrimPrefix(key, "vertex_ai/")
		switch base, tag, cut := strings.Cut(name, "@"); {
		case cut && tag == "default":
			return canonicalization{ProviderVertexAI, canonicalRef{base, ""}, 1}, true
		case cut:
			return canonicalization{ProviderVertexAI, canonicalRef{base + "-" + tag, ""}, 0}, true
		case strings.Contains(name, "/"):
			// publisher MaaS path: openai/gpt-oss-120b-maas
			bare := strings.TrimSuffix(name[strings.LastIndexByte(name, '/')+1:], "-maas")
			return canonicalization{ProviderVertexAI, canonicalRef{bare, ""}, 1}, true
		default:
			return canonicalization{ProviderVertexAI, canonicalRef{name, ""}, 0}, true
		}
	case ProviderAzure.owns(lp), ProviderAzureAI.owns(lp):
		p := ProviderAzure
		if ProviderAzureAI.owns(lp) {
			p = ProviderAzureAI
		}
		rest, ok := strings.CutPrefix(key, string(p)+"/")
		if !ok {
			return canonicalization{}, false
		}
		region := ""
		if seg, r2, ok := strings.Cut(rest, "/"); ok {
			region, rest = seg, r2
		}
		if strings.Contains(rest, "/") {
			return canonicalization{}, false // deeper nesting (image-quality keys) is not a chat model
		}
		if p == ProviderAzure {
			if tail, ok := strings.CutPrefix(rest, "gpt-35"); ok {
				rest = "gpt-3.5" + tail
			}
		}
		return canonicalization{p, canonicalRef{rest, region}, 0}, true
	}
	return canonicalization{}, false
}

// canonicalIndexes maps, per provider, the vendor's canonical model name
// (plus region) to the pricing key — canonicalize applied to the whole
// table. When several keys claim one canonical name (Bedrock lists
// "anthropic.claude-haiku-4-5-20251001-v1:0" AND "…@20251001"; Azure lists
// both "gpt-35" and "gpt-3.5" spellings of some models), the standard form
// wins by canonicalize's priority order, and a tie between distinct keys at
// equal priority drops the name entirely — fail closed, never guess between
// possibly-differently-priced entries. Dropped or outranked spellings
// remain reachable via their native ids.
var canonicalIndexes = sync.OnceValue(func() map[Provider]map[canonicalRef]string {
	type candidate struct {
		key  string
		prio int
	}
	best := map[Provider]map[canonicalRef]candidate{}
	dead := map[Provider]map[canonicalRef]bool{}
	for key, tiers := range table() {
		c, ok := canonicalize(key, tiers[TierStandard].litellmProvider)
		if !ok {
			continue
		}
		if best[c.provider] == nil {
			best[c.provider], dead[c.provider] = map[canonicalRef]candidate{}, map[canonicalRef]bool{}
		}
		if dead[c.provider][c.ref] {
			continue
		}
		cur, taken := best[c.provider][c.ref]
		switch {
		case !taken || c.prio < cur.prio:
			best[c.provider][c.ref] = candidate{key, c.prio}
		case c.prio == cur.prio && key != cur.key:
			delete(best[c.provider], c.ref) // equal-priority conflict: fail closed
			dead[c.provider][c.ref] = true
		}
	}
	out := make(map[Provider]map[canonicalRef]string, len(best))
	for p, refs := range best {
		out[p] = make(map[canonicalRef]string, len(refs))
		for ref, c := range refs {
			out[p][ref] = c.key
		}
	}
	return out
})
