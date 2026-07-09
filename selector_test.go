package llmcost

import (
	"math/big"
	"regexp"
	"strings"
	"testing"
)

// TestSelectorKeyGrammar encodes the NATIVE-id grammar per provider, pinned
// against real vendored keys: direct ids verbatim, azure/azure_ai with an
// optional region segment (matched case-insensitively), vertex_ai prefix
// with the Vertex id ("@" version suffixes included) verbatim, and Bedrock
// AWS ids — cross-region inference-profile prefixes included — verbatim.
// Every constructed key must also resolve via RatesFor: Key never returns
// an unverified key.
func TestSelectorKeyGrammar(t *testing.T) {
	for _, tc := range []struct {
		sel  ModelSelector
		want string
	}{
		{ModelSelector{ProviderOpenAI, "gpt-5.4", ""}, "gpt-5.4"},
		{ModelSelector{ProviderAnthropic, "claude-sonnet-4-5", ""}, "claude-sonnet-4-5"},
		{ModelSelector{ProviderAzure, "gpt-5.4", ""}, "azure/gpt-5.4"},
		{ModelSelector{ProviderAzure, "gpt-5.4", "us"}, "azure/us/gpt-5.4"},
		{ModelSelector{ProviderAzure, "gpt-5.5", "EU"}, "azure/eu/gpt-5.5"},
		{ModelSelector{ProviderAzureAI, "claude-opus-4-6", ""}, "azure_ai/claude-opus-4-6"},
		{ModelSelector{ProviderBedrock, "anthropic.claude-sonnet-4-5-20250929-v1:0", ""}, "anthropic.claude-sonnet-4-5-20250929-v1:0"},
		{ModelSelector{ProviderBedrock, "us.anthropic.claude-sonnet-4-5-20250929-v1:0", ""}, "us.anthropic.claude-sonnet-4-5-20250929-v1:0"},
		{ModelSelector{ProviderBedrock, "openai.gpt-oss-120b-1:0", ""}, "openai.gpt-oss-120b-1:0"},
		{ModelSelector{ProviderVertexAI, "claude-sonnet-4-5", ""}, "vertex_ai/claude-sonnet-4-5"},
		{ModelSelector{ProviderVertexAI, "claude-sonnet-4-5@20250929", ""}, "vertex_ai/claude-sonnet-4-5@20250929"},
	} {
		got, ok := tc.sel.Key()
		if !ok || got != tc.want {
			t.Errorf("%+v: Key = %q, %v; want %q, true", tc.sel, got, ok, tc.want)
			continue
		}
		if _, ok := RatesFor(got, TierStandard); !ok {
			t.Errorf("%+v: key %q does not resolve via RatesFor", tc.sel, got)
		}
	}
}

// TestSelectorCanonicalNames encodes the CANONICAL vendor-name resolution —
// the caller uses the vendor's own model name and the selector inverts the
// serving cloud's renaming scheme, one pinned canary per bespoke rule:
// Bedrock's vendor prefix and artifact versions (dated -v1:0, undated -v1,
// versionless -1:0), geo-profile and aws-region key forms, the
// undated→unique-dated resolution (and its refusal when ambiguous),
// Vertex's @date and @default, Azure's gpt-35 spelling and its
// dual-spelling determinism.
func TestSelectorCanonicalNames(t *testing.T) {
	for _, tc := range []struct {
		sel  ModelSelector
		want string
	}{
		// Bedrock: vendor prefix + dated artifact version.
		{ModelSelector{ProviderBedrock, "claude-sonnet-4-5-20250929", ""}, "anthropic.claude-sonnet-4-5-20250929-v1:0"},
		// Undated canonical resolves to the UNIQUE dated variant.
		{ModelSelector{ProviderBedrock, "claude-sonnet-4-5", ""}, "anthropic.claude-sonnet-4-5-20250929-v1:0"},
		// Region selects the cross-region inference profile, case-insensitively.
		{ModelSelector{ProviderBedrock, "claude-sonnet-4-5", "US"}, "us.anthropic.claude-sonnet-4-5-20250929-v1:0"},
		// Undated AWS id with an artifact version (-v1 is the artifact here,
		// not the model version — contrast claude-v2:1 below).
		{ModelSelector{ProviderBedrock, "claude-opus-4-6", ""}, "anthropic.claude-opus-4-6-v1"},
		{ModelSelector{ProviderBedrock, "claude-opus-4-6", "global"}, "global.anthropic.claude-opus-4-6-v1"},
		// The standard -v1:0 id outranks its "@date" twin.
		{ModelSelector{ProviderBedrock, "claude-haiku-4-5-20251001", ""}, "anthropic.claude-haiku-4-5-20251001-v1:0"},
		// aws-region key form, and the vendor-prefixed id outranks the
		// vendorless oddity upstream lists beside it.
		{ModelSelector{ProviderBedrock, "claude-sonnet-4-5-20250929", "us-gov-west-1"}, "bedrock/us-gov-west-1/anthropic.claude-sonnet-4-5-20250929-v1:0"},
		// OpenAI-on-Bedrock versionless artifact suffix.
		{ModelSelector{ProviderBedrock, "gpt-oss-120b", ""}, "openai.gpt-oss-120b-1:0"},
		// Legacy Claude 1/2: vN is the MODEL version and is part of the name.
		{ModelSelector{ProviderBedrock, "claude-v2:1", ""}, "anthropic.claude-v2:1"},
		// Vertex: @date form.
		{ModelSelector{ProviderVertexAI, "claude-sonnet-4-5-20250929", ""}, "vertex_ai/claude-sonnet-4-5@20250929"},
		// Vertex: the explicit undated key outranks its @default twin.
		{ModelSelector{ProviderVertexAI, "claude-opus-4-6", ""}, "vertex_ai/claude-opus-4-6"},
		// Vertex publisher MaaS path.
		{ModelSelector{ProviderVertexAI, "gpt-oss-120b", ""}, "vertex_ai/openai/gpt-oss-120b-maas"},
		// Azure's gpt-35 spelling of the vendor's gpt-3.5 name.
		{ModelSelector{ProviderAzure, "gpt-3.5-turbo-1106", ""}, "azure/gpt-35-turbo-1106"},
		// Azure lists BOTH spellings of some models: the vendor spelling
		// resolves natively; the gpt-35 twin stays reachable natively too.
		{ModelSelector{ProviderAzure, "gpt-3.5-turbo-0125", ""}, "azure/gpt-3.5-turbo-0125"},
		{ModelSelector{ProviderAzure, "gpt-35-turbo-0125", ""}, "azure/gpt-35-turbo-0125"},
		// Azure AI Foundry claude names are already vendor-canonical.
		{ModelSelector{ProviderAzureAI, "claude-haiku-4-5", ""}, "azure_ai/claude-haiku-4-5"},
	} {
		got, ok := tc.sel.Key()
		if !ok || got != tc.want {
			t.Errorf("%+v: Key = %q, %v; want %q, true", tc.sel, got, ok, tc.want)
		}
	}
	// Ambiguous undated canonical: Bedrock serves two dated claude-3-5-sonnet
	// variants; the selector must refuse to guess which the caller meant.
	if key, ok := (ModelSelector{ProviderBedrock, "claude-3-5-sonnet", ""}).Key(); ok {
		t.Errorf("ambiguous undated canonical resolved to %q; want ok=false", key)
	}
}

// TestSelectorNeverGuesses encodes the no-fuzzing contract that separates
// this from LiteLLM's alias layer: a selector that cannot be priced FAILS —
// no region fallback (azure/us/ bills ~10% above azure/, so falling back
// underbills), no provider inference, no key surgery, and no region on
// providers whose regional pricing lives elsewhere (OpenAI residency and
// Anthropic geo are usage-side).
func TestSelectorNeverGuesses(t *testing.T) {
	for name, sel := range map[string]ModelSelector{
		"unknown region must not fall back to the global key": {ProviderAzure, "gpt-5.4", "japaneast"},
		"unknown model": {ProviderAzure, "no-such-model", ""},
		"region on openai (residency is usage-side)":                   {ProviderOpenAI, "gpt-5.4", "eu"},
		"region on anthropic (geo is usage-side)":                      {ProviderAnthropic, "claude-sonnet-4-5", "us"},
		"region composes with canonical names, not native bedrock ids": {ProviderBedrock, "anthropic.claude-sonnet-4-5-20250929-v1:0", "us"},
		"unknown provider":                      {Provider("gemini"), "gemini-2.5-pro", ""},
		"empty model":                           {ProviderOpenAI, "", ""},
		"direct id through the wrong provider":  {ProviderVertexAI, "gpt-5.4", ""},
		"canonical name the cloud never serves": {ProviderBedrock, "gpt-5.4", ""},
		// The provider-ownership check: keys that CONSTRUCT and RESOLVE, but
		// to an entry billed by a different provider, must fail — never
		// silently bill that provider's rates.
		"prefixed key smuggled into Model":    {ProviderOpenAI, "azure/gpt-5.4", ""},
		"bedrock id through anthropic":        {ProviderAnthropic, "anthropic.claude-sonnet-4-5-20250929-v1:0", ""},
		"cross-vendor: claude through openai": {ProviderOpenAI, "claude-sonnet-4-5", ""},
		"cross-vendor: gpt through anthropic": {ProviderAnthropic, "gpt-5.4", ""},
	} {
		if key, ok := sel.Key(); ok {
			t.Errorf("%s: %+v resolved to %q; want ok=false", name, sel, key)
		}
	}
}

// TestSelectorCanonicalCoverage is THE cross-provider guarantee and the sync
// gate: literally every priceable key served by Bedrock, Vertex, Azure, and
// Azure AI is selectable, and every canonical vendor name selects exactly
// the right key. For each cloud-owned key in the vendored table:
//
//   - it must be inside the canonicalization scheme (a cloud key
//     canonicalize cannot place is a new upstream naming form — extend
//     canonicalize, never strand the model);
//   - the canonical (name, region) must round-trip: the vendor name selects
//     exactly this key;
//   - a key OUTRANKED by a standard twin (Bedrock's "@date" beside -v1:0,
//     Vertex's @default beside the undated key, vendorless oddities) must
//     bill IDENTICALLY to the winner — otherwise canonical selection would
//     misbill — and must stay reachable via its native id;
//   - a key dropped by an equal-priority conflict (Azure's dual gpt-35 /
//     gpt-3.5 spellings, Bedrock's dual bedrock/-prefixed ids) must stay
//     reachable via its native id.
//
// Direct OpenAI/Anthropic keys must resolve natively. The guarantee is
// scoped to the Anthropic/OpenAI vocabulary (claude|gpt keys) — other
// vendors hosted on the same clouds are outside the selector's contract.
// Per-provider floors keep the gate honest: a filter regression that
// silently skips a family fails loudly.
func TestSelectorCanonicalCoverage(t *testing.T) {
	vendorModel := regexp.MustCompile(`claude|gpt`)
	counts := map[Provider]int{}
	for key, tiers := range table() {
		r := tiers[TierStandard] // standard anchors membership: present for every entry
		if !vendorModel.MatchString(key) {
			continue
		}
		if ProviderOpenAI.owns(r.litellmProvider) || ProviderAnthropic.owns(r.litellmProvider) {
			p := ProviderOpenAI
			if ProviderAnthropic.owns(r.litellmProvider) {
				p = ProviderAnthropic
			}
			counts[p]++
			if got, ok := (ModelSelector{p, key, ""}).Key(); !ok || got != key {
				t.Errorf("%s: direct key not natively reachable", key)
			}
			continue
		}
		cloud := ProviderBedrock.owns(r.litellmProvider) || ProviderVertexAI.owns(r.litellmProvider) ||
			ProviderAzure.owns(r.litellmProvider) || ProviderAzureAI.owns(r.litellmProvider)
		c, ok := canonicalize(key, r.litellmProvider)
		if !ok {
			if cloud {
				t.Errorf("%s (litellm_provider %s): cloud key outside the canonicalization scheme — new upstream naming form?", key, r.litellmProvider)
			}
			continue
		}
		counts[c.provider]++
		native := nativeSelectorFor(c.provider, key, c.ref.region)
		winner, live := canonicalIndexes()[c.provider][c.ref]
		switch {
		case live && winner == key:
			if got, ok := (ModelSelector{c.provider, c.ref.name, c.ref.region}).Key(); !ok || got != key {
				t.Errorf("%s: canonical %+v resolved to %q, %v; want this key — round trip broken", key, c.ref, got, ok)
			}
		case live: // outranked twin of the winner
			if !tiersEqual(table()[key], table()[winner]) {
				t.Errorf("%s: outranked by %s with DIFFERENT rates — canonical selection would misbill", key, winner)
			}
			if got, ok := native.Key(); !ok || got != key {
				t.Errorf("%s: outranked twin not natively reachable via %+v", key, native)
			}
		default: // dropped by equal-priority conflict
			if got, ok := native.Key(); !ok || got != key {
				t.Errorf("%s: dropped from the canonical index and not natively reachable via %+v", key, native)
			}
		}
	}
	// Floors per provider — the gate must actually gate.
	for p, floor := range map[Provider]int{
		ProviderOpenAI: 100, ProviderAnthropic: 20, ProviderBedrock: 80,
		ProviderVertexAI: 30, ProviderAzure: 100, ProviderAzureAI: 15,
	} {
		if counts[p] < floor {
			t.Errorf("%s: coverage gate checked only %d keys (floor %d) — the classification has regressed", p, counts[p], floor)
		}
	}
}

// nativeSelectorFor reconstructs the native-id selector that reaches key.
func nativeSelectorFor(p Provider, key, region string) ModelSelector {
	switch p {
	case ProviderBedrock:
		return ModelSelector{p, key, ""} // AWS ids (aws-region key forms included) are table keys verbatim
	default:
		rest := strings.TrimPrefix(key, string(p)+"/")
		if region != "" {
			if r2, ok := strings.CutPrefix(rest, region+"/"); ok {
				return ModelSelector{p, r2, region}
			}
		}
		return ModelSelector{p, rest, ""}
	}
}

// TestSelectorCanonicalVendorNames encodes that the canonicalization
// actually produces VENDOR vocabulary, not cloud-flavored junk: every
// Anthropic/OpenAI canonical name derived from a cloud key must belong to a
// model family the vendor itself lists (its direct keys, dated or not), or
// appear on the pinned exception list below with a reason. A new cloud key
// whose canonical form matches neither fails the weekly sync — that is the
// demand-review moment for a new naming scheme. Stale exceptions fail too,
// so the list can only shrink.
func TestSelectorCanonicalVendorNames(t *testing.T) {
	stripDate := regexp.MustCompile(`-\d{8}$`)
	directFamilies := map[string]bool{}
	for key, tiers := range table() {
		if lp := tiers[TierStandard].litellmProvider; ProviderOpenAI.owns(lp) || ProviderAnthropic.owns(lp) {
			directFamilies[stripDate.ReplaceAllString(key, "")] = true
		}
	}
	exceptions := map[Provider]map[string]bool{
		// Vendor-delisted models the clouds still serve: the vendor's direct
		// API (and so LiteLLM's direct key set) dropped them, the clouds kept
		// them. The names are still the vendor's own.
		ProviderBedrock: setOf(
			"claude-3-5-haiku-20241022", "claude-3-5-sonnet-20240620", "claude-3-5-sonnet-20241022",
			"claude-3-sonnet-20240229",
			// Legacy Claude 1/2 family, where vN is the model version.
			"claude-instant-v1", "claude-v1", "claude-v2:1",
			// Open-weight OpenAI models, never served on api.openai.com.
			"gpt-oss-120b", "gpt-oss-20b", "gpt-oss-safeguard-120b", "gpt-oss-safeguard-20b",
		),
		ProviderVertexAI: setOf(
			"claude-3-5-haiku", "claude-3-5-haiku-20241022", "claude-3-5-sonnet", "claude-3-5-sonnet-20240620",
			"claude-3-sonnet", "claude-3-sonnet-20240229",
			"gpt-oss-120b", "gpt-oss-20b",
		),
		ProviderAzureAI: setOf("gpt-oss-120b"),
		// Azure-only OpenAI entries: models or dated variants OpenAI's direct
		// key set no longer (or never) lists, still sold on Azure.
		ProviderAzure: setOf(
			"gpt-3.5-turbo-16k-0613", "gpt-4-32k", "gpt-4-32k-0613", "gpt-4-turbo-vision-preview",
			"gpt-4.5-preview", "gpt-4o-realtime-preview-2024-10-01",
			"gpt-5.1-chat", "gpt-5.1-chat-2025-11-13", "gpt-5.1-codex-2025-11-13", "gpt-5.1-codex-mini-2025-11-13",
			"gpt-5.2-chat", "gpt-5.2-chat-2025-12-11", "gpt-5.3-chat",
			"gpt-audio-1.5-2026-02-23", "gpt-realtime-1.5-2026-02-23",
		),
	}
	vendorModel := regexp.MustCompile(`claude|gpt`)
	used := map[Provider]map[string]bool{}
	for key, tiers := range table() {
		c, ok := canonicalize(key, tiers[TierStandard].litellmProvider)
		if !ok || !vendorModel.MatchString(c.ref.name) {
			continue
		}
		if directFamilies[stripDate.ReplaceAllString(c.ref.name, "")] {
			continue
		}
		if exceptions[c.provider][c.ref.name] {
			if used[c.provider] == nil {
				used[c.provider] = map[string]bool{}
			}
			used[c.provider][c.ref.name] = true
			continue
		}
		t.Errorf("%s: canonical %q (from %s) is not a vendor model family and not a pinned exception — new cloud naming form? extend canonicalize or the exception list after verifying the name", c.provider, c.ref.name, key)
	}
	for p, names := range exceptions {
		for name := range names {
			if !used[p][name] {
				t.Errorf("%s: exception %q is stale — upstream no longer lists it; remove it", p, name)
			}
		}
	}
}

func setOf(names ...string) map[string]bool {
	s := make(map[string]bool, len(names))
	for _, n := range names {
		s[n] = true
	}
	return s
}

func ratEq(a, b *big.Rat) bool {
	if a == nil || b == nil {
		return a == b
	}
	return a.Cmp(b) == 0
}

func tierRatesEqual(a, b TierRates) bool {
	return ratEq(a.Input, b.Input) && ratEq(a.CacheRead, b.CacheRead) &&
		ratEq(a.CacheCreation, b.CacheCreation) && ratEq(a.CacheCreation1h, b.CacheCreation1h) &&
		ratEq(a.Output, b.Output)
}

func ratMapEqual(a, b map[string]*big.Rat) bool {
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		if !ratEq(v, b[k]) {
			return false
		}
	}
	return true
}

// ratesEqual compares two models' complete pricing exactly.
func ratesEqual(a, b Rates) bool {
	if !tierRatesEqual(a.Base, b.Base) || len(a.Tiers) != len(b.Tiers) ||
		!ratEq(a.Fast, b.Fast) || !ratMapEqual(a.Geo, b.Geo) || !ratMapEqual(a.RegionalUplift, b.RegionalUplift) {
		return false
	}
	for i := range a.Tiers {
		if a.Tiers[i].AbovePromptTokens != b.Tiers[i].AbovePromptTokens || !tierRatesEqual(a.Tiers[i].TierRates, b.Tiers[i].TierRates) {
			return false
		}
	}
	return true
}

// tiersEqual compares two models' complete pricing across every service
// tier — an outranked twin that differs only in flex/priority rates would
// misbill tier-priced usage just the same.
func tiersEqual(a, b map[ServiceTier]Rates) bool {
	if len(a) != len(b) {
		return false
	}
	for tier, ra := range a {
		rb, ok := b[tier]
		if !ok || !ratesEqual(ra, rb) {
			return false
		}
	}
	return true
}
