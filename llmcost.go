package llmcost

import (
	"cmp"
	_ "embed"
	"encoding/json"
	"fmt"
	"math/big"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
)

// litellmJSON is LiteLLM's pricing map, vendored byte-identical from
// BerriAI/litellm at the commit pinned in VENDORED_FROM. It is embedded whole
// (rather than trimmed to the providers we use) so the weekly sync workflow is
// a pure fetch-and-compare with no trim logic to drift.
//
//go:embed model_prices_and_context_window.json
var litellmJSON []byte

// Nls is the billing unit: 1 nls = 1/100,000 USD — the unit also known as a
// centimill.
type Nls = int64

const nlsPerUSD = 100_000

// ServiceTier selects which of a model's per-request rate variants price a
// response. OpenAI bills the same request differently by processing tier:
// flex (cheaper, slower) and priority (pricier, faster) each publish their
// own per-token rates, carried in LiteLLM's data as *_flex and *_priority
// field variants. Only the three constants below resolve — an unknown or
// empty tier is unpriced (ok=false), never a silent standard fallback — and
// there is no cross-tier fallback: a model without rates at the requested
// tier fails rather than billing another tier's rates (a ~2× error in
// either direction). LiteLLM's *_batches variants (Batch API) are not
// service tiers and are not modeled.
type ServiceTier string

const (
	TierStandard ServiceTier = "standard"
	TierFlex     ServiceTier = "flex"     // LiteLLM's *_flex fields
	TierPriority ServiceTier = "priority" // LiteLLM's *_priority fields
)

// ClaudeUsage holds the RAW token counts of one Anthropic API response, named
// after the API's usage fields. Anthropic reports input/cache-read/output as
// DISJOINT counts:
//
//	InputTokens                = usage.input_tokens                — uncached input ONLY, excludes both cache fields
//	CacheReadInputTokens       = usage.cache_read_input_tokens     — cache hits, billed at the cache-read rate (0.1× input)
//	CacheCreationInputTokens   = usage.cache_creation_input_tokens — TOTAL cache writes across both TTLs
//	CacheCreation1hInputTokens = usage.cache_creation.ephemeral_1h_input_tokens — the 1-hour-TTL SUBSET of the total
//	OutputTokens               = usage.output_tokens
//
// Copy the fields straight from the response; no arithmetic is needed or
// wanted. Cache writes are priced by TTL: the 1h subset bills at the model's
// 1-hour cache-write rate (2× input) and the remainder — the API's
// ephemeral_5m_input_tokens, computed internally as total minus 1h — at the
// default 5-minute rate (1.25× input). A response without the cache_creation
// breakdown leaves CacheCreation1hInputTokens zero and all writes bill at the
// 5-minute rate. CacheCreation1hInputTokens > CacheCreationInputTokens is
// impossible in a real response and panics. The request's total prompt size —
// which decides context-window pricing tiers — is InputTokens +
// CacheReadInputTokens + CacheCreationInputTokens, computed internally.
//
// Two non-count fields carry the request's price-multiplying modes:
//
//	Speed        = the REQUEST's speed parameter — "" (default) or "fast"
//	InferenceGeo = usage.inference_geo — the region the request was pinned to,
//	               or "global"/"not_available" when unpinned
//
// Fast mode and pinned-region inference multiply the uncached-input and
// output rates (fast is 2×–6× depending on the model, "us" is 1.1×); cache
// reads and writes bill unscaled, matching Anthropic's fast/geo pricing and
// LiteLLM's calculator. A mode the model has no multiplier for — fast on a
// model without fast pricing, a pinned geo without a geo factor, or an
// unrecognized Speed value — fails to price rather than billing standard
// rates: those premiums reach 6×, and silently dropping one is exactly the
// kind of underbilling this module exists to prevent.
type ClaudeUsage struct {
	InputTokens                int64
	CacheReadInputTokens       int64
	CacheCreationInputTokens   int64
	CacheCreation1hInputTokens int64
	OutputTokens               int64
	Speed                      string
	InferenceGeo               string
}

// OpenAIUsage holds the RAW token counts of one OpenAI API response, named
// after the API's usage fields. OpenAI reports OVERLAPPING counts:
//
//	InputTokens       = usage.input_tokens (a.k.a. prompt_tokens) — total input, INCLUDES the cached portion
//	CachedInputTokens = input_tokens_details.cached_tokens        — the cached subset of InputTokens
//	OutputTokens      = usage.output_tokens                       — total output; reasoning tokens are a subset, already included
//
// Copy the fields straight from the response; the uncached remainder
// (InputTokens - CachedInputTokens) is computed internally. Do NOT
// pre-subtract — that is this module's job, and doing it twice under-bills.
// CachedInputTokens > InputTokens is impossible in a real response and
// panics. OpenAI does not bill cache writes, so there is no cache-creation
// field.
//
// DataResidency is the data-residency region of the host that served the
// request: "eu" for eu.api.openai.com, "us" for us.api.openai.com, "" for
// the global api.openai.com. Regionalized requests bill EVERY token —
// cache reads included — at the model's regional-processing uplift (1.1× on
// the models OpenAI regionally prices). A model without an uplift for the
// region bills at standard rates: unlike Anthropic's response-asserted
// fast/geo modes, residency is a transport fact that holds for every model,
// and OpenAI publishes the uplift only for the models it regionally prices —
// LiteLLM's calculator prices absent uplifts the same way.
type OpenAIUsage struct {
	InputTokens       int64
	CachedInputTokens int64
	OutputTokens      int64
	DataResidency     string
}

// TierRates are exact USD-per-token prices for the five billable components.
// CacheCreation is the default 5-minute-TTL cache-write rate
// (cache_creation_input_token_cost); CacheCreation1h is the 1-hour-TTL rate
// (cache_creation_input_token_cost_above_1hr). A nil rate means the provider
// does not price that component (e.g. OpenAI models have no cache-write
// rates); costing usage with a positive count on a nil component fails rather
// than billing zero.
type TierRates struct {
	Input, CacheRead, CacheCreation, CacheCreation1h, Output *big.Rat
}

// Tier is a context-window pricing tier: when a request's total prompt tokens
// STRICTLY exceed AbovePromptTokens, the ENTIRE request — every input, cache,
// and output token, not just the excess — bills at these rates. That is how
// the providers bill long context (e.g. Anthropic's 1M-context beta bills the
// whole request at the >200k rates) and how LiteLLM's own cost calculator
// interprets its *_above_Xk_tokens fields. Rates are fully resolved at parse
// time: a component upstream leaves untiered inherits the base rate.
type Tier struct {
	AbovePromptTokens int64
	TierRates
}

// Rates are a model's exact USD-per-token prices, parsed from LiteLLM's
// decimal literals (never through float64).
//
// Fast and Geo are Anthropic price multipliers from LiteLLM's
// provider_specific_entry: Fast scales uncached input and output when the
// request ran with speed "fast" (2×–6× depending on the model), Geo maps a
// pinned inference region (lowercased, e.g. "us") to its multiplier (1.1×).
// Cache reads and writes are never scaled by either. RegionalUplift maps an
// OpenAI data-residency region ("eu", "us") to the model's
// regional_processing_uplift_multiplier_<region>, a flat uplift on ALL token
// costs — cache included — for requests served by a regionalized host. Nil
// or missing entries mean the model does not price that mode.
type Rates struct {
	Base           TierRates
	Tiers          []Tier // ascending by AbovePromptTokens; empty when untiered
	Fast           *big.Rat
	Geo            map[string]*big.Rat
	RegionalUplift map[string]*big.Rat

	// litellmProvider is the entry's upstream litellm_provider value —
	// which service bills this key. [ModelSelector.Key] checks it so a
	// selector can never resolve another provider's entry; absent upstream
	// means unowned and no selector matches (fail closed).
	litellmProvider string
}

// Usage is one response's raw token usage. It is implemented only by
// [ClaudeUsage] and [OpenAIUsage] — the interface is sealed, so the only way
// to supply usage is to hand a provider type its RAW reported counts. Each
// provider type normalizes its own reporting convention (Anthropic's
// disjoint counts, OpenAI's overlapping ones) into the module's disjoint
// billable components; callers never do that arithmetic.
type Usage interface {
	// disjoint validates the raw counts and normalizes them into disjoint
	// billable components. Counts impossible in a real response — negative,
	// or a subset exceeding its total — are caller bugs and panic.
	disjoint() components
	// premium resolves the usage's price-multiplying modes (fast, pinned
	// geo, data residency) against the model's rates. ok is false when the
	// usage reports a mode the model has no multiplier for — such usage must
	// fail, never bill at standard rates.
	premium(r Rates) (p premium, ok bool)
}

// premium is a response's resolved price multipliers: tokens scales uncached
// input and output, cache scales cache reads and writes. Nil means unscaled
// (1×). Anthropic's fast/geo premiums leave cache nil; OpenAI's regional
// uplift sets both to the same factor.
type premium struct {
	tokens, cache *big.Rat
}

// Cost prices one response in nls at the standard service tier. model is
// the LiteLLM pricing key. Cost normalizes the provider's raw usage into
// disjoint components, resolves the context-window tier from the total
// prompt size, computes rate × tokens exactly per component (scaled by any
// fast/geo/residency multiplier the usage triggers), sums in USD, and
// ceiling-rounds only the final total, so sub-nls token costs accumulate
// instead of truncating to zero and any non-zero usage costs at least 1 nls.
// ok is false if the model is unknown, unpriced, lacks a rate for a
// component the usage reports, or lacks a multiplier for a mode the usage
// reports.
func Cost(model string, u Usage) (Nls, bool) {
	return CostTier(model, TierStandard, u)
}

// CostTier prices one response in nls at a service tier, with the same
// semantics as [Cost] applied to the tier's own rates: the context-window
// tier is resolved within the service tier, and ok is false if the model is
// unknown, the model has no priceable rates at that tier (e.g. a model
// LiteLLM lists no *_flex fields for), the tier is not one of the
// [ServiceTier] constants, or the tier lacks a rate for a component the
// usage reports. There is deliberately no fallback to another tier's rates.
func CostTier(model string, tier ServiceTier, u Usage) (Nls, bool) {
	c := u.disjoint() // before the lookup: impossible counts panic even for unknown models and tiers
	r, ok := table()[model][tier]
	if !ok {
		return 0, false
	}
	p, ok := u.premium(r)
	if !ok {
		return 0, false
	}
	return cost(r, c, p)
}

func (u ClaudeUsage) disjoint() components {
	if u.InputTokens < 0 || u.CacheReadInputTokens < 0 || u.CacheCreationInputTokens < 0 || u.CacheCreation1hInputTokens < 0 || u.OutputTokens < 0 {
		panic(fmt.Sprintf("llmcost: negative token counts: %+v", u))
	}
	if u.CacheCreation1hInputTokens > u.CacheCreationInputTokens {
		panic(fmt.Sprintf("llmcost: CacheCreation1hInputTokens exceeds CacheCreationInputTokens (the 1h count is a subset of the total): %+v", u))
	}
	return components{
		input:           u.InputTokens, // already disjoint: Anthropic's input_tokens excludes cache activity
		cacheRead:       u.CacheReadInputTokens,
		cacheCreation:   u.CacheCreationInputTokens - u.CacheCreation1hInputTokens,
		cacheCreation1h: u.CacheCreation1hInputTokens,
		output:          u.OutputTokens,
	}
}

// premium composes the fast-mode and pinned-geo multipliers, which scale
// uncached input and output only — Anthropic bills cache reads and writes at
// their normal rates in both modes, and LiteLLM's calculator excludes cache
// costs from these multipliers the same way. "global" and "not_available"
// are the API's unpinned geos and never carry a premium; any other geo, and
// speed "fast", require a factor in the model's rates and fail without one.
func (u ClaudeUsage) premium(r Rates) (premium, bool) {
	m := big.NewRat(1, 1)
	switch u.Speed {
	case "", "standard":
	case "fast":
		if r.Fast == nil {
			return premium{}, false
		}
		m.Mul(m, r.Fast)
	default:
		return premium{}, false // a speed this module cannot price must never bill standard rates
	}
	switch geo := strings.ToLower(u.InferenceGeo); geo {
	case "", "global", "not_available":
	default:
		f, ok := r.Geo[geo]
		if !ok {
			return premium{}, false
		}
		m.Mul(m, f)
	}
	if m.Cmp(big.NewRat(1, 1)) == 0 {
		return premium{}, true
	}
	return premium{tokens: m}, true
}

func (u OpenAIUsage) disjoint() components {
	if u.InputTokens < 0 || u.CachedInputTokens < 0 || u.OutputTokens < 0 {
		panic(fmt.Sprintf("llmcost: negative token counts: %+v", u))
	}
	if u.CachedInputTokens > u.InputTokens {
		panic(fmt.Sprintf("llmcost: CachedInputTokens exceeds InputTokens (cached is a subset of input in OpenAI usage): %+v", u))
	}
	return components{
		input:     u.InputTokens - u.CachedInputTokens, // OpenAI's input_tokens includes the cached subset
		cacheRead: u.CachedInputTokens,
		output:    u.OutputTokens,
	}
}

// premium applies the model's regional-processing uplift to every component
// — OpenAI's uplift is a flat multiplier on all token costs, cache reads
// included. A region the model has no uplift for bills at standard rates
// (see the DataResidency field doc for why absent ≠ unpriced here).
func (u OpenAIUsage) premium(r Rates) (premium, bool) {
	if f, ok := r.RegionalUplift[strings.ToLower(u.DataResidency)]; ok {
		return premium{tokens: f, cache: f}, true
	}
	return premium{}, true
}

// RatesFor returns the raw standard-tier per-token rates for model (a
// LiteLLM pricing key), for callers that want them. ok is false if the
// model is unknown or has no positive rates. The returned rats are copies;
// mutating them cannot corrupt the shared table.
func RatesFor(model string) (Rates, bool) {
	return RatesForTier(model, TierStandard)
}

// RatesForTier returns the raw per-token rates for model at a service tier.
// ok is false if the model is unknown, has no priceable rates at that tier,
// or the tier is not one of the [ServiceTier] constants. The returned rats
// are copies; mutating them cannot corrupt the shared table.
func RatesForTier(model string, tier ServiceTier) (Rates, bool) {
	r, ok := table()[model][tier]
	if !ok {
		return Rates{}, false
	}
	out := Rates{
		Base:            r.Base.clone(),
		Tiers:           make([]Tier, len(r.Tiers)),
		Fast:            cpRat(r.Fast),
		Geo:             cloneRatMap(r.Geo),
		RegionalUplift:  cloneRatMap(r.RegionalUplift),
		litellmProvider: r.litellmProvider,
	}
	for i, t := range r.Tiers {
		out.Tiers[i] = Tier{AbovePromptTokens: t.AbovePromptTokens, TierRates: t.clone()}
	}
	return out, true
}

// components are one response's token counts normalized to disjoint billable
// parts: input is UNCACHED input only, cacheCreation is 5-minute-TTL writes
// only (the 1h subset already split out into cacheCreation1h).
type components struct {
	input, cacheRead, cacheCreation, cacheCreation1h, output int64
}

// promptTokens is the request's total prompt size — the context-window tier
// discriminator. Both providers gate long-context pricing on total input
// including cached tokens, whatever their TTL.
func (c components) promptTokens() int64 {
	return c.input + c.cacheRead + c.cacheCreation + c.cacheCreation1h
}

// cost is the pure core: resolve the tier from total prompt size, then
// Σ rate × tokens × multiplier exactly in USD and ceiling-round the total to
// nls. Multipliers scale the tier-resolved rates — tier selection itself is
// by token counts alone. A positive count on a component the tier has no
// rate for fails — a model that cannot price reported usage must never bill
// it as zero.
func cost(r Rates, c components, p premium) (Nls, bool) {
	t := r.tier(c.promptTokens())
	total := new(big.Rat)
	for _, part := range []struct {
		rate   *big.Rat
		tokens int64
		mult   *big.Rat // nil = unscaled
	}{
		{t.Input, c.input, p.tokens},
		{t.CacheRead, c.cacheRead, p.cache},
		{t.CacheCreation, c.cacheCreation, p.cache},
		{t.CacheCreation1h, c.cacheCreation1h, p.cache},
		{t.Output, c.output, p.tokens},
	} {
		if part.tokens == 0 {
			continue
		}
		if part.rate == nil {
			return 0, false
		}
		usd := new(big.Rat).Mul(part.rate, new(big.Rat).SetInt64(part.tokens))
		if part.mult != nil {
			usd.Mul(usd, part.mult)
		}
		total.Add(total, usd)
	}
	return ceilNls(total), true
}

// tier returns the rates the ENTIRE request bills at: the highest tier whose
// threshold the total prompt size strictly exceeds, else the base rates.
func (r Rates) tier(promptTokens int64) TierRates {
	for _, t := range slices.Backward(r.Tiers) {
		if promptTokens > t.AbovePromptTokens {
			return t.TierRates
		}
	}
	return r.Base
}

func (t TierRates) clone() TierRates {
	return TierRates{Input: cpRat(t.Input), CacheRead: cpRat(t.CacheRead), CacheCreation: cpRat(t.CacheCreation), CacheCreation1h: cpRat(t.CacheCreation1h), Output: cpRat(t.Output)}
}

func cpRat(r *big.Rat) *big.Rat {
	if r == nil {
		return nil
	}
	return new(big.Rat).Set(r)
}

func cloneRatMap(m map[string]*big.Rat) map[string]*big.Rat {
	if m == nil {
		return nil
	}
	out := make(map[string]*big.Rat, len(m))
	for k, v := range m {
		out[k] = cpRat(v)
	}
	return out
}

// priceable reports whether a component set can bill: positive input and
// output rates, and no negative cache rates. Nil and zero cache rates are
// deliberately distinct: ABSENT upstream means the component is unpriced and
// using it fails, while an EXPLICIT 0 is upstream asserting the component is
// free and is honored — real free tiers exist (DeepSeek and OpenAI-via-
// gateway entries list cache writes at 0 because those providers genuinely
// don't bill them), and LiteLLM's own calculator prices explicit zeros as
// free. Consumers that need stronger guarantees for the specific models they
// bill should assert those models' rates via [RatesFor] in their own tests.
func (t TierRates) priceable() bool {
	pos := func(r *big.Rat) bool { return r != nil && r.Sign() > 0 }
	nonneg := func(r *big.Rat) bool { return r == nil || r.Sign() >= 0 }
	return pos(t.Input) && pos(t.Output) && nonneg(t.CacheRead) && nonneg(t.CacheCreation) && nonneg(t.CacheCreation1h)
}

// ceilNls converts a non-negative exact USD amount to nls, rounding up.
func ceilNls(usd *big.Rat) Nls {
	nls := usd.Mul(usd, new(big.Rat).SetInt64(nlsPerUSD)) // usd is freshly built by cost(); mutating it here is fine
	q, rem := new(big.Int).QuoRem(nls.Num(), nls.Denom(), new(big.Int))
	if rem.Sign() > 0 {
		q.Add(q, big.NewInt(1))
	}
	if !q.IsInt64() {
		panic(fmt.Sprintf("llmcost: cost %s nls overflows int64", q))
	}
	return q.Int64()
}

// table lazily parses the embedded LiteLLM JSON into per-model, per-service-
// tier rates. A malformed embed is impossible past CI and panics.
var table = sync.OnceValue(func() map[string]map[ServiceTier]Rates {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(litellmJSON, &raw); err != nil {
		panic(fmt.Errorf("llmcost: parse embedded LiteLLM data: %w", err))
	}
	models := make(map[string]map[ServiceTier]Rates, len(raw))
	for model, spec := range raw {
		if m, ok := parseModel(spec); ok {
			models[model] = m
		}
	}
	return models
})

// serviceSuffixes maps each ServiceTier to the key suffix LiteLLM appends to
// that tier's rate variants (input_cost_per_token_flex,
// cache_read_input_token_cost_priority, …). The standard tier is the bare
// key.
var serviceSuffixes = map[ServiceTier]string{TierStandard: "", TierFlex: "_flex", TierPriority: "_priority"}

// tierKey matches LiteLLM's context-window-tier-defining keys
// (input_cost_per_token_above_200k_tokens, ..._above_128k_tokens,
// ..._above_272k_tokens), with an optional service-tier suffix
// (..._above_272k_tokens_priority) captured so each service tier's parse
// accepts only its own keys. Anchoring to the end still excludes every
// other variant family (e.g. *_batches, which prices the Batch API — not a
// per-request service tier), exactly as LiteLLM's own calculator excludes
// them from threshold detection.
var tierKey = regexp.MustCompile(`^input_cost_per_token_above_(\d+)(k?)_tokens(_flex|_priority)?$`)

// parseModel builds per-service-tier Rates from one LiteLLM model spec. ok
// is false for entries that cannot bill at the STANDARD tier — no pricing,
// zero/negative rates, or rows like "sample_spec" — which therefore must not
// resolve at all: standard rates anchor table membership (LiteLLM has no
// tier-only models; such an entry is upstream garbage). Flex and priority
// parse independently from their *_flex / *_priority field variants and are
// present only when priceable in their own right — a tier never inherits
// another tier's rates, so a model without flex rates simply has no flex
// entry and fails to price flex usage.
func parseModel(spec json.RawMessage) (map[ServiceTier]Rates, bool) {
	var fields map[string]json.RawMessage
	if json.Unmarshal(spec, &fields) != nil {
		return nil, false
	}
	std, ok := parseTier(fields, "")
	if !ok {
		return nil, false
	}
	m := map[ServiceTier]Rates{TierStandard: std}
	for tier, svc := range serviceSuffixes {
		if tier == TierStandard {
			continue
		}
		if r, ok := parseTier(fields, svc); ok {
			m[tier] = r
		}
	}
	return m, true
}

// parseTier builds one service tier's Rates from a model's fields, reading
// only keys carrying that tier's suffix (svc, per serviceSuffixes). Key
// names compose as <component><ctx-suffix><svc-suffix> — e.g.
// "input_cost_per_token" + "_above_272k_tokens" + "_priority". Cost fields
// decode as json.Number and convert to big.Rat from their decimal literals,
// so rates are exact. Context-window tiers come from the tierKey family
// whose service suffix matches svc; each tier's components resolve to the
// tiered rate when upstream defines one and inherit THIS service tier's
// base rate otherwise — never another service tier's.
func parseTier(fields map[string]json.RawMessage, svc string) (Rates, bool) {
	rates := func(ctx string, base TierRates) TierRates {
		or := func(r, fallback *big.Rat) *big.Rat {
			if r != nil {
				return r
			}
			return fallback
		}
		return TierRates{
			Input:         or(ratField(fields, "input_cost_per_token"+ctx+svc), base.Input),
			CacheRead:     or(ratField(fields, "cache_read_input_token_cost"+ctx+svc), base.CacheRead),
			CacheCreation: or(ratField(fields, "cache_creation_input_token_cost"+ctx+svc), base.CacheCreation),
			// The 1h write rate tiers too: cache_creation_input_token_cost_above_1hr_above_200k_tokens.
			CacheCreation1h: or(ratField(fields, "cache_creation_input_token_cost_above_1hr"+ctx+svc), base.CacheCreation1h),
			Output:          or(ratField(fields, "output_cost_per_token"+ctx+svc), base.Output),
		}
	}
	r := Rates{Base: rates("", TierRates{})}
	if !r.Base.priceable() {
		return Rates{}, false
	}
	if raw, ok := fields["litellm_provider"]; ok {
		_ = json.Unmarshal(raw, &r.litellmProvider) // non-string stays "": unowned, no selector matches
	}
	r.Fast, r.Geo = parseProviderMultipliers(fields)
	for key := range fields {
		if region, ok := strings.CutPrefix(key, "regional_processing_uplift_multiplier_"); ok {
			if f := ratField(fields, key); f != nil && f.Sign() > 0 {
				if r.RegionalUplift == nil {
					r.RegionalUplift = map[string]*big.Rat{}
				}
				r.RegionalUplift[strings.ToLower(region)] = f
			}
			continue
		}
		m := tierKey.FindStringSubmatch(key)
		if m == nil || m[3] != svc {
			continue
		}
		threshold, err := strconv.ParseInt(m[1], 10, 64)
		if err != nil || threshold <= 0 {
			continue
		}
		if m[2] == "k" {
			threshold *= 1000
		}
		tier := Tier{AbovePromptTokens: threshold, TierRates: rates("_above_"+m[1]+m[2]+"_tokens", r.Base)}
		if !tier.priceable() {
			continue // garbage tier rates upstream — never bill from them
		}
		r.Tiers = append(r.Tiers, tier)
	}
	slices.SortFunc(r.Tiers, func(a, b Tier) int { return cmp.Compare(a.AbovePromptTokens, b.AbovePromptTokens) })
	return r, true
}

// parseProviderMultipliers reads LiteLLM's provider_specific_entry object:
// numeric fields are price multipliers — "fast" for fast mode, any other for
// a regional-inference geo (e.g. "us": 1.1). Non-numeric fields (e.g.
// bedrock_invocation_schema) are not multipliers and are skipped. A
// non-positive factor is garbage and dropped, so the mode fails to price
// rather than billing scaled-toward-zero.
func parseProviderMultipliers(fields map[string]json.RawMessage) (fast *big.Rat, geo map[string]*big.Rat) {
	raw, ok := fields["provider_specific_entry"]
	if !ok {
		return nil, nil
	}
	var entries map[string]json.RawMessage
	if json.Unmarshal(raw, &entries) != nil {
		return nil, nil
	}
	for key := range entries {
		f := ratField(entries, key)
		if f == nil || f.Sign() <= 0 {
			continue
		}
		if key == "fast" {
			fast = f
			continue
		}
		if geo == nil {
			geo = map[string]*big.Rat{}
		}
		geo[strings.ToLower(key)] = f
	}
	return fast, geo
}

// ratField converts one JSON number field to an exact rational; nil if
// absent, non-numeric, or unparseable.
func ratField(fields map[string]json.RawMessage, key string) *big.Rat {
	raw, ok := fields[key]
	if !ok {
		return nil
	}
	var n json.Number
	if json.Unmarshal(raw, &n) != nil {
		return nil
	}
	if r, ok := new(big.Rat).SetString(string(n)); ok {
		return r
	}
	return nil
}
