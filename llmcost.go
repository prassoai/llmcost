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
type ClaudeUsage struct {
	InputTokens                int64
	CacheReadInputTokens       int64
	CacheCreationInputTokens   int64
	CacheCreation1hInputTokens int64
	OutputTokens               int64
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
type OpenAIUsage struct {
	InputTokens       int64
	CachedInputTokens int64
	OutputTokens      int64
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
type Rates struct {
	Base  TierRates
	Tiers []Tier // ascending by AbovePromptTokens; empty when untiered
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
}

// Cost prices one response in nls. model is the LiteLLM pricing key. Cost
// normalizes the provider's raw usage into disjoint components, resolves the
// context-window tier from the total prompt size, computes rate × tokens
// exactly per component, sums in USD, and ceiling-rounds only the final
// total, so sub-nls token costs accumulate instead of truncating to zero and
// any non-zero usage costs at least 1 nls. ok is false if the model is
// unknown, unpriced, or lacks a rate for a component the usage reports.
func Cost(model string, u Usage) (Nls, bool) {
	r, ok := table()[model]
	if !ok {
		return 0, false
	}
	return cost(r, u.disjoint())
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

// RatesFor returns the raw per-token rates for model (a LiteLLM pricing
// key), for callers that want them. ok is false if the model is unknown or
// has no positive rates. The returned rats are copies; mutating them cannot
// corrupt the shared table.
func RatesFor(model string) (Rates, bool) {
	r, ok := table()[model]
	if !ok {
		return Rates{}, false
	}
	out := Rates{Base: r.Base.clone(), Tiers: make([]Tier, len(r.Tiers))}
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
// Σ rate × tokens exactly in USD and ceiling-round the total to nls. A
// positive count on a component the tier has no rate for fails — a model that
// cannot price reported usage must never bill it as zero.
func cost(r Rates, c components) (Nls, bool) {
	t := r.tier(c.promptTokens())
	total := new(big.Rat)
	for _, part := range []struct {
		rate   *big.Rat
		tokens int64
	}{
		{t.Input, c.input},
		{t.CacheRead, c.cacheRead},
		{t.CacheCreation, c.cacheCreation},
		{t.CacheCreation1h, c.cacheCreation1h},
		{t.Output, c.output},
	} {
		if part.tokens == 0 {
			continue
		}
		if part.rate == nil {
			return 0, false
		}
		total.Add(total, new(big.Rat).Mul(part.rate, new(big.Rat).SetInt64(part.tokens)))
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
	cp := func(r *big.Rat) *big.Rat {
		if r == nil {
			return nil
		}
		return new(big.Rat).Set(r)
	}
	return TierRates{Input: cp(t.Input), CacheRead: cp(t.CacheRead), CacheCreation: cp(t.CacheCreation), CacheCreation1h: cp(t.CacheCreation1h), Output: cp(t.Output)}
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

// table lazily parses the embedded LiteLLM JSON into per-model rates. A
// malformed embed is impossible past CI and panics.
var table = sync.OnceValue(func() map[string]Rates {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(litellmJSON, &raw); err != nil {
		panic(fmt.Errorf("llmcost: parse embedded LiteLLM data: %w", err))
	}
	models := make(map[string]Rates, len(raw))
	for model, spec := range raw {
		if r, ok := parseModel(spec); ok {
			models[model] = r
		}
	}
	return models
})

// tierKey matches LiteLLM's tier-defining keys (input_cost_per_token_above_200k_tokens,
// ..._above_128k_tokens, ..._above_272_tokens). Anchoring to the end excludes
// service-tier variants like *_tokens_priority, which price OpenAI's
// priority/flex tiers — out of scope (standard tier only), exactly as
// LiteLLM's own calculator excludes them from threshold detection.
var tierKey = regexp.MustCompile(`^input_cost_per_token_above_(\d+)(k?)_tokens$`)

// parseModel builds Rates from one LiteLLM model spec. ok is false for
// entries that cannot bill — no pricing, zero/negative rates, or rows like
// "sample_spec" — which therefore must not resolve. Cost fields decode as
// json.Number and convert to big.Rat from their decimal literals, so rates
// are exact. Tiers come from the *_above_Xk_tokens key family, keyed on the
// input-cost key (as in LiteLLM's calculator); each tier's components resolve
// to the tiered rate when upstream defines one and inherit the base rate
// otherwise.
func parseModel(spec json.RawMessage) (Rates, bool) {
	var fields map[string]json.RawMessage
	if json.Unmarshal(spec, &fields) != nil {
		return Rates{}, false
	}
	rates := func(suffix string, base TierRates) TierRates {
		or := func(r, fallback *big.Rat) *big.Rat {
			if r != nil {
				return r
			}
			return fallback
		}
		return TierRates{
			Input:         or(ratField(fields, "input_cost_per_token"+suffix), base.Input),
			CacheRead:     or(ratField(fields, "cache_read_input_token_cost"+suffix), base.CacheRead),
			CacheCreation: or(ratField(fields, "cache_creation_input_token_cost"+suffix), base.CacheCreation),
			// The 1h write rate tiers too: cache_creation_input_token_cost_above_1hr_above_200k_tokens.
			CacheCreation1h: or(ratField(fields, "cache_creation_input_token_cost_above_1hr"+suffix), base.CacheCreation1h),
			Output:          or(ratField(fields, "output_cost_per_token"+suffix), base.Output),
		}
	}
	r := Rates{Base: rates("", TierRates{})}
	if !r.Base.priceable() {
		return Rates{}, false
	}
	for key := range fields {
		m := tierKey.FindStringSubmatch(key)
		if m == nil {
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
