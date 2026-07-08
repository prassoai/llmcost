package llmcost

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"math/big"
	"sync"
)

// litellmJSON is LiteLLM's pricing map, vendored byte-identical from
// BerriAI/litellm at the commit pinned in VENDORED_FROM. It is embedded whole
// (rather than trimmed to the providers we use) so the weekly sync workflow is
// a pure fetch-and-compare with no trim logic to drift.
//
//go:embed model_prices_and_context_window.json
var litellmJSON []byte

// Nls is back's billing unit: 1 nls = 1/100,000 USD (back's protos call the
// same unit "centimills").
type Nls = int64

const nlsPerUSD = 100_000

// Usage counts the tokens of one response. Counts are disjoint
// (Anthropic-style): InputTokens is uncached input only, CachedInputTokens is
// cache reads billed at the cache-read rate. For OpenAI-style APIs, where
// prompt_tokens includes cached_tokens, subtract before constructing a Usage.
type Usage struct {
	InputTokens, CachedInputTokens, OutputTokens int64
}

// Rates are a model's exact USD-per-token prices, parsed from LiteLLM's
// decimal literals (never through float64).
type Rates struct {
	Input, CachedInput, Output *big.Rat
}

// Cost prices a whole response in nls. It computes rate × tokens exactly per
// component, sums in USD, and ceiling-rounds only the final total — matching
// back's ceiling convention and guaranteeing that sub-nls token costs
// accumulate instead of truncating to zero. ok is false if the model is
// unknown or has no positive rates. Negative token counts are a caller bug
// and panic.
func Cost(model string, u Usage) (Nls, bool) {
	if u.InputTokens < 0 || u.CachedInputTokens < 0 || u.OutputTokens < 0 {
		panic(fmt.Sprintf("llmcost: negative token counts: %+v", u))
	}
	r, ok := table()[resolve(model)]
	if !ok {
		return 0, false
	}
	return ceilNls(usd(r, u)), true
}

// RatesFor returns the raw per-token rates for model, for callers that want
// them. ok is false if the model is unknown or has no positive rates. The
// returned rats are copies; mutating them cannot corrupt the shared table.
func RatesFor(model string) (Rates, bool) {
	r, ok := table()[resolve(model)]
	if !ok {
		return Rates{}, false
	}
	return Rates{
		Input:       new(big.Rat).Set(r.Input),
		CachedInput: new(big.Rat).Set(r.CachedInput),
		Output:      new(big.Rat).Set(r.Output),
	}, true
}

// resolve maps an internal model id to its LiteLLM key; ids not in the alias
// map are tried as LiteLLM keys directly.
func resolve(model string) string {
	if key, ok := aliases[model]; ok {
		return key
	}
	return model
}

// usd is the pure core: Σ rate × tokens, exact in USD.
func usd(r Rates, u Usage) *big.Rat {
	total := new(big.Rat).Mul(r.Input, new(big.Rat).SetInt64(u.InputTokens))
	total.Add(total, new(big.Rat).Mul(r.CachedInput, new(big.Rat).SetInt64(u.CachedInputTokens)))
	return total.Add(total, new(big.Rat).Mul(r.Output, new(big.Rat).SetInt64(u.OutputTokens)))
}

// ceilNls converts a non-negative exact USD amount to nls, rounding up.
func ceilNls(usd *big.Rat) Nls {
	nls := usd.Mul(usd, new(big.Rat).SetInt64(nlsPerUSD)) // usd is freshly built by usd(); mutating it here is fine
	q, rem := new(big.Int).QuoRem(nls.Num(), nls.Denom(), new(big.Int))
	if rem.Sign() > 0 {
		q.Add(q, big.NewInt(1))
	}
	if !q.IsInt64() {
		panic(fmt.Sprintf("llmcost: cost %s nls overflows int64", q))
	}
	return q.Int64()
}

// table lazily parses the embedded LiteLLM JSON into per-model rates, keeping
// only models with positive input and output rates — entries without pricing
// (free models, the "sample_spec" documentation row) must fail lookup so
// callers fail loudly rather than bill zero. Cost fields are decoded as
// json.Number and converted to big.Rat from their decimal literals, so rates
// are exact. A malformed embed is impossible past CI and panics.
var table = sync.OnceValue(func() map[string]Rates {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(litellmJSON, &raw); err != nil {
		panic(fmt.Errorf("llmcost: parse embedded LiteLLM data: %w", err))
	}
	models := make(map[string]Rates, len(raw))
	for model, spec := range raw {
		var costs struct {
			Input  json.Number `json:"input_cost_per_token"`
			Cached json.Number `json:"cache_read_input_token_cost"`
			Output json.Number `json:"output_cost_per_token"`
		}
		if json.Unmarshal(spec, &costs) != nil {
			continue // row whose cost fields aren't numbers — not priceable
		}
		in, out := rat(costs.Input), rat(costs.Output)
		if in == nil || out == nil || in.Sign() <= 0 || out.Sign() <= 0 {
			continue // unpriced or free — must not resolve
		}
		cached := rat(costs.Cached)
		if cached == nil {
			cached = in // no cache-read discount: cached tokens bill at the full input rate
		}
		if cached.Sign() < 0 {
			continue // garbage upstream rate — must not produce negative bills
		}
		models[model] = Rates{Input: in, CachedInput: cached, Output: out}
	}
	return models
})

// rat converts a JSON number literal to an exact rational; nil if absent or
// unparseable.
func rat(n json.Number) *big.Rat {
	if r, ok := new(big.Rat).SetString(string(n)); ok {
		return r
	}
	return nil
}
