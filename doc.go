// Package llmcost prices LLM inference from vendored LiteLLM data.
//
// It is a shared module consumed by murmur, llm-gateway, and back: data plus a
// lookup and an exact multiply, nothing clever.
//
// # Unit
//
// Costs are [Nls] (int64), where 1 nls = 1/100,000 USD. This is back's billing
// unit — back's protos call the same unit "centimills" (see
// billingpb/usage_service.proto: "centimills (1 centimills = 1/100,000 of a
// dollar)").
//
// # Precision and rounding
//
// LiteLLM rates are tiny USD-per-token decimals, so a single token costs a
// fraction of an nls. [Cost] therefore prices the whole response exactly:
// rates are parsed from their decimal literals into [math/big.Rat] (never
// through float64), multiplied by token counts, and summed in USD. Only the
// final total is converted to nls, ceiling-rounded — matching back's
// convention of ceiling-rounding its margin'd total. Ceiling at the total
// means many cheap tokens accumulate correctly instead of truncating to zero,
// and any non-zero usage costs at least 1 nls — never silently free.
//
// # Interface
//
//	cost, ok := llmcost.Cost("claude-opus-4-8", llmcost.Usage{
//		InputTokens:       1200,
//		CachedInputTokens: 45000,
//		OutputTokens:      800,
//	})
//
// [Cost] returns ok=false for models it cannot price. [RatesFor] exposes the
// raw per-token rates for callers that want them.
//
// [Usage] follows the Anthropic convention of disjoint counts: InputTokens is
// uncached input only, and CachedInputTokens (cache reads) is NOT a subset of
// it. Callers of OpenAI-style APIs — where prompt_tokens includes
// cached_tokens — must subtract before constructing a Usage. Cache writes are
// intentionally out of scope.
//
// # Model names
//
// Model ids resolve through an alias map (internal id → LiteLLM key) owned by
// this module; ids not in the map are tried as LiteLLM keys directly, so
// arbitrary upstream keys (e.g. "gpt-4o") also price. A model prices only if
// LiteLLM lists positive input and output rates for it — entries without
// pricing (or with zero rates) return ok=false so callers fail loudly rather
// than bill zero. A model with no cache-read discount bills cached tokens at
// the full input rate.
//
// # Vendored data
//
// model_prices_and_context_window.json is embedded byte-identical from
// BerriAI/litellm (MIT; see THIRD_PARTY_LICENSES) at the commit pinned in
// VENDORED_FROM. A weekly GitHub Actions workflow re-vendors it and opens a
// PR; the validation gate (TestAliasesResolve) fails that PR if any aliased
// model no longer resolves to non-zero rates. Merges to main are auto-tagged
// with the next patch version so consumers get a real Go module tag.
package llmcost
