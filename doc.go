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
// # Provider-explicit usage — the part callers must not get wrong
//
// The two providers report token counts DIFFERENTLY, so [Cost] takes a
// sealed [Usage] interface implemented only by [ClaudeUsage] and
// [OpenAIUsage]. Each provider type takes that provider's RAW reported
// counts and normalizes them into disjoint billable components internally.
// Never wrap one provider's counts in the other's type, and never
// pre-subtract.
//
// Anthropic reports disjoint counts — input_tokens EXCLUDES cache activity:
//
//	llmcost.Cost("claude-opus-4-8", llmcost.ClaudeUsage{
//		InputTokens:                u.InputTokens,          // uncached input only
//		CacheReadInputTokens:       u.CacheReadInputTokens, // billed at the cache-read rate
//		CacheCreationInputTokens:   u.CacheCreationInputTokens, // TOTAL cache writes, both TTLs
//		CacheCreation1hInputTokens: u.CacheCreation.Ephemeral1hInputTokens, // the 1h-TTL subset
//		OutputTokens:               u.OutputTokens,
//	})
//
// OpenAI reports overlapping counts — input_tokens INCLUDES cached_tokens:
//
//	llmcost.Cost("gpt-5.4", llmcost.OpenAIUsage{
//		InputTokens:       u.InputTokens,       // total input, cached included
//		CachedInputTokens: u.CachedInputTokens, // the cached subset of InputTokens
//		OutputTokens:      u.OutputTokens,      // reasoning tokens already included
//	})
//
// Cost returns (Nls, bool): ok=false whenever the response cannot be priced —
// unknown model, unpriced model, or usage on a component the model has no
// rate for. Nothing ever silently bills zero or falls back to another
// component's rate. [RatesFor] exposes the raw per-token rates (base and
// tiers) for callers that want them.
//
// # Cache pricing
//
// Cache reads bill at the model's cache_read_input_token_cost. Cache writes
// bill by TTL: the 1-hour subset of ClaudeUsage.CacheCreationInputTokens at
// cache_creation_input_token_cost_above_1hr (2× input) and the 5-minute
// remainder at cache_creation_input_token_cost (1.25× input). OpenAI bills
// reads only. These are the providers' real rates, and there is deliberately
// NO fallback in any direction: a model without a rate for a component —
// including a 5m-only model handed 1h writes — fails to price usage that
// reports those tokens.
//
// # Context-window tiers
//
// Some models price by context size (LiteLLM's *_above_Xk_tokens fields —
// e.g. Anthropic's 1M-context beta above 200k, GPT-5.4/5.5 above 272k,
// Gemini above 200k). The semantics, verified against both LiteLLM's cost
// calculator and the providers' billing: the tier is selected by the
// request's TOTAL prompt size (uncached input + cache reads + cache writes;
// for OpenAI that is exactly the reported InputTokens), the threshold is
// strict (>, not >=), and once exceeded the ENTIRE request — every input,
// cache, and output token — bills at the tier's rates, not just the excess.
// A component upstream leaves untiered inherits its base rate. OpenAI's
// priority/flex service tiers are out of scope (standard tier only).
//
// # Precision and rounding
//
// LiteLLM rates are tiny USD-per-token decimals, so a single token costs a
// fraction of an nls. Rates are parsed from their decimal literals into
// [math/big.Rat] (never through float64), multiplied by token counts, and
// summed in USD. Only the final total is converted to nls, ceiling-rounded —
// matching back's convention of ceiling-rounding its margin'd total. Ceiling
// at the total means many cheap tokens accumulate correctly instead of
// truncating to zero, and any non-zero usage costs at least 1 nls — never
// silently free.
//
// # Model names
//
// Model ids resolve through an alias map (internal id → LiteLLM key) owned by
// this module; ids not in the map are tried as LiteLLM keys directly, so
// arbitrary upstream keys (e.g. "gpt-4o") also price. A model prices only if
// LiteLLM lists positive input and output rates for it — entries without
// pricing (or with zero rates) return ok=false.
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
