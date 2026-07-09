// Package llmcost prices LLM inference from vendored LiteLLM data: data plus
// a lookup and an exact multiply, nothing clever.
//
// # Unit
//
// Costs are [Nls] (int64), where 1 nls = 1/100,000 USD — the unit also known
// as a centimill.
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
//		Speed:                      speed,          // the REQUEST's speed param: "" or "fast"
//		InferenceGeo:               u.InferenceGeo, // usage.inference_geo when present
//	})
//
// OpenAI reports overlapping counts — input_tokens INCLUDES cached_tokens:
//
//	llmcost.Cost("gpt-5.4", llmcost.OpenAIUsage{
//		InputTokens:       u.InputTokens,       // total input, cached included
//		CachedInputTokens: u.CachedInputTokens, // the cached subset of InputTokens
//		OutputTokens:      u.OutputTokens,      // reasoning tokens already included
//		DataResidency:     residency,           // "eu"/"us" when the host was eu./us.api.openai.com
//		ServiceTier:       tier,                // "" = standard; TierFlex / TierPriority bill the tier's own rates
//	})
//
// Cost returns (Nls, bool): ok=false whenever the response cannot be priced —
// unknown model, unpriced model, usage on a component the model has no rate
// for, or a mode (fast, pinned geo) the model has no multiplier for. Nothing
// ever silently bills zero or falls back to another component's rate.
// [RatesFor] exposes the raw per-token rates (base, tiers, and multipliers)
// for callers that want them.
//
// # Service tiers
//
// OpenAI bills the same request differently by processing tier: flex
// (cheaper, slower) and priority (pricier, faster) publish their own
// per-token rates, carried in LiteLLM's data as *_flex and *_priority field
// variants. The tier rides on [OpenAIUsage.ServiceTier] — zero value
// standard — so every provider-specific pricing input lives on its
// provider's usage type and a flex Claude request is inexpressible;
// [RatesFor] takes the [ServiceTier] explicitly (a raw lookup with no
// usage in hand).
//
//	llmcost.Cost("gpt-5.5", llmcost.OpenAIUsage{..., ServiceTier: llmcost.TierFlex})
//
// The no-fallback rule extends across tiers: a model without priceable rates
// at the requested tier (e.g. gpt-5.3-codex has no flex rates), a tier
// missing a rate for a component the usage reports, or an unrecognized
// non-empty [OpenAIUsage.ServiceTier] value — all return ok=false. Flex or
// priority usage is never billed at standard rates; that would be a ~2×
// error in either direction. Context-window tiers exist within a service
// tier (*_above_Xk_tokens_priority) and resolve there with the same
// semantics, inheriting that service tier's base rates. Standard rates
// anchor table membership: a model without them does not resolve at any
// tier. LiteLLM's *_batches variants price the Batch API, not a per-request
// service tier, and are not modeled.
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
// A component upstream leaves untiered inherits its base rate — the base of
// the service tier being priced, never another service tier's.
//
// # Price multipliers — fast mode, pinned geo, data residency
//
// Three request modes multiply the per-token rates; each is priced exactly
// as LiteLLM's calculator prices it from the same data.
//
// Anthropic's fast mode (request speed:"fast") bills uncached input and
// output at the model's fast multiplier — 6× on claude-opus-4-6/4-7, 2× on
// claude-opus-4-8 — and Anthropic's pinned-region inference
// (usage.inference_geo, e.g. "us") at the model's geo multiplier (1.1×).
// The two compose multiplicatively; cache reads and writes bill UNSCALED in
// both modes. A fast response on a model without fast pricing, a pinned geo
// without a factor, or an unrecognized Speed value fails to price: these
// premiums reach 6×, so billing standard rates on a pricing-data lag is
// exactly the silent underbill this module exists to prevent. The unpinned
// geos "global" and "not_available" carry no premium.
//
// OpenAI's data residency (OpenAIUsage.DataResidency — "eu"/"us" when the
// request was served by eu./us.api.openai.com) bills EVERY component, cache
// reads included, at the model's regional-processing uplift (1.1× on the
// gpt-5.4/5.5 family). Unlike the response-asserted Anthropic modes,
// residency is a caller-stated transport fact that holds for every model
// while OpenAI publishes the uplift only for the models it regionally
// prices — so a model without an uplift for the region bills at standard
// rates rather than failing, matching LiteLLM.
//
// Multipliers scale the tier-resolved rates; tier selection itself is a
// pure function of token counts.
//
// # Precision and rounding
//
// LiteLLM rates are tiny USD-per-token decimals, so a single token costs a
// fraction of an nls. Rates are parsed from their decimal literals into
// [math/big.Rat] (never through float64), multiplied by token counts, and
// summed in USD. Only the final total is converted to nls, ceiling-rounded.
// Ceiling at the total means many cheap tokens accumulate correctly instead
// of truncating to zero, and any non-zero usage costs at least 1 nls — never
// silently free.
//
// # Model names and providers
//
// Model ids are LiteLLM pricing keys, looked up directly (e.g.
// "claude-opus-4-8", "gpt-5.4", "codex-mini-latest"). Provider differences
// live in the KEY, not in code: the same vendor model bills differently per
// serving provider — "gpt-5.4" (OpenAI direct), "azure/gpt-5.4",
// "azure/us/gpt-5.4" (Azure US data zone, ~10% premium baked into the
// rates), "vertex_ai/claude-sonnet-4-5", or the AWS model id verbatim on
// Bedrock — and each is its own table entry.
//
// [ModelSelector] owns the key GRAMMAR for Anthropic and OpenAI models
// across direct, Bedrock, Vertex, Azure OpenAI, and Azure AI Foundry.
// Key() resolves the provider's NATIVE id verbatim, or the VENDOR's
// canonical model name through the cloud's bespoke renaming scheme —
// Bedrock's vendor prefixes, artifact versions (-v1:0), geo profiles and
// aws-region key forms; Vertex's @date and @default; Azure's gpt-35
// spelling — so {Bedrock, "claude-sonnet-4-5-20250929"} selects
// "anthropic.claude-sonnet-4-5-20250929-v1:0" and
// {Bedrock, "claude-sonnet-4-5", "us"} the us. inference profile.
// Resolution is deterministic and verified: never guessing (an undated name
// ambiguous across several dated variants fails), never falling back (a
// missing Azure region key fails rather than billing the cheaper global
// key), and never crossing providers (the resolved entry's own
// litellm_provider must match). TestSelectorCanonicalCoverage gates every
// data sync on the whole scheme: each cloud-served Anthropic/OpenAI key
// must remain selectable by its vendor name. This module still owns no
// fuzzy alias layer: WHICH selectors a consumer bills is the consumer's
// policy, and the consumer should test that each of its selectors resolves
// — that test is the guarantee that a data sync can't silently drop a model
// it depends on. A model prices only if LiteLLM lists positive input and
// output rates for it — entries without pricing (or with zero rates) return
// ok=false.
//
// # Vendored data
//
// model_prices_and_context_window.json is embedded byte-identical from
// BerriAI/litellm (MIT; see THIRD_PARTY_LICENSES) at the commit pinned in
// VENDORED_FROM. A weekly GitHub Actions workflow re-vendors it and opens a
// PR; the validation gate (go test: schema parse, table invariants, data
// canaries) fails that PR if upstream restructures the pricing schema or
// drops the canary models. Merges to main are auto-tagged with the next
// patch version so consumers get a real Go module tag — and consumers'
// resolution tests catch dropped models when they bump.
package llmcost
