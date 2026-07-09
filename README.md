# llmcost

Shared Go module: LLM inference cost from vendored LiteLLM pricing data (MIT).
Data plus a lookup and an exact multiply — nothing clever.

Costs are in **nls**: 1 nls = 1/100,000 USD (the unit also known as a
centimill). Model ids are LiteLLM pricing keys, looked up directly.

## Usage — provider-explicit, raw counts in

The two providers report token counts **differently**. `Cost` takes a sealed
`Usage` interface implemented only by `ClaudeUsage` and `OpenAIUsage`; each
takes its provider's RAW usage fields and normalizes to disjoint billable
components inside. Never pre-subtract, never mix shapes.

**Anthropic** (disjoint counts — `input_tokens` EXCLUDES cache activity):

```go
cost, ok := llmcost.Cost("claude-opus-4-8", llmcost.ClaudeUsage{
    InputTokens:                1200,  // usage.input_tokens — uncached input only
    CacheReadInputTokens:       45000, // usage.cache_read_input_tokens
    CacheCreationInputTokens:   3000,  // usage.cache_creation_input_tokens — TOTAL writes, both TTLs
    CacheCreation1hInputTokens: 1000,  // usage.cache_creation.ephemeral_1h_input_tokens — 1h subset
    OutputTokens:               800,   // usage.output_tokens
})
```

**OpenAI / codex** (overlapping counts — `input_tokens` INCLUDES cached):

```go
cost, ok := llmcost.Cost("gpt-5.4", llmcost.OpenAIUsage{
    InputTokens:       46200, // usage.input_tokens — total, cached included
    CachedInputTokens: 45000, // input_tokens_details.cached_tokens — subset
    OutputTokens:      800,   // reasoning tokens are a subset, already included
})
```

`ok` is false whenever the response can't be priced — unknown model, or usage
on a component the model has no rate for. Nothing silently bills zero.

## Service tiers — flex and priority

OpenAI bills the same request differently by processing tier: **flex**
(cheaper, slower) and **priority** (pricier, faster) publish their own rates
(LiteLLM's `*_flex` / `*_priority` variants). `CostTier` selects one;
`Cost` is exactly the standard-tier view.

```go
cost, ok := llmcost.CostTier("gpt-5.5", llmcost.TierFlex, llmcost.OpenAIUsage{
    InputTokens:       46200,
    CachedInputTokens: 45000,
    OutputTokens:      800,
})
```

The no-fallback rule extends across tiers: a model without rates at the
requested tier (gpt-5.3-codex has no flex), a tier missing a component the
usage reports, or a tier string that isn't one of `TierStandard` /
`TierFlex` / `TierPriority` all return `ok=false` — flex/priority usage is
never silently billed at standard rates (~2× off either way). Priority has
its own context-window tiers (`*_above_Xk_tokens_priority`), resolved with
the same semantics. LiteLLM's `*_batches` variants (Batch API) are not
modeled.

## Semantics

- **Real cache rates, no fallback.** Cache reads bill at the model's
  `cache_read_input_token_cost` (0.1× input for Anthropic; OpenAI reports
  reads only). Cache writes bill **by TTL**: the 1-hour subset at
  `cache_creation_input_token_cost_above_1hr` (2× input) and the 5-minute
  remainder at `cache_creation_input_token_cost` (1.25× input) — the 1h rate
  participates in context-window tiers too. A model missing a rate for a
  reported component *fails* to price — never approximated at another rate.
- **Context-window tiers.** Models with LiteLLM `*_above_Xk_tokens` pricing
  (Anthropic's 1M-context beta >200k, GPT-5.4/5.5 >272k, Gemini >200k) are
  tiered on the request's **total prompt size** (uncached + cache reads +
  cache writes; strict `>`), and once over the threshold the **entire
  request** bills at the tier's rates — not just the excess. Verified against
  LiteLLM's own cost calculator and the providers' billing. Untiered
  components inherit the priced service tier's base rates — never another
  service tier's.
- **Exact math, ceiling at the total.** Rates parse from decimal literals
  into `math/big.Rat` — never through float64. The response is priced exactly
  in USD and converted to nls only at the final total, **ceiling-rounded**.
  Sub-nls token costs accumulate correctly; any non-zero usage costs at
  least 1 nls.
- **No alias layer.** Model ids are LiteLLM keys. A consumer with internal
  model ids owns its own id → LiteLLM key mapping, and should test that every
  id it bills resolves here (`RatesFor`) — that test is the consumer's
  guarantee that a data sync can't silently drop a model it depends on.

`RatesFor(model)` / `RatesForTier(model, tier)` expose the raw per-token
rates — base and tiers — for callers that want them. See `doc.go` for the
full contract.

## Vendored data

`model_prices_and_context_window.json` is embedded byte-identical from
[BerriAI/litellm](https://github.com/BerriAI/litellm) (MIT — see
`THIRD_PARTY_LICENSES`) at the commit pinned in `VENDORED_FROM`.

A weekly workflow (`.github/workflows/sync.yml`) re-fetches upstream, and if
the data changed, bumps the pin and opens a PR. The validation gate —
`go test`, run both in the sync job and by CI — fails the update if upstream
restructures the pricing schema or drops the canary models' pricing
(LiteLLM has shipped a broken cost map before). Consumers' own resolution
tests catch dropped models when they bump the module version.

Every merge to main is auto-tagged with the next patch version
(`.github/workflows/tag.yml`), so consumers always have a real module tag to
pin. Bump minor/major by pushing a tag by hand.
