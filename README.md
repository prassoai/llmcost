# llmcost

Shared Go module: LLM inference cost from vendored LiteLLM pricing data (MIT).
Consumed by murmur, llm-gateway, and back. Data plus a lookup and an exact
multiply — nothing clever.

Costs are in **nls**: 1 nls = 1/100,000 USD (back's billing unit — its protos
call it "centimills").

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
  components inherit base rates; OpenAI priority/flex tiers are out of scope.
- **Exact math, ceiling at the total.** Rates parse from decimal literals
  into `math/big.Rat` — never through float64. The response is priced exactly
  in USD and converted to nls only at the final total, **ceiling-rounded**
  (matching back, which ceiling-rounds its margin'd total). Sub-nls token
  costs accumulate correctly; any non-zero usage costs at least 1 nls.
- **Aliases.** Internal model ids (`claude-opus-4-8`, `gpt-5.4`,
  `codex-mini`, …) resolve through the alias map in `aliases.go` — the single
  place to register a new internal id. Ids outside the map are tried as
  LiteLLM keys directly, so arbitrary upstream keys (`gpt-4o`) also price.

`RatesFor(model)` exposes the raw per-token rates — base and tiers — for
callers that want them. See `doc.go` for the full contract.

## Vendored data

`model_prices_and_context_window.json` is embedded byte-identical from
[BerriAI/litellm](https://github.com/BerriAI/litellm) (MIT — see
`THIRD_PARTY_LICENSES`) at the commit pinned in `VENDORED_FROM`.

A weekly workflow (`.github/workflows/sync.yml`) re-fetches upstream, and if
the data changed, bumps the pin and opens a PR. The validation gate —
`TestAliasesResolve`, run both in the sync job and by CI — fails the update if
any aliased model no longer resolves to non-zero rates, catching upstream
renames or a broken cost map before it reaches a consumer.

Every merge to main is auto-tagged with the next patch version
(`.github/workflows/tag.yml`), so consumers always have a real module tag to
pin. Bump minor/major by pushing a tag by hand.
