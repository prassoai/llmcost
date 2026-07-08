# llmcost

Shared Go module: LLM inference cost from vendored LiteLLM pricing data (MIT).
Consumed by murmur, llm-gateway, and back. Data plus a lookup and an exact
multiply — nothing clever.

Costs are in **nls**: 1 nls = 1/100,000 USD (back's billing unit — its protos
call it "centimills").

## Usage — provider-explicit, raw counts in

The two providers report token counts **differently**. Each gets its own
entry point taking that provider's RAW usage fields; normalization happens
inside. Never pre-subtract, never mix shapes.

**Anthropic** (disjoint counts — `input_tokens` EXCLUDES cache activity):

```go
cost, ok := llmcost.ClaudeCost("claude-opus-4-8", llmcost.ClaudeUsage{
    InputTokens:              1200,  // usage.input_tokens — uncached input only
    CacheReadInputTokens:     45000, // usage.cache_read_input_tokens
    CacheCreationInputTokens: 3000,  // usage.cache_creation_input_tokens
    OutputTokens:             800,   // usage.output_tokens
})
```

**OpenAI / codex** (overlapping counts — `input_tokens` INCLUDES cached):

```go
cost, ok := llmcost.OpenAICost("gpt-5.4", llmcost.OpenAIUsage{
    InputTokens:       46200, // usage.input_tokens — total, cached included
    CachedInputTokens: 45000, // input_tokens_details.cached_tokens — subset
    OutputTokens:      800,   // reasoning tokens are a subset, already included
})
```

`ok` is false whenever the response can't be priced — unknown model, or usage
on a component the model has no rate for. Nothing silently bills zero.

## Semantics

- **Real cache rates, no fallback.** Cache reads bill at the model's
  `cache_read_input_token_cost`, cache writes at its
  `cache_creation_input_token_cost` (Anthropic: 0.1× and 1.25× input;
  OpenAI reports reads only). A model missing a cache rate *fails* to price
  usage reporting those tokens — it is never approximated at the input rate.
  Anthropic's 1-hour-TTL write premium is out of scope (single write count,
  billed at the 5-minute rate).
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
