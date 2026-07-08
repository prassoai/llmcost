# llmcost

Shared Go module: LLM inference cost from vendored LiteLLM pricing data (MIT).
Consumed by murmur, llm-gateway, and back. Data plus a lookup and an exact
multiply — nothing clever.

```go
import "github.com/prassoai/llmcost"

cost, ok := llmcost.Cost("claude-opus-4-8", llmcost.Usage{
    InputTokens:       1200,  // uncached input
    CachedInputTokens: 45000, // cache reads (disjoint from InputTokens)
    OutputTokens:      800,
})
// cost is in nls: 1 nls = 1/100,000 USD (back's billing unit — its protos
// call it "centimills"). ok is false if the model can't be priced.
```

## Semantics

- **Exact math.** Rates are parsed from LiteLLM's decimal literals into
  `math/big.Rat` — never through float64. The whole response is priced
  exactly in USD and converted to nls only at the final total,
  **ceiling-rounded** (matching back, which ceiling-rounds its margin'd
  total). Sub-nls token costs accumulate correctly; any non-zero usage costs
  at least 1 nls.
- **Fail loud.** Unknown models, and models LiteLLM lists without positive
  input and output rates, return `ok=false`. Nothing ever silently bills zero.
- **Aliases.** Internal model ids (`claude-opus-4-8`, `gpt-5.4`,
  `codex-mini`, …) resolve through the alias map in `aliases.go` — the single
  place to register a new internal id. Ids outside the map are tried as
  LiteLLM keys directly, so arbitrary upstream keys (`gpt-4o`) also price.
- **Token counts are Anthropic-style disjoint.** `InputTokens` excludes cache
  reads. OpenAI-style callers (where `prompt_tokens` includes
  `cached_tokens`) must subtract before constructing a `Usage`.

See `doc.go` for the full contract.

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
