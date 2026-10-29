# List prices (`listprice`)

The full design is murmuration's
[`DESIGN_LLM_INFERENCE_LIST_PRICING.md`](https://github.com/prassoai/murmuration/blob/main/docs/proposals/DESIGN_LLM_INFERENCE_LIST_PRICING.md)
(prassoai/murmuration#13471). This records what llmcost implements of it
(Phase 1).

## What ships here

- **`listprice`** — the one inference price definition murmur and Macroscope
  share: `price = ceil(cost × (10000 + MarkupBasisPoints) / 10000)` with
  `MarkupBasisPoints = 500`, where cost is llmcost's. `Of` prices a call and
  reports its `Source` (llmcost, override, or degraded); `Priceable` answers
  before the call; `Markup` re-derives a price from a recorded cost;
  `CurrentVersion` names the price book (`<module version>+litellm.<commit8>`).
- **`listprice/published.go`** — the curated models sold by name (product id
  → llmcost key, which is where aliases live) and rate overrides. Every entry
  must price at the standard tier; CI enforces it.
- **`listprice.Book()`** — the rate card as exact USD per million tokens, cost
  and price per model, service tier, speed, context tier and component.
  Consumers render the public price page and the cost-vs-price audit table
  from it.
- **`Models(provider)`, `LiteLLMCommit()`** — enumeration and provenance the
  book needs from the root package.
- **Daily refresh** — `sync.yml` runs daily; `cmd/pricediff` classifies each
  sync's effect on the book, and the class sets the merge policy (README).

## Degradation, deliberately narrow

Only response-only dimensions degrade: an inference geo or a 1-hour cache
write the model has no rate for prices at the base rate, `SourceDegraded`,
because the call already happened and must not be free. Request-time modes —
the model, the service tier, fast mode — never degrade; `Priceable` refuses
them before the call.

## Deferred

Macroscope's aliases (`llmcostConfig()` in back) move into `published.go`
when back adopts `listprice` (the design's Phase 5).
