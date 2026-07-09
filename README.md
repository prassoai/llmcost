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
    Speed:                      "",    // the REQUEST's speed param — "fast" bills the 2×–6× fast premium
    InferenceGeo:               "",    // usage.inference_geo — a pinned region bills its premium (us: 1.1×)
})
```

**OpenAI / codex** (overlapping counts — `input_tokens` INCLUDES cached):

```go
cost, ok := llmcost.Cost("gpt-5.4", llmcost.OpenAIUsage{
    InputTokens:       46200, // usage.input_tokens — total, cached included
    CachedInputTokens: 45000, // input_tokens_details.cached_tokens — subset
    OutputTokens:      800,   // reasoning tokens are a subset, already included
    DataResidency:     "",    // "eu"/"us" when served by eu./us.api.openai.com — bills the 1.1× uplift
    ServiceTier:       "",    // "" = standard; TierFlex/TierPriority bill that tier's rates
})
```

`ok` is false whenever the response can't be priced — unknown model, or usage
on a component the model has no rate for. Nothing silently bills zero.

## Service tiers — flex and priority

OpenAI bills the same request differently by processing tier: **flex**
(cheaper, slower) and **priority** (pricier, faster) publish their own rates
(LiteLLM's `*_flex` / `*_priority` variants). The tier rides on
`OpenAIUsage.ServiceTier` (zero value = standard) — Claude usage has no tier
knob.

```go
cost, ok := llmcost.Cost("gpt-5.5", llmcost.OpenAIUsage{
    InputTokens:       46200,
    CachedInputTokens: 45000,
    OutputTokens:      800,
    ServiceTier:       llmcost.TierFlex,
})
```

The no-fallback rule extends across tiers: a model without rates at the
requested tier (gpt-5.3-codex has no flex), a tier missing a component the
usage reports, or an unrecognized non-empty `ServiceTier` value all return
`ok=false` — flex/priority usage is
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
- **Price multipliers, no silent standard rates.** Anthropic's fast mode
  (`speed: "fast"`, ×6 on opus-4-6/4-7, ×2 on opus-4-8) and pinned-region
  inference (`usage.inference_geo`, ×1.1 for `us`) multiply uncached input
  and output — cache traffic is never scaled — and compose multiplicatively.
  A mode the model has no factor for *fails* to price: a fast premium is up
  to 6×, so billing standard rates on a data lag is a silent 6× underbill.
  OpenAI's data residency (`eu.`/`us.api.openai.com`) uplifts **every**
  component including cache reads (×1.1 on gpt-5.4/5.5); models OpenAI
  doesn't regionally price bill standard, as in LiteLLM. Multipliers scale
  the rates of whichever service tier is being priced.
- **Exact math, ceiling at the total.** Rates parse from decimal literals
  into `math/big.Rat` — never through float64. The response is priced exactly
  in USD and converted to nls only at the final total, **ceiling-rounded**.
  Sub-nls token costs accumulate correctly; any non-zero usage costs at
  least 1 nls.
- **Provider in the key, grammar in `ModelSelector`, no fuzzy aliasing.**
  The same vendor model bills differently per serving provider —
  `gpt-5.4` vs `azure/gpt-5.4` vs `azure/us/gpt-5.4` (Azure data zones carry
  a ~10% premium in the rates), Claude direct vs Bedrock AWS ids vs
  `vertex_ai/…` — each its own key. `ModelSelector{Provider, Model, Region}.Key()`
  resolves the provider's **native id** verbatim or the **vendor's canonical
  name** through each cloud's bespoke renaming scheme (Bedrock's
  `anthropic.` prefix, `-v1:0` artifact versions, `us.` geo profiles;
  Vertex's `@date`/`@default`; Azure's `gpt-35` spelling):
  `{Bedrock, "claude-sonnet-4-5", "us"}` →
  `us.anthropic.claude-sonnet-4-5-20250929-v1:0`. Resolution is
  **deterministic and verified** — an ambiguous undated name fails, a
  missing region key fails (never the cheaper global key), a cross-provider
  key fails — and `TestSelectorCanonicalCoverage` gates every data sync on
  the whole scheme: each cloud-served Anthropic/OpenAI key must stay
  selectable by its vendor name. Which selectors you bill is your policy:
  test that each resolves — that test is your guarantee that a data sync
  can't silently drop a model you depend on.

`RatesFor(model, tier)` exposes the raw per-token
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
