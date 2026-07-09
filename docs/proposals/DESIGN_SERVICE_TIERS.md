# Service-tier pricing (flex / priority)

## Problem statement

llmcost prices only OpenAI's standard service tier. The vendored LiteLLM
data carries per-request tier variants — `*_flex` (≈half price, slower) on
23 models and `*_priority` (≈2×, faster) on 78, including
context-window-tiered priority rates (`input_cost_per_token_above_272k_
tokens_priority`) — but `parseModel` deliberately excludes them, so a
consumer running requests at a non-standard tier cannot price them at all.
Concretely: murmuration's codex backend exposes flex/priority-equivalent
service tiers and today records $0.00 for every such turn, with a
documented "when llmcost grows tier support" TODO pointing here
(murmuration `docs/proposals/DESIGN_CODEX_COST.md`, R6).

## Requirements

- **R1** A caller can price a response at OpenAI's `flex` or `priority`
  service tier with the same exactness guarantees as standard pricing
  (big.Rat arithmetic, ceiling-round only the final total).
- **R2** The change is additive and backward-compatible: `Cost` and
  `RatesFor` keep their signatures and semantics, and are exactly the
  standard-tier views of the new API.
- **R3** No cross-tier fallback: pricing at a tier the model has no rates
  for — or usage reporting a component the tier has no rate for — returns
  `ok=false`. Flex/priority usage is never billed at standard rates (a
  silent ~2× error in either direction).
- **R4** Context-window tiers work within a service tier: the
  `*_above_Xk_tokens_priority` family prices with the same
  strict-threshold, whole-request-re-rates semantics as standard
  context tiers, and a component the context tier leaves untiered inherits
  that service tier's base rate (never standard's).
- **R5** Only recognized tiers resolve: an unknown or empty tier string
  returns `ok=false`, never a silent standard fallback.
- **R6** Standard rates remain the table-membership anchor: a model
  without priceable standard rates does not resolve at any tier
  (unchanged behavior — LiteLLM has no tier-only models; such an entry is
  upstream garbage).
- **R7** The weekly-sync validation gate extends to tiers: table
  invariants cover every parsed service tier, and data canaries pin known
  tier shapes (gpt-5.5 flex+priority, gpt-5.3-codex priority-only) so a
  sync that drops or restructures tier fields fails the gate.
- **R8** LiteLLM's `*_batches` variants (Batch API) stay out of scope:
  they price asynchronous batch jobs, not the per-request service tiers
  this module models. The parser must keep excluding them.

## Proposed design

One new exported type and two new functions; everything else is the
existing machinery parameterized by a key suffix.

```go
// ServiceTier selects which of a model's per-request rate variants price
// a response.
type ServiceTier string

const (
    TierStandard ServiceTier = "standard"
    TierFlex     ServiceTier = "flex"     // LiteLLM's *_flex fields
    TierPriority ServiceTier = "priority" // LiteLLM's *_priority fields
)

func CostTier(model string, tier ServiceTier, u Usage) (Nls, bool)     // (satisfies R1, R3, R5)
func RatesForTier(model string, tier ServiceTier) (Rates, bool)

func Cost(model string, u Usage) (Nls, bool)  // ≡ CostTier(model, TierStandard, u)   (satisfies R2)
func RatesFor(model string) (Rates, bool)     // ≡ RatesForTier(model, TierStandard)  (satisfies R2)
```

- **Table shape.** `table()` becomes `map[string]map[ServiceTier]Rates`.
  `parseModel` parses the standard tier exactly as today and gates
  membership on it (R6), then independently parses `_flex` and
  `_priority`, storing whichever are priceable. LiteLLM key names compose
  as `<component><ctx-suffix><svc-suffix>` (e.g. `input_cost_per_token` +
  `_above_272k_tokens` + `_priority`), so the existing `rates` helper
  gains a service-suffix parameter and nothing else changes. (R1, R4)
- **Tier-key detection.** The context-tier regex grows an optional
  service-suffix group — `^input_cost_per_token_above_(\d+)(k?)_tokens
  (_flex|_priority)?$` — and each service tier's parse accepts only keys
  whose group matches its own suffix. `_batches` (and any future suffix)
  matches neither alternative and stays excluded, preserving today's
  standard-tier behavior byte-for-byte. (R4, R8)
- **No cross-tier inheritance.** A service tier's `Rates` are built only
  from that tier's fields: a component absent at the tier is nil and fails
  on positive counts, exactly like the existing absent-component rule.
  Within a service tier, context tiers inherit that tier's base — the
  existing intra-tier rule, now scoped per service tier. Rationale: the
  absent-component philosophy already in doc.go ("nothing ever silently
  bills zero or falls back to another component's rate") extends to tiers;
  inheriting standard's cache-read into flex would overbill 2×. (R3, R4)
- **Lookup.** `CostTier` validates usage first (impossible counts panic
  even for unknown models/tiers, as today), then `table()[model][tier]` —
  an unknown model, missing tier, or unrecognized tier string is one
  uniform `ok=false`. (R3, R5)
- **Composition with price multipliers.** Fast/geo/residency premiums
  (added on main in parallel with this design) resolve against the
  service tier's own `Rates` and scale its tier-resolved rates — the same
  premium × context-tier composition as standard, applied per service
  tier. Each tier's parse carries the model's multipliers and provider
  attribution, and `ModelSelector` reads them from the standard tier,
  which anchors membership (R6).
- **Validation gate.** `TestTableInvariants` iterates every model × parsed
  tier (priceable rates, strictly-ascending positive thresholds, standard
  always present); `TestVendoredDataCanaries` pins gpt-5.5 (flex and
  priority resolve), gpt-5.3-codex (priority resolves, flex does not);
  `TestCostMatchesRatesFor` extends across tiers. Fixture tests pin exact
  flex/priority costs and the priority context-window tier. (R7)

## Caller-facing interface

The Go API is the caller surface. Pricing a flex-tier response:

```go
nls, ok := llmcost.CostTier("gpt-5.5", llmcost.TierFlex, llmcost.OpenAIUsage{
    InputTokens:       u.InputTokens,       // total input, cached included
    CachedInputTokens: u.CachedInputTokens,
    OutputTokens:      u.OutputTokens,
})
```

`ok=false` when the model has no flex rates (e.g. gpt-5.3-codex) — the
caller decides whether that is "record zero and log" (murmuration's
choice) or an error. Existing callers are untouched: `Cost`/`RatesFor`
compile and behave identically.

Consumers own the mapping from their tier vocabulary to these constants
(same rule as model ids): murmuration's follow-up maps its codex
`service_tier` values (`flex` → `TierFlex`, `fast` → `TierPriority`) and
drops its non-standard-tier zero-cost gate.

## Open questions

- **Tag.** This is an additive API change: it deserves a minor tag
  (v0.2.0) rather than the auto-patch the merge workflow applies. The
  maintainer pushes minor tags manually (tag.yml preserves them).
- **Batch pricing (R8).** If a consumer ever needs Batch API pricing, the
  same mechanism extends (`TierBatch`/`_batches`), but batch jobs have
  different usage semantics (no interactive turn) — out of scope until a
  real consumer exists.
