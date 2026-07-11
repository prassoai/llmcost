# Consumer customization layer (overrides)

## Problem statement

llmcost prices from a vendored LiteLLM snapshot that is updated weekly.
Between snapshots — and sometimes permanently, when a provider's published
pricing diverges from what LiteLLM ships — consumers carry bespoke
workaround code to correct the rates their billing uses. back currently
maintains three independent mechanisms in `services/cost/costmodel/
llmcost_live.go`:

1. **Date-gated rate overrides (`isDateGated` / `sonnet5StandardPricingStart`).**
   claude-sonnet-5 launched with introductory pricing; standard rates take
   effect on 2026-09-01. The static snapshot cannot represent either side
   deterministically, so back routes the model to its own native-pricing
   path. The override must express: "for model X, use rate schedule R keyed
   by the usage timestamp."

2. **Tier-structure suppression (`isTierDivergent`).**
   LiteLLM v0.1.7 gives the GPT-5.6 family (gpt-5.6, gpt-5.6-sol,
   gpt-5.6-terra, gpt-5.6-luna) a 272k context tier (2x input / 1.5x
   output above 272k) that OpenAI does NOT publish. back suppresses it to
   avoid overbilling >272k requests. The override must express: "this model
   has no context tiers; treat as flat."

3. **Model aliases (`llmcostModelAliases`).**
   back maps `claude-sonnet-4-0` -> `claude-sonnet-4-20250514`. This is a
   consumer-vocabulary mapping: the consumer's internal model id resolves
   to the LiteLLM pricing key. llmcost should own the full resolution
   chain — alias -> direct key lookup — so the consumer declares the
   mapping once and all pricing flows through it.

These are ad-hoc, imperative, scattered across back's codebase, and every
new model or pricing correction requires a back code PR instead of a
data/config update. llm-gateway (murmuration) does not carry these
workarounds today but will need equivalent customizability as it takes over
more pricing.

## Goal

A single, declarative customization layer inside llmcost. Consumers pass a
config struct at init time that declares their overrides. The layer:

- Takes precedence over the embedded LiteLLM snapshot (override > snapshot).
- Collapses all three existing back mechanisms into one API.
- Is backward compatible: no overrides = today's behavior. The package-level
  `Cost` and `RatesFor` functions remain unchanged.
- Is consumed by back, murmuration (llm-gateway), and any future consumer.

## Requirements

- **R1.** A consumer can declare model aliases (consumer-internal id ->
  LiteLLM pricing key). Resolution order: alias -> direct key lookup.
  Aliases do not chain (A->B->C is rejected at construction).
- **R2.** A consumer can suppress context-window tiers for a model,
  forcing flat (base-rate-only) pricing regardless of what the snapshot
  contains for that model. Applied per model, across all service tiers.
- **R3.** A consumer can declare a date-gated rate schedule for a model:
  a sorted list of (effective-at, rates) entries. The latest entry whose
  effective time is <= the usage timestamp is used. A non-zero timestamp
  with no matching entry is unpriced (ok=false). A zero timestamp on a
  scheduled model is unpriced — the consumer declared time-dependent
  pricing and must provide a timestamp.
- **R4.** A consumer can declare static rate overrides for a model,
  replacing the snapshot's rates entirely. This is the degenerate case of
  R3 with a single entry effective since the epoch.
- **R5.** Overrides compose with the existing sealed-Usage, exact-math,
  service-tier, and multiplier machinery. An overridden model behaves
  identically to a snapshot model in every other respect: provider-shaped
  usage normalizes the same way, fast/geo/residency premiums apply the
  same way, ceiling rounding works the same way.
- **R6.** The package-level `Cost(model, usage)` and `RatesFor(model,
  tier)` remain unchanged — they use the unoverridden snapshot. The
  override layer is opt-in via a new `Table` type.
- **R7.** Construction validates the config and panics on malformed
  input: empty alias targets, alias chains/cycles, overlapping
  `Rates`+`RateSchedule`, unpriceable override rates. Misconfiguration
  fails fast and loud at init, never at runtime.
- **R8.** The override layer adds no new dependencies beyond the standard
  library. The `time` package is the only new import.

## Proposed design

### Public API surface

One new type (`Table`), one config type (`Config`), two per-model override
types (`ModelOverride`, `ScheduledRates`), one helper (`MustParseRat`), and
methods that mirror the existing package-level functions with an added
timestamp parameter.

```go
package llmcost

import (
    "math/big"
    "time"
)

// Config declares consumer-specific customizations applied on top of the
// embedded LiteLLM snapshot when constructing a [Table].
type Config struct {
    // Aliases maps consumer-internal model ids to LiteLLM pricing keys.
    // When Table.Cost or Table.RatesFor receives a model string that
    // matches an alias key, the alias target is used for the table
    // lookup instead. Aliases do not chain: if A maps to B, B is looked
    // up directly — it is never re-checked against the alias map.
    //
    // Example: {"claude-sonnet-4-0": "claude-sonnet-4-20250514"}
    Aliases map[string]string

    // Models contains per-model overrides keyed by the LiteLLM pricing
    // key (the post-alias-resolution key). An override for a model that
    // does not exist in the snapshot is valid — it defines a new model.
    Models map[string]ModelOverride
}

// ModelOverride customizes a single model's pricing behavior. At most
// one of Rates and RateSchedule may be set (enforced at construction).
// FlattenTiers composes only with snapshot-derived rates: when Rates or
// RateSchedule provides explicit rates, the consumer controls the tier
// structure directly (omit Tiers from the provided Rates for flat pricing).
type ModelOverride struct {
    // FlattenTiers, when true, suppresses all context-window tiers from
    // the snapshot for this model, at every service tier. The model
    // prices as if it has no *_above_Xk_tokens fields — base rates
    // only, regardless of prompt size. Has no effect when Rates or
    // RateSchedule is set (the consumer controls the Tiers slice in
    // those cases).
    FlattenTiers bool

    // Rates, when non-nil, replaces the model's snapshot rates entirely.
    // The map is keyed by ServiceTier; a tier absent from the map is
    // unpriced at that tier (ok=false), matching the no-cross-tier-
    // fallback rule. At minimum, TierStandard must be present.
    // Mutually exclusive with RateSchedule.
    Rates map[ServiceTier]Rates

    // RateSchedule, when non-empty, provides time-dependent rate
    // overrides. Must be sorted by EffectiveAt (ascending); [New]
    // validates this. At query time, the latest entry whose EffectiveAt
    // <= the usage timestamp is used. When the timestamp is non-zero but
    // precedes all entries, or the timestamp is zero, the model is
    // unpriced (ok=false) — the consumer declared time-dependent pricing
    // and must provide a timestamp.
    // Mutually exclusive with Rates.
    RateSchedule []ScheduledRates
}

// ScheduledRates is one entry in a date-gated rate schedule.
type ScheduledRates struct {
    // EffectiveAt is the instant these rates take effect. Rates are
    // effective for usage timestamps >= EffectiveAt and < the next
    // entry's EffectiveAt (or indefinitely for the last entry).
    EffectiveAt time.Time

    // Rates is the complete per-service-tier rate set effective from
    // EffectiveAt. Keyed by ServiceTier; at minimum TierStandard must
    // be present. Tiers absent from the map are unpriced.
    Rates map[ServiceTier]Rates
}

// Table is a configured pricing table: the embedded LiteLLM snapshot
// plus consumer-declared overrides. The zero value is the unoverridden
// snapshot (identical to the package-level functions). Construct with
// [New] to apply overrides.
//
// A Table is safe for concurrent use — it is immutable after construction.
type Table struct {
    // unexported: holds the parsed config, pre-validated alias map,
    // and references to the snapshot (the existing sync.OnceValue table).
}

// New returns a Table with the given config applied on top of the
// embedded LiteLLM snapshot. Panics if the config is malformed:
//
//   - an alias with an empty target
//   - an alias whose target is itself an alias key (no chaining)
//   - a ModelOverride with both Rates and RateSchedule set
//   - a RateSchedule not sorted by EffectiveAt
//   - a RateSchedule or Rates entry missing TierStandard
//   - override rates that are not priceable (non-positive input or
//     output, negative cache rates — the same bar as the snapshot)
//
// Misconfiguration is a bug and fails at init, not at query time.
func New(cfg Config) *Table

// Cost prices one response using the configured table. model is resolved
// through aliases first, then looked up in overrides, then the snapshot.
// at is the usage timestamp, consulted only when the resolved model has a
// RateSchedule — otherwise it is ignored and may be zero. All other
// semantics — provider-shaped usage normalization, service-tier
// resolution, multiplier composition, exact math, ceiling rounding —
// are identical to the package-level [Cost].
func (t *Table) Cost(model string, u Usage, at time.Time) (Nls, bool)

// RatesFor returns the raw per-token rates for model at a service tier,
// with alias resolution and overrides applied. at is consulted only for
// models with a RateSchedule. The returned rats are copies.
func (t *Table) RatesFor(model string, tier ServiceTier, at time.Time) (Rates, bool)

// MustParseRat parses a decimal string (e.g. "3e-6", "0.0000025") into
// an exact *big.Rat, panicking on malformed input. This is a convenience
// for declaring override rates in Config literals without verbose
// big.Rat construction.
func MustParseRat(s string) *big.Rat
```

### Resolution and precedence

For every `Table.Cost(model, usage, at)` call:

```
1. Alias resolution
   model ∈ cfg.Aliases?  →  model = cfg.Aliases[model]
   (one step only; the target is never re-checked against aliases)

2. Override lookup (cfg.Models[model])
   a. RateSchedule non-empty AND at non-zero?
      → latest entry where EffectiveAt <= at
      → no match? → ok=false (unpriced)
   b. RateSchedule non-empty AND at zero?
      → ok=false (timestamp required for scheduled models)
   c. Rates non-nil?
      → use Rates[tier]
   d. FlattenTiers AND snapshot has model?
      → snapshot rates with Tiers cleared
   e. No override?
      → snapshot rates (existing behavior)

3. Price using resolved rates
   Same pipeline as today: usage.disjoint(), usage.tier(),
   table lookup by service tier, usage.premium(rates),
   cost(rates, components, premium), ceilNls.
```

### How each back mechanism maps

#### 1. Date-gated rates (claude-sonnet-5)

```go
// back's sonnet5StandardPricingStart = 2026-09-01
table := llmcost.New(llmcost.Config{
    Models: map[string]llmcost.ModelOverride{
        "claude-sonnet-5": {
            RateSchedule: []llmcost.ScheduledRates{
                {
                    // Introductory pricing from launch
                    EffectiveAt: time.Date(2026, 6, 1, 0, 0, 0, 0, time.UTC),
                    Rates: map[llmcost.ServiceTier]llmcost.Rates{
                        llmcost.TierStandard: {
                            Base: llmcost.TierRates{
                                Input:         llmcost.MustParseRat("3e-6"),
                                CacheRead:     llmcost.MustParseRat("3e-7"),
                                CacheCreation: llmcost.MustParseRat("3.75e-6"),
                                Output:        llmcost.MustParseRat("15e-6"),
                            },
                        },
                    },
                },
                {
                    // Standard pricing kicks in 2026-09-01
                    EffectiveAt: time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC),
                    Rates: map[llmcost.ServiceTier]llmcost.Rates{
                        llmcost.TierStandard: {
                            Base: llmcost.TierRates{
                                Input:         llmcost.MustParseRat("4e-6"),
                                CacheRead:     llmcost.MustParseRat("4e-7"),
                                CacheCreation: llmcost.MustParseRat("5e-6"),
                                Output:        llmcost.MustParseRat("20e-6"),
                            },
                        },
                    },
                },
            },
        },
    },
})

// Usage from June 2026 bills at introductory rates:
nls, ok := table.Cost("claude-sonnet-5", usage, time.Date(2026, 7, 15, 0, 0, 0, 0, time.UTC))

// Usage from October 2026 bills at standard rates:
nls, ok = table.Cost("claude-sonnet-5", usage, time.Date(2026, 10, 1, 0, 0, 0, 0, time.UTC))
```

Replaces: back's `isDateGated()`, `sonnet5StandardPricingStart`, and the
native-pricing routing path. The consumer passes the usage timestamp (from
the API response or the billing record) and the schedule resolves the
correct rates. No imperative code, no special-case routing.

#### 2. Tier-structure suppression (GPT-5.6 family)

```go
table := llmcost.New(llmcost.Config{
    Models: map[string]llmcost.ModelOverride{
        "gpt-5.6":       {FlattenTiers: true},
        "gpt-5.6-sol":   {FlattenTiers: true},
        "gpt-5.6-terra": {FlattenTiers: true},
        "gpt-5.6-luna":  {FlattenTiers: true},
    },
})

// A 300k-token prompt bills at base rates, not the 272k tier:
nls, ok := table.Cost("gpt-5.6", llmcost.OpenAIUsage{
    InputTokens: 300000, OutputTokens: 1000,
}, time.Time{}) // zero timestamp — no schedule, ignored
```

Replaces: back's `isTierDivergent()` exclusion list. The snapshot's 272k
context tier for these models is suppressed — the model is treated as
flat. When the snapshot is eventually corrected (LiteLLM removes the
bogus tier), the consumer drops the override and the behavior is the same.

#### 3. Model aliases (claude-sonnet-4-0)

```go
table := llmcost.New(llmcost.Config{
    Aliases: map[string]string{
        "claude-sonnet-4-0": "claude-sonnet-4-20250514",
    },
})

// The consumer's internal id resolves to the snapshot key:
nls, ok := table.Cost("claude-sonnet-4-0", usage, time.Time{})
// Equivalent to: llmcost.Cost("claude-sonnet-4-20250514", usage)
```

Replaces: back's `llmcostModelAliases` map. The consumer declares the
mapping once in the config; every `Table.Cost` and `Table.RatesFor` call
resolves it transparently.

### Composing all three

A single `Config` can combine all three mechanisms:

```go
table := llmcost.New(llmcost.Config{
    Aliases: map[string]string{
        "claude-sonnet-4-0": "claude-sonnet-4-20250514",
    },
    Models: map[string]llmcost.ModelOverride{
        "claude-sonnet-5": {
            RateSchedule: []llmcost.ScheduledRates{
                {EffectiveAt: launchDate, Rates: introRates},
                {EffectiveAt: standardDate, Rates: standardRates},
            },
        },
        "gpt-5.6":       {FlattenTiers: true},
        "gpt-5.6-sol":   {FlattenTiers: true},
        "gpt-5.6-terra": {FlattenTiers: true},
        "gpt-5.6-luna":  {FlattenTiers: true},
    },
})
```

Models not mentioned in the config use the snapshot unchanged. The
package-level `Cost` and `RatesFor` are completely unaffected.

### Interaction with ModelSelector

`ModelSelector` is orthogonal to the override layer. It resolves a
provider + vendor model name into a LiteLLM pricing key via the cloud's
renaming grammar. Aliases are a different concern: they map consumer-
internal vocabulary to LiteLLM keys.

The consumer can compose both:

```go
// Use ModelSelector to find the key for a Bedrock model:
key, ok := llmcost.ModelSelector{
    Provider: llmcost.ProviderBedrock,
    Model:    "claude-sonnet-4-5",
}.Key()

// Use an alias for a consumer-internal name:
table := llmcost.New(llmcost.Config{
    Aliases: map[string]string{"our-sonnet": key},
})
nls, ok := table.Cost("our-sonnet", usage, time.Time{})
```

`ModelSelector` does not consult aliases — it remains a pure key-grammar
function operating on the snapshot. This separation keeps both mechanisms
simple and independently testable.

### Internal implementation sketch

The `Table` struct holds a reference to the parsed snapshot (the existing
`sync.OnceValue` table) and the validated `Config`. No data is copied or
merged at construction — resolution happens at query time:

```go
type Table struct {
    aliases  map[string]string
    models   map[string]modelOverride // validated, schedule sorted
    snapshot func() map[string]map[ServiceTier]Rates // the existing table()
}

func (t *Table) resolve(model string) string {
    if target, ok := t.aliases[model]; ok {
        return target
    }
    return model
}

func (t *Table) rates(model string, tier ServiceTier, at time.Time) (Rates, bool) {
    model = t.resolve(model)

    if ovr, ok := t.models[model]; ok {
        switch {
        case len(ovr.schedule) > 0:
            return ovr.scheduledRates(tier, at)
        case ovr.rates != nil:
            r, ok := ovr.rates[tier]
            return r, ok
        case ovr.flattenTiers:
            r, ok := t.snapshot()[model][tier]
            if ok {
                r.Tiers = nil
            }
            return r, ok
        }
    }

    r, ok := t.snapshot()[model][tier]
    return r, ok
}
```

The zero-value `Table` has nil aliases and models, so `resolve` returns
the model unchanged and `rates` falls through to the snapshot — the
package-level functions can delegate to a zero `Table` internally.

## Backward compatibility

- **Package-level functions unchanged.** `Cost(model, usage)` and
  `RatesFor(model, tier)` keep their existing signatures and behavior.
  They operate on the unoverridden snapshot. No consumer needs to change
  anything when bumping to a version with the override layer.
- **No new required dependencies.** The `time` package is the only new
  import; it is already transitively available in every Go program.
- **Additive API.** `Table`, `Config`, `ModelOverride`, `ScheduledRates`,
  `MustParseRat`, and `New` are new exports. Nothing is removed or renamed.
- **Opt-in.** A consumer that does not call `New` never touches the
  override layer. The package-level functions are the default path.

## Consumer migration sketch

### back

Replace the three mechanisms in `services/cost/costmodel/llmcost_live.go`
with a single `llmcost.New(cfg)` call at init. The `Config` is built from
back's existing constants (model lists, date thresholds, alias maps),
turning imperative switch-case workaround code into a declarative struct
literal. The `isDateGated`, `isTierDivergent`, and `llmcostModelAliases`
code paths are deleted.

### murmuration (llm-gateway)

Currently calls `llmcost.Cost` directly with no workarounds. When
murmuration needs any of these customizations — or new ones — it
constructs a `Table` and calls `table.Cost` instead. No change required
until a customization is needed.

### llm-gateway (standalone)

Same as murmuration. The `Table` is constructed at service init with
whatever overrides the gateway needs and threaded through the pricing
call sites.

## Testing strategy

### Unit tests in llmcost

Requirements tests (one per R1-R8), following the existing pattern:

- **TestAliasResolution**: alias resolves, unknown alias falls through,
  alias target is never re-aliased (no chaining).
- **TestFlattenTiers**: a model with snapshot tiers prices flat when
  FlattenTiers is set; the snapshot's tier rates are ignored.
- **TestRateSchedule**: time-dependent resolution — intro period uses
  intro rates, standard period uses standard rates, before-all-entries
  fails, zero timestamp fails.
- **TestStaticRateOverride**: override rates replace the snapshot entirely;
  the model prices from the override, not the snapshot.
- **TestOverridePrecedence**: schedule > static > flatten+snapshot >
  snapshot. Each level tested by constructing the appropriate Config.
- **TestOverrideWithMultipliers**: an overridden Claude model with
  fast/geo multipliers defined in the override's Rates prices the
  premiums correctly (R5 — overrides compose with existing machinery).
- **TestOverrideServiceTiers**: override rates keyed by ServiceTier; a
  tier absent from the override map fails (ok=false), matching the
  existing no-cross-tier-fallback contract.
- **TestMalformedConfigPanics**: each validation case in New panics —
  empty alias target, alias chain, both Rates and RateSchedule,
  unsorted schedule, missing TierStandard, unpriceable rates.
- **TestZeroTableIsSnapshot**: `(&Table{}).Cost(model, usage, time.Time{})`
  equals `Cost(model, usage)` for every model.
- **TestMustParseRat**: valid decimals parse exactly; invalid input
  panics.

### Consumer tests

Consumers should test that their Config declares every model they bill
and that each resolves. The existing pattern — murmuration's
`TestCodexModelsPriceable` — extends naturally: the test constructs the
consumer's Table and asserts that each model prices at the expected
tiers.

## Open questions

1. **Override rates and service tiers.** The proposed design has override
   rates keyed by `map[ServiceTier]Rates`. Back's current cases are all
   standard-tier-only (Anthropic models, GPT-5.6 flat), so the map is
   always `{TierStandard: ...}`. Should we simplify to a single `Rates`
   (standard only) and add the service-tier dimension later if needed?
   Pro: simpler API for the current cases. Con: if a future override
   needs flex/priority rates, the API changes. **Recommendation:** keep
   the map — the cost is one map literal, and it future-proofs the API
   without adding complexity to the resolution logic.

2. **Override rates and price multipliers.** When a model has override
   rates AND the snapshot has fast/geo multipliers, should the multipliers
   carry over from the snapshot, or must the override declare them
   explicitly? The proposed design uses the override's `Rates` struct,
   which includes Fast/Geo/RegionalUplift fields — so the override
   controls the multipliers too. This means an override that wants the
   snapshot's multipliers must re-declare them. **Recommendation:** this
   is the right default. Multipliers are part of the rate structure, and
   an override that replaces rates should be explicit about what it
   replaces. A consumer that wants only to change the base prices and
   keep the multipliers should use `Table.RatesFor` on the snapshot to
   read them, then copy them into the override. A convenience for this
   (e.g. "inherit multipliers from snapshot") adds API surface for a rare
   case — not worth it initially.

3. **FlattenTiers across service tiers.** The proposed design says
   FlattenTiers suppresses tiers at every service tier. The GPT-5.6 case
   only needs standard (those models have no flex/priority in LiteLLM
   today). Is across-all-tiers correct, or should it be per-tier?
   **Recommendation:** across all tiers. If the snapshot's tier structure
   is wrong for standard, it is wrong for flex/priority too (the 272k
   tier came from the same upstream data). Per-tier flatten adds
   complexity for no known use case.

4. **Alias targets that are not direct LiteLLM keys.** Should an alias
   target be required to exist in the snapshot (or as an override), or
   can it be any string that eventually resolves through ModelSelector?
   The proposed design requires the alias target to be a LiteLLM pricing
   key (a direct table key). ModelSelector resolution is the caller's
   responsibility before aliasing. **Recommendation:** keep this
   constraint. Aliases are a simple rename layer; coupling them to
   ModelSelector's provider grammar makes both harder to reason about
   and test.

5. **Tag strategy.** This adds new exports but does not change existing
   signatures. A minor tag (v0.2.0) signals "new API surface, existing
   API unchanged" — the same bar as the service-tier addition. The
   auto-patch workflow (tag.yml) should not be used for this; the
   maintainer pushes a minor tag manually.

6. **Should `Table.Cost` accept `time.Time` or a more restricted
   timestamp type?** `time.Time` is the natural choice and composes
   with the standard library. A newtype (e.g. `UsageTime`) would make
   the "this is the usage timestamp, not wall-clock time" intent
   clearer but adds ceremony. **Recommendation:** use `time.Time`.
   The doc comment on `Table.Cost` makes the semantics clear.

7. **Should overrides for nonexistent snapshot models be allowed?** The
   proposed design allows it — a `ModelOverride` with `Rates` or
   `RateSchedule` for a model not in the snapshot is valid and defines
   a new model. This is useful when a new model ships before the
   weekly snapshot sync picks it up. **Recommendation:** allow it.
   Validation should still require priceable rates.

## Non-goals

- **Full LiteLLM replacement.** The override layer supplements the
  snapshot; it does not replace the vendored data pipeline. The weekly
  sync remains the primary source of pricing truth.
- **Dynamic reloading.** The `Table` is immutable after construction.
  Consumers that need to change overrides at runtime construct a new
  `Table` and swap the pointer. No hot-reload machinery.
- **Provider-shaped overrides.** Overrides use `Rates` (the internal
  rate structure), not `ClaudeUsage`/`OpenAIUsage`. The provider
  distinction is a usage-normalization concern, not a rate concern.
- **Touching back's code.** This PR is the design proposal for llmcost.
  back's migration to use it is a follow-up PR after design sign-off.
