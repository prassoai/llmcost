# Consumer customization layer (overrides)

## Problem statement

llmcost prices from a vendored LiteLLM snapshot that is updated weekly.
Between snapshots — and sometimes permanently, when a provider's published
pricing diverges from what LiteLLM ships — consumers carry bespoke
workaround code to correct the rates their billing uses.

The original v0.2.0 API (`ModelOverride` with `FlattenTiers`, `Rates
map[ServiceTier]Rates`, `RateSchedule []ScheduledRates`) was too bespoke.
Three distinct mechanisms for what amounts to two simple operations:
overwrite a model's rates, or remove a model.

## Goal

Collapse the override interface to two primitives — overwrite and remove —
and let consumer-side concerns (time-dependent pricing, per-service-tier
overrides) stay with the consumer.

## Revised design

### Public API surface

```go
package llmcost

import "math/big"

// Config declares consumer-specific customizations applied on top of the
// embedded LiteLLM snapshot when constructing a [Table].
type Config struct {
    // Aliases maps consumer-internal model ids to LiteLLM pricing keys.
    // Aliases do not chain: if A maps to B, B is looked up directly.
    Aliases map[string]string

    // Overrides customizes per-model pricing, keyed by LiteLLM pricing
    // key (the post-alias-resolution key).
    //
    // Non-nil *Rates = overwrite: replaces the model's rates wholesale
    // at the standard service tier. A caller flattens tiers by providing
    // Rates with no Tiers field — no dedicated flag needed.
    //
    // nil = remove: suppresses the model entirely (ok=false).
    //
    // An override for a model not in the snapshot defines a new model.
    Overrides map[string]*Rates
}

// Table is a configured pricing table: the embedded LiteLLM snapshot
// plus consumer-declared overrides. The zero value is the unoverridden
// snapshot (identical to the package-level functions). Construct with
// [New] to apply overrides.
//
// A Table is safe for concurrent use — it is immutable after construction.
type Table struct { /* unexported */ }

// New returns a Table with the given config applied. Panics on malformed
// input: empty alias targets, alias chaining, alias targets not in the
// snapshot or Overrides, unpriceable override rates.
func New(cfg Config) *Table

// Cost prices one response with alias and override resolution.
func (t *Table) Cost(model string, u Usage) (Nls, bool)

// RatesFor returns the raw per-token rates with alias and override resolution.
func (t *Table) RatesFor(model string, tier ServiceTier) (Rates, bool)

// MustParseRat parses a decimal string into an exact *big.Rat, panicking
// on malformed input.
func MustParseRat(s string) *big.Rat
```

### What changed from v0.2.0

| v0.2.0 | v0.3.0 |
|--------|--------|
| `Config.Models map[string]ModelOverride` | `Config.Overrides map[string]*Rates` |
| `ModelOverride.FlattenTiers bool` | Omit `Tiers` from the override `Rates` |
| `ModelOverride.Rates map[ServiceTier]Rates` | `*Rates` (standard tier only) |
| `ModelOverride.RateSchedule []ScheduledRates` | Removed — consumer's concern |
| `ScheduledRates` type | Removed |
| `Table.Cost(model, usage, at time.Time)` | `Table.Cost(model, usage)` |
| `Table.RatesFor(model, tier, at time.Time)` | `Table.RatesFor(model, tier)` |

### Resolution and precedence

For every `Table.Cost(model, usage)` call:

```
1. Alias resolution
   model in cfg.Aliases?  ->  model = cfg.Aliases[model]
   (one step only; the target is never re-checked against aliases)

2. Override lookup (cfg.Overrides[model])
   a. Key exists, value nil?
      -> ok=false (model suppressed)
   b. Key exists, value non-nil?
      -> use the override Rates (standard tier only)
   c. Key absent?
      -> snapshot rates (existing behavior)

3. Price using resolved rates
   Same pipeline as today: usage.disjoint(), usage.tier(),
   tier lookup, usage.premium(rates), cost(rates, components, premium),
   ceilNls.
```

### How each back mechanism maps

#### 1. Tier-structure suppression (GPT-5.6 family)

```go
// Read the snapshot's base rates, provide them without Tiers.
snap, _ := llmcost.RatesFor("gpt-5.6", llmcost.TierStandard)
table := llmcost.New(llmcost.Config{
    Overrides: map[string]*llmcost.Rates{
        "gpt-5.6": {Base: snap.Base},
        // Tiers omitted — flat pricing. Same for sol/terra/luna.
    },
})
```

Replaces: back's `isTierDivergent()` exclusion list.

#### 2. Date-gated rates (claude-sonnet-5)

Time-dependent pricing is intentionally NOT in the module. The consumer
manages it — back builds the `Config` with the correct `*Rates` for the
current period, or handles sonnet-5 natively. The static override API
stays simple.

#### 3. Model aliases (claude-sonnet-4-0)

```go
table := llmcost.New(llmcost.Config{
    Aliases: map[string]string{
        "claude-sonnet-4-0": "claude-sonnet-4-20250514",
    },
})
```

Unchanged from v0.2.0.

#### 4. Model removal

```go
table := llmcost.New(llmcost.Config{
    Overrides: map[string]*llmcost.Rates{
        "unwanted-model": nil, // Cost/RatesFor return ok=false
    },
})
```

New primitive — previously required workaround code.

### Construction-time validation

`New` panics on:

- Empty alias target.
- Alias chaining (target is itself an alias key).
- Alias target not in the snapshot or `Config.Overrides`.
- Non-nil override with unpriceable rates: non-positive input or output,
  negative cache rates, non-positive multipliers (Fast, Geo,
  RegionalUplift), context-window tiers with non-positive or
  non-ascending thresholds or unpriceable tier rates.

Nil overrides are valid — no validation needed for a removal.

### Immutability guarantees

- `New` deep-copies all mutable config state (maps, `*big.Rat` values)
  — the caller cannot corrupt the Table after construction.
- Read methods (`Cost`, `RatesFor`) return clones — the caller cannot
  corrupt the Table or the snapshot by mutating returned values.
- The zero-value Table is the unoverridden snapshot.
- A Table is safe for concurrent use.

## Backward compatibility

- **Package-level functions unchanged.** `Cost(model, usage)` and
  `RatesFor(model, tier)` keep their existing signatures and behavior.
- **Additive API.** `Table`, `Config`, `MustParseRat`, and `New` are
  exports. `ModelOverride`, `ScheduledRates`, and the `at time.Time`
  parameters on `Table.Cost`/`Table.RatesFor` are removed (breaking
  change from v0.2.0, hence the minor bump).

## Non-goals

- **Time-dependent pricing in the module.** With overwrite/remove as the
  only primitives, date-gated pricing is the consumer's concern.
- **Per-service-tier overrides.** The override applies at the standard
  tier. Consumers needing flex/priority overrides can use multiple
  override entries or handle it in their own layer.
- **Dynamic reloading.** The Table is immutable. Consumers swap the
  pointer.
