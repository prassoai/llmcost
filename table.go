package llmcost

import (
	"fmt"
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
	// key (the post-alias-resolution key). An override with Rates or
	// RateSchedule for a model not in the snapshot is valid — it defines
	// a new model. A FlattenTiers-only override requires the model to
	// exist in the snapshot (there must be tiers to flatten).
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
	aliases map[string]string
	models  map[string]modelOverride
}

// modelOverride is the validated, deep-copied internal form of [ModelOverride].
type modelOverride struct {
	flattenTiers bool
	rates        map[ServiceTier]Rates
	schedule     []scheduledRates
}

// scheduledRates is the internal form of [ScheduledRates].
type scheduledRates struct {
	effectiveAt time.Time
	rates       map[ServiceTier]Rates
}

// MustParseRat parses a decimal string (e.g. "3e-6", "0.0000025") into
// an exact *big.Rat, panicking on malformed input. This is a convenience
// for declaring override rates in Config literals without verbose
// big.Rat construction.
func MustParseRat(s string) *big.Rat {
	r, ok := new(big.Rat).SetString(s)
	if !ok {
		panic(fmt.Sprintf("llmcost: invalid rate %q", s))
	}
	return r
}

// New returns a Table with the given config applied on top of the
// embedded LiteLLM snapshot. Panics if the config is malformed:
//
//   - an alias with an empty target
//   - an alias whose target is itself an alias key (no chaining)
//   - an alias whose target is not a snapshot key and not a
//     Config.Models key (catches typos at init, not at query time;
//     models not yet in the snapshot are declared in Config.Models)
//   - a snapshot-dependent ModelOverride (FlattenTiers only, no Rates
//     or RateSchedule) for a model not in the snapshot — FlattenTiers
//     modifies snapshot rates, so there must be something to flatten
//   - a ModelOverride with both Rates and RateSchedule set
//   - a RateSchedule not sorted by EffectiveAt
//   - a RateSchedule or Rates entry missing TierStandard
//   - override rates that violate the same invariants the snapshot
//     parser enforces (TestTableInvariants): non-positive input or
//     output, negative cache rates, non-positive multipliers (Fast,
//     Geo, RegionalUplift), and context-window tiers with non-positive
//     or non-ascending thresholds or unpriceable tier rates
//
// Misconfiguration is a bug and fails at init, not at query time.
func New(cfg Config) *Table {
	snap := table()
	t := &Table{}

	if len(cfg.Aliases) > 0 {
		t.aliases = make(map[string]string, len(cfg.Aliases))
		for k, v := range cfg.Aliases {
			if v == "" {
				panic(fmt.Sprintf("llmcost: alias %q has empty target", k))
			}
			if _, chain := cfg.Aliases[v]; chain {
				panic(fmt.Sprintf("llmcost: alias %q -> %q chains (target is itself an alias key)", k, v))
			}
			if _, inSnap := snap[v]; !inSnap {
				if _, inModels := cfg.Models[v]; !inModels {
					panic(fmt.Sprintf("llmcost: alias %q -> %q: target not in snapshot or Config.Models", k, v))
				}
			}
			t.aliases[k] = v
		}
	}

	if len(cfg.Models) > 0 {
		t.models = make(map[string]modelOverride, len(cfg.Models))
		for model, ovr := range cfg.Models {
			if ovr.Rates != nil && len(ovr.RateSchedule) > 0 {
				panic(fmt.Sprintf("llmcost: %s: both Rates and RateSchedule set", model))
			}
			if ovr.Rates == nil && len(ovr.RateSchedule) == 0 {
				if _, inSnap := snap[model]; !inSnap {
					panic(fmt.Sprintf("llmcost: %s: snapshot-dependent override for model not in snapshot", model))
				}
			}

			var mo modelOverride
			mo.flattenTiers = ovr.FlattenTiers

			if ovr.Rates != nil {
				if _, ok := ovr.Rates[TierStandard]; !ok {
					panic(fmt.Sprintf("llmcost: %s: Rates missing TierStandard", model))
				}
				mo.rates = cloneAndValidateServiceTierRates(model, ovr.Rates)
			}

			if len(ovr.RateSchedule) > 0 {
				mo.schedule = make([]scheduledRates, len(ovr.RateSchedule))
				for i, sr := range ovr.RateSchedule {
					if i > 0 && !sr.EffectiveAt.After(ovr.RateSchedule[i-1].EffectiveAt) {
						panic(fmt.Sprintf("llmcost: %s: RateSchedule not sorted ascending by EffectiveAt", model))
					}
					if _, ok := sr.Rates[TierStandard]; !ok {
						panic(fmt.Sprintf("llmcost: %s: RateSchedule[%d] missing TierStandard", model, i))
					}
					mo.schedule[i] = scheduledRates{
						effectiveAt: sr.EffectiveAt,
						rates:       cloneAndValidateServiceTierRates(model, sr.Rates),
					}
				}
			}

			t.models[model] = mo
		}
	}

	return t
}

// cloneAndValidateServiceTierRates deep-copies and validates a per-service-tier
// rate map. Each tier's rates are validated against the same invariants the
// snapshot parser enforces.
func cloneAndValidateServiceTierRates(model string, m map[ServiceTier]Rates) map[ServiceTier]Rates {
	out := make(map[ServiceTier]Rates, len(m))
	for tier, r := range m {
		if _, ok := serviceSuffixes[tier]; !ok {
			panic(fmt.Sprintf("llmcost: %s: unknown service tier %q", model, tier))
		}
		validateOverrideRates(model, tier, r)
		out[tier] = r.clone()
	}
	return out
}

// validateOverrideRates panics if r violates the invariants TestTableInvariants
// asserts for every parsed snapshot entry: priceable base rates, strictly
// ascending positive tier thresholds with priceable rates, positive multipliers.
func validateOverrideRates(model string, tier ServiceTier, r Rates) {
	if !r.Base.priceable() {
		panic(fmt.Sprintf("llmcost: %s/%s: unpriceable base rates", model, tier))
	}
	for i, t := range r.Tiers {
		if t.AbovePromptTokens <= 0 || (i > 0 && t.AbovePromptTokens <= r.Tiers[i-1].AbovePromptTokens) {
			panic(fmt.Sprintf("llmcost: %s/%s: tier thresholds not strictly ascending and positive", model, tier))
		}
		if !t.priceable() {
			panic(fmt.Sprintf("llmcost: %s/%s: unpriceable tier rates", model, tier))
		}
	}
	if r.Fast != nil && r.Fast.Sign() <= 0 {
		panic(fmt.Sprintf("llmcost: %s/%s: non-positive fast multiplier", model, tier))
	}
	for key, f := range r.Geo {
		if f == nil || f.Sign() <= 0 {
			panic(fmt.Sprintf("llmcost: %s/%s: non-positive geo multiplier %q", model, tier, key))
		}
	}
	for key, f := range r.RegionalUplift {
		if f == nil || f.Sign() <= 0 {
			panic(fmt.Sprintf("llmcost: %s/%s: non-positive regional uplift %q", model, tier, key))
		}
	}
}

// clone returns a deep copy of r. The caller can mutate the returned
// Rates without corrupting the original.
func (r Rates) clone() Rates {
	out := Rates{
		Base:            r.Base.clone(),
		Fast:            cpRat(r.Fast),
		Geo:             cloneRatMap(r.Geo),
		RegionalUplift:  cloneRatMap(r.RegionalUplift),
		litellmProvider: r.litellmProvider,
	}
	if len(r.Tiers) > 0 {
		out.Tiers = make([]Tier, len(r.Tiers))
		for i, t := range r.Tiers {
			out.Tiers[i] = Tier{AbovePromptTokens: t.AbovePromptTokens, TierRates: t.TierRates.clone()}
		}
	}
	return out
}

func (t *Table) resolve(model string) string {
	if target, ok := t.aliases[model]; ok {
		return target
	}
	return model
}

// rates resolves alias, applies overrides, and returns a CLONE of the
// resolved rates. Every code path returns a deep copy — callers can
// mutate the returned *big.Rat values without corrupting the Table or
// the shared snapshot.
func (t *Table) rates(model string, tier ServiceTier, at time.Time) (Rates, bool) {
	model = t.resolve(model)
	if ovr, ok := t.models[model]; ok {
		switch {
		case len(ovr.schedule) > 0:
			if at.IsZero() {
				return Rates{}, false
			}
			idx := -1
			for i := len(ovr.schedule) - 1; i >= 0; i-- {
				if !ovr.schedule[i].effectiveAt.After(at) {
					idx = i
					break
				}
			}
			if idx < 0 {
				return Rates{}, false
			}
			r, ok := ovr.schedule[idx].rates[tier]
			if !ok {
				return Rates{}, false
			}
			return r.clone(), true
		case ovr.rates != nil:
			r, ok := ovr.rates[tier]
			if !ok {
				return Rates{}, false
			}
			return r.clone(), true
		case ovr.flattenTiers:
			r, ok := table()[model][tier]
			if !ok {
				return Rates{}, false
			}
			r = r.clone()
			r.Tiers = nil
			return r, true
		}
	}
	r, ok := table()[model][tier]
	if !ok {
		return Rates{}, false
	}
	return r.clone(), true
}

// Cost prices one response using the configured table. model is resolved
// through aliases first, then looked up in overrides, then the snapshot.
// at is the usage timestamp, consulted only when the resolved model has a
// RateSchedule — otherwise it is ignored and may be zero. All other
// semantics — provider-shaped usage normalization, service-tier
// resolution, multiplier composition, exact math, ceiling rounding —
// are identical to the package-level [Cost].
func (t *Table) Cost(model string, u Usage, at time.Time) (Nls, bool) {
	c := u.disjoint()
	tier, ok := u.tier()
	if !ok {
		return 0, false
	}
	r, ok := t.rates(model, tier, at)
	if !ok {
		return 0, false
	}
	p, ok := u.premium(r)
	if !ok {
		return 0, false
	}
	return cost(r, c, p)
}

// RatesFor returns the raw per-token rates for model at a service tier,
// with alias resolution and overrides applied. at is consulted only for
// models with a RateSchedule. The returned rats are copies.
func (t *Table) RatesFor(model string, tier ServiceTier, at time.Time) (Rates, bool) {
	return t.rates(model, tier, at)
}
