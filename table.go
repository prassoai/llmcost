package llmcost

import (
	"fmt"
	"math/big"
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

	// Overrides customizes per-model pricing, keyed by LiteLLM pricing
	// key (the post-alias-resolution key).
	//
	// A non-nil *Rates replaces the model's rates wholesale. The model
	// prices from the override at the standard service tier; other
	// service tiers are unpriced for the overridden model. A caller
	// flattens tiers simply by providing Rates with no Tiers field —
	// no dedicated flag is needed.
	//
	// A nil value suppresses the model entirely: RatesFor and Cost
	// return ok=false for it, even if the model exists in the snapshot.
	//
	// An override for a model not in the snapshot defines a new model.
	// Models absent from the map use the snapshot unchanged.
	Overrides map[string]*Rates
}

// Table is a configured pricing table: the embedded LiteLLM snapshot
// plus consumer-declared overrides. The zero value is the unoverridden
// snapshot (identical to the package-level functions). Construct with
// [New] to apply overrides.
//
// A Table is safe for concurrent use — it is immutable after construction.
type Table struct {
	aliases   map[string]string
	overrides map[string]*Rates
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
//     Config.Overrides key (catches typos at init, not at query time;
//     models not yet in the snapshot are declared in Config.Overrides)
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
				if _, inOverrides := cfg.Overrides[v]; !inOverrides {
					panic(fmt.Sprintf("llmcost: alias %q -> %q: target not in snapshot or Config.Overrides", k, v))
				}
			}
			t.aliases[k] = v
		}
	}

	if len(cfg.Overrides) > 0 {
		t.overrides = make(map[string]*Rates, len(cfg.Overrides))
		for model, r := range cfg.Overrides {
			if r == nil {
				t.overrides[model] = nil // suppress
				continue
			}
			validateOverrideRates(model, TierStandard, *r)
			cloned := r.clone()
			t.overrides[model] = &cloned
		}
	}

	return t
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
func (t *Table) rates(model string, tier ServiceTier) (Rates, bool) {
	model = t.resolve(model)
	if t.overrides != nil {
		if r, exists := t.overrides[model]; exists {
			if r == nil {
				return Rates{}, false // suppressed
			}
			if tier != TierStandard {
				return Rates{}, false // override provides standard only
			}
			return r.clone(), true
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
// All semantics — provider-shaped usage normalization, service-tier
// resolution, multiplier composition, exact math, ceiling rounding —
// are identical to the package-level [Cost].
func (t *Table) Cost(model string, u Usage) (Nls, bool) {
	c := u.disjoint()
	tier, ok := u.tier()
	if !ok {
		return 0, false
	}
	r, ok := t.rates(model, tier)
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
// with alias resolution and overrides applied. The returned rats are copies.
func (t *Table) RatesFor(model string, tier ServiceTier) (Rates, bool) {
	return t.rates(model, tier)
}
