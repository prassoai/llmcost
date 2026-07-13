package llmcost

import (
	"math/big"
	"sync"
	"testing"
)

// TestAliasResolution encodes R1: aliases resolve consumer-internal model ids
// to LiteLLM pricing keys. The target is looked up directly — it is never
// re-checked against the alias map (no chaining). An unknown alias falls
// through to the model key unchanged.
func TestAliasResolution(t *testing.T) {
	tbl := New(Config{
		Aliases: map[string]string{
			"my-sonnet": "claude-sonnet-4-5",
		},
	})
	u := ClaudeUsage{InputTokens: 1000, OutputTokens: 100}

	// Alias resolves: Table.Cost("my-sonnet") == Cost("claude-sonnet-4-5").
	got, ok := tbl.Cost("my-sonnet", u)
	want, wantOK := Cost("claude-sonnet-4-5", u)
	if !ok || !wantOK || got != want {
		t.Fatalf("alias cost = %d, %v; snapshot cost = %d, %v; want equal and ok", got, ok, want, wantOK)
	}

	// Non-alias key falls through unchanged.
	got2, ok2 := tbl.Cost("claude-sonnet-4-5", u)
	if !ok2 || got2 != want {
		t.Fatalf("non-alias cost = %d, %v; want %d, true", got2, ok2, want)
	}

	// Unknown model returns ok=false.
	if _, ok3 := tbl.Cost("no-such-model", u); ok3 {
		t.Fatal("unknown model resolved; want ok=false")
	}

	// RatesFor resolves aliases the same way.
	rAlias, okA := tbl.RatesFor("my-sonnet", TierStandard)
	rDirect, okD := RatesFor("claude-sonnet-4-5", TierStandard)
	if !okA || !okD {
		t.Fatalf("RatesFor alias ok=%v, direct ok=%v; want both true", okA, okD)
	}
	if rAlias.Base.Input.Cmp(rDirect.Base.Input) != 0 {
		t.Fatal("RatesFor via alias returned different rates than the snapshot")
	}
}

// TestOverwriteRates encodes the overwrite primitive: a non-nil *Rates in
// Config.Overrides replaces the model's snapshot rates entirely. The model
// prices from the override, not the snapshot.
func TestOverwriteRates(t *testing.T) {
	tbl := New(Config{
		Overrides: map[string]*Rates{
			"claude-sonnet-4-5": {
				Base: TierRates{
					Input:         MustParseRat("10e-6"),
					CacheRead:     MustParseRat("1e-6"),
					CacheCreation: MustParseRat("12.5e-6"),
					Output:        MustParseRat("50e-6"),
				},
			},
		},
	})

	u := ClaudeUsage{InputTokens: 1000, CacheReadInputTokens: 30000, CacheCreationInputTokens: 2000, OutputTokens: 100}
	// 1000×10e-6 + 30000×1e-6 + 2000×12.5e-6 + 100×50e-6
	// = 0.01 + 0.03 + 0.025 + 0.005 = $0.07 = 7000 nls
	got, ok := tbl.Cost("claude-sonnet-4-5", u)
	if !ok || got != 7000 {
		t.Fatalf("override cost = %d, %v; want 7000, true", got, ok)
	}
	snap, snapOK := Cost("claude-sonnet-4-5", u)
	if !snapOK || snap == got {
		t.Fatalf("snapshot cost = %d should differ from override cost %d", snap, got)
	}
}

// TestRemoveModel encodes the remove primitive: a nil value in
// Config.Overrides suppresses the model entirely — RatesFor and Cost
// return ok=false, even though the model exists in the snapshot.
func TestRemoveModel(t *testing.T) {
	tbl := New(Config{
		Overrides: map[string]*Rates{
			"claude-sonnet-4-5": nil,
		},
	})

	if _, ok := tbl.Cost("claude-sonnet-4-5", ClaudeUsage{InputTokens: 1000, OutputTokens: 100}); ok {
		t.Fatal("suppressed model resolved via Cost; want ok=false")
	}
	if _, ok := tbl.RatesFor("claude-sonnet-4-5", TierStandard); ok {
		t.Fatal("suppressed model resolved via RatesFor; want ok=false")
	}
	// The snapshot is unaffected — package-level Cost still prices it.
	if _, ok := Cost("claude-sonnet-4-5", ClaudeUsage{InputTokens: 1000, OutputTokens: 100}); !ok {
		t.Fatal("package-level Cost failed for model suppressed only in Table")
	}
}

// TestAliasToSuppressedModel encodes that an alias whose target is
// suppressed (nil override) resolves to the suppressed model and
// returns ok=false. The alias is valid at construction (the target
// exists as an Overrides key), but the model is suppressed at query
// time.
func TestAliasToSuppressedModel(t *testing.T) {
	tbl := New(Config{
		Aliases: map[string]string{
			"my-sonnet": "suppressed-model",
		},
		Overrides: map[string]*Rates{
			"suppressed-model": nil,
		},
	})

	if _, ok := tbl.Cost("my-sonnet", ClaudeUsage{InputTokens: 1000, OutputTokens: 100}); ok {
		t.Fatal("alias to suppressed model resolved; want ok=false")
	}
}

// TestOverrideNewModel encodes that an override for a model not in the
// snapshot defines a new model. The model is priceable through the Table
// but unknown to the package-level functions.
func TestOverrideNewModel(t *testing.T) {
	tbl := New(Config{
		Overrides: map[string]*Rates{
			"custom-model": {
				Base: TierRates{
					Input:  MustParseRat("3e-6"),
					Output: MustParseRat("15e-6"),
				},
			},
		},
	})

	// 1000×3e-6 + 100×15e-6 = $0.0045 = 450 nls
	got, ok := tbl.Cost("custom-model", ClaudeUsage{InputTokens: 1000, OutputTokens: 100})
	if !ok || got != 450 {
		t.Fatalf("new model cost = %d, %v; want 450, true", got, ok)
	}
	if _, ok := Cost("custom-model", ClaudeUsage{InputTokens: 1000, OutputTokens: 100}); ok {
		t.Fatal("new model resolved via package-level Cost; want ok=false")
	}
}

// TestOverrideFlattensTiersByOmission encodes that a caller can flatten
// context-window tiers simply by providing Rates with no Tiers field.
// This replaces the old FlattenTiers flag with a simpler mechanism:
// the override controls the rate structure directly.
func TestOverrideFlattensTiersByOmission(t *testing.T) {
	// Read the snapshot's base rates for claude-sonnet-4-5 and reproduce
	// them as an override WITHOUT context-window tiers.
	snap, ok := RatesFor("claude-sonnet-4-5", TierStandard)
	if !ok {
		t.Fatal("snapshot model not found")
	}
	if len(snap.Tiers) == 0 {
		t.Fatal("test assumes claude-sonnet-4-5 has context-window tiers")
	}
	tbl := New(Config{
		Overrides: map[string]*Rates{
			"claude-sonnet-4-5": {
				Base: snap.Base,
				// Tiers intentionally omitted — flat pricing.
			},
		},
	})

	// 300k total prompt (250k input + 50k cache read) would exceed the
	// snapshot's 200k tier. With the flat override, base rates apply.
	u := ClaudeUsage{InputTokens: 250000, CacheReadInputTokens: 50000, OutputTokens: 1000}
	flatCost, flatOK := tbl.Cost("claude-sonnet-4-5", u)
	snapCost, snapOK := Cost("claude-sonnet-4-5", u)
	if !flatOK || !snapOK {
		t.Fatalf("flat ok=%v, snap ok=%v; want both true", flatOK, snapOK)
	}
	if flatCost >= snapCost {
		t.Fatalf("flat cost %d >= snapshot cost %d; want lower (base rates only)", flatCost, snapCost)
	}

	// The override's RatesFor must have nil Tiers.
	r, ok := tbl.RatesFor("claude-sonnet-4-5", TierStandard)
	if !ok {
		t.Fatal("RatesFor flat override failed")
	}
	if r.Tiers != nil {
		t.Fatalf("flat override Tiers = %v; want nil", r.Tiers)
	}
}

// TestOverrideWithMultipliers encodes that overridden models compose with
// the existing fast/geo multiplier machinery. Multipliers in the override's
// Rates scale uncached input and output exactly as they do for snapshot models.
func TestOverrideWithMultipliers(t *testing.T) {
	tbl := New(Config{
		Overrides: map[string]*Rates{
			"custom-with-mult": {
				Base: TierRates{
					Input:         MustParseRat("5e-6"),
					CacheRead:     MustParseRat("5e-7"),
					CacheCreation: MustParseRat("6.25e-6"),
					Output:        MustParseRat("25e-6"),
				},
				Fast: MustParseRat("6"),
				Geo:  map[string]*big.Rat{"us": MustParseRat("1.1")},
			},
		},
	})

	u := ClaudeUsage{InputTokens: 1000, CacheReadInputTokens: 30000, CacheCreationInputTokens: 2000, OutputTokens: 100}

	// Standard: 1000×5e-6 + 30000×5e-7 + 2000×6.25e-6 + 100×25e-6
	// = 0.005 + 0.015 + 0.0125 + 0.0025 = $0.035 = 3500 nls
	if got, ok := tbl.Cost("custom-with-mult", u); !ok || got != 3500 {
		t.Fatalf("standard = %d, %v; want 3500, true", got, ok)
	}

	// Fast: uncached input and output ×6, cache unscaled.
	// (1000×5e-6 + 100×25e-6) × 6 + 30000×5e-7 + 2000×6.25e-6
	// = (0.005 + 0.0025) × 6 + 0.015 + 0.0125 = 0.045 + 0.0275 = $0.0725 = 7250 nls
	u.Speed = "fast"
	if got, ok := tbl.Cost("custom-with-mult", u); !ok || got != 7250 {
		t.Fatalf("fast = %d, %v; want 7250, true", got, ok)
	}

	// Geo (us, 1.1×): uncached input and output ×1.1, cache unscaled.
	// (1000×5e-6 + 100×25e-6) × 1.1 + 30000×5e-7 + 2000×6.25e-6
	// = 0.0075 × 1.1 + 0.0275 = 0.00825 + 0.0275 = $0.03575 = 3575 nls
	u.Speed = ""
	u.InferenceGeo = "us"
	if got, ok := tbl.Cost("custom-with-mult", u); !ok || got != 3575 {
		t.Fatalf("geo us = %d, %v; want 3575, true", got, ok)
	}
}

// TestOverrideStandardTierOnly encodes that overrides provide rates for
// the standard service tier only. Non-standard tiers on an overridden
// model return ok=false — the override replaces the model's rates
// wholesale, and the single *Rates applies at the standard tier.
func TestOverrideStandardTierOnly(t *testing.T) {
	tbl := New(Config{
		Overrides: map[string]*Rates{
			"custom": {Base: TierRates{
				Input:  MustParseRat("5e-6"),
				Output: MustParseRat("30e-6"),
			}},
		},
	})

	// Standard: 600×5e-6 + 100×30e-6 = $0.006 = 600 nls.
	if got, ok := tbl.Cost("custom", OpenAIUsage{InputTokens: 600, OutputTokens: 100}); !ok || got != 600 {
		t.Fatalf("standard = %d, %v; want 600, true", got, ok)
	}

	// Flex: unpriced — the override provides standard only.
	if _, ok := tbl.Cost("custom", OpenAIUsage{InputTokens: 600, OutputTokens: 100, ServiceTier: TierFlex}); ok {
		t.Fatal("flex on overridden model resolved; want ok=false")
	}

	// Priority: unpriced.
	if _, ok := tbl.Cost("custom", OpenAIUsage{InputTokens: 600, OutputTokens: 100, ServiceTier: TierPriority}); ok {
		t.Fatal("priority on overridden model resolved; want ok=false")
	}
}

// TestNonOverriddenModelKeepsTiers encodes that a model not mentioned
// in Overrides keeps its snapshot service tiers intact.
func TestNonOverriddenModelKeepsTiers(t *testing.T) {
	tbl := New(Config{
		Overrides: map[string]*Rates{
			"some-other-model": {Base: TierRates{Input: MustParseRat("1e-6"), Output: MustParseRat("2e-6")}},
		},
	})

	// gpt-5.5 has flex and priority tiers in the snapshot. They must
	// remain accessible through a Table that doesn't override it.
	for _, tier := range []ServiceTier{TierStandard, TierFlex, TierPriority} {
		tblR, tblOK := tbl.RatesFor("gpt-5.5", tier)
		snapR, snapOK := RatesFor("gpt-5.5", tier)
		if tblOK != snapOK {
			t.Errorf("gpt-5.5/%s: Table ok=%v, snapshot ok=%v", tier, tblOK, snapOK)
			continue
		}
		if tblOK && tblR.Base.Input.Cmp(snapR.Base.Input) != 0 {
			t.Errorf("gpt-5.5/%s: Table rates differ from snapshot", tier)
		}
	}
}

// TestMalformedConfigPanics encodes that every malformed config case
// panics at construction. One subtest per validation rule.
func TestMalformedConfigPanics(t *testing.T) {
	mustPanicNew := func(name string, cfg Config) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("did not panic")
				}
			}()
			New(cfg)
		})
	}

	mustPanicNew("empty alias target", Config{
		Aliases: map[string]string{"a": ""},
	})

	mustPanicNew("alias chain", Config{
		Aliases: map[string]string{"a": "b", "b": "claude-sonnet-4-5"},
	})

	mustPanicNew("alias target not in snapshot or overrides", Config{
		Aliases: map[string]string{"a": "no-such-model-xyz"},
	})

	mustPanicNew("unpriceable rates: non-positive input", Config{
		Overrides: map[string]*Rates{
			"x": {Base: TierRates{Input: MustParseRat("-1e-6"), Output: MustParseRat("15e-6")}},
		},
	})

	mustPanicNew("unpriceable rates: zero output", Config{
		Overrides: map[string]*Rates{
			"x": {Base: TierRates{Input: MustParseRat("3e-6"), Output: new(big.Rat)}},
		},
	})

	mustPanicNew("unpriceable rates: negative cache", Config{
		Overrides: map[string]*Rates{
			"x": {Base: TierRates{
				Input: MustParseRat("3e-6"), CacheRead: MustParseRat("-1e-7"), Output: MustParseRat("15e-6"),
			}},
		},
	})

	mustPanicNew("non-positive fast multiplier", Config{
		Overrides: map[string]*Rates{
			"x": {
				Base: TierRates{Input: MustParseRat("3e-6"), Output: MustParseRat("15e-6")},
				Fast: new(big.Rat),
			},
		},
	})

	mustPanicNew("non-positive geo multiplier", Config{
		Overrides: map[string]*Rates{
			"x": {
				Base: TierRates{Input: MustParseRat("3e-6"), Output: MustParseRat("15e-6")},
				Geo:  map[string]*big.Rat{"us": MustParseRat("-1")},
			},
		},
	})

	mustPanicNew("non-positive regional uplift", Config{
		Overrides: map[string]*Rates{
			"x": {
				Base:           TierRates{Input: MustParseRat("3e-6"), Output: MustParseRat("15e-6")},
				RegionalUplift: map[string]*big.Rat{"eu": new(big.Rat)},
			},
		},
	})

	mustPanicNew("non-positive tier threshold", Config{
		Overrides: map[string]*Rates{
			"x": {
				Base:  TierRates{Input: MustParseRat("3e-6"), Output: MustParseRat("15e-6")},
				Tiers: []Tier{{AbovePromptTokens: 0, TierRates: TierRates{Input: MustParseRat("6e-6"), Output: MustParseRat("30e-6")}}},
			},
		},
	})

	mustPanicNew("non-ascending tier thresholds", Config{
		Overrides: map[string]*Rates{
			"x": {
				Base: TierRates{Input: MustParseRat("3e-6"), Output: MustParseRat("15e-6")},
				Tiers: []Tier{
					{AbovePromptTokens: 200000, TierRates: TierRates{Input: MustParseRat("6e-6"), Output: MustParseRat("30e-6")}},
					{AbovePromptTokens: 100000, TierRates: TierRates{Input: MustParseRat("4e-6"), Output: MustParseRat("20e-6")}},
				},
			},
		},
	})

	mustPanicNew("unpriceable tier rates", Config{
		Overrides: map[string]*Rates{
			"x": {
				Base:  TierRates{Input: MustParseRat("3e-6"), Output: MustParseRat("15e-6")},
				Tiers: []Tier{{AbovePromptTokens: 200000, TierRates: TierRates{Input: MustParseRat("-1e-6"), Output: MustParseRat("30e-6")}}},
			},
		},
	})
}

// TestZeroTableIsSnapshot encodes that the zero-value Table is the
// unoverridden snapshot — identical to the package-level functions.
func TestZeroTableIsSnapshot(t *testing.T) {
	var tbl Table
	for _, model := range []string{"claude-opus-4-8", "claude-sonnet-4-5", "gpt-5.4", "gpt-5.5"} {
		for _, u := range []Usage{
			ClaudeUsage{InputTokens: 1000, CacheReadInputTokens: 500, CacheCreationInputTokens: 200, OutputTokens: 100},
			OpenAIUsage{InputTokens: 1000, CachedInputTokens: 400, OutputTokens: 100},
		} {
			got, gotOK := tbl.Cost(model, u)
			want, wantOK := Cost(model, u)
			if got != want || gotOK != wantOK {
				t.Errorf("%s: Table.Cost = %d, %v; Cost = %d, %v", model, got, gotOK, want, wantOK)
			}
		}
		for _, tier := range []ServiceTier{TierStandard, TierFlex, TierPriority} {
			gotR, gotOK := tbl.RatesFor(model, tier)
			wantR, wantOK := RatesFor(model, tier)
			if gotOK != wantOK {
				t.Errorf("%s/%s: Table.RatesFor ok=%v; RatesFor ok=%v", model, tier, gotOK, wantOK)
				continue
			}
			if gotOK && gotR.Base.Input.Cmp(wantR.Base.Input) != 0 {
				t.Errorf("%s/%s: Table.RatesFor input rate differs from RatesFor", model, tier)
			}
		}
	}
}

// TestMustParseRat encodes the convenience helper: valid decimals parse to
// exact rationals; invalid input panics.
func TestMustParseRat(t *testing.T) {
	if r := MustParseRat("3e-6"); r.Cmp(big.NewRat(3, 1_000_000)) != 0 {
		t.Fatalf("MustParseRat(3e-6) = %v; want 3/1000000", r)
	}
	if r := MustParseRat("0.0000025"); r.Cmp(big.NewRat(1, 400_000)) != 0 {
		t.Fatalf("MustParseRat(0.0000025) = %v; want 1/400000", r)
	}
	if r := MustParseRat("1.1"); r.Cmp(big.NewRat(11, 10)) != 0 {
		t.Fatalf("MustParseRat(1.1) = %v; want 11/10", r)
	}

	t.Run("invalid panics", func(t *testing.T) {
		defer func() {
			if recover() == nil {
				t.Fatal("MustParseRat(invalid) did not panic")
			}
		}()
		MustParseRat("not-a-number")
	})
}

// TestConfigDeepCopy encodes the immutability guarantee: mutating the input
// Config after New — maps, *big.Rat values — does not change billing
// results from the constructed Table.
func TestConfigDeepCopy(t *testing.T) {
	inputRate := MustParseRat("3e-6")
	outputRate := MustParseRat("15e-6")
	cfg := Config{
		Aliases: map[string]string{"my-model": "claude-sonnet-4-5"},
		Overrides: map[string]*Rates{
			"custom-model": {Base: TierRates{
				Input:  inputRate,
				Output: outputRate,
			}},
		},
	}

	tbl := New(cfg)
	u := ClaudeUsage{InputTokens: 1000, OutputTokens: 100}
	before, beforeOK := tbl.Cost("custom-model", u)
	beforeAlias, beforeAliasOK := tbl.Cost("my-model", u)

	// Mutate the original *big.Rat values.
	inputRate.SetInt64(999)
	outputRate.SetInt64(999)

	// Mutate the alias map.
	cfg.Aliases["my-model"] = "gpt-5.4"
	delete(cfg.Aliases, "my-model")

	// Mutate the override map.
	delete(cfg.Overrides, "custom-model")

	// Table is unaffected.
	after, afterOK := tbl.Cost("custom-model", u)
	if !beforeOK || !afterOK || after != before {
		t.Fatalf("config mutation changed Table.Cost: %d -> %d", before, after)
	}
	afterAlias, afterAliasOK := tbl.Cost("my-model", u)
	if !beforeAliasOK || !afterAliasOK || afterAlias != beforeAlias {
		t.Fatalf("alias mutation changed Table.Cost: %d -> %d", beforeAlias, afterAlias)
	}
}

// TestReturnedRatesAreCopies encodes that mutating a Table.RatesFor return
// value — base rats, tier rats, Fast, Geo, RegionalUplift — does not
// corrupt the Table or the snapshot.
func TestReturnedRatesAreCopies(t *testing.T) {
	tbl := New(Config{
		Overrides: map[string]*Rates{
			"custom": {
				Base: TierRates{
					Input:         MustParseRat("5e-6"),
					CacheRead:     MustParseRat("5e-7"),
					CacheCreation: MustParseRat("6.25e-6"),
					Output:        MustParseRat("25e-6"),
				},
				Fast:           MustParseRat("6"),
				Geo:            map[string]*big.Rat{"us": MustParseRat("1.1")},
				RegionalUplift: map[string]*big.Rat{"eu": MustParseRat("1.1")},
			},
		},
	})

	u := ClaudeUsage{InputTokens: 1000, OutputTokens: 100}
	before, _ := tbl.Cost("custom", u)

	// Mutate every *big.Rat in the returned Rates.
	r, _ := tbl.RatesFor("custom", TierStandard)
	for _, rat := range []*big.Rat{r.Base.Input, r.Base.CacheRead, r.Base.CacheCreation, r.Base.Output, r.Fast} {
		if rat != nil {
			rat.SetInt64(999)
		}
	}
	for _, m := range []map[string]*big.Rat{r.Geo, r.RegionalUplift} {
		for _, rat := range m {
			rat.SetInt64(999)
		}
	}

	after, ok := tbl.Cost("custom", u)
	if !ok || after != before {
		t.Fatalf("mutating RatesFor return changed Cost: %d -> %d", before, after)
	}

	// Also verify snapshot rates are not corrupted via Table reads.
	snapBefore, _ := Cost("claude-sonnet-4-5", ClaudeUsage{InputTokens: 1000, OutputTokens: 100})
	r2, _ := tbl.RatesFor("claude-sonnet-4-5", TierStandard)
	if r2.Base.Input != nil {
		r2.Base.Input.SetInt64(999)
	}
	snapAfter, _ := Cost("claude-sonnet-4-5", ClaudeUsage{InputTokens: 1000, OutputTokens: 100})
	if snapAfter != snapBefore {
		t.Fatalf("mutating Table.RatesFor return corrupted snapshot: %d -> %d", snapBefore, snapAfter)
	}
}

// TestConcurrentCost encodes the concurrency guarantee: concurrent Table.Cost
// calls on the same Table produce consistent results. Run with -race.
func TestConcurrentCost(t *testing.T) {
	tbl := New(Config{
		Aliases: map[string]string{"my-sonnet": "claude-sonnet-4-5"},
		Overrides: map[string]*Rates{
			"custom": {Base: TierRates{
				Input:  MustParseRat("3e-6"),
				Output: MustParseRat("15e-6"),
			}},
			"claude-sonnet-4-5": nil, // suppressed
		},
	})

	u := ClaudeUsage{InputTokens: 1000, OutputTokens: 100}
	want, wantOK := tbl.Cost("custom", u)

	var wg sync.WaitGroup
	for range 100 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 100 {
				got, ok := tbl.Cost("custom", u)
				if got != want || ok != wantOK {
					t.Errorf("concurrent Cost = %d, %v; want %d, %v", got, ok, want, wantOK)
					return
				}
				// Suppressed model must return ok=false consistently.
				if _, ok := tbl.Cost("claude-sonnet-4-5", u); ok {
					t.Error("suppressed model resolved concurrently")
					return
				}
				// Alias resolves to suppressed model.
				if _, ok := tbl.Cost("my-sonnet", u); ok {
					t.Error("alias to suppressed model resolved concurrently")
					return
				}
				tbl.RatesFor("custom", TierStandard)
			}
		}()
	}
	wg.Wait()
}
