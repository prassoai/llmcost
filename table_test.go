package llmcost

import (
	"math/big"
	"sync"
	"testing"
	"time"
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
	got, ok := tbl.Cost("my-sonnet", u, time.Time{})
	want, wantOK := Cost("claude-sonnet-4-5", u)
	if !ok || !wantOK || got != want {
		t.Fatalf("alias cost = %d, %v; snapshot cost = %d, %v; want equal and ok", got, ok, want, wantOK)
	}

	// Non-alias key falls through unchanged.
	got2, ok2 := tbl.Cost("claude-sonnet-4-5", u, time.Time{})
	if !ok2 || got2 != want {
		t.Fatalf("non-alias cost = %d, %v; want %d, true", got2, ok2, want)
	}

	// Unknown model returns ok=false.
	if _, ok3 := tbl.Cost("no-such-model", u, time.Time{}); ok3 {
		t.Fatal("unknown model resolved; want ok=false")
	}

	// RatesFor resolves aliases the same way.
	rAlias, okA := tbl.RatesFor("my-sonnet", TierStandard, time.Time{})
	rDirect, okD := RatesFor("claude-sonnet-4-5", TierStandard)
	if !okA || !okD {
		t.Fatalf("RatesFor alias ok=%v, direct ok=%v; want both true", okA, okD)
	}
	if rAlias.Base.Input.Cmp(rDirect.Base.Input) != 0 {
		t.Fatal("RatesFor via alias returned different rates than the snapshot")
	}
}

// TestFlattenTiers encodes R2: FlattenTiers suppresses all context-window
// tiers from the snapshot, forcing flat (base-rate-only) pricing regardless
// of prompt size.
func TestFlattenTiers(t *testing.T) {
	tbl := New(Config{
		Models: map[string]ModelOverride{
			"claude-sonnet-4-5": {FlattenTiers: true},
		},
	})

	// 300k total prompt (250k input + 50k cache read) exceeds the 200k tier.
	// Flattened: base rates  → 250000×3e-6 + 50000×3e-7 + 1000×1.5e-5
	//                        = 0.75 + 0.015 + 0.015 = $0.78 = 78000 nls
	// Snapshot:  tier rates  → 250000×6e-6 + 50000×6e-7 + 1000×2.25e-5
	//                        = 1.5 + 0.03 + 0.0225 = $1.5525 = 155250 nls
	u := ClaudeUsage{InputTokens: 250000, CacheReadInputTokens: 50000, OutputTokens: 1000}
	flatCost, flatOK := tbl.Cost("claude-sonnet-4-5", u, time.Time{})
	snapCost, snapOK := Cost("claude-sonnet-4-5", u)
	if !flatOK || !snapOK {
		t.Fatalf("flat ok=%v, snap ok=%v; want both true", flatOK, snapOK)
	}
	if flatCost != 78000 {
		t.Fatalf("flattened cost = %d; want 78000 (base rates)", flatCost)
	}
	if snapCost != 155250 {
		t.Fatalf("snapshot cost = %d; want 155250 (tier rates)", snapCost)
	}

	// Tiers field is nil on the flattened rates.
	r, ok := tbl.RatesFor("claude-sonnet-4-5", TierStandard, time.Time{})
	if !ok {
		t.Fatal("RatesFor flattened model failed")
	}
	if r.Tiers != nil {
		t.Fatalf("flattened Tiers = %v; want nil", r.Tiers)
	}
}

// TestRateSchedule encodes R3: time-dependent resolution — intro period
// uses intro rates, standard period uses standard rates, before-all-entries
// is unpriced, zero timestamp is unpriced.
func TestRateSchedule(t *testing.T) {
	introDate := time.Date(2026, 6, 1, 0, 0, 0, 0, time.UTC)
	standardDate := time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC)

	tbl := New(Config{
		Models: map[string]ModelOverride{
			"scheduled-model": {
				RateSchedule: []ScheduledRates{
					{
						EffectiveAt: introDate,
						Rates: map[ServiceTier]Rates{
							TierStandard: {
								Base: TierRates{
									Input:  MustParseRat("3e-6"),
									Output: MustParseRat("15e-6"),
								},
							},
						},
					},
					{
						EffectiveAt: standardDate,
						Rates: map[ServiceTier]Rates{
							TierStandard: {
								Base: TierRates{
									Input:  MustParseRat("4e-6"),
									Output: MustParseRat("20e-6"),
								},
							},
						},
					},
				},
			},
		},
	})

	u := ClaudeUsage{InputTokens: 1000, OutputTokens: 100}

	// Intro period (July 2026): 1000×3e-6 + 100×15e-6 = $0.0045 = 450 nls.
	if got, ok := tbl.Cost("scheduled-model", u, time.Date(2026, 7, 15, 0, 0, 0, 0, time.UTC)); !ok || got != 450 {
		t.Fatalf("intro cost = %d, %v; want 450, true", got, ok)
	}

	// Standard period (October 2026): 1000×4e-6 + 100×20e-6 = $0.006 = 600 nls.
	if got, ok := tbl.Cost("scheduled-model", u, time.Date(2026, 10, 1, 0, 0, 0, 0, time.UTC)); !ok || got != 600 {
		t.Fatalf("standard cost = %d, %v; want 600, true", got, ok)
	}

	// Before all entries: unpriced.
	if _, ok := tbl.Cost("scheduled-model", u, time.Date(2026, 5, 1, 0, 0, 0, 0, time.UTC)); ok {
		t.Fatal("before-all-entries resolved; want ok=false")
	}

	// Zero timestamp: unpriced (consumer declared time-dependent pricing).
	if _, ok := tbl.Cost("scheduled-model", u, time.Time{}); ok {
		t.Fatal("zero timestamp on scheduled model resolved; want ok=false")
	}
}

// TestStaticRateOverride encodes R4: override rates replace the snapshot
// entirely. The model prices from the override, not the snapshot.
func TestStaticRateOverride(t *testing.T) {
	tbl := New(Config{
		Models: map[string]ModelOverride{
			"claude-sonnet-4-5": {
				Rates: map[ServiceTier]Rates{
					TierStandard: {
						Base: TierRates{
							Input:         MustParseRat("10e-6"),
							CacheRead:     MustParseRat("1e-6"),
							CacheCreation: MustParseRat("12.5e-6"),
							Output:        MustParseRat("50e-6"),
						},
					},
				},
			},
		},
	})

	u := ClaudeUsage{InputTokens: 1000, CacheReadInputTokens: 30000, CacheCreationInputTokens: 2000, OutputTokens: 100}
	// 1000×10e-6 + 30000×1e-6 + 2000×12.5e-6 + 100×50e-6
	// = 0.01 + 0.03 + 0.025 + 0.005 = $0.07 = 7000 nls
	got, ok := tbl.Cost("claude-sonnet-4-5", u, time.Time{})
	if !ok || got != 7000 {
		t.Fatalf("override cost = %d, %v; want 7000, true", got, ok)
	}
	snap, snapOK := Cost("claude-sonnet-4-5", u)
	if !snapOK || snap == got {
		t.Fatalf("snapshot cost = %d should differ from override cost %d", snap, got)
	}
}

// TestOverridePrecedence encodes the resolution order: RateSchedule >
// static Rates > FlattenTiers+snapshot > raw snapshot. Each level is tested
// by constructing the appropriate Config so the costs are distinguishable.
func TestOverridePrecedence(t *testing.T) {
	tbl := New(Config{
		Models: map[string]ModelOverride{
			// Level 1: schedule.
			"custom-scheduled": {
				RateSchedule: []ScheduledRates{{
					EffectiveAt: time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC),
					Rates: map[ServiceTier]Rates{
						TierStandard: {Base: TierRates{
							Input:  MustParseRat("1e-6"),
							Output: MustParseRat("2e-6"),
						}},
					},
				}},
			},
			// Level 2: static rates.
			"custom-static": {
				Rates: map[ServiceTier]Rates{
					TierStandard: {Base: TierRates{
						Input:  MustParseRat("3e-6"),
						Output: MustParseRat("6e-6"),
					}},
				},
			},
			// Level 3: flatten + snapshot.
			"claude-sonnet-4-5": {FlattenTiers: true},
			// Level 4: gpt-5.4 — no override, raw snapshot.
		},
	})

	u := ClaudeUsage{InputTokens: 1000, OutputTokens: 100}
	at := time.Date(2026, 7, 1, 0, 0, 0, 0, time.UTC)

	// Schedule: 1000×1e-6 + 100×2e-6 = $0.0012 = 120 nls.
	if got, ok := tbl.Cost("custom-scheduled", u, at); !ok || got != 120 {
		t.Fatalf("schedule cost = %d, %v; want 120, true", got, ok)
	}

	// Static: 1000×3e-6 + 100×6e-6 = $0.0036 = 360 nls.
	if got, ok := tbl.Cost("custom-static", u, at); !ok || got != 360 {
		t.Fatalf("static cost = %d, %v; want 360, true", got, ok)
	}

	// Flatten: differs from snapshot at long context (300k > 200k tier).
	uLong := ClaudeUsage{InputTokens: 250000, CacheReadInputTokens: 50000, OutputTokens: 100}
	flatCost, flatOK := tbl.Cost("claude-sonnet-4-5", uLong, at)
	snapCost, snapOK := Cost("claude-sonnet-4-5", uLong)
	if !flatOK || !snapOK {
		t.Fatalf("flatten ok=%v, snap ok=%v", flatOK, snapOK)
	}
	if flatCost >= snapCost {
		t.Fatalf("flatten cost %d >= snapshot cost %d; want lower (base rates)", flatCost, snapCost)
	}

	// Snapshot (no override): Table.Cost == package-level Cost.
	oaiU := OpenAIUsage{InputTokens: 1000, OutputTokens: 100}
	gotSnap, okSnap := tbl.Cost("gpt-5.4", oaiU, at)
	wantSnap, wantOKSnap := Cost("gpt-5.4", oaiU)
	if okSnap != wantOKSnap || gotSnap != wantSnap {
		t.Fatalf("snapshot via table = %d, %v; package-level = %d, %v", gotSnap, okSnap, wantSnap, wantOKSnap)
	}
}

// TestOverrideWithMultipliers encodes R5: overridden models compose with the
// existing fast/geo multiplier machinery. Multipliers in the override's Rates
// scale uncached input and output exactly as they do for snapshot models.
func TestOverrideWithMultipliers(t *testing.T) {
	tbl := New(Config{
		Models: map[string]ModelOverride{
			"custom-with-mult": {
				Rates: map[ServiceTier]Rates{
					TierStandard: {
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
			},
		},
	})

	u := ClaudeUsage{InputTokens: 1000, CacheReadInputTokens: 30000, CacheCreationInputTokens: 2000, OutputTokens: 100}

	// Standard: 1000×5e-6 + 30000×5e-7 + 2000×6.25e-6 + 100×25e-6
	// = 0.005 + 0.015 + 0.0125 + 0.0025 = $0.035 = 3500 nls
	if got, ok := tbl.Cost("custom-with-mult", u, time.Time{}); !ok || got != 3500 {
		t.Fatalf("standard = %d, %v; want 3500, true", got, ok)
	}

	// Fast: uncached input and output ×6, cache unscaled.
	// (1000×5e-6 + 100×25e-6) × 6 + 30000×5e-7 + 2000×6.25e-6
	// = (0.005 + 0.0025) × 6 + 0.015 + 0.0125 = 0.045 + 0.0275 = $0.0725 = 7250 nls
	u.Speed = "fast"
	if got, ok := tbl.Cost("custom-with-mult", u, time.Time{}); !ok || got != 7250 {
		t.Fatalf("fast = %d, %v; want 7250, true", got, ok)
	}

	// Geo (us, 1.1×): uncached input and output ×1.1, cache unscaled.
	// (1000×5e-6 + 100×25e-6) × 1.1 + 30000×5e-7 + 2000×6.25e-6
	// = 0.0075 × 1.1 + 0.0275 = 0.00825 + 0.0275 = $0.03575 = 3575 nls
	u.Speed = ""
	u.InferenceGeo = "us"
	if got, ok := tbl.Cost("custom-with-mult", u, time.Time{}); !ok || got != 3575 {
		t.Fatalf("geo us = %d, %v; want 3575, true", got, ok)
	}
}

// TestOverrideServiceTiers encodes the service-tier dimension of overrides:
// rates are keyed by ServiceTier; a tier absent from the override map fails
// (ok=false), matching the no-cross-tier-fallback contract.
func TestOverrideServiceTiers(t *testing.T) {
	tbl := New(Config{
		Models: map[string]ModelOverride{
			"custom-tiered": {
				Rates: map[ServiceTier]Rates{
					TierStandard: {Base: TierRates{
						Input:  MustParseRat("5e-6"),
						Output: MustParseRat("30e-6"),
					}},
					TierFlex: {Base: TierRates{
						Input:  MustParseRat("2.5e-6"),
						Output: MustParseRat("15e-6"),
					}},
				},
			},
		},
	})

	// Standard: 600×5e-6 + 100×30e-6 = $0.006 = 600 nls.
	if got, ok := tbl.Cost("custom-tiered", OpenAIUsage{InputTokens: 600, OutputTokens: 100}, time.Time{}); !ok || got != 600 {
		t.Fatalf("standard = %d, %v; want 600, true", got, ok)
	}

	// Flex: 600×2.5e-6 + 100×15e-6 = $0.003 = 300 nls.
	if got, ok := tbl.Cost("custom-tiered", OpenAIUsage{InputTokens: 600, OutputTokens: 100, ServiceTier: TierFlex}, time.Time{}); !ok || got != 300 {
		t.Fatalf("flex = %d, %v; want 300, true", got, ok)
	}

	// Priority (absent from override): unpriced.
	if _, ok := tbl.Cost("custom-tiered", OpenAIUsage{InputTokens: 600, OutputTokens: 100, ServiceTier: TierPriority}, time.Time{}); ok {
		t.Fatal("priority resolved despite absent from override; want ok=false")
	}
}

// TestMalformedConfigPanics encodes R7: every malformed config case panics
// at construction. One subtest per validation rule.
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

	stdRates := map[ServiceTier]Rates{
		TierStandard: {Base: TierRates{
			Input:  MustParseRat("3e-6"),
			Output: MustParseRat("15e-6"),
		}},
	}

	mustPanicNew("empty alias target", Config{
		Aliases: map[string]string{"a": ""},
	})

	mustPanicNew("alias chain", Config{
		Aliases: map[string]string{"a": "b", "b": "claude-sonnet-4-5"},
	})

	mustPanicNew("alias target not in snapshot or models", Config{
		Aliases: map[string]string{"a": "no-such-model-xyz"},
	})

	mustPanicNew("snapshot-dependent override on nonexistent model", Config{
		Models: map[string]ModelOverride{"no-such-model-xyz": {FlattenTiers: true}},
	})

	mustPanicNew("empty override on nonexistent model", Config{
		Models: map[string]ModelOverride{"no-such-model-xyz": {}},
	})

	mustPanicNew("both Rates and RateSchedule", Config{
		Models: map[string]ModelOverride{"x": {
			Rates: stdRates,
			RateSchedule: []ScheduledRates{
				{EffectiveAt: time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC), Rates: stdRates},
			},
		}},
	})

	mustPanicNew("unsorted schedule", Config{
		Models: map[string]ModelOverride{"x": {
			RateSchedule: []ScheduledRates{
				{EffectiveAt: time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC), Rates: stdRates},
				{EffectiveAt: time.Date(2026, 6, 1, 0, 0, 0, 0, time.UTC), Rates: stdRates},
			},
		}},
	})

	mustPanicNew("missing TierStandard in Rates", Config{
		Models: map[string]ModelOverride{"x": {
			Rates: map[ServiceTier]Rates{
				TierFlex: {Base: TierRates{Input: MustParseRat("3e-6"), Output: MustParseRat("15e-6")}},
			},
		}},
	})

	mustPanicNew("missing TierStandard in RateSchedule", Config{
		Models: map[string]ModelOverride{"x": {
			RateSchedule: []ScheduledRates{{
				EffectiveAt: time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC),
				Rates: map[ServiceTier]Rates{
					TierFlex: {Base: TierRates{Input: MustParseRat("3e-6"), Output: MustParseRat("15e-6")}},
				},
			}},
		}},
	})

	mustPanicNew("unknown service tier in Rates", Config{
		Models: map[string]ModelOverride{"x": {
			Rates: map[ServiceTier]Rates{
				TierStandard: {Base: TierRates{Input: MustParseRat("3e-6"), Output: MustParseRat("15e-6")}},
				"turbo":      {Base: TierRates{Input: MustParseRat("3e-6"), Output: MustParseRat("15e-6")}},
			},
		}},
	})

	mustPanicNew("unpriceable rates: non-positive input", Config{
		Models: map[string]ModelOverride{"x": {
			Rates: map[ServiceTier]Rates{
				TierStandard: {Base: TierRates{Input: MustParseRat("-1e-6"), Output: MustParseRat("15e-6")}},
			},
		}},
	})

	mustPanicNew("unpriceable rates: zero output", Config{
		Models: map[string]ModelOverride{"x": {
			Rates: map[ServiceTier]Rates{
				TierStandard: {Base: TierRates{Input: MustParseRat("3e-6"), Output: new(big.Rat)}},
			},
		}},
	})

	mustPanicNew("unpriceable rates: negative cache", Config{
		Models: map[string]ModelOverride{"x": {
			Rates: map[ServiceTier]Rates{
				TierStandard: {Base: TierRates{
					Input: MustParseRat("3e-6"), CacheRead: MustParseRat("-1e-7"), Output: MustParseRat("15e-6"),
				}},
			},
		}},
	})

	mustPanicNew("non-positive fast multiplier", Config{
		Models: map[string]ModelOverride{"x": {
			Rates: map[ServiceTier]Rates{
				TierStandard: {
					Base: TierRates{Input: MustParseRat("3e-6"), Output: MustParseRat("15e-6")},
					Fast: new(big.Rat),
				},
			},
		}},
	})

	mustPanicNew("non-positive geo multiplier", Config{
		Models: map[string]ModelOverride{"x": {
			Rates: map[ServiceTier]Rates{
				TierStandard: {
					Base: TierRates{Input: MustParseRat("3e-6"), Output: MustParseRat("15e-6")},
					Geo:  map[string]*big.Rat{"us": MustParseRat("-1")},
				},
			},
		}},
	})

	mustPanicNew("non-positive regional uplift", Config{
		Models: map[string]ModelOverride{"x": {
			Rates: map[ServiceTier]Rates{
				TierStandard: {
					Base:           TierRates{Input: MustParseRat("3e-6"), Output: MustParseRat("15e-6")},
					RegionalUplift: map[string]*big.Rat{"eu": new(big.Rat)},
				},
			},
		}},
	})

	mustPanicNew("non-positive tier threshold", Config{
		Models: map[string]ModelOverride{"x": {
			Rates: map[ServiceTier]Rates{
				TierStandard: {
					Base:  TierRates{Input: MustParseRat("3e-6"), Output: MustParseRat("15e-6")},
					Tiers: []Tier{{AbovePromptTokens: 0, TierRates: TierRates{Input: MustParseRat("6e-6"), Output: MustParseRat("30e-6")}}},
				},
			},
		}},
	})

	mustPanicNew("non-ascending tier thresholds", Config{
		Models: map[string]ModelOverride{"x": {
			Rates: map[ServiceTier]Rates{
				TierStandard: {
					Base: TierRates{Input: MustParseRat("3e-6"), Output: MustParseRat("15e-6")},
					Tiers: []Tier{
						{AbovePromptTokens: 200000, TierRates: TierRates{Input: MustParseRat("6e-6"), Output: MustParseRat("30e-6")}},
						{AbovePromptTokens: 100000, TierRates: TierRates{Input: MustParseRat("4e-6"), Output: MustParseRat("20e-6")}},
					},
				},
			},
		}},
	})

	mustPanicNew("unpriceable tier rates", Config{
		Models: map[string]ModelOverride{"x": {
			Rates: map[ServiceTier]Rates{
				TierStandard: {
					Base:  TierRates{Input: MustParseRat("3e-6"), Output: MustParseRat("15e-6")},
					Tiers: []Tier{{AbovePromptTokens: 200000, TierRates: TierRates{Input: MustParseRat("-1e-6"), Output: MustParseRat("30e-6")}}},
				},
			},
		}},
	})
}

// TestZeroTableIsSnapshot encodes R6: the zero-value Table is the
// unoverridden snapshot — identical to the package-level functions.
func TestZeroTableIsSnapshot(t *testing.T) {
	var tbl Table
	for _, model := range []string{"claude-opus-4-8", "claude-sonnet-4-5", "gpt-5.4", "gpt-5.5"} {
		for _, u := range []Usage{
			ClaudeUsage{InputTokens: 1000, CacheReadInputTokens: 500, CacheCreationInputTokens: 200, OutputTokens: 100},
			OpenAIUsage{InputTokens: 1000, CachedInputTokens: 400, OutputTokens: 100},
		} {
			got, gotOK := tbl.Cost(model, u, time.Time{})
			want, wantOK := Cost(model, u)
			if got != want || gotOK != wantOK {
				t.Errorf("%s: Table.Cost = %d, %v; Cost = %d, %v", model, got, gotOK, want, wantOK)
			}
		}
		for _, tier := range []ServiceTier{TierStandard, TierFlex, TierPriority} {
			gotR, gotOK := tbl.RatesFor(model, tier, time.Time{})
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
// Config after New — maps, RateSchedule slice, *big.Rat values — does not
// change billing results from the constructed Table.
func TestConfigDeepCopy(t *testing.T) {
	inputRate := MustParseRat("3e-6")
	outputRate := MustParseRat("15e-6")
	cfg := Config{
		Aliases: map[string]string{"my-model": "claude-sonnet-4-5"},
		Models: map[string]ModelOverride{
			"custom-model": {
				Rates: map[ServiceTier]Rates{
					TierStandard: {Base: TierRates{
						Input:  inputRate,
						Output: outputRate,
					}},
				},
			},
		},
	}

	tbl := New(cfg)
	u := ClaudeUsage{InputTokens: 1000, OutputTokens: 100}
	before, beforeOK := tbl.Cost("custom-model", u, time.Time{})
	beforeAlias, beforeAliasOK := tbl.Cost("my-model", u, time.Time{})

	// Mutate the original *big.Rat values.
	inputRate.SetInt64(999)
	outputRate.SetInt64(999)

	// Mutate the alias map.
	cfg.Aliases["my-model"] = "gpt-5.4"
	delete(cfg.Aliases, "my-model")

	// Mutate the model override map.
	delete(cfg.Models, "custom-model")

	// Table is unaffected.
	after, afterOK := tbl.Cost("custom-model", u, time.Time{})
	if !beforeOK || !afterOK || after != before {
		t.Fatalf("config mutation changed Table.Cost: %d -> %d", before, after)
	}
	afterAlias, afterAliasOK := tbl.Cost("my-model", u, time.Time{})
	if !beforeAliasOK || !afterAliasOK || afterAlias != beforeAlias {
		t.Fatalf("alias mutation changed Table.Cost: %d -> %d", beforeAlias, afterAlias)
	}
}

// TestReturnedRatesAreCopies encodes that mutating a Table.RatesFor return
// value — base rats, tier rats, Fast, Geo, RegionalUplift — does not
// corrupt the Table or the snapshot. Same discipline as the existing
// TestRatesForReturnsCopies.
func TestReturnedRatesAreCopies(t *testing.T) {
	tbl := New(Config{
		Models: map[string]ModelOverride{
			"custom": {
				Rates: map[ServiceTier]Rates{
					TierStandard: {
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
			},
		},
	})

	u := ClaudeUsage{InputTokens: 1000, OutputTokens: 100}
	before, _ := tbl.Cost("custom", u, time.Time{})

	// Mutate every *big.Rat in the returned Rates.
	r, _ := tbl.RatesFor("custom", TierStandard, time.Time{})
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

	after, ok := tbl.Cost("custom", u, time.Time{})
	if !ok || after != before {
		t.Fatalf("mutating RatesFor return changed Cost: %d -> %d", before, after)
	}

	// Also verify snapshot rates are not corrupted via Table reads.
	snapBefore, _ := Cost("claude-sonnet-4-5", ClaudeUsage{InputTokens: 1000, OutputTokens: 100})
	r2, _ := tbl.RatesFor("claude-sonnet-4-5", TierStandard, time.Time{})
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
		Models: map[string]ModelOverride{
			"custom": {
				Rates: map[ServiceTier]Rates{
					TierStandard: {Base: TierRates{
						Input:  MustParseRat("3e-6"),
						Output: MustParseRat("15e-6"),
					}},
				},
			},
			"claude-sonnet-4-5": {FlattenTiers: true},
		},
	})

	u := ClaudeUsage{InputTokens: 1000, OutputTokens: 100}
	want, wantOK := tbl.Cost("custom", u, time.Time{})

	var wg sync.WaitGroup
	for range 100 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 100 {
				got, ok := tbl.Cost("custom", u, time.Time{})
				if got != want || ok != wantOK {
					t.Errorf("concurrent Cost = %d, %v; want %d, %v", got, ok, want, wantOK)
					return
				}
				tbl.Cost("my-sonnet", u, time.Time{})
				tbl.Cost("claude-sonnet-4-5", ClaudeUsage{InputTokens: 250000, OutputTokens: 100}, time.Time{})
				tbl.RatesFor("custom", TierStandard, time.Time{})
			}
		}()
	}
	wg.Wait()
}
