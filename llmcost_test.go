package llmcost

import (
	"encoding/json"
	"math/big"
	"testing"
)

// opusRates mirrors claude-opus-4-8's published prices ($5/M input, $0.50/M
// cache read, $25/M output) as a fixture, so the exact-math tests are
// independent of the vendored data — a weekly price sync must never turn
// these red.
func opusRates() Rates {
	return Rates{Input: mustRat("5e-6"), CachedInput: mustRat("5e-7"), Output: mustRat("2.5e-5")}
}

func mustRat(s string) *big.Rat {
	r, ok := new(big.Rat).SetString(s)
	if !ok {
		panic("bad rat literal: " + s)
	}
	return r
}

// TestExactCost encodes the core requirement: a response is priced as
// Σ rate × tokens computed exactly, and a total that lands exactly on an nls
// boundary is returned as-is (ceiling only rounds fractional totals).
// 1000×5e-6 + 2000×5e-7 + 500×2.5e-5 = $0.0185 = exactly 1850 nls.
func TestExactCost(t *testing.T) {
	if got := ceilNls(usd(opusRates(), Usage{InputTokens: 1000, CachedInputTokens: 2000, OutputTokens: 500})); got != 1850 {
		t.Fatalf("cost = %d nls, want 1850", got)
	}
}

// TestCeilingRounding encodes the rounding rule: the final total — and only
// the final total — is ceiling-rounded, matching back's convention of
// ceiling-rounding its margin'd total. One output token at $2.5e-5 is 2.5 nls
// and must bill as 3, and any non-zero usage must cost at least 1 nls: a
// single cached token at $5e-7 is 0.05 nls, never free.
func TestCeilingRounding(t *testing.T) {
	if got := ceilNls(usd(opusRates(), Usage{OutputTokens: 1})); got != 3 {
		t.Fatalf("1 output token = %d nls, want 3 (ceil of 2.5)", got)
	}
	if got := ceilNls(usd(opusRates(), Usage{CachedInputTokens: 1})); got != 1 {
		t.Fatalf("1 cached token = %d nls, want 1 (ceil of 0.05)", got)
	}
}

// TestSubNlsAccumulation encodes the precision requirement that motivates
// big.Rat: many tokens each worth a fraction of an nls must sum exactly.
// 400 cached tokens at $2.5e-7 each are 0.025 nls apiece — per-token flooring
// would truncate to 0 and per-token ceiling would inflate to 400; the exact
// total is $1e-4 = exactly 10 nls.
func TestSubNlsAccumulation(t *testing.T) {
	r := Rates{Input: mustRat("2.5e-6"), CachedInput: mustRat("2.5e-7"), Output: mustRat("1.5e-5")}
	if got := ceilNls(usd(r, Usage{CachedInputTokens: 400})); got != 10 {
		t.Fatalf("400 cached tokens = %d nls, want exactly 10", got)
	}
}

// TestZeroUsageIsFree encodes that ceiling never invents cost: a response
// with no tokens costs 0 nls.
func TestZeroUsageIsFree(t *testing.T) {
	if got, ok := Cost("claude-opus-4-8", Usage{}); !ok || got != 0 {
		t.Fatalf("Cost(zero usage) = %d, %v; want 0, true", got, ok)
	}
}

// TestUnknownModel encodes the fail-loud contract: a model the module cannot
// price returns ok=false — callers must never mistake "unknown" for "free".
// LiteLLM's "sample_spec" documentation row and its zero-rate entries must
// not resolve either.
func TestUnknownModel(t *testing.T) {
	for _, model := range []string{"no-such-model", "sample_spec", ""} {
		if _, ok := Cost(model, Usage{InputTokens: 1}); ok {
			t.Errorf("Cost(%q) resolved; want ok=false", model)
		}
		if _, ok := RatesFor(model); ok {
			t.Errorf("RatesFor(%q) resolved; want ok=false", model)
		}
	}
}

// TestAliasesResolve is the validation gate for the weekly data sync: every
// internal model id in the alias map must resolve against the vendored
// LiteLLM data with strictly positive input, cached-input, and output rates.
// If upstream renames or drops a priced model — LiteLLM has shipped a broken
// cost map before — this test fails the sync PR instead of letting a consumer
// silently bill zero.
func TestAliasesResolve(t *testing.T) {
	for id := range aliases {
		r, ok := RatesFor(id)
		if !ok {
			t.Errorf("alias %q no longer resolves in vendored data", id)
			continue
		}
		if r.Input.Sign() <= 0 || r.CachedInput.Sign() <= 0 || r.Output.Sign() <= 0 {
			t.Errorf("alias %q has non-positive rates: in=%v cached=%v out=%v", id, r.Input, r.CachedInput, r.Output)
		}
	}
}

// TestAliasMapping encodes that aliasing is transparent: an internal id and
// the LiteLLM key it maps to price identically.
func TestAliasMapping(t *testing.T) {
	u := Usage{InputTokens: 12345, CachedInputTokens: 678, OutputTokens: 910}
	got, ok := Cost("codex-mini", u)
	want, ok2 := Cost("codex-mini-latest", u)
	if !ok || !ok2 || got != want {
		t.Fatalf("Cost(codex-mini) = %d, %v; Cost(codex-mini-latest) = %d, %v; want equal and ok", got, ok, want, ok2)
	}
}

// TestDirectLiteLLMKey encodes that ids outside the alias map fall through to
// direct LiteLLM key lookup, so gateways can price arbitrary upstream models.
func TestDirectLiteLLMKey(t *testing.T) {
	if _, inAliases := aliases["gpt-4o"]; inAliases {
		t.Fatal("gpt-4o joined the alias map; pick a different direct key for this test")
	}
	if _, ok := Cost("gpt-4o", Usage{InputTokens: 1}); !ok {
		t.Fatal("Cost(gpt-4o) did not resolve via direct LiteLLM key lookup")
	}
}

// TestCostMatchesRatesFor encodes that the two exported views never disagree:
// Cost(model, u) is exactly the ceiling of the total derived from RatesFor.
func TestCostMatchesRatesFor(t *testing.T) {
	u := Usage{InputTokens: 3117, CachedInputTokens: 41775, OutputTokens: 977}
	for id := range aliases {
		r, ok := RatesFor(id)
		if !ok {
			continue // TestAliasesResolve reports this
		}
		got, ok := Cost(id, u)
		if want := ceilNls(usd(r, u)); !ok || got != want {
			t.Errorf("Cost(%q) = %d, %v; want %d from RatesFor", id, got, ok, want)
		}
	}
}

// TestRatesForReturnsCopies encodes that the shared rate table is immutable:
// a caller mutating the rats RatesFor returned must not corrupt later
// lookups (or costs) of the same model.
func TestRatesForReturnsCopies(t *testing.T) {
	before, _ := Cost("claude-opus-4-8", Usage{InputTokens: 1000})
	r, _ := RatesFor("claude-opus-4-8")
	r.Input.SetInt64(999)
	r.CachedInput.SetInt64(999)
	r.Output.SetInt64(999)
	if after, _ := Cost("claude-opus-4-8", Usage{InputTokens: 1000}); after != before {
		t.Fatalf("mutating RatesFor result changed Cost: %d -> %d", before, after)
	}
}

// TestNegativeTokensPanic encodes that negative token counts are a caller
// bug, rejected loudly rather than priced as a negative bill.
func TestNegativeTokensPanic(t *testing.T) {
	for _, u := range []Usage{{InputTokens: -1}, {CachedInputTokens: -1}, {OutputTokens: -1}} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("Cost(%+v) did not panic", u)
				}
			}()
			Cost("claude-opus-4-8", u)
		}()
	}
}

// TestRatParsesDecimalLiteralsExactly encodes the no-float64 requirement:
// rates come out of the JSON as exact rationals of their decimal literals.
// 2.5e-7 is exactly 1/4,000,000 — a value float64 cannot represent.
func TestRatParsesDecimalLiteralsExactly(t *testing.T) {
	if r := rat(json.Number("2.5e-7")); r == nil || r.Cmp(big.NewRat(1, 4_000_000)) != 0 {
		t.Fatalf("rat(2.5e-7) = %v, want exactly 1/4000000", r)
	}
	if r := rat(json.Number("")); r != nil {
		t.Fatalf("rat(absent) = %v, want nil", r)
	}
}
