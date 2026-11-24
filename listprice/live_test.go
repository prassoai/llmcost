package listprice

import (
	"math/big"
	"slices"
	"testing"

	"github.com/prassoai/llmcost"
)

// TestLiveList pins GPT-Live's list prices: OpenAI's $0.05 a minute active and
// $0.02 a minute standby, billed per second, each marked up. Sixty active and
// thirty standby seconds cost $0.06 and are priced at $0.063. Both rates are
// published for both models as per-second rows, and a caller that reports no
// standby is billed for active time alone.
func TestLiveList(t *testing.T) {
	for _, model := range []string{"gpt-live-1", "gpt-live-1.1"} {
		t.Run(model, func(t *testing.T) {
			got, ok := Of(model, llmcost.LiveUsage{ActiveSeconds: 60, StandbySeconds: 30})
			if !ok || got != (Price{Cost: 6000, Price: 6300, Source: SourceOverride}) {
				t.Fatalf("Of(60 active, 30 standby) = %+v, %v; want cost 6000, price 6300", got, ok)
			}
			if got, ok := Of(model, llmcost.LiveUsage{ActiveSeconds: 60}); !ok || got.Cost != 5000 || got.Price != 5250 {
				t.Fatalf("Of(60 active) = %+v, %v; want cost 5000, price 5250", got, ok)
			}
			if !Priceable(model, llmcost.TierStandard, "") {
				t.Fatal("not priceable")
			}
			rows := slices.DeleteFunc(Book(), func(r Rate) bool { return r.Model != model })
			want := map[Component]*big.Rat{ComponentActiveSecond: big.NewRat(5, 6000), ComponentStandbySecond: big.NewRat(2, 6000)}
			if len(rows) != len(want) {
				t.Fatalf("book rows = %+v, want one per rate", rows)
			}
			for _, r := range rows {
				if !r.Published || r.Cost.Cmp(want[r.Component]) != 0 || r.Price.Cmp(new(big.Rat).Mul(want[r.Component], big.NewRat(10_500, 10_000))) != 0 {
					t.Errorf("row %v: cost %s price %s", r.Component, r.Cost.RatString(), r.Price.RatString())
				}
			}
		})
	}
}

// TestGPTLiveOverrideStillNeeded fails once LiteLLM publishes a GPT-Live
// standby rate, when the override in published.go should be revisited.
func TestGPTLiveOverrideStillNeeded(t *testing.T) {
	for _, model := range []string{"gpt-live-1", "gpt-live-1.1"} {
		r, ok := llmcost.RatesFor(model, llmcost.TierStandard)
		if !ok || r.PerSecond == nil {
			t.Fatalf("%s: LiteLLM no longer prices active seconds", model)
		}
		if r.PerStandbySecond != nil {
			t.Errorf("%s: LiteLLM now publishes a standby rate; revisit the override", model)
		}
	}
}
