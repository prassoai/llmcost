package listprice

import (
	"math/big"
	"slices"
	"testing"

	"github.com/prassoai/llmcost"
)

// TestLiveList pins the list price of a GPT-Live voice session: OpenAI's
// $0.05 a minute plus the markup, billed per second, and published in the
// book as a per-second row rather than a per-million-token one.
func TestLiveList(t *testing.T) {
	got, ok := Of("gpt-live-1", llmcost.LiveUsage{Seconds: 60})
	if !ok || got != (Price{Cost: 5000, Price: 5250, Source: SourceLLMCost}) {
		t.Fatalf("Of(gpt-live-1, 60 s) = %+v, %v", got, ok)
	}
	if !Priceable("gpt-live-1", llmcost.TierStandard, "") {
		t.Fatal("gpt-live-1 is not priceable")
	}
	rows := slices.DeleteFunc(slices.Clone(Book()), func(r Rate) bool { return r.Model != "gpt-live-1" })
	if len(rows) != 1 || !rows[0].Published || rows[0].Component != ComponentSecond || rows[0].Cost.Cmp(llmcost.MustParseRat("0.000833333333333")) != 0 {
		t.Fatalf("gpt-live-1 book rows = %+v", rows)
	}
	if want := new(big.Rat).Mul(rows[0].Cost, big.NewRat(10_500, 10_000)); rows[0].Price.Cmp(want) != 0 {
		t.Fatalf("gpt-live-1 price = %s, want %s", rows[0].Price.FloatString(12), want.FloatString(12))
	}
}
