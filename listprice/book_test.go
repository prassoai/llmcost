package listprice

import (
	"math/big"
	"testing"

	"github.com/prassoai/llmcost"
)

// TestBook pins the price book's invariants: published models come first and
// all appear, every list price is its cost marked up exactly, fast rows exist
// only where the model has a fast multiplier, and rates are unique per
// model, tier, speed, context tier and component.
func TestBook(t *testing.T) {
	rows := Book()
	type key struct {
		model     string
		tier      llmcost.ServiceTier
		fast      bool
		above     int64
		component Component
	}
	seen := map[key]bool{}
	models := map[string]bool{}
	pastPublished := map[llmcost.Provider]bool{}
	for _, r := range rows {
		k := key{r.Model, r.Tier, r.Fast, r.AbovePromptTokens, r.Component}
		if seen[k] {
			t.Errorf("duplicate rate %+v", k)
		}
		seen[k] = true
		models[r.Model] = true
		if !r.Published {
			pastPublished[r.Provider] = true
		} else if pastPublished[r.Provider] {
			t.Errorf("published %s listed after unpublished %s models", r.Model, r.Provider)
		}
		if want := new(big.Rat).Mul(r.Cost, big.NewRat(10_500, 10_000)); r.Price.Cmp(want) != 0 || r.Cost.Sign() < 0 {
			t.Errorf("%+v: price %s, want %s", k, r.Price.RatString(), want.RatString())
		}
		if r.Fast && !Priceable(r.Model, r.Tier, "fast") {
			t.Errorf("%+v: fast row for a model without fast mode", k)
		}
		if r.Provider != llmcost.ProviderAnthropic && r.Provider != llmcost.ProviderOpenAI {
			t.Errorf("%+v: provider %s", k, r.Provider)
		}
	}
	for _, m := range published {
		if !models[m.id] {
			t.Errorf("published model %s missing from the book", m.id)
		}
	}
}

// TestBookMatchesOf requires the book to be the rate card [Of] charges: a
// million input tokens of a published model costs its input row exactly.
func TestBookMatchesOf(t *testing.T) {
	for _, r := range Book() {
		if r.Model != "claude-opus-5" || r.Component != ComponentInput || r.AbovePromptTokens != 0 {
			continue
		}
		speed := ""
		if r.Fast {
			speed = "fast"
		}
		p, ok := Of(r.Model, llmcost.ClaudeUsage{InputTokens: 1_000_000, Speed: speed})
		if want := new(big.Rat).Mul(r.Cost, big.NewRat(100_000, 1)); !ok || !want.IsInt() || want.Num().Int64() != int64(p.Cost) {
			t.Errorf("%s fast=%v: Of costs %d nls, book says $%s", r.Model, r.Fast, p.Cost, r.Cost.FloatString(4))
		}
	}
}
