package listprice

import (
	"math/big"
	"slices"
	"testing"

	"github.com/prassoai/llmcost"
)

// TestBook pins the price book's invariants: published models come first and
// all appear, every list price is its cost marked up by MarkupBasisPoints
// exactly, fast rows exist only where the model has a fast multiplier, and a
// model has one rate per tier, speed, context tier and component whichever
// provider serves it.
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
			pastPublished[r.PricedBy] = true
		} else if pastPublished[r.PricedBy] {
			t.Errorf("published %s listed after unpublished %s models", r.Model, r.PricedBy)
		}
		if want := new(big.Rat).Mul(r.Cost, big.NewRat(10_000+MarkupBasisPoints, 10_000)); r.Price.Cmp(want) != 0 || r.Cost.Sign() < 0 {
			t.Errorf("%+v: price %s, want %s", k, r.Price.RatString(), want.RatString())
		}
		if r.Fast && !Priceable(r.Model, r.Tier, "fast") {
			t.Errorf("%+v: fast row for a model without fast mode", k)
		}
		if r.PricedBy != llmcost.ProviderAnthropic && r.PricedBy != llmcost.ProviderOpenAI {
			t.Errorf("%+v: priced by %s", k, r.PricedBy)
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

// TestCostsCarryEveryPrice requires the book's prices to be recomputable from
// its costs: every rate's cost is its pricing provider's cost for the same
// model, tier, speed, context tier and component. Costs are unique per
// provider and model, and only providers llmcost prices appear.
func TestCostsCarryEveryPrice(t *testing.T) {
	type key struct {
		provider  llmcost.Provider
		model     string
		tier      llmcost.ServiceTier
		fast      bool
		above     int64
		component Component
	}
	costs := map[key]*big.Rat{}
	for _, c := range Costs() {
		k := key{c.Provider, c.Model, c.Tier, c.Fast, c.AbovePromptTokens, c.Component}
		if _, dup := costs[k]; dup {
			t.Errorf("duplicate cost %+v", k)
		}
		if !slices.Contains(costers, c.Provider) || c.Key == "" || c.Cost.Sign() < 0 {
			t.Errorf("malformed cost %+v", c)
		}
		costs[k] = c.Cost
	}
	for _, r := range Book() {
		k := key{r.PricedBy, r.Model, r.Tier, r.Fast, r.AbovePromptTokens, r.Component}
		if c, ok := costs[k]; !ok || c.Cmp(r.Cost) != 0 {
			t.Errorf("%+v: price set by cost %s, the pricing provider's cost is %v", k, r.Cost.RatString(), c)
		}
	}
}

// TestCostsSpanProviders requires a model's costs to cover every cloud that
// serves it, each under that cloud's own key: claude-sonnet-4-5 is priced by
// Anthropic and also served by Bedrock, Vertex AI and Azure AI Foundry.
func TestCostsSpanProviders(t *testing.T) {
	keys := map[llmcost.Provider]string{}
	for _, c := range Costs() {
		if c.Model == "claude-sonnet-4-5" && c.Tier == llmcost.TierStandard && !c.Fast && c.AbovePromptTokens == 0 && c.Component == ComponentInput {
			keys[c.Provider] = c.Key
		}
	}
	for _, p := range []llmcost.Provider{llmcost.ProviderAnthropic, llmcost.ProviderBedrock, llmcost.ProviderVertexAI, llmcost.ProviderAzureAI} {
		if want, ok := (llmcost.ModelSelector{Provider: p, Model: "claude-sonnet-4-5"}).Key(); !ok || keys[p] != want {
			t.Errorf("claude-sonnet-4-5 on %s: cost keyed %q, want %q", p, keys[p], want)
		}
	}
	if k, ok := keys[llmcost.ProviderOpenAI]; ok {
		t.Errorf("claude-sonnet-4-5 costed on openai under %q", k)
	}
}
