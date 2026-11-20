package listprice

import (
	"math/big"
	"testing"

	"github.com/prassoai/llmcost"
)

// TestOfWebSearch pins the monetary result of a call with hosted web
// searches: its cost is llmcost's all-in cost (tokens plus $0.01 = 1,000 nls
// per search), its price that cost marked up, and the search part of each is
// exactly the fee — 1,000 nls cost and 1,050 nls price per search — so the
// token and search parts a relay sends separately sum to the charged total.
// Zero searches carry no search part.
func TestOfWebSearch(t *testing.T) {
	for _, tc := range []struct {
		model    string
		usage    llmcost.Usage
		searches int64
	}{
		{"claude-opus-5", llmcost.ClaudeUsage{InputTokens: 12_345, CacheReadInputTokens: 50_000, OutputTokens: 2_000, WebSearchRequests: 2}, 2},
		{"claude-opus-5", llmcost.ClaudeUsage{InputTokens: 10, OutputTokens: 10, Speed: "fast", WebSearchRequests: 1}, 1},
		{"gpt-5.5", llmcost.OpenAIUsage{InputTokens: 300_000, CachedInputTokens: 100_000, OutputTokens: 9_000, WebSearchRequests: 2}, 2},
		{"gpt-5.5", llmcost.OpenAIUsage{InputTokens: 1_000, OutputTokens: 7, ServiceTier: llmcost.TierPriority, WebSearchRequests: 3}, 3},
		{"gpt-5.5", llmcost.OpenAIUsage{InputTokens: 1_000, OutputTokens: 7}, 0},
	} {
		all, _ := llmcost.Cost(tc.model, tc.usage)
		got, ok := Of(tc.model, tc.usage)
		want := Price{Cost: all, Price: Markup(all), WebSearchCost: 1_000 * tc.searches, WebSearchPrice: 1_050 * tc.searches, Source: SourceLLMCost}
		if !ok || got != want {
			t.Errorf("Of(%s, %+v) = %+v, %v; want %+v", tc.model, tc.usage, got, ok, want)
		}
		tokens, _ := llmcost.Cost(tc.model, withoutWebSearch(tc.usage))
		if got.Cost-got.WebSearchCost != tokens || got.Price-got.WebSearchPrice != Markup(tokens) {
			t.Errorf("Of(%s, %+v): token part %d/%d, want the search-free call's %d/%d", tc.model, tc.usage,
				got.Cost-got.WebSearchCost, got.Price-got.WebSearchPrice, tokens, Markup(tokens))
		}
	}
}

// TestOfDegradesUnpricedWebSearch requires a call that reported searches on
// a model with no search fee — gpt-5.4, which LiteLLM lists without one — to
// be charged its tokens rather than recorded as free, flagged as degraded.
// Refusing such a request before it runs is the caller's job.
func TestOfDegradesUnpricedWebSearch(t *testing.T) {
	u := llmcost.OpenAIUsage{InputTokens: 1_000, OutputTokens: 1_000, WebSearchRequests: 2}
	if _, ok := llmcost.Cost("gpt-5.4", u); ok {
		t.Fatal("gpt-5.4 now prices web search; pick a model without a search fee")
	}
	tokens, _ := llmcost.Cost("gpt-5.4", withoutWebSearch(u))
	if got, ok := Of("gpt-5.4", u); !ok || got != (Price{Cost: tokens, Price: Markup(tokens), Source: SourceDegraded}) {
		t.Errorf("Of = %+v, %v; want the tokens alone (%d), degraded", got, ok, tokens)
	}
	// Degrading the search keeps every dimension that still prices: a
	// us-pinned Claude call with searches on a fixture model with a geo
	// rate but no search fee keeps its geo premium.
	b := newBook(
		[]model{{"acme-1", "acme-key-1", llmcost.ProviderAnthropic}},
		map[string]override{"acme-key-1": {llmcost.ProviderAnthropic, llmcost.Rates{
			Base: llmcost.TierRates{Input: llmcost.MustParseRat("1e-6"), Output: llmcost.MustParseRat("2e-6")},
			Geo:  map[string]*big.Rat{"us": llmcost.MustParseRat("1.1")},
		}}},
	)
	if got, ok := b.of("acme-1", llmcost.ClaudeUsage{InputTokens: 1_000_000, InferenceGeo: "us", WebSearchRequests: 1}); !ok || got != (Price{Cost: 110_000, Price: 115_500, Source: SourceDegraded}) {
		t.Errorf("of(geo + unpriced search) = %+v, %v; want $1.10 geo-priced tokens, degraded", got, ok)
	}
}

// TestWebSearchFeesSplitExactly encodes the invariant Price's doc relies on:
// every search fee the book can charge is a whole multiple of 20 nls, so
// [Markup] of the fee is a whole number of nls and the token and search
// parts of a price sum to exactly the marked-up total. A sync that breaks it
// must revisit how the relay splits a price.
func TestWebSearchFeesSplitExactly(t *testing.T) {
	seen := map[string]bool{}
	for _, c := range Costs() {
		if seen[c.Key] {
			continue
		}
		seen[c.Key] = true
		r, _ := std.table.RatesFor(c.Key, c.Tier)
		if r.WebSearch == nil {
			continue
		}
		nls := new(big.Rat).Mul(r.WebSearch, big.NewRat(100_000, 1))
		if !nls.IsInt() || new(big.Int).Rem(nls.Num(), big.NewInt(20)).Sign() != 0 {
			t.Errorf("%s: web search fee %s USD is not a whole multiple of 20 nls", c.Key, r.WebSearch.FloatString(6))
		}
	}
}
