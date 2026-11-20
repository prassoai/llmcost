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
		sans, _ := withoutWebSearch(tc.usage)
		tokens, _ := llmcost.Cost(tc.model, sans)
		if got.Cost-got.WebSearchCost != tokens || got.Price-got.WebSearchPrice != Markup(tokens) {
			t.Errorf("Of(%s, %+v): token part %d/%d, want the search-free call's %d/%d", tc.model, tc.usage,
				got.Cost-got.WebSearchCost, got.Price-got.WebSearchPrice, tokens, Markup(tokens))
		}
	}
}

// TestOfWebSearchOnEveryProviderModel requires a search on a model LiteLLM
// lists no search fee for — gpt-5.4 — to be charged OpenAI's list fee for the
// tool like any other, not dropped or degraded.
func TestOfWebSearchOnEveryProviderModel(t *testing.T) {
	u := llmcost.OpenAIUsage{InputTokens: 1_000, OutputTokens: 100, WebSearchRequests: 2}
	if got, ok := Of("gpt-5.4", u); !ok || got != (Price{Cost: 2_400, Price: 2_520, WebSearchCost: 2_000, WebSearchPrice: 2_100, Source: SourceLLMCost}) {
		t.Errorf("Of(gpt-5.4, 2 searches) = %+v, %v; want tokens 400 + searches 2000, marked up", got, ok)
	}
}

// TestOfRefusesUnpricedWebSearch requires searches the book has no price for
// — the preview tool's, or a model whose search content bills as a fixed
// block — to fail rather than degrade: the request declares the search tool,
// so the caller refuses such a call before it runs, and a silent fallback
// would charge the wrong fee. A degraded dimension elsewhere in the call
// keeps its search fee.
func TestOfRefusesUnpricedWebSearch(t *testing.T) {
	for model, u := range map[string]llmcost.Usage{
		"gpt-5.5":     llmcost.OpenAIUsage{InputTokens: 1_000, WebSearchRequests: 1, WebSearchPreview: true},
		"gpt-4o-mini": llmcost.OpenAIUsage{InputTokens: 1_000, WebSearchRequests: 1},
	} {
		if got, ok := Of(model, u); ok {
			t.Errorf("Of(%s, %+v) = %+v, want unpriced", model, u, got)
		}
	}
	u := llmcost.ClaudeUsage{InputTokens: 1_000, OutputTokens: 1_000, InferenceGeo: "us", WebSearchRequests: 1}
	if _, ok := llmcost.Cost("claude-sonnet-4-5", u); ok {
		t.Fatal("claude-sonnet-4-5 now prices geo us; pick a model without a geo rate")
	}
	if got, ok := Of("claude-sonnet-4-5", u); !ok || got.Source != SourceDegraded || got.WebSearchCost != 1_000 || got.WebSearchPrice != 1_050 {
		t.Errorf("Of(geo + search) = %+v, %v; want degraded with the search fee kept", got, ok)
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
