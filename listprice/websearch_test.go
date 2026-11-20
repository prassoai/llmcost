package listprice

import (
	"math/big"
	"slices"
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

// TestBookWebSearch requires the book to state the search fee [Of] charges:
// a model and service tier carry a web search row exactly when Of prices a
// search on them, and the row's price per million searches is a million
// times what Of charges for one. Published models list OpenAI's and
// Anthropic's $10 per 1,000 marked up to $10.50; gpt-4o-mini, whose search
// content bills as a fixed token block, lists none.
func TestBookWebSearch(t *testing.T) {
	type at struct {
		model string
		tier  llmcost.ServiceTier
	}
	rows := map[at]*big.Rat{}
	tiers := map[at]bool{}
	for _, r := range Book() {
		tiers[at{r.Model, r.Tier}] = true
		if r.Component == ComponentWebSearch {
			if r.Fast || r.AbovePromptTokens != 0 {
				t.Errorf("%s %s: web search row at fast=%v above=%d, want one base row per tier", r.Model, r.Tier, r.Fast, r.AbovePromptTokens)
			}
			rows[at{r.Model, r.Tier}] = r.Price
		}
	}
	searched := func(model string, tier llmcost.ServiceTier) (Price, bool) {
		if tier == llmcost.TierStandard {
			if p, ok := Of(model, llmcost.ClaudeUsage{InputTokens: 1, WebSearchRequests: 1}); ok {
				return p, true
			}
		}
		return Of(model, llmcost.OpenAIUsage{InputTokens: 1, ServiceTier: tier, WebSearchRequests: 1})
	}
	for k := range tiers {
		p, ok := searched(k.model, k.tier)
		row, listed := rows[k]
		switch {
		case ok != listed:
			t.Errorf("%s %s: Of prices a search %v, book lists a search price %v", k.model, k.tier, ok, listed)
		case listed && new(big.Rat).Mul(row, big.NewRat(1, 10)).Cmp(big.NewRat(p.WebSearchPrice, 1)) != 0:
			t.Errorf("%s %s: book prices a million searches at $%s, Of charges %d nls for one", k.model, k.tier, row.FloatString(2), p.WebSearchPrice)
		}
	}
	for _, m := range []string{"claude-opus-5-5", "gpt-5.5"} {
		if got := rows[at{m, llmcost.TierStandard}]; got == nil || got.Cmp(big.NewRat(10_500, 1)) != 0 {
			t.Errorf("%s: web search $%v per million searches, want $10,500", m, got)
		}
	}
	if got, ok := rows[at{"gpt-4o-mini", llmcost.TierStandard}]; ok || !tiers[at{"gpt-4o-mini", llmcost.TierStandard}] {
		t.Errorf("gpt-4o-mini: listed %v with web search price %v, want listed without one", tiers[at{"gpt-4o-mini", llmcost.TierStandard}], got)
	}
}

// TestChatSearchModelsHaveNoPrice requires OpenAI's Chat Completions search
// models to be absent from the book and refused by Of and Priceable: they
// search on every request without reporting a count, so a gateway refuses
// them on its own key, and the price page — rendered from the book — must not
// list a model that key refuses. ChatSearchModel is the one predicate both
// sides use.
func TestChatSearchModelsHaveNoPrice(t *testing.T) {
	var vendored []string
	for _, k := range llmcost.Models(llmcost.ProviderOpenAI) {
		if ChatSearchModel(k) {
			vendored = append(vendored, k)
		}
	}
	for _, m := range []string{"gpt-4o-search-preview", "gpt-4o-mini-search-preview", "gpt-5-search-api", "gpt-5-search-api-2025-10-14"} {
		if !slices.Contains(vendored, m) {
			t.Errorf("%s: not a vendored ChatSearchModel", m)
		}
	}
	for _, r := range Book() {
		if slices.Contains(vendored, r.Model) {
			t.Errorf("book lists %s", r.Model)
		}
	}
	for _, c := range Costs() {
		if slices.Contains(vendored, c.Model) {
			t.Errorf("costs list %s on %s", c.Model, c.Provider)
		}
	}
	for _, m := range vendored {
		if Priceable(m, llmcost.TierStandard, "") {
			t.Errorf("%s is priceable", m)
		}
		if p, ok := Of(m, llmcost.OpenAIUsage{InputTokens: 1_000}); ok {
			t.Errorf("Of(%s) = %+v, want unpriced", m, p)
		}
	}
	for _, m := range []string{"gpt-5", "gpt-4o-mini", "gpt-5-search", "o3-deep-research"} {
		if ChatSearchModel(m) {
			t.Errorf("ChatSearchModel(%s) = true", m)
		}
	}
}

// TestSearchRefused requires SearchRefused to name exactly the models that
// can search but have no search price. A price page lists them as needing the
// customer's own key, which is only true of a model whose endpoint has a
// search tool: legacy completion models also lack a search price but cannot
// search on any key, and models with a price are not refused.
func TestSearchRefused(t *testing.T) {
	for m, want := range map[string]bool{
		"gpt-4o-mini": true, "gpt-4o-mini-2024-07-18": true, "gpt-4.1-mini": true, "gpt-4.1-mini-2025-04-14": true,
		"babbage-002": false, "davinci-002": false, "gpt-3.5-turbo-instruct": false,
		"gpt-5.5": false, "claude-opus-5-5": false, "gpt-4o-mini-search-preview": false, "no-such-model": false,
	} {
		if got := SearchRefused(m); got != want {
			t.Errorf("SearchRefused(%s) = %v, want %v", m, got, want)
		}
	}
	for _, r := range Book() {
		if SearchRefused(r.Model) && r.Component == ComponentWebSearch {
			t.Errorf("%s: refused search but the book prices it", r.Model)
		}
	}
}
