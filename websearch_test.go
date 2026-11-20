package llmcost

import (
	"math/big"
	"strings"
	"testing"
)

// withWebSearch adds LiteLLM's search_context_cost_per_query to a fixture
// spec, the same fee at every search_context_size as the vendored data
// carries for every Anthropic and GPT-5+ entry ($10 per 1,000 searches).
func withWebSearch(spec, fee string) string {
	return strings.Replace(spec, "{", `{
	"search_context_cost_per_query": {"search_context_size_low": `+fee+`, "search_context_size_medium": `+fee+`, "search_context_size_high": `+fee+`},`, 1)
}

// TestWebSearchFee encodes the provider-hosted web search fee: each search
// the response reports bills the model's flat per-search fee on top of its
// tokens — $0.01 = 1,000 nls on Anthropic's $10 per 1,000 searches — and
// zero searches bill nothing extra. Two searches with opus-4-6's tokens
// (1000×5e-6 + 100×2.5e-5 = 750 nls) are 750 + 2×1,000 = 2,750 nls.
func TestWebSearchFee(t *testing.T) {
	r := mustParse(t, withWebSearch(opus46Spec, "0.01"))
	for searches, want := range map[int64]Nls{0: 750, 1: 1750, 2: 2750} {
		if got, ok := price(r, ClaudeUsage{InputTokens: 1000, OutputTokens: 100, WebSearchRequests: searches}); !ok || got != want {
			t.Errorf("%d searches = %d, %v; want %d, true", searches, got, ok, want)
		}
	}
	o := mustParse(t, withWebSearch(gpt54Spec, "0.01"))
	if got, ok := price(o, OpenAIUsage{InputTokens: 1000, CachedInputTokens: 400, OutputTokens: 100, WebSearchRequests: 2}); !ok || got != 310+2000 {
		t.Errorf("openai 2 searches = %d, %v; want 2310, true", got, ok)
	}
}

// TestWebSearchFeeIsFlat encodes that the per-search fee is never scaled:
// fast mode and pinned geo multiply Anthropic's tokens, data residency and a
// context-window tier OpenAI's, but a search costs the same flat fee under
// every one of them. Each case's tokens are priced without searches and the
// fee must add exactly 2,000 nls for two searches.
func TestWebSearchFeeIsFlat(t *testing.T) {
	claude := mustParse(t, withWebSearch(opus46Spec, "0.01"))
	openai := mustParse(t, withWebSearch(gpt54Spec, "0.01"))
	for name, tc := range map[string]struct {
		r          Rates
		with, sans Usage
	}{
		"fast+us": {claude,
			ClaudeUsage{InputTokens: 1000, OutputTokens: 100, Speed: "fast", InferenceGeo: "us", WebSearchRequests: 2},
			ClaudeUsage{InputTokens: 1000, OutputTokens: 100, Speed: "fast", InferenceGeo: "us"}},
		"eu residency": {openai,
			OpenAIUsage{InputTokens: 1000, OutputTokens: 100, DataResidency: "eu", WebSearchRequests: 2},
			OpenAIUsage{InputTokens: 1000, OutputTokens: 100, DataResidency: "eu"}},
		"above 272k": {openai,
			OpenAIUsage{InputTokens: 300_000, OutputTokens: 100, WebSearchRequests: 2},
			OpenAIUsage{InputTokens: 300_000, OutputTokens: 100}},
		"priority": {table()["gpt-5.5"][TierPriority],
			OpenAIUsage{InputTokens: 1000, OutputTokens: 100, ServiceTier: TierPriority, WebSearchRequests: 2},
			OpenAIUsage{InputTokens: 1000, OutputTokens: 100, ServiceTier: TierPriority}},
	} {
		with, ok1 := price(tc.r, tc.with)
		sans, ok2 := price(tc.r, tc.sans)
		if !ok1 || !ok2 || with-sans != 2000 {
			t.Errorf("%s: %d with searches, %d without (%v, %v); want exactly 2000 apart", name, with, sans, ok1, ok2)
		}
	}
}

// TestWebSearchWithoutFeeFails encodes the no-silent-zero rule for searches:
// a response reporting searches on a model with no search fee fails to price
// rather than billing the searches as free, while the same model's
// search-free usage still prices.
func TestWebSearchWithoutFeeFails(t *testing.T) {
	r := mustParse(t, opus46Spec)
	if _, ok := price(r, ClaudeUsage{InputTokens: 1, WebSearchRequests: 1}); ok {
		t.Error("searches priced on a model with no search fee")
	}
	if _, ok := price(r, ClaudeUsage{InputTokens: 1}); !ok {
		t.Error("search-free usage failed to price")
	}
}

// TestParseWebSearch encodes which search_context_cost_per_query shapes
// price: one positive fee at every size. A fee that varies by size would
// need the request's search_context_size, which no usage carries, so it is
// unpriced rather than billing one size's fee for another; zero, negative,
// unparseable and absent fees are unpriced too.
func TestParseWebSearch(t *testing.T) {
	for name, tc := range map[string]struct {
		field string
		want  *big.Rat
	}{
		"uniform":    {`"search_context_cost_per_query": {"search_context_size_low": 0.01, "search_context_size_medium": 0.01, "search_context_size_high": 0.01},`, big.NewRat(1, 100)},
		"varies":     {`"search_context_cost_per_query": {"search_context_size_low": 0.03, "search_context_size_medium": 0.035, "search_context_size_high": 0.05},`, nil},
		"zero":       {`"search_context_cost_per_query": {"search_context_size_low": 0, "search_context_size_medium": 0, "search_context_size_high": 0},`, nil},
		"negative":   {`"search_context_cost_per_query": {"search_context_size_medium": -0.01},`, nil},
		"not number": {`"search_context_cost_per_query": {"search_context_size_medium": "ten"},`, nil},
		"not object": {`"search_context_cost_per_query": 0.01,`, nil},
		"absent":     {``, nil},
	} {
		got := mustParse(t, strings.Replace(sonnet45Spec, "{", "{"+tc.field, 1)).WebSearch
		if (got == nil) != (tc.want == nil) || (got != nil && got.Cmp(tc.want) != 0) {
			t.Errorf("%s: WebSearch = %v, want %v", name, got, tc.want)
		}
	}
}

// TestVendoredWebSearchFees pins the vendored search fees murmur bills: $10
// per 1,000 searches on current Anthropic and GPT-5+ models, at every
// service tier. If a sync moves one, a published search price moves with it —
// verify against the provider's pricing page before merging.
func TestVendoredWebSearchFees(t *testing.T) {
	for model, tiers := range map[string][]ServiceTier{
		"claude-opus-5-5": {TierStandard},
		"claude-opus-4-8": {TierStandard},
		"gpt-5.5":         {TierStandard, TierFlex, TierPriority},
		"gpt-6.2-sol":     {TierStandard},
	} {
		for _, tier := range tiers {
			r, ok := RatesFor(model, tier)
			if !ok || r.WebSearch == nil || r.WebSearch.Cmp(big.NewRat(1, 100)) != 0 {
				t.Errorf("%s at %s: WebSearch = %v, %v; want 0.01", model, tier, r.WebSearch, ok)
			}
		}
	}
	if got, ok := Cost("claude-opus-4-8", ClaudeUsage{WebSearchRequests: 2}); !ok || got != 2000 {
		t.Errorf("Cost(claude-opus-4-8, 2 searches) = %d, %v; want 2000, true", got, ok)
	}
}

// TestNegativeWebSearchesPanic encodes that a negative search count is an
// impossible response and a caller bug, like a negative token count.
func TestNegativeWebSearchesPanic(t *testing.T) {
	for _, u := range []Usage{ClaudeUsage{WebSearchRequests: -1}, OpenAIUsage{WebSearchRequests: -1}} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%+v did not panic", u)
				}
			}()
			Cost("claude-opus-4-8", u)
		}()
	}
}

// TestWebSearchOverride encodes that a consumer override can restate the
// search fee with the rest of a model's rates, that RatesFor returns a copy
// of it, and that New rejects a non-positive fee as it does any other
// garbage rate.
func TestWebSearchOverride(t *testing.T) {
	tbl := New(Config{Overrides: map[string]*Rates{"acme": {
		Base:      TierRates{Input: MustParseRat("1e-6"), Output: MustParseRat("2e-6")},
		WebSearch: MustParseRat("0.025"),
	}}})
	if got, ok := tbl.Cost("acme", OpenAIUsage{WebSearchRequests: 2}); !ok || got != 5000 {
		t.Errorf("override Cost = %d, %v; want 5000, true", got, ok)
	}
	r, _ := tbl.RatesFor("acme", TierStandard)
	r.WebSearch.SetInt64(999)
	if got, _ := tbl.Cost("acme", OpenAIUsage{WebSearchRequests: 2}); got != 5000 {
		t.Errorf("mutating RatesFor's fee changed Cost to %d", got)
	}
	defer func() {
		if recover() == nil {
			t.Error("New accepted a zero web search fee")
		}
	}()
	New(Config{Overrides: map[string]*Rates{"acme": {
		Base:      TierRates{Input: MustParseRat("1e-6"), Output: MustParseRat("2e-6")},
		WebSearch: new(big.Rat),
	}}})
}
