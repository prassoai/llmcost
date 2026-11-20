package llmcost

import (
	"math/big"
	"testing"
)

// parseKey parses one fixture spec under key, through [Parse] — the path that
// gives a key its provider's web search fee.
func parseKey(t *testing.T, key, spec string) Rates {
	t.Helper()
	snap, err := Parse([]byte(`{"` + key + `": ` + spec + `}`))
	if err != nil {
		t.Fatal(err)
	}
	return snap.models[key][TierStandard]
}

// TestWebSearchFee encodes the provider-hosted web search fee: each search
// the response reports bills the model's flat per-search fee on top of its
// tokens — $0.01 = 1,000 nls on Anthropic's $10 per 1,000 searches — and
// zero searches bill nothing extra. Two searches with opus-4-6's tokens
// (1000×5e-6 + 100×2.5e-5 = 750 nls) are 750 + 2×1,000 = 2,750 nls.
func TestWebSearchFee(t *testing.T) {
	r := parseKey(t, "claude-opus-4-6", opus46Spec)
	for searches, want := range map[int64]Nls{0: 750, 1: 1750, 2: 2750} {
		if got, ok := price(r, ClaudeUsage{InputTokens: 1000, OutputTokens: 100, WebSearchRequests: searches}); !ok || got != want {
			t.Errorf("%d searches = %d, %v; want %d, true", searches, got, ok, want)
		}
	}
	o := parseKey(t, "gpt-5.4", gpt54Spec)
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
	claude := parseKey(t, "claude-opus-4-6", opus46Spec)
	openai := parseKey(t, "gpt-5.4", gpt54Spec)
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

// TestWebSearchFeeOwnership pins who sets the search fee: the provider's list
// fee for its standard tool, on every key Anthropic or OpenAI bills — not
// LiteLLM's per-model search_context_cost_per_query, which is absent for
// gpt-5.4 and carries the retired preview tool's $25 per 1,000 for gpt-4o.
// Keys another provider bills (Azure, Vertex AI, Fireworks) and the OpenAI
// models whose search content bills as a fixed token block price no search.
func TestWebSearchFeeOwnership(t *testing.T) {
	fee := big.NewRat(1, 100)
	for _, tc := range []struct {
		model string
		tier  ServiceTier
		want  *big.Rat
	}{
		{"claude-opus-5-5", TierStandard, fee},
		{"claude-opus-4-8", TierStandard, fee},
		{"claude-haiku-4-5", TierStandard, fee},
		{"gpt-5.4", TierStandard, fee},
		{"gpt-5.4", TierPriority, fee},
		{"gpt-5.5", TierFlex, fee},
		{"gpt-6.2-sol", TierStandard, fee},
		{"gpt-4o", TierStandard, fee},
		{"gpt-4o-mini", TierStandard, nil},
		{"gpt-4.1-mini-2025-04-14", TierStandard, nil},
		{"azure/gpt-5.4", TierStandard, nil},
		{"vertex_ai/claude-sonnet-4-5", TierStandard, nil},
		{"fireworks_ai/accounts/fireworks/models/deepseek-v4-flash", TierStandard, nil},
	} {
		r, ok := RatesFor(tc.model, tc.tier)
		if !ok {
			t.Errorf("%s at %s no longer resolves", tc.model, tc.tier)
			continue
		}
		if (r.WebSearch == nil) != (tc.want == nil) || (r.WebSearch != nil && r.WebSearch.Cmp(tc.want) != 0) {
			t.Errorf("%s at %s: WebSearch = %v, want %v", tc.model, tc.tier, r.WebSearch, tc.want)
		}
	}
	if got, ok := Cost("gpt-5.4", OpenAIUsage{InputTokens: 1000, OutputTokens: 100, WebSearchRequests: 2}); !ok || got != 400+2000 {
		t.Errorf("Cost(gpt-5.4, 2 searches) = %d, %v; want 2400, true", got, ok)
	}
	if _, ok := Cost("gpt-4o-mini", OpenAIUsage{InputTokens: 1000, WebSearchRequests: 1}); ok {
		t.Error("searches priced on gpt-4o-mini, whose search content bills as a fixed block")
	}
	if _, ok := Cost("gpt-4o-mini", OpenAIUsage{InputTokens: 1000}); !ok {
		t.Error("gpt-4o-mini's search-free usage failed to price")
	}
}

// TestWebSearchPreviewFails encodes the explicit rule for the web_search_preview
// tool: its fee depends on the model, so usage reporting preview searches
// fails to price rather than billing the standard tool's fee, while the same
// call without searches prices normally.
func TestWebSearchPreviewFails(t *testing.T) {
	if _, ok := Cost("gpt-5.5", OpenAIUsage{InputTokens: 1000, WebSearchRequests: 1, WebSearchPreview: true}); ok {
		t.Error("preview searches priced")
	}
	if _, ok := Cost("gpt-5.5", OpenAIUsage{InputTokens: 1000, WebSearchPreview: true}); !ok {
		t.Error("a preview-tool call that searched nothing failed to price")
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
