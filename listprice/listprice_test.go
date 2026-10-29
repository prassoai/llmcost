package listprice

import (
	"math"
	"math/big"
	"regexp"
	"slices"
	"testing"

	"github.com/prassoai/llmcost"
)

// TestMarkup pins the list-price arithmetic: cost plus 5%, rounded up to a
// whole nls, so any non-zero cost carries a non-zero markup and anyone holding
// a recorded cost re-derives exactly the recorded price.
func TestMarkup(t *testing.T) {
	for cost, want := range map[llmcost.Nls]llmcost.Nls{0: 0, 1: 2, 20: 21, 100: 105, 101: 107, 1_000_000: 1_050_000} {
		if got := Markup(cost); got != want {
			t.Errorf("Markup(%d) = %d, want %d", cost, got, want)
		}
	}
}

// TestMarkupIsExactUpToOverflow requires Markup to be the exact ceiling of
// cost×(1+MarkupBasisPoints/10⁴) for every cost whose price fits an int64 —
// including costs where cost×MarkupBasisPoints alone would overflow — and to
// panic rather than wrap for a negative cost or one whose price does not fit.
func TestMarkupIsExactUpToOverflow(t *testing.T) {
	exact := func(cost llmcost.Nls) *big.Int {
		n := new(big.Int).Mul(big.NewInt(int64(cost)), big.NewInt(10_000+MarkupBasisPoints))
		n.Add(n, big.NewInt(9_999))
		return n.Div(n, big.NewInt(10_000))
	}
	// The largest cost whose price fits: floor(MaxInt64×10⁴/(10⁴+bps)).
	limit := new(big.Int).Div(new(big.Int).Mul(big.NewInt(math.MaxInt64), big.NewInt(10_000)), big.NewInt(10_000+MarkupBasisPoints))
	if exact(llmcost.Nls(limit.Int64())).Cmp(big.NewInt(math.MaxInt64)) > 0 {
		limit.Sub(limit, big.NewInt(1))
	}
	top := llmcost.Nls(limit.Int64())
	for _, cost := range []llmcost.Nls{0, 1, 9_999, 10_000, 10_001, 19, 21, math.MaxInt64 / MarkupBasisPoints, math.MaxInt64/MarkupBasisPoints + 1, top - 10_001, top - 1, top} {
		if got, want := big.NewInt(int64(Markup(cost))), exact(cost); got.Cmp(want) != 0 {
			t.Errorf("Markup(%d) = %s, want %s", cost, got, want)
		}
	}
	for _, cost := range []llmcost.Nls{-1, top + 1, math.MaxInt64} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("Markup(%d) did not panic", cost)
				}
			}()
			Markup(cost)
		}()
	}
}

// TestOf requires a call's list price to be its llmcost cost marked up: the
// cost is exactly what llmcost charges for the same usage, so the price book
// adds a markup and nothing else.
func TestOf(t *testing.T) {
	for _, tc := range []struct {
		model string
		usage llmcost.Usage
	}{
		{"claude-opus-5", llmcost.ClaudeUsage{InputTokens: 12_345, CacheReadInputTokens: 50_000, CacheCreationInputTokens: 4_000, CacheCreation1hInputTokens: 1_000, OutputTokens: 2_000, InferenceGeo: "us"}},
		{"claude-opus-5", llmcost.ClaudeUsage{InputTokens: 10, OutputTokens: 10, Speed: "fast"}},
		{"gpt-5.5", llmcost.OpenAIUsage{InputTokens: 300_000, CachedInputTokens: 100_000, OutputTokens: 9_000}},
		{"gpt-5.5", llmcost.OpenAIUsage{InputTokens: 1_000, OutputTokens: 1_000, ServiceTier: llmcost.TierFlex}},
		{"claude-haiku-5-5", llmcost.ClaudeUsage{InputTokens: 1}},
	} {
		want, ok := llmcost.Cost(tc.model, tc.usage)
		if !ok {
			t.Fatalf("llmcost.Cost(%s, %+v) unpriced; the fixture must price", tc.model, tc.usage)
		}
		got, ok := Of(tc.model, tc.usage)
		if !ok || got != (Price{Cost: want, Price: Markup(want), Source: SourceLLMCost}) {
			t.Errorf("Of(%s, %+v) = %+v, %v; want cost %d marked up from llmcost", tc.model, tc.usage, got, ok, want)
		}
	}
}

// TestOfDegradesResponseOnlyDimensions requires a call that already happened
// to be priced, never recorded as free, when its response reports a geo the
// model has no rate for: it prices at the base rate and says so.
func TestOfDegradesResponseOnlyDimensions(t *testing.T) {
	u := llmcost.ClaudeUsage{InputTokens: 1_000, OutputTokens: 1_000, InferenceGeo: "us"}
	if _, ok := llmcost.Cost("claude-sonnet-4-5", u); ok {
		t.Fatal("claude-sonnet-4-5 now prices geo us; pick a model without a geo rate")
	}
	base := u
	base.InferenceGeo = ""
	want, _ := llmcost.Cost("claude-sonnet-4-5", base)
	if got, ok := Of("claude-sonnet-4-5", u); !ok || got != (Price{Cost: want, Price: Markup(want), Source: SourceDegraded}) || got.Price <= 0 {
		t.Errorf("Of = %+v, %v; want the base-rate cost %d, degraded", got, ok, want)
	}
}

// TestOfRefuses requires the modes a request states up front — the model,
// the service tier, fast mode — to fail rather than degrade: those are known
// before the call, so Priceable refuses them and a silent fallback would
// underbill by up to the fast multiplier.
func TestOfRefuses(t *testing.T) {
	for _, tc := range []struct {
		model string
		usage llmcost.Usage
	}{
		{"no-such-model", llmcost.ClaudeUsage{InputTokens: 1}},
		{"claude-opus-4-6", llmcost.ClaudeUsage{InputTokens: 1, Speed: "fast", InferenceGeo: "us"}},
		{"gpt-5.3-codex", llmcost.OpenAIUsage{InputTokens: 1, ServiceTier: llmcost.TierFlex}},
	} {
		if got, ok := Of(tc.model, tc.usage); ok {
			t.Errorf("Of(%s, %+v) = %+v, want unpriced", tc.model, tc.usage, got)
		}
	}
}

// TestPriceable pins the pre-flight answer for each request mode.
func TestPriceable(t *testing.T) {
	for _, tc := range []struct {
		model string
		tier  llmcost.ServiceTier
		speed string
		want  bool
	}{
		{"claude-opus-5", llmcost.TierStandard, "", true},
		{"claude-opus-5", llmcost.TierStandard, "fast", true},
		{"claude-opus-4-6", llmcost.TierStandard, "fast", false},
		{"claude-opus-5", llmcost.TierStandard, "turbo", false},
		{"gpt-5.5", llmcost.TierPriority, "", true},
		{"gpt-5.3-codex", llmcost.TierFlex, "", false},
		{"no-such-model", llmcost.TierStandard, "", false},
	} {
		if got := Priceable(tc.model, tc.tier, tc.speed); got != tc.want {
			t.Errorf("Priceable(%s, %s, %q) = %v, want %v", tc.model, tc.tier, tc.speed, got, tc.want)
		}
	}
}

// TestPublishedPriceable requires every published model to price at the
// standard tier from its own provider's key: a data sync that drops one, or a
// curation typo, must fail here rather than refuse a model sold by name.
func TestPublishedPriceable(t *testing.T) {
	for _, m := range published {
		if !Priceable(m.id, llmcost.TierStandard, "") {
			t.Errorf("published model %q does not price", m.id)
		}
		if _, ok := overrides[m.key]; !ok && !slices.Contains(llmcost.Models(m.provider), m.key) {
			t.Errorf("published model %q prices from %q, which %s does not bill", m.id, m.key, m.provider)
		}
	}
}

// TestCurrentVersion pins the identifier format: the module version and the
// first eight hex digits of the vendored LiteLLM commit.
func TestCurrentVersion(t *testing.T) {
	v := CurrentVersion()
	if v.LiteLLMCommit != llmcost.LiteLLMCommit() || v.Module != "(devel)" {
		t.Errorf("CurrentVersion() = %+v, want this module unreleased at the vendored commit", v)
	}
	if !regexp.MustCompile(`^\(devel\)\+litellm\.[0-9a-f]{8}$`).MatchString(v.String()) {
		t.Errorf("String() = %q", v.String())
	}
}

// TestCuration exercises aliases and overrides through a fixture book: an
// alias prices as its key, an override's rates price and are attributed to
// it, and usage the override cannot price exactly degrades.
func TestCuration(t *testing.T) {
	b := newBook(
		[]model{{"acme-1", "acme-key-1", llmcost.ProviderAnthropic}},
		map[string]override{"acme-key-1": {llmcost.ProviderAnthropic, llmcost.Rates{Base: llmcost.TierRates{
			Input: llmcost.MustParseRat("1e-6"), Output: llmcost.MustParseRat("2e-6"), CacheCreation: llmcost.MustParseRat("1.25e-6"),
		}}}},
	)
	if got, ok := b.of("acme-1", llmcost.ClaudeUsage{InputTokens: 1_000_000}); !ok || got != (Price{Cost: 100_000, Price: 105_000, Source: SourceOverride}) {
		t.Errorf("of(alias) = %+v, %v; want the override's $1/MTok, marked up", got, ok)
	}
	if got, ok := b.of("acme-1", llmcost.ClaudeUsage{CacheCreationInputTokens: 1_000_000, CacheCreation1hInputTokens: 1_000_000}); !ok || got != (Price{Cost: 125_000, Price: 131_250, Source: SourceDegraded}) {
		t.Errorf("of(1h writes, no 1h rate) = %+v, %v; want the 5m rate, degraded", got, ok)
	}
	rows := b.rates()
	if len(rows) == 0 || rows[0].Model != "acme-1" || rows[0].Key != "acme-key-1" || !rows[0].Published || rows[0].Source != SourceOverride {
		t.Fatalf("first row = %+v, want the published alias priced from its override", rows[0])
	}
	for _, r := range rows {
		if r.Key == "acme-key-1" && r.Model != "acme-1" {
			t.Errorf("override key listed again unpublished: %+v", r)
		}
	}
}

// TestNewBookRejectsDuplicates requires a curation that publishes one id
// twice to fail at init.
func TestNewBookRejectsDuplicates(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("newBook accepted a duplicate id")
		}
	}()
	newBook([]model{{"gpt-5.5", "gpt-5.5", llmcost.ProviderOpenAI}, {"gpt-5.5", "gpt-5.5", llmcost.ProviderOpenAI}}, nil)
}
