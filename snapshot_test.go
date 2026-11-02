package llmcost

import (
	"maps"
	"slices"
	"strings"
	"testing"
)

// TestParseEmbeddedMatchesTable encodes the Parse contract: parsing the
// vendored bytes yields exactly the snapshot the embedded path builds — the
// same keys and the same rates at every service tier, Azure backfill
// included. A consumer that fetches the pinned commit's file prices with
// the same table the module ships.
func TestParseEmbeddedMatchesTable(t *testing.T) {
	s, err := Parse(litellmJSON)
	if err != nil {
		t.Fatalf("Parse(embedded): %v", err)
	}
	if got, want := slices.Sorted(maps.Keys(s.models)), slices.Sorted(maps.Keys(table())); !slices.Equal(got, want) {
		t.Fatalf("Parse(embedded) has %d keys, embedded table %d", len(got), len(want))
	}
	for key, tiers := range table() {
		if !tiersEqual(s.models[key], tiers) {
			t.Errorf("%s: Parse(embedded) rates differ from the embedded table", key)
		}
	}
}

// TestNewFromEmbeddedEqualsNew encodes the NewFrom contract: NewFrom of the
// parsed vendored bytes is the same cost model as New for the same Config —
// every model prices identically at every service tier, with aliases and
// overrides (overwrite and suppress) applied the same way.
func TestNewFromEmbeddedEqualsNew(t *testing.T) {
	s, err := Parse(litellmJSON)
	if err != nil {
		t.Fatalf("Parse(embedded): %v", err)
	}
	cfg := Config{
		Aliases: map[string]string{"my-sonnet": "claude-sonnet-4-5", "my-model": "custom-model"},
		Overrides: map[string]*Rates{
			"custom-model": {Base: TierRates{Input: MustParseRat("1e-6"), Output: MustParseRat("2e-6")}},
			"gpt-5.4":      nil,
		},
	}
	want, got := New(cfg), NewFrom(s, cfg)
	models := append(slices.Collect(maps.Keys(table())), "my-sonnet", "my-model", "custom-model", "no-such-model")
	usages := []Usage{
		ClaudeUsage{InputTokens: 300_000, CacheReadInputTokens: 1000, CacheCreationInputTokens: 500, CacheCreation1hInputTokens: 200, OutputTokens: 100},
		OpenAIUsage{InputTokens: 2000, CachedInputTokens: 500, OutputTokens: 100, ServiceTier: TierFlex},
		FireworksUsage{InputTokens: 2000, OutputTokens: 100},
	}
	for _, model := range models {
		for tier := range serviceSuffixes {
			wr, wok := want.RatesFor(model, tier)
			gr, gok := got.RatesFor(model, tier)
			if wok != gok || !ratesEqual(wr, gr) {
				t.Errorf("RatesFor(%q, %s): NewFrom = (%v, %v), New = (%v, %v)", model, tier, gr, gok, wr, wok)
			}
		}
		for _, u := range usages {
			wc, wok := want.Cost(model, u)
			gc, gok := got.Cost(model, u)
			if wc != gc || wok != gok {
				t.Errorf("Cost(%q, %T): NewFrom = (%d, %v), New = (%d, %v)", model, u, gc, gok, wc, wok)
			}
		}
	}
}

// fetched is a minimal LiteLLM pricing map standing in for bytes fetched
// from a different upstream commit than the vendored one.
const fetched = `{
	"sample_spec": {"input_cost_per_token": 0, "output_cost_per_token": 0},
	"fetched-model": {"litellm_provider": "openai", "input_cost_per_token": 1e-6, "output_cost_per_token": 4e-6}
}`

// TestNewFromPricesOnlyTheGivenSnapshot encodes why NewFrom exists: a Table
// built from fetched bytes prices from those bytes alone. A model present
// only in the fetched data prices; a model present only in the embedded
// snapshot does not — the embedded data never leaks in as a fallback — and
// alias targets validate against the given snapshot, not the embedded one.
func TestNewFromPricesOnlyTheGivenSnapshot(t *testing.T) {
	s, err := Parse([]byte(fetched))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	tbl := NewFrom(s, Config{Aliases: map[string]string{"mine": "fetched-model"}})
	// 1000 × 1e-6 + 100 × 4e-6 = 0.0014 USD = 140 nls.
	if got, ok := tbl.Cost("mine", OpenAIUsage{InputTokens: 1000, OutputTokens: 100}); !ok || got != 140 {
		t.Errorf("Cost(fetched-model) = (%d, %v), want (140, true)", got, ok)
	}
	if _, ok := tbl.Cost("claude-sonnet-4-5", ClaudeUsage{InputTokens: 1000, OutputTokens: 100}); ok {
		t.Error("Cost(claude-sonnet-4-5) priced from the embedded snapshot; want ok=false")
	}
	if _, ok := tbl.RatesFor("sample_spec", TierStandard); ok {
		t.Error("RatesFor(sample_spec) ok; unpriceable entries must not parse")
	}
	defer func() {
		if recover() == nil {
			t.Error("NewFrom accepted an alias to an embedded-only key; want panic")
		}
	}()
	NewFrom(s, Config{Aliases: map[string]string{"x": "claude-sonnet-4-5"}})
}

// TestParseRejectsUnusableData encodes that Parse fails loudly on data that
// cannot be a price list: malformed JSON, and well-formed JSON with no
// priceable model (an upstream schema change). An empty Snapshot would
// silently unprice every model.
func TestParseRejectsUnusableData(t *testing.T) {
	for _, data := range []string{
		``,
		`{"truncated":`,
		`[]`,
		`null`,
		`{}`,
		`{"sample_spec": {"input_cost_per_token": 0, "output_cost_per_token": 0}}`,
		`{"renamed": {"input_price": 1e-6, "output_price": 4e-6}}`,
	} {
		if s, err := Parse([]byte(data)); err == nil || s != nil {
			t.Errorf("Parse(%q) = (%v, %v), want (nil, error)", data, s, err)
		} else if !strings.HasPrefix(err.Error(), "llmcost: ") {
			t.Errorf("Parse(%q) error %q lacks the llmcost: prefix", data, err)
		}
	}
}
