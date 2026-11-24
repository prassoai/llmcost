package llmcost

import (
	"math/big"
	"slices"
	"testing"
)

// TestGPTLivePricing pins GPT-Live's time billing: OpenAI bills a voice
// session per second at $0.05 a minute, so a minute costs 5,000 nls, a single
// second rounds up to a whole nls, and token usage — which GPT-Live never
// reports — has no rate to bill at.
func TestGPTLivePricing(t *testing.T) {
	for _, model := range []string{"gpt-live-1", "gpt-live-1.1"} {
		t.Run(model, func(t *testing.T) {
			for _, tc := range []struct {
				seconds int64
				want    Nls
			}{{60, 5000}, {1, 84}, {0, 0}, {3600, 300000}} {
				if got, ok := Cost(model, LiveUsage{ActiveSeconds: tc.seconds}); !ok || got != tc.want {
					t.Errorf("Cost(%d s) = %d, %v; want %d, true", tc.seconds, got, ok, tc.want)
				}
			}
			if _, ok := Cost(model, OpenAIUsage{InputTokens: 1}); ok {
				t.Error("token usage priced on a time-billed model")
			}
		})
	}
}

// TestLiveUsageRefusedByTokenModels proves time usage never bills a model
// without a per-second rate, and that per-second rates are read only for
// GPT-Live: other time-billed LiteLLM entries (transcription, provisioned
// throughput) stay out of the table rather than pricing usage they do not
// model.
func TestLiveUsageRefusedByTokenModels(t *testing.T) {
	if _, ok := Cost("gpt-5.5", LiveUsage{ActiveSeconds: 60}); ok {
		t.Error("time usage priced on a token-billed model")
	}
	for _, key := range []string{"gpt-live-transcribe", "whisper-1", "gpt-realtime-translate"} {
		if _, ok := RatesFor(key, TierStandard); ok {
			t.Errorf("%s: a non-GPT-Live time-billed entry is in the table", key)
		}
	}
	if !slices.Contains(Models(ProviderOpenAI), "gpt-live-1") {
		t.Error("gpt-live-1 is not an OpenAI-billed key")
	}
}

// TestLiveUsageNegativePanics: a negative second count is impossible in a
// real event and is a caller bug, like a negative token count.
func TestLiveUsageNegativePanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("negative seconds did not panic")
		}
	}()
	Cost("gpt-live-1", LiveUsage{ActiveSeconds: -1})
}

// TestLiveRatesValidated: a per-second rate never makes malformed rates
// priceable. A non-positive per-second rate is unpriceable, and a key that
// supplies token rates alongside one must supply priceable token rates —
// otherwise a negative token rate would bill a negative cost. Overrides are
// held to the same rule at construction.
func TestLiveRatesValidated(t *testing.T) {
	sec := MustParseRat("0.01")
	for name, r := range map[string]Rates{
		"negative per second": {PerSecond: MustParseRat("-0.01")},
		"zero per second":     {PerSecond: new(big.Rat)},
		"bad token rates":     {Base: TierRates{Input: MustParseRat("-1"), Output: MustParseRat("1")}, PerSecond: sec},
	} {
		if r.priceable() {
			t.Errorf("%s: priceable", name)
		}
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s: override accepted", name)
				}
			}()
			New(Config{Overrides: map[string]*Rates{"x": &r}})
		}()
	}
	if !(Rates{PerSecond: sec}).priceable() {
		t.Error("a per-second-only key is not priceable")
	}
	parsed, ok := parseModel([]byte(`{"litellm_provider":"openai","supported_endpoints":["/v1/live/sessions"],"input_cost_per_second":-0.01}`))
	if ok {
		t.Errorf("a negative per-second entry parsed: %+v", parsed)
	}
}

// TestLiveStandbySeconds: standby seconds bill at the standby rate, beside
// active seconds at the active rate. LiteLLM publishes no standby rate, so the
// snapshot refuses standby seconds rather than billing them as active, and a
// standby rate without an active one, or a non-positive one, is unpriceable.
func TestLiveStandbySeconds(t *testing.T) {
	if _, ok := Cost("gpt-live-1", LiveUsage{ActiveSeconds: 60, StandbySeconds: 30}); ok {
		t.Error("standby seconds priced without a standby rate")
	}
	table := New(Config{Overrides: map[string]*Rates{"live": {PerSecond: big.NewRat(5, 6000), PerStandbySecond: big.NewRat(2, 6000)}}})
	if got, ok := table.Cost("live", LiveUsage{ActiveSeconds: 60, StandbySeconds: 30}); !ok || got != 6000 {
		t.Errorf("Cost(60 active, 30 standby) = %d, %v; want 6000 ($0.06), true", got, ok)
	}
	for name, r := range map[string]Rates{
		"standby without active": {PerStandbySecond: big.NewRat(2, 6000)},
		"zero standby":           {PerSecond: big.NewRat(5, 6000), PerStandbySecond: new(big.Rat)},
	} {
		if r.priceable() {
			t.Errorf("%s: priceable", name)
		}
	}
}
