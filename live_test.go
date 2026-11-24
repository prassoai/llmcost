package llmcost

import (
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
				if got, ok := Cost(model, LiveUsage{Seconds: tc.seconds}); !ok || got != tc.want {
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
	if _, ok := Cost("gpt-5.5", LiveUsage{Seconds: 60}); ok {
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
	Cost("gpt-live-1", LiveUsage{Seconds: -1})
}
