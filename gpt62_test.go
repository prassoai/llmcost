package llmcost

import (
	"testing"
)

// GPT-6.2 must bill cached turns and long contexts without borrowing unpublished tiers.
func TestGPT62Pricing(t *testing.T) {
	for _, model := range []struct {
		name        string
		short, long int64
		priority    bool
	}{
		{"gpt-6.2-astra", 100800, 588000, true},
		{"gpt-6.2-astra-law", 100800, 588000, false},
		{"gpt-6.2-sol", 25200, 147000, true},
	} {
		for _, context := range []struct {
			name  string
			usage OpenAIUsage
			want  int64
		}{
			{"short", OpenAIUsage{InputTokens: 120000, CachedInputTokens: 20000, CacheWriteTokens: 20000, OutputTokens: 4000}, model.short},
			{"long", OpenAIUsage{InputTokens: 400000, CachedInputTokens: 100000, CacheWriteTokens: 100000, OutputTokens: 10000}, model.long},
		} {
			t.Run(model.name+"/"+context.name, func(t *testing.T) {
				if got, ok := Cost(model.name, context.usage); !ok || got != context.want {
					t.Fatalf("Cost = %d, %v; want %d, true", got, ok, context.want)
				}
				usage := context.usage
				usage.ServiceTier = TierPriority
				if got, ok := Cost(model.name, usage); ok != model.priority || ok && got != 2*context.want {
					t.Fatalf("priority Cost = %d, %v; want %d, %v", got, ok, 2*context.want, model.priority)
				}
				usage.ServiceTier = TierFlex
				if _, ok := Cost(model.name, usage); ok {
					t.Fatal("unpublished flex pricing must be refused")
				}
			})
		}
	}
}
