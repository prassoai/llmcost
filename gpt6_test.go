package llmcost

import "testing"

// TestGPT6SolLunaPricing keeps new Codex choices billable across cache usage,
// long contexts, and service tiers at OpenAI's published September 22 rates:
// https://developers.openai.com/api/docs/models/gpt-6-sol
// https://developers.openai.com/api/docs/models/gpt-6-luna
func TestGPT6SolLunaPricing(t *testing.T) {
	for _, model := range []struct {
		name        string
		short, long int64
	}{
		{"gpt-6-sol", 25400, 149000},
		{"gpt-6-luna", 1270, 7450},
	} {
		for _, context := range []struct {
			name  string
			usage OpenAIUsage
			want  int64
		}{
			{"short", OpenAIUsage{InputTokens: 120000, CachedInputTokens: 20000, CacheWriteTokens: 20000, OutputTokens: 4000}, model.short},
			{"long", OpenAIUsage{InputTokens: 400000, CachedInputTokens: 100000, CacheWriteTokens: 100000, OutputTokens: 10000}, model.long},
		} {
			for _, tier := range []struct {
				name                   ServiceTier
				numerator, denominator int64
			}{
				{TierStandard, 1, 1},
				{TierFlex, 1, 2},
				{TierPriority, 2, 1},
			} {
				t.Run(model.name+"/"+context.name+"/"+string(tier.name), func(t *testing.T) {
					usage := context.usage
					usage.ServiceTier = tier.name
					want := context.want * tier.numerator / tier.denominator
					if got, ok := Cost(model.name, usage); !ok || got != want {
						t.Fatalf("Cost = %d, %v; want %d, true", got, ok, want)
					}
				})
			}
		}
	}
}
