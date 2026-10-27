package llmcost

import "testing"

// TestGPT6FamilyPricing keeps new Codex choices billable across cache usage,
// long contexts, and service tiers at OpenAI's published rates:
// https://developers.openai.com/api/docs/models/gpt-6-sol
// https://developers.openai.com/api/docs/models/gpt-6-luna
// https://developers.openai.com/api/docs/models/gpt-6.1-sol
// https://developers.openai.com/api/docs/models/gpt-6.1-terra
//
// gpt-6.1-sol matches gpt-6-sol everywhere but the cache-read rate, which
// OpenAI halved to $0.10/M — the one number a consumer bumping to this
// snapshot must not silently inherit from the older twin. gpt-6.1-terra is
// gpt-6.1-sol's card halved on every component, so it keeps 6.1's 5%
// cached-input ratio rather than the 10% the rest of GPT-6 charges.
func TestGPT6FamilyPricing(t *testing.T) {
	for _, model := range []struct {
		name        string
		short, long int64
	}{
		{"gpt-6-sol", 25400, 149000},
		{"gpt-6-luna", 1270, 7450},
		{"gpt-6.1-sol", 25200, 147000},
		{"gpt-6.1-terra", 12600, 73500},
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
