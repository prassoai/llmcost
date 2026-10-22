package llmcost

import "testing"

// TestClaude55Pricing pins Anthropic's published prices, including both cache
// write TTLs and US residency. Haiku's 100k boundary counts every prompt token;
// output tokens do not push a request into the higher tier.
// Source: https://platform.claude.com/docs/en/about-claude/pricing (2026-10-22).
func TestClaude55Pricing(t *testing.T) {
	for _, tt := range []struct {
		name, model string
		usage       ClaudeUsage
		want        Nls
	}{
		{"sonnet components", "claude-sonnet-5-5", ClaudeUsage{InputTokens: 10000, CacheReadInputTokens: 10000, CacheCreationInputTokens: 20000, CacheCreation1hInputTokens: 10000, OutputTokens: 10000}, 18600},
		{"sonnet US", "claude-sonnet-5-5", ClaudeUsage{InputTokens: 10000, CacheReadInputTokens: 10000, CacheCreationInputTokens: 20000, CacheCreation1hInputTokens: 10000, OutputTokens: 10000, InferenceGeo: "us"}, 20460},
		{"haiku components", "claude-haiku-5-5", ClaudeUsage{InputTokens: 10000, CacheReadInputTokens: 10000, CacheCreationInputTokens: 20000, CacheCreation1hInputTokens: 10000, OutputTokens: 10000}, 935},
		{"haiku boundary", "claude-haiku-5-5", ClaudeUsage{InputTokens: 70000, CacheReadInputTokens: 10000, CacheCreationInputTokens: 20000, CacheCreation1hInputTokens: 10000, OutputTokens: 10000}, 1535},
		{"haiku cached prompt above boundary", "claude-haiku-5-5", ClaudeUsage{InputTokens: 70000, CacheReadInputTokens: 20000, CacheCreationInputTokens: 20000, CacheCreation1hInputTokens: 10000, OutputTokens: 10000}, 7725},
		{"haiku US above boundary", "claude-haiku-5-5", ClaudeUsage{InputTokens: 70000, CacheReadInputTokens: 20000, CacheCreationInputTokens: 20000, CacheCreation1hInputTokens: 10000, OutputTokens: 10000, InferenceGeo: "us"}, 8498},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got, ok := Cost(tt.model, tt.usage); !ok || got != tt.want {
				t.Fatalf("Cost = %d, %v; want %d, true", got, ok, tt.want)
			}
		})
	}
}
