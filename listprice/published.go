package listprice

import "github.com/prassoai/llmcost"

// published is the curated set of models sold by name, in the order the
// price page lists them. Every entry must price at the standard tier
// (TestPublishedPriceable), so a data sync that drops one fails CI instead
// of turning a sold model into a refusal.
var published = []model{
	{"claude-opus-5-5", "claude-opus-5-5", llmcost.ProviderAnthropic},
	{"claude-sonnet-5-5", "claude-sonnet-5-5", llmcost.ProviderAnthropic},
	{"claude-haiku-5-5", "claude-haiku-5-5", llmcost.ProviderAnthropic},
	{"claude-fable-5-1", "claude-fable-5-1", llmcost.ProviderAnthropic},
	{"claude-fable-5", "claude-fable-5", llmcost.ProviderAnthropic},
	{"claude-opus-5", "claude-opus-5", llmcost.ProviderAnthropic},
	{"claude-opus-4-8", "claude-opus-4-8", llmcost.ProviderAnthropic},
	{"claude-opus-4-6", "claude-opus-4-6", llmcost.ProviderAnthropic},
	{"claude-sonnet-4-6", "claude-sonnet-4-6", llmcost.ProviderAnthropic},
	{"claude-sonnet-4-5", "claude-sonnet-4-5", llmcost.ProviderAnthropic},
	{"gpt-6.1-sol", "gpt-6.1-sol", llmcost.ProviderOpenAI},
	{"gpt-6-astra", "gpt-6-astra", llmcost.ProviderOpenAI},
	{"gpt-6-sol", "gpt-6-sol", llmcost.ProviderOpenAI},
	{"gpt-6-luna", "gpt-6-luna", llmcost.ProviderOpenAI},
	{"gpt-5.6-sol", "gpt-5.6-sol", llmcost.ProviderOpenAI},
	{"gpt-5.6-terra", "gpt-5.6-terra", llmcost.ProviderOpenAI},
	{"gpt-5.6-luna", "gpt-5.6-luna", llmcost.ProviderOpenAI},
	{"gpt-5.5", "gpt-5.5", llmcost.ProviderOpenAI},
	{"gpt-5.4", "gpt-5.4", llmcost.ProviderOpenAI},
	{"gpt-5.4-mini", "gpt-5.4-mini", llmcost.ProviderOpenAI},
	{"gpt-5.3-codex", "gpt-5.3-codex", llmcost.ProviderOpenAI},
	{"gpt-5.2", "gpt-5.2", llmcost.ProviderOpenAI},
}

// overrides restate rates, keyed by llmcost key. Every entry makes a price
// that no data sync can correct, so each must say why the snapshot is wrong
// and carry a test that fails once it no longer is. There are none: every
// published model prices from LiteLLM.
var overrides = map[string]override{}

var std = newBook(published, overrides)
