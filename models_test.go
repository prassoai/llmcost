package llmcost

import (
	"regexp"
	"slices"
	"testing"
)

// TestModels pins the enumeration contract consumers build price books from:
// sorted, provider-owned, every key priceable, and no other provider's keys —
// a Bedrock or Azure twin listed under Anthropic or OpenAI would publish a
// rate the direct API never bills.
func TestModels(t *testing.T) {
	for p, want := range map[Provider]string{ProviderAnthropic: "claude-opus-5", ProviderOpenAI: "gpt-5.5"} {
		keys := Models(p)
		if !slices.IsSorted(keys) {
			t.Errorf("Models(%s) is not sorted", p)
		}
		if !slices.Contains(keys, want) {
			t.Errorf("Models(%s) lacks %q", p, want)
		}
		for _, k := range keys {
			r, ok := RatesFor(k, TierStandard)
			if !ok || !p.owns(r.litellmProvider) {
				t.Errorf("Models(%s) lists %q, which %s does not price (ok=%v, provider=%q)", p, k, p, ok, r.litellmProvider)
			}
		}
	}
	if slices.Contains(Models(ProviderOpenAI), "azure/gpt-5.4") {
		t.Error("Models(openai) lists an Azure key")
	}
}

// TestLiteLLMCommit requires the pin to be a full commit SHA: it is half of a
// price book's identity, so an abbreviated or empty pin would let two
// different snapshots claim the same version.
func TestLiteLLMCommit(t *testing.T) {
	if c := LiteLLMCommit(); !regexp.MustCompile(`^[0-9a-f]{40}$`).MatchString(c) {
		t.Errorf("LiteLLMCommit() = %q, want a 40-hex SHA", c)
	}
}
