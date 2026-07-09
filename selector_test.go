package llmcost

import (
	"regexp"
	"slices"
	"strings"
	"testing"
)

// TestSelectorKeyGrammar encodes the key-construction grammar per provider,
// pinned against real vendored keys: direct ids verbatim, azure/azure_ai
// with an optional region segment (matched case-insensitively), vertex_ai
// prefix with the Vertex id ("@" version suffixes included) verbatim, and
// Bedrock AWS ids — cross-region inference-profile prefixes included —
// verbatim. Every constructed key must also resolve via RatesFor: Key never
// returns an unverified key.
func TestSelectorKeyGrammar(t *testing.T) {
	for _, tc := range []struct {
		sel  ModelSelector
		want string
	}{
		{ModelSelector{ProviderOpenAI, "gpt-5.4", ""}, "gpt-5.4"},
		{ModelSelector{ProviderAnthropic, "claude-sonnet-4-5", ""}, "claude-sonnet-4-5"},
		{ModelSelector{ProviderAzure, "gpt-5.4", ""}, "azure/gpt-5.4"},
		{ModelSelector{ProviderAzure, "gpt-5.4", "us"}, "azure/us/gpt-5.4"},
		{ModelSelector{ProviderAzure, "gpt-5.5", "EU"}, "azure/eu/gpt-5.5"},
		{ModelSelector{ProviderAzureAI, "claude-opus-4-6", ""}, "azure_ai/claude-opus-4-6"},
		{ModelSelector{ProviderBedrock, "anthropic.claude-sonnet-4-5-20250929-v1:0", ""}, "anthropic.claude-sonnet-4-5-20250929-v1:0"},
		{ModelSelector{ProviderBedrock, "us.anthropic.claude-sonnet-4-5-20250929-v1:0", ""}, "us.anthropic.claude-sonnet-4-5-20250929-v1:0"},
		{ModelSelector{ProviderBedrock, "openai.gpt-oss-120b-1:0", ""}, "openai.gpt-oss-120b-1:0"},
		{ModelSelector{ProviderVertexAI, "claude-sonnet-4-5", ""}, "vertex_ai/claude-sonnet-4-5"},
		{ModelSelector{ProviderVertexAI, "claude-sonnet-4-5@20250929", ""}, "vertex_ai/claude-sonnet-4-5@20250929"},
	} {
		got, ok := tc.sel.Key()
		if !ok || got != tc.want {
			t.Errorf("%+v: Key = %q, %v; want %q, true", tc.sel, got, ok, tc.want)
			continue
		}
		if _, ok := RatesFor(got); !ok {
			t.Errorf("%+v: key %q does not resolve via RatesFor", tc.sel, got)
		}
	}
}

// TestSelectorNeverGuesses encodes the no-fuzzing contract that separates
// this from LiteLLM's alias layer: a selector that cannot be priced FAILS —
// no region fallback (azure/us/ bills ~10% above azure/, so falling back
// underbills), no provider inference, no key surgery, and no region on
// providers whose regional pricing lives elsewhere (OpenAI residency and
// Anthropic geo are usage-side; Bedrock ids carry their own prefix).
func TestSelectorNeverGuesses(t *testing.T) {
	for name, sel := range map[string]ModelSelector{
		"unknown region must not fall back to the global key": {ProviderAzure, "gpt-5.4", "japaneast"},
		"unknown model": {ProviderAzure, "no-such-model", ""},
		"region on openai (residency is usage-side)":     {ProviderOpenAI, "gpt-5.4", "eu"},
		"region on anthropic (geo is usage-side)":        {ProviderAnthropic, "claude-sonnet-4-5", "us"},
		"region on bedrock (ids carry their own prefix)": {ProviderBedrock, "anthropic.claude-sonnet-4-5-20250929-v1:0", "us"},
		"unknown provider":                               {Provider("gemini"), "gemini-2.5-pro", ""},
		"empty model":                                    {ProviderOpenAI, "", ""},
		"direct id through the wrong provider":           {ProviderVertexAI, "gpt-5.4", ""},
		// The provider-ownership check: keys that CONSTRUCT and RESOLVE, but
		// to an entry billed by a different provider, must fail — never
		// silently bill that provider's rates.
		"prefixed key smuggled into Model":    {ProviderOpenAI, "azure/gpt-5.4", ""},
		"direct openai key through bedrock":   {ProviderBedrock, "gpt-5.4", ""},
		"bedrock id through anthropic":        {ProviderAnthropic, "anthropic.claude-sonnet-4-5-20250929-v1:0", ""},
		"cross-vendor: claude through openai": {ProviderOpenAI, "claude-sonnet-4-5", ""},
		"cross-vendor: gpt through anthropic": {ProviderAnthropic, "gpt-5.4", ""},
	} {
		if key, ok := sel.Key(); ok {
			t.Errorf("%s: %+v resolved to %q; want ok=false", name, sel, key)
		}
	}
}

// TestSelectorCoversVendoredKeys is the sync gate for the key grammar: every
// PRICEABLE Anthropic/OpenAI-vendor key in the vendored data — direct,
// Bedrock, Vertex, Azure, Azure AI — must be reachable by some selector.
// Keys are classified by their entry's own litellm_provider (never by key
// shape — Bedrock ids include vendor-prefixless and "us-gov." geo forms that
// shape-guessing misses). If upstream restructures its key scheme (new
// nesting, a new prefix style), this fails the sync PR instead of silently
// stranding those models behind an unconstructible key.
func TestSelectorCoversVendoredKeys(t *testing.T) {
	vendorModel := regexp.MustCompile(`claude|gpt`)
	covered := 0
	for key, r := range table() {
		if !vendorModel.MatchString(key) {
			continue
		}
		var sels []ModelSelector
		switch lp := r.litellmProvider; {
		case ProviderOpenAI.owns(lp):
			sels = []ModelSelector{{Provider: ProviderOpenAI, Model: key}}
		case ProviderAnthropic.owns(lp):
			sels = []ModelSelector{{Provider: ProviderAnthropic, Model: key}}
		case ProviderBedrock.owns(lp):
			sels = []ModelSelector{{Provider: ProviderBedrock, Model: key}}
		case ProviderAzure.owns(lp), ProviderAzureAI.owns(lp), ProviderVertexAI.owns(lp):
			parts := strings.SplitN(key, "/", 3)
			p := Provider(parts[0])
			sels = []ModelSelector{{Provider: p, Model: strings.TrimPrefix(key, parts[0]+"/")}}
			if len(parts) == 3 {
				sels = append(sels, ModelSelector{Provider: p, Model: parts[2], Region: parts[1]})
			}
		default:
			continue // other hosts of these vendors' models (openrouter, gateways, …) are out of the selector's scope
		}
		covered++
		if !slices.ContainsFunc(sels, func(s ModelSelector) bool { k, ok := s.Key(); return ok && k == key }) {
			t.Errorf("%s (litellm_provider %s): not reachable by any selector — key grammar drifted from upstream", key, r.litellmProvider)
		}
	}
	// The gate must actually gate: hundreds of Anthropic/OpenAI keys exist
	// across the five providers. A collapse here means the filter regressed
	// and the test is no longer checking anything.
	if covered < 100 {
		t.Errorf("coverage gate checked only %d keys; the key filter has regressed", covered)
	}
}
