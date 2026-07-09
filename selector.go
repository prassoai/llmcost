package llmcost

import "strings"

// Provider is the service that actually served a request — the biller. The
// same vendor model bills differently per provider (and per Azure data
// zone: azure/us/gpt-5.4 carries a ~10% premium over azure/gpt-5.4), so the
// provider is part of model identity here. The constants cover Anthropic
// and OpenAI models everywhere LiteLLM prices them: direct, Bedrock,
// Vertex, and Azure.
type Provider string

const (
	// ProviderOpenAI is api.openai.com: keys are the model id verbatim
	// ("gpt-5.4"). Regional data-residency hosts bill via
	// [OpenAIUsage.DataResidency], not the key — Region must be empty.
	ProviderOpenAI Provider = "openai"
	// ProviderAnthropic is api.anthropic.com: keys are the model id verbatim
	// ("claude-sonnet-4-5"). Regional premiums are asserted by the response
	// ([ClaudeUsage.InferenceGeo]), not the key — Region must be empty.
	ProviderAnthropic Provider = "anthropic"
	// ProviderAzure is Azure OpenAI Service: keys are azure/{model}, or
	// azure/{region}/{model} for the data-zone deployments ("us", "eu")
	// whose premium is baked into the region key's rates.
	ProviderAzure Provider = "azure"
	// ProviderAzureAI is Azure AI Foundry — Anthropic (and other non-OpenAI)
	// models on Azure: keys are azure_ai/{model}, or
	// azure_ai/{region}/{model} where upstream prices a region.
	ProviderAzureAI Provider = "azure_ai"
	// ProviderBedrock is AWS Bedrock: the AWS model id IS the key, verbatim —
	// including any cross-region inference-profile prefix
	// ("us.anthropic.claude-sonnet-4-5-20250929-v1:0"); Region must be empty.
	ProviderBedrock Provider = "bedrock"
	// ProviderVertexAI is Google Vertex AI: keys are vertex_ai/{model} with
	// the caller's Vertex model id verbatim — "@" version suffixes and
	// publisher paths included ("claude-sonnet-4-5@20250929").
	ProviderVertexAI Provider = "vertex_ai"
)

// ModelSelector identifies a model by how it was served, in the caller's own
// vocabulary: the provider that billed the request, the provider's native
// model id, and — only for providers that price by region key — the region.
//
// [ModelSelector.Key] constructs the LiteLLM pricing key DETERMINISTICALLY
// and verifies it resolves; it never guesses and never falls back. This is
// the module's whole alias policy: the key grammar (which is coupled to the
// vendored data and maintained with it) lives here, while WHICH selectors
// exist stays with the caller — who should test that every selector they
// bill resolves, so a vendored-data sync that drops a key fails their build
// instead of production.
type ModelSelector struct {
	Provider Provider
	// Model is the model id exactly as the caller's API client names it:
	// "gpt-5.4", "claude-sonnet-4-5",
	// "anthropic.claude-sonnet-4-5-20250929-v1:0", "claude-sonnet-4-5@20250929".
	Model string
	// Region is the region segment of region-keyed providers (Azure data
	// zones "us"/"eu"; matched case-insensitively). It must be empty for
	// providers that do not price by region key: OpenAI residency is
	// [OpenAIUsage.DataResidency], Anthropic geo is
	// [ClaudeUsage.InferenceGeo], and Bedrock ids carry their own geo prefix.
	Region string
}

// Key returns the verified LiteLLM pricing key for the selector. ok is false
// when the selector is malformed — empty model, Region on a provider that
// does not price by region key, unknown provider — when the constructed key
// does not resolve to a priceable model, or when the resolved entry belongs
// to a DIFFERENT provider (by the entry's own litellm_provider). The
// ownership check closes the verbatim pass-through hole: a prefixed key
// smuggled into Model ({ProviderOpenAI, "azure/gpt-5.4"}) or a cross-vendor
// id ({ProviderBedrock, "gpt-5.4"}) constructs a key that resolves — but to
// an entry another provider bills, and fails here rather than silently
// billing that provider's rates. There is no fallback in any direction: an
// Azure data-zone deployment whose region key is missing upstream fails
// rather than billing the ~10%-cheaper global key.
func (s ModelSelector) Key() (string, bool) {
	key, ok := s.key()
	if !ok {
		return "", false
	}
	r, ok := table()[key]
	if !ok || !s.Provider.owns(r.litellmProvider) {
		return "", false
	}
	return key, true
}

// owns reports whether a table entry with the given upstream
// litellm_provider is billed by this provider. LiteLLM subdivides some
// providers, so each case lists its variants; an empty value (entry without
// litellm_provider) is unowned and never matches.
func (p Provider) owns(litellmProvider string) bool {
	switch p {
	case ProviderOpenAI:
		return litellmProvider == "openai" || litellmProvider == "text-completion-openai"
	case ProviderAnthropic:
		return litellmProvider == "anthropic"
	case ProviderAzure:
		return litellmProvider == "azure" || litellmProvider == "azure_text"
	case ProviderAzureAI:
		return litellmProvider == "azure_ai"
	case ProviderBedrock:
		return litellmProvider == "bedrock" || litellmProvider == "bedrock_converse"
	case ProviderVertexAI:
		// vertex_ai, vertex_ai-anthropic_models, vertex_ai-openai_models, …
		return strings.HasPrefix(litellmProvider, "vertex_ai")
	}
	return false
}

// key is the pure grammar: provider-prefixed deterministic construction,
// no table lookup.
func (s ModelSelector) key() (string, bool) {
	if s.Model == "" {
		return "", false
	}
	region := strings.ToLower(s.Region)
	switch s.Provider {
	case ProviderOpenAI, ProviderAnthropic, ProviderBedrock:
		if region != "" {
			return "", false
		}
		return s.Model, true
	case ProviderAzure, ProviderAzureAI, ProviderVertexAI:
		if region == "" {
			return string(s.Provider) + "/" + s.Model, true
		}
		return string(s.Provider) + "/" + region + "/" + s.Model, true
	}
	return "", false
}
