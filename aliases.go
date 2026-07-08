package llmcost

// aliases maps every internal model id we bill for — the ids murmur spawns
// agents with, llm-gateway routes, and back's rate tables key on — to its
// LiteLLM pricing key.
//
// Identity entries are intentional, not redundant: presence in this map is
// the module's guarantee that the id prices. TestAliasesResolve asserts every
// entry resolves against the vendored data with positive input, cache-read,
// and output rates (plus cache-creation for the claude ids), and the weekly
// sync workflow relies on that test as its validation gate — if upstream
// renames or drops a key, the sync PR goes red instead of a consumer silently
// billing zero. When a new internal model id enters service, add it here (and
// only here).
var aliases = map[string]string{
	// Anthropic — claude backend.
	"claude-fable-5":            "claude-fable-5",
	"claude-haiku-4-5":          "claude-haiku-4-5",
	"claude-haiku-4-5-20251001": "claude-haiku-4-5-20251001",
	"claude-opus-4-1":           "claude-opus-4-1",
	"claude-opus-4-5":           "claude-opus-4-5",
	"claude-opus-4-6":           "claude-opus-4-6",
	"claude-opus-4-7":           "claude-opus-4-7",
	"claude-opus-4-8":           "claude-opus-4-8",
	"claude-sonnet-4-5":         "claude-sonnet-4-5",
	"claude-sonnet-4-6":         "claude-sonnet-4-6",

	// OpenAI — codex backend.
	"codex-mini":        "codex-mini-latest",
	"gpt-5":             "gpt-5",
	"gpt-5-codex":       "gpt-5-codex",
	"gpt-5.1":           "gpt-5.1",
	"gpt-5.1-codex":     "gpt-5.1-codex",
	"gpt-5.1-codex-max": "gpt-5.1-codex-max",
	"gpt-5.2":           "gpt-5.2",
	"gpt-5.2-codex":     "gpt-5.2-codex",
	"gpt-5.3-codex":     "gpt-5.3-codex",
	"gpt-5.4":           "gpt-5.4",
	"gpt-5.4-mini":      "gpt-5.4-mini",
	"gpt-5.5":           "gpt-5.5",
}
