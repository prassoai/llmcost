package llmcost

import (
	_ "embed"
	"fmt"
	"regexp"
	"slices"
)

//go:embed VENDORED_FROM
var vendoredFrom string

// litellmCommit is VENDORED_FROM's pin. A pin file without one is impossible
// past CI and panics at init.
var litellmCommit = func() string {
	m := regexp.MustCompile(`(?m)^commit: ([0-9a-f]{40})$`).FindStringSubmatch(vendoredFrom)
	if m == nil {
		panic(fmt.Sprintf("llmcost: VENDORED_FROM pins no commit:\n%s", vendoredFrom))
	}
	return m[1]
}()

// LiteLLMCommit returns the BerriAI/litellm commit the embedded pricing data
// was vendored from. With this module's version it identifies exactly which
// rates a build prices with.
func LiteLLMCommit() string { return litellmCommit }

// Models returns, sorted, every snapshot key that p bills: the keys
// [RatesFor] prices at [TierStandard] whose upstream litellm_provider p owns
// — the same ownership [ModelSelector.Key] enforces. Consumer overrides are
// not snapshot keys and are not listed.
func Models(p Provider) []string {
	return slices.Sorted(func(yield func(string) bool) {
		for key, tiers := range table() {
			if p.owns(tiers[TierStandard].litellmProvider) && !yield(key) {
				return
			}
		}
	})
}
