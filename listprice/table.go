package listprice

import (
	"fmt"
	"slices"

	"github.com/prassoai/llmcost"
)

// model is one published model: the id customers request it under, and the
// llmcost key that prices it. An id that differs from its key is an alias.
type model struct {
	id, key  string
	provider llmcost.Provider
}

// override restates a key's rates, for a model LiteLLM prices wrongly or not
// at all. It prices the standard service tier only.
type override struct {
	provider llmcost.Provider
	rates    llmcost.Rates
}

// book is a price book: the llmcost table its published aliases and overrides
// configure, and the curated set it publishes.
type book struct {
	table     *llmcost.Table
	aliases   map[string]string
	published []model
	overrides map[string]override
}

// newBook panics on a malformed curation — a duplicate id, a provider the
// book does not list, or anything [llmcost.New] rejects — because a price
// book that cannot be built must fail at init, not price. A published model
// whose key its provider no longer bills is suppressed rather than rejected:
// the vendored data removed the model, so it drops out of the book like any
// other. Suppressing its id, not just leaving out its alias, keeps an id that
// is itself a snapshot key from pricing at that key's rates instead.
func newBook(published []model, overrides map[string]override) *book {
	rates := make(map[string]*llmcost.Rates, len(overrides))
	for key, o := range overrides {
		rates[key] = &o.rates
	}
	aliases := map[string]string{}
	seen := map[string]bool{}
	for _, m := range published {
		switch {
		case seen[m.id]:
			panic(fmt.Sprintf("listprice: model %q is published twice", m.id))
		case !slices.Contains(pricers, m.provider):
			panic(fmt.Sprintf("listprice: model %q has unlisted provider %q", m.id, m.provider))
		case !billed(m, overrides):
			rates[m.id] = nil
		case m.id != m.key:
			aliases[m.id] = m.key
		}
		seen[m.id] = true
	}
	return &book{
		table:     llmcost.New(llmcost.Config{Aliases: aliases, Overrides: rates}),
		aliases:   aliases,
		published: published,
		overrides: overrides,
	}
}

// billed reports whether m's key prices: its provider bills it, or an
// override restates it.
func billed(m model, overrides map[string]override) bool {
	_, ok := overrides[m.key]
	return ok || slices.Contains(llmcost.Models(m.provider), m.key)
}

func (b *book) source(model string) Source {
	key, ok := b.aliases[model]
	if !ok {
		key = model
	}
	if _, ok := b.overrides[key]; ok {
		return SourceOverride
	}
	return SourceLLMCost
}

func (b *book) of(model string, u llmcost.Usage) (Price, bool) {
	for i, d := range append([]llmcost.Usage{u}, degraded(u)...) {
		c, ok := b.table.Cost(model, d)
		if !ok {
			continue
		}
		// Dropping a component never makes usage that priced unpriceable.
		tokens, _ := b.table.Cost(model, withoutWebSearch(d))
		src := SourceDegraded
		if i == 0 {
			src = b.source(model)
		}
		return Price{Cost: c, Price: Markup(c), WebSearchCost: c - tokens, WebSearchPrice: Markup(c - tokens), Source: src}, true
	}
	return Price{}, false
}

// degraded lists u's re-pricings with the response-reported dimensions
// dropped, least lossy first: Anthropic's geo, its 1-hour cache-write split,
// both, then each of those with the web searches dropped as well. OpenAI
// usage drops only its web searches; Fireworks usage has no such dimension.
func degraded(u llmcost.Usage) []llmcost.Usage {
	switch u := u.(type) {
	case llmcost.ClaudeUsage:
		var out []llmcost.Usage
		for drop := 1; drop < 8; drop++ {
			d := u
			if drop&1 != 0 {
				d.InferenceGeo = ""
			}
			if drop&2 != 0 {
				d.CacheCreation1hInputTokens = 0
			}
			if drop&4 != 0 {
				d.WebSearchRequests = 0
			}
			out = append(out, d)
		}
		return out
	case llmcost.OpenAIUsage:
		return []llmcost.Usage{withoutWebSearch(u)}
	}
	return nil
}

// withoutWebSearch is u with its web searches dropped: the call's tokens alone.
func withoutWebSearch(u llmcost.Usage) llmcost.Usage {
	switch u := u.(type) {
	case llmcost.ClaudeUsage:
		u.WebSearchRequests = 0
		return u
	case llmcost.OpenAIUsage:
		u.WebSearchRequests = 0
		return u
	}
	return u
}

func (b *book) priceable(model string, tier llmcost.ServiceTier, speed string) bool {
	r, ok := b.table.RatesFor(model, tier)
	switch speed {
	case "", "standard":
		return ok
	case "fast":
		return ok && r.Fast != nil
	}
	return false
}
