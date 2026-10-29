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
// book that cannot be built must fail at init, not price.
func newBook(published []model, overrides map[string]override) *book {
	aliases := map[string]string{}
	seen := map[string]bool{}
	for _, m := range published {
		switch {
		case seen[m.id]:
			panic(fmt.Sprintf("listprice: model %q is published twice", m.id))
		case !slices.Contains(pricers, m.provider):
			panic(fmt.Sprintf("listprice: model %q has unlisted provider %q", m.id, m.provider))
		case m.id != m.key:
			aliases[m.id] = m.key
		}
		seen[m.id] = true
	}
	rates := make(map[string]*llmcost.Rates, len(overrides))
	for key, o := range overrides {
		rates[key] = &o.rates
	}
	return &book{
		table:     llmcost.New(llmcost.Config{Aliases: aliases, Overrides: rates}),
		aliases:   aliases,
		published: published,
		overrides: overrides,
	}
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
	if c, ok := b.table.Cost(model, u); ok {
		return Price{Cost: c, Price: Markup(c), Source: b.source(model)}, true
	}
	for _, d := range degraded(u) {
		if c, ok := b.table.Cost(model, d); ok {
			return Price{Cost: c, Price: Markup(c), Source: SourceDegraded}, true
		}
	}
	return Price{}, false
}

// degraded lists u's re-pricings with the response-reported dimensions
// dropped, least lossy first: Anthropic's geo, its 1-hour cache-write split,
// then both. Other usage has no such dimension.
func degraded(u llmcost.Usage) []llmcost.Usage {
	c, ok := u.(llmcost.ClaudeUsage)
	if !ok {
		return nil
	}
	noGeo, no1h := c, c
	noGeo.InferenceGeo, no1h.CacheCreation1hInputTokens = "", 0
	neither := noGeo
	neither.CacheCreation1hInputTokens = 0
	return []llmcost.Usage{noGeo, no1h, neither}
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
