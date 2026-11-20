package listprice

import (
	"fmt"
	"regexp"
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
	for _, k := range llmcost.Models(llmcost.ProviderOpenAI) {
		if ChatSearchModel(k) {
			rates[k] = nil
		}
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
	if p, ok := b.price(model, u, b.source(model)); ok {
		return p, true
	}
	for _, d := range degraded(u) {
		if p, ok := b.price(model, d, SourceDegraded); ok {
			return p, true
		}
	}
	return Price{}, false
}

// price is u's all-in price, with the part its web searches take. A call
// that searched is priced twice — all-in and tokens alone — and the
// difference is the search fee; one that did not has no search part.
func (b *book) price(model string, u llmcost.Usage, src Source) (Price, bool) {
	c, ok := b.table.Cost(model, u)
	if !ok {
		return Price{}, false
	}
	p := Price{Cost: c, Price: Markup(c), Source: src}
	if tokens, searched := withoutWebSearch(u); searched {
		t, _ := b.table.Cost(model, tokens) // dropping a component never makes usage that priced unpriceable
		p.WebSearchCost, p.WebSearchPrice = c-t, Markup(c-t)
	}
	return p, true
}

// degraded lists u's re-pricings with the response-reported dimensions
// dropped, least lossy first: Anthropic's geo, its 1-hour cache-write split,
// then both. Other usage has no such dimension. Web searches are not one: the
// request declares the search tool, so a caller refuses an unpriceable one
// before the call, like an unpriced service tier.
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

// withoutWebSearch is u with its web searches dropped — the call's tokens
// alone — and whether it reported any.
func withoutWebSearch(u llmcost.Usage) (llmcost.Usage, bool) {
	switch v := u.(type) {
	case llmcost.ClaudeUsage:
		v.WebSearchRequests = 0
		return v, v != u
	case llmcost.OpenAIUsage:
		v.WebSearchRequests, v.WebSearchPreview = 0, false
		return v, v != u
	}
	return u, false
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

// chatSearchModel matches OpenAI's dedicated Chat Completions search models —
// gpt-4o-search-preview, gpt-4o-mini-search-preview, gpt-5-search-api and
// their dated snapshots.
var chatSearchModel = regexp.MustCompile(`-search-(preview|api)(-\d{4}-\d{2}-\d{2})?$`)

// ChatSearchModel reports whether model is one of OpenAI's dedicated Chat
// Completions search models, which search the web on every request and report
// no search count, so no call to one can be charged its search fee. The book
// has no price for them: [Book] leaves them out and [Of] and [Priceable]
// refuse them. A gateway uses it to name the refusal before a call.
func ChatSearchModel(model string) bool { return chatSearchModel.MatchString(model) }
