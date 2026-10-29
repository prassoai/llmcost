package listprice

import (
	"fmt"
	"math/big"
	"slices"

	"github.com/prassoai/llmcost"
)

// Component is one billable token class.
type Component uint8

const (
	ComponentInput Component = iota + 1
	ComponentOutput
	ComponentCacheRead
	// ComponentCacheWrite5m is the default 5-minute-TTL cache write, and
	// OpenAI's only cache write.
	ComponentCacheWrite5m
	ComponentCacheWrite1h
)

func (c Component) String() string {
	switch c {
	case ComponentInput:
		return "input"
	case ComponentOutput:
		return "output"
	case ComponentCacheRead:
		return "cache_read"
	case ComponentCacheWrite5m:
		return "cache_write_5m"
	case ComponentCacheWrite1h:
		return "cache_write_1h"
	}
	return fmt.Sprintf("Component(%d)", c)
}

// Rate is one list price: what murmur charges per million tokens of one
// component, for one model at one service tier, speed and context tier. A
// model has one price whichever provider serves it: the price is set by
// PricedBy's cost, and every provider's own cost is a [ProviderCost].
type Rate struct {
	// Model is the id the model is requested under: a published model's id,
	// otherwise its llmcost key.
	Model string
	// Published marks the curated set of models sold by name.
	Published bool
	// PricedBy is the provider whose listed cost sets the price.
	PricedBy llmcost.Provider
	Tier     llmcost.ServiceTier
	// Fast is Anthropic's fast mode: input and output carry the model's fast
	// multiplier, cache components bill unscaled.
	Fast bool
	// AbovePromptTokens is the context tier's floor: the row prices requests
	// whose total prompt exceeds it. 0 is the base tier.
	AbovePromptTokens int64
	Component         Component
	// Cost is PricedBy's cost and Price is Cost with [MarkupBasisPoints]
	// applied, unrounded; both are exact USD per million tokens.
	Cost, Price *big.Rat
}

// ProviderCost is what one provider charges per million tokens of one
// component, for one book model at one service tier, speed and context tier.
type ProviderCost struct {
	Provider llmcost.Provider
	// Model is the book model, as [Rate.Model] names it; Key is the
	// provider's llmcost key for it.
	Model, Key string
	// Source is [SourceLLMCost] or [SourceOverride].
	Source            Source
	Tier              llmcost.ServiceTier
	Fast              bool
	AbovePromptTokens int64
	Component         Component
	// Cost is exact USD per million tokens.
	Cost *big.Rat
}

// Book enumerates the list prices, ordered by pricing provider, then the
// published models in their curated order, then every other priceable key
// sorted. Each model contributes a row per service tier, speed, context tier
// and priced component. It covers everything [Of] prices at a base inference
// geo, so a model it lists no rate for has no list price.
func Book() []Rate { return std.rates() }

// Costs enumerates what every provider llmcost prices charges for the book's
// models, in [Book]'s model order and, per model, its pricing provider first.
// A provider appears for a model when its global (unregioned) key for the
// model resolves: [llmcost.ModelSelector] maps the pricing key's vendor name
// onto each cloud's own naming.
func Costs() []ProviderCost { return std.costs() }

// pricers are the providers whose models the book prices, in order.
var pricers = []llmcost.Provider{llmcost.ProviderAnthropic, llmcost.ProviderOpenAI}

// costers are the providers [Costs] reports, in order. Regional keys (Azure
// data zones, Bedrock geo profiles) are left out: each is its global key plus
// a premium.
var costers = []llmcost.Provider{
	llmcost.ProviderAnthropic, llmcost.ProviderOpenAI, llmcost.ProviderBedrock, llmcost.ProviderVertexAI,
	llmcost.ProviderAzure, llmcost.ProviderAzureAI, llmcost.ProviderFireworks,
}

var (
	perMillion = big.NewRat(1_000_000, 1)
	markup     = big.NewRat(10_000+MarkupBasisPoints, 10_000)
)

// entry is one book model: the id it is requested under, the key its pricing
// provider bills it by, and whether it is published.
type entry struct {
	model
	published bool
}

// entries lists the book's models in [Book] order.
func (b *book) entries() []entry {
	var out []entry
	for _, p := range pricers {
		covered := map[string]bool{}
		for _, m := range b.published {
			if m.provider == p {
				out = append(out, entry{m, true})
				covered[m.key] = true
			}
		}
		keys := llmcost.Models(p)
		for k, o := range b.overrides {
			if o.provider == p && !slices.Contains(keys, k) {
				keys = append(keys, k)
			}
		}
		slices.Sort(keys)
		for _, k := range keys {
			if !covered[k] {
				out = append(out, entry{model{k, k, p}, false})
			}
		}
	}
	return out
}

func (b *book) rates() []Rate {
	var out []Rate
	for _, e := range b.entries() {
		for _, c := range b.priced(e.provider, e.id, e.key) {
			out = append(out, Rate{
				Model: e.id, Published: e.published, PricedBy: e.provider,
				Tier: c.Tier, Fast: c.Fast, AbovePromptTokens: c.AbovePromptTokens, Component: c.Component,
				Cost: c.Cost, Price: new(big.Rat).Mul(c.Cost, markup),
			})
		}
	}
	return out
}

func (b *book) costs() []ProviderCost {
	var out []ProviderCost
	for _, e := range b.entries() {
		out = append(out, b.priced(e.provider, e.id, e.key)...)
		for _, p := range costers {
			if p == e.provider {
				continue
			}
			if key, ok := (llmcost.ModelSelector{Provider: p, Model: e.key}).Key(); ok {
				out = append(out, b.priced(p, e.id, key)...)
			}
		}
	}
	return out
}

// priced lists provider p's costs for model, billed by key, at every service
// tier, speed, context tier and component key prices.
func (b *book) priced(p llmcost.Provider, model, key string) []ProviderCost {
	var out []ProviderCost
	for _, tier := range []llmcost.ServiceTier{llmcost.TierStandard, llmcost.TierFlex, llmcost.TierPriority} {
		r, ok := b.table.RatesFor(key, tier)
		if !ok {
			continue
		}
		speeds := []*big.Rat{nil}
		if r.Fast != nil {
			speeds = append(speeds, r.Fast)
		}
		ctx := append([]llmcost.Tier{{TierRates: r.Base}}, r.Tiers...)
		for _, fast := range speeds {
			for _, t := range ctx {
				for _, c := range []struct {
					component Component
					rate      *big.Rat
					scaled    bool
				}{
					{ComponentInput, t.Input, true},
					{ComponentOutput, t.Output, true},
					{ComponentCacheRead, t.CacheRead, false},
					{ComponentCacheWrite5m, t.CacheCreation, false},
					{ComponentCacheWrite1h, t.CacheCreation1h, false},
				} {
					if c.rate == nil {
						continue
					}
					cost := new(big.Rat).Mul(c.rate, perMillion)
					if fast != nil && c.scaled {
						cost.Mul(cost, fast)
					}
					out = append(out, ProviderCost{
						Provider: p, Model: model, Key: key, Source: b.source(key),
						Tier: tier, Fast: fast != nil, AbovePromptTokens: t.AbovePromptTokens, Component: c.component,
						Cost: cost,
					})
				}
			}
		}
	}
	return out
}
