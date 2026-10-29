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

// Rate is one price-book row: what the provider charges per million tokens of
// one component, and the list price, for one model at one service tier,
// speed and context tier.
type Rate struct {
	Provider llmcost.Provider
	// Model is the id the model is requested under: a published model's id,
	// otherwise its llmcost key.
	Model string
	Key   string
	// Published marks the curated set of models sold by name.
	Published bool
	// Source is [SourceLLMCost] or [SourceOverride].
	Source Source
	Tier   llmcost.ServiceTier
	// Fast is Anthropic's fast mode: input and output carry the model's fast
	// multiplier, cache components bill unscaled.
	Fast bool
	// AbovePromptTokens is the context tier's floor: the row prices requests
	// whose total prompt exceeds it. 0 is the base tier.
	AbovePromptTokens int64
	Component         Component
	// Cost and Price are exact USD per million tokens; Price is Cost with
	// [MarkupBasisPoints] applied, unrounded.
	Cost, Price *big.Rat
}

// Book enumerates the price book, ordered by provider, then the published
// models in their curated order, then every other priceable key sorted. Each
// model contributes a row per service tier, speed, context tier and priced
// component. It covers everything [Of] prices at a base inference geo, so
// a model it lists no rate for has no list price.
func Book() []Rate { return std.rates() }

// providers are the first-party providers the book lists, in order.
var providers = []llmcost.Provider{llmcost.ProviderAnthropic, llmcost.ProviderOpenAI}

var (
	perMillion = big.NewRat(1_000_000, 1)
	markup     = big.NewRat(10_000+MarkupBasisPoints, 10_000)
)

func (b *book) rates() []Rate {
	var out []Rate
	for _, p := range providers {
		covered := map[string]bool{}
		for _, m := range b.published {
			if m.provider == p {
				out = append(out, b.rows(p, m.id, m.key, true)...)
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
				out = append(out, b.rows(p, k, k, false)...)
			}
		}
	}
	return out
}

func (b *book) rows(p llmcost.Provider, model, key string, published bool) []Rate {
	var out []Rate
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
					out = append(out, Rate{
						Provider: p, Model: model, Key: key, Published: published, Source: b.source(key),
						Tier: tier, Fast: fast != nil, AbovePromptTokens: t.AbovePromptTokens, Component: c.component,
						Cost: cost, Price: new(big.Rat).Mul(cost, markup),
					})
				}
			}
		}
	}
	return out
}
