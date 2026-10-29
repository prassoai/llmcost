package listprice

import (
	"cmp"
	"fmt"
	"runtime/debug"
	"slices"
	"sync"

	"github.com/prassoai/llmcost"
)

// MarkupBasisPoints is the markup on provider list cost that every list price
// includes: 500 basis points, 5%.
const MarkupBasisPoints = 500

// Source records which rates priced a call or a [Rate].
type Source uint8

const (
	// SourceLLMCost prices from the key's vendored LiteLLM rates.
	SourceLLMCost Source = iota + 1
	// SourceOverride prices from a reviewed rate this package restates.
	SourceOverride
	// SourceDegraded priced a call that reported a dimension its model has no
	// rate for — a pinned inference geo, or 1-hour cache writes — with that
	// dimension dropped. Only [Of] returns it; it never appears in a [Rate].
	SourceDegraded
)

func (s Source) String() string {
	switch s {
	case SourceLLMCost:
		return "llmcost"
	case SourceOverride:
		return "override"
	case SourceDegraded:
		return "degraded"
	}
	return fmt.Sprintf("Source(%d)", s)
}

// Price is one call's list price and the provider cost it derives from, in
// nls. Price is always [Markup](Cost).
type Price struct {
	Cost, Price llmcost.Nls
	Source      Source
}

// Markup returns the list price of cost: cost plus [MarkupBasisPoints],
// rounded up to a whole nls. Anyone holding a recorded cost re-derives its
// price with it.
func Markup(cost llmcost.Nls) llmcost.Nls {
	return cost + (cost*MarkupBasisPoints+9_999)/10_000
}

// Of prices one call by its llmcost [llmcost.Usage]. model is the id the
// call was requested under: a published model's id, or an llmcost key.
//
// A call whose usage prices exactly is [SourceLLMCost] or [SourceOverride].
// A call that cannot is re-priced with the response-reported dimensions a
// request cannot foresee dropped — Anthropic's inference_geo to the base
// rate, 1-hour cache writes to the 5-minute rate — and is [SourceDegraded]:
// the call was already made, and recording it as free would be the one wrong
// answer. Nothing else degrades: an unknown model, an unpriced service tier
// or an unpriced fast mode is ok=false, and those are what [Priceable]
// refuses before a call is made. Impossible counts panic, as in llmcost.
func Of(model string, u llmcost.Usage) (Price, bool) { return std.of(model, u) }

// Priceable reports whether a request for model at tier and speed (the
// Anthropic speed parameter: "" or "fast") can be list-priced, before any
// usage exists. A request it rejects would have no published price.
func Priceable(model string, tier llmcost.ServiceTier, speed string) bool {
	return std.priceable(model, tier, speed)
}

// Version identifies a price book: this module's version as the consuming
// build resolved it, and the LiteLLM commit its rates are vendored from.
// Every rate and markup the book applies is fixed by the pair.
type Version struct {
	Module        string
	LiteLLMCommit string
}

// String is the version's stable identifier, e.g. "v0.5.0+litellm.4846d9c2".
func (v Version) String() string { return v.Module + "+litellm." + v.LiteLLMCommit[:8] }

// CurrentVersion is the version of the price book compiled into this binary.
// A build that replaces this module with a local directory reports
// "(devel)": such a build is not a release and its prices are not a book.
func CurrentVersion() Version { return current() }

const modulePath = "github.com/prassoai/llmcost"

var current = sync.OnceValue(func() Version {
	bi, ok := debug.ReadBuildInfo()
	if !ok {
		panic("listprice: binary carries no module build info")
	}
	mod := &bi.Main
	if mod.Path != modulePath {
		i := slices.IndexFunc(bi.Deps, func(d *debug.Module) bool { return d.Path == modulePath })
		if i < 0 {
			panic("listprice: " + modulePath + " is not in the binary's build info")
		}
		mod = bi.Deps[i]
		if mod.Replace != nil {
			mod = mod.Replace
		}
	}
	return Version{Module: cmp.Or(mod.Version, "(devel)"), LiteLLMCommit: llmcost.LiteLLMCommit()}
})
