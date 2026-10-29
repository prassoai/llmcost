// Command pricediff classifies how a LiteLLM re-vendor changes the
// [listprice] book, so the daily sync knows whether it may merge unattended.
//
//	pricediff dump > book.json          # the book this checkout compiles in
//	pricediff class base.json head.json # one of the classes below
//	pricediff report base.json head.json # the PR-body markdown
//
// Classes, most severe first: published-removed (a published rate vanished;
// the sync must not merge), published-price-change (a published rate moved or
// appeared beside existing ones; a human approves), published-added (a
// published model gained its first rates), unpublished-only, none.
package main

import (
	"cmp"
	"encoding/json"
	"fmt"
	"io"
	"math/big"
	"os"
	"slices"
	"strings"

	"github.com/prassoai/llmcost/listprice"
)

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "pricediff:", err)
		os.Exit(2)
	}
}

func run(args []string, w io.Writer) error {
	switch {
	case len(args) == 1 && args[0] == "dump":
		return json.NewEncoder(w).Encode(rows(listprice.Book()))
	case len(args) == 3 && (args[0] == "class" || args[0] == "report"):
		base, err := load(args[1])
		if err != nil {
			return fmt.Errorf("loading base: %w", err)
		}
		head, err := load(args[2])
		if err != nil {
			return fmt.Errorf("loading head: %w", err)
		}
		d := diff(base, head)
		if args[0] == "class" {
			_, err := fmt.Fprintln(w, d.class())
			return err
		}
		_, err = io.WriteString(w, d.report())
		return err
	}
	return fmt.Errorf("usage: pricediff dump | pricediff {class|report} BASE HEAD")
}

// row is one [listprice.Rate], serialized so two checkouts' books compare.
type row struct {
	Provider  string `json:"provider"`
	Model     string `json:"model"`
	Published bool   `json:"published"`
	Tier      string `json:"tier"`
	Fast      bool   `json:"fast"`
	Above     int64  `json:"above_prompt_tokens"`
	Component string `json:"component"`
	// Cost is exact USD per million tokens. It encodes as a big.Rat string,
	// so a malformed one fails decoding.
	Cost *big.Rat `json:"cost"`
}

// key identifies a rate. It includes the provider, so a model moving to
// another provider is a removal, never an unchanged rate.
type key struct {
	provider, model, tier, component string
	fast                             bool
	above                            int64
}

func (r row) key() key { return key{r.Provider, r.Model, r.Tier, r.Component, r.Fast, r.Above} }

func rows(book []listprice.Rate) []row {
	out := make([]row, len(book))
	for i, r := range book {
		out[i] = row{string(r.Provider), r.Model, r.Published, string(r.Tier), r.Fast, r.AbovePromptTokens, r.Component.String(), r.Cost}
	}
	return out
}

func load(path string) ([]row, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading %s: %w", path, err)
	}
	var out []row
	if err := json.Unmarshal(b, &out); err != nil {
		return nil, fmt.Errorf("decoding %s: %w", path, err)
	}
	if i := slices.IndexFunc(out, func(r row) bool { return r.Cost == nil }); i >= 0 {
		return nil, fmt.Errorf("decoding %s: row %d has no cost", path, i)
	}
	return out, nil
}

// change is one published rate that differs between base and head; a nil
// side is a rate that does not exist there.
type change struct{ base, head *row }

type delta struct {
	removed, repriced, added []change
	unpublished              int
}

// diff compares two books. A published model is one published at head, or at
// base if head dropped it; a sync never edits the curation, so the two agree.
func diff(base, head []row) delta {
	index := func(rs []row) map[key]*row {
		m := make(map[key]*row, len(rs))
		for i := range rs {
			m[rs[i].key()] = &rs[i]
		}
		return m
	}
	b, h := index(base), index(head)
	priced := map[string]bool{}
	for _, r := range base {
		priced[r.Model] = true
	}
	var d delta
	for _, r := range base {
		switch hr := h[r.key()]; {
		case !r.Published:
			if hr == nil || hr.Cost.Cmp(r.Cost) != 0 {
				d.unpublished++
			}
		case hr == nil:
			d.removed = append(d.removed, change{base: &r})
		case hr.Cost.Cmp(r.Cost) != 0:
			d.repriced = append(d.repriced, change{&r, hr})
		}
	}
	for _, r := range head {
		switch {
		case b[r.key()] != nil:
		case !r.Published:
			d.unpublished++
		case priced[r.Model]:
			d.repriced = append(d.repriced, change{head: &r})
		default:
			d.added = append(d.added, change{head: &r})
		}
	}
	return d
}

func (d delta) class() string {
	switch {
	case len(d.removed) > 0:
		return "published-removed"
	case len(d.repriced) > 0:
		return "published-price-change"
	case len(d.added) > 0:
		return "published-added"
	case d.unpublished > 0:
		return "unpublished-only"
	}
	return "none"
}

func (d delta) report() string {
	var s strings.Builder
	fmt.Fprintf(&s, "Price book change: **%s**", d.class())
	if d.unpublished > 0 {
		fmt.Fprintf(&s, " (%d unpublished rates changed)", d.unpublished)
	}
	s.WriteString("\n")
	changes := slices.Concat(d.removed, d.repriced, d.added)
	if len(changes) == 0 {
		return s.String()
	}
	s.WriteString("\nPublished list prices, USD per 1M tokens:\n\n| Model | Tier | Speed | Prompt above | Component | Before | After |\n|---|---|---|---|---|---|---|\n")
	slices.SortStableFunc(changes, func(x, y change) int {
		a, b := cmp.Or(x.base, x.head), cmp.Or(y.base, y.head)
		return cmp.Or(strings.Compare(a.Model, b.Model), strings.Compare(a.Tier, b.Tier), strings.Compare(a.Component, b.Component))
	})
	for _, c := range changes {
		r := cmp.Or(c.base, c.head)
		speed := "standard"
		if r.Fast {
			speed = "fast"
		}
		fmt.Fprintf(&s, "| %s | %s | %s | %d | %s | %s | %s |\n", r.Model, r.Tier, speed, r.Above, r.Component, price(c.base), price(c.head))
	}
	return s.String()
}

// price renders a row's list price exactly, or "—" for a rate that does not
// exist on that side.
func price(r *row) string {
	if r == nil {
		return "—"
	}
	p := strings.TrimRight(new(big.Rat).Mul(r.Cost, big.NewRat(10_000+listprice.MarkupBasisPoints, 10_000)).FloatString(10), "0")
	return "$" + strings.TrimSuffix(p, ".")
}
