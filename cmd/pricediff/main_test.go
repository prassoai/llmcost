package main

import (
	"bytes"
	"encoding/json"
	"math/big"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func rat(s string) *big.Rat {
	r, ok := new(big.Rat).SetString(s)
	if !ok {
		panic(s)
	}
	return r
}

func published(model, component, cost string) row {
	return row{Provider: "openai", Model: model, Published: true, Tier: "standard", Component: component, Cost: rat(cost)}
}

func unpublished(model, cost string) row {
	return row{Provider: "openai", Model: model, Tier: "standard", Component: "input", Cost: rat(cost)}
}

// TestClass pins the merge policy's inputs: which book changes a sync may
// merge unattended, which need a human, and which must block.
func TestClass(t *testing.T) {
	base := []row{published("gpt-5.5", "input", "5"), published("gpt-5.5", "output", "40"), unpublished("gpt-old", "1")}
	for _, tc := range []struct {
		name string
		head []row
		want string
	}{
		{"identical", base, "none"},
		{"unpublished repriced", []row{base[0], base[1], unpublished("gpt-old", "2")}, "unpublished-only"},
		{"unpublished added and removed", []row{base[0], base[1], unpublished("gpt-new", "1")}, "unpublished-only"},
		{"published model resolves", append(base[:3:3], published("gpt-6", "input", "1")), "published-added"},
		{"published repriced", []row{base[0], published("gpt-5.5", "output", "45"), base[2]}, "published-price-change"},
		{"published model gains a component", append(base[:3:3], published("gpt-5.5", "cache_read", "1/2")), "published-price-change"},
		{"published rate vanishes", []row{base[0], base[2]}, "published-removed"},
		{"removal outranks repricing", []row{published("gpt-5.5", "input", "6"), base[2]}, "published-removed"},
		{"equal costs in another form", []row{published("gpt-5.5", "input", "10/2"), base[1], base[2]}, "none"},
		{"published model moves provider", []row{moved(base[0]), moved(base[1]), base[2]}, "published-removed"},
	} {
		if got := diff(base, tc.head).class(); got != tc.want {
			t.Errorf("%s: class %s, want %s", tc.name, got, tc.want)
		}
	}
}

// TestReport requires the PR body to show each published list price before
// and after, markup included, so the approver reviews what customers pay.
func TestReport(t *testing.T) {
	got := diff(
		[]row{published("gpt-5.5", "input", "5"), published("gpt-5.5", "output", "40")},
		[]row{published("gpt-5.5", "input", "1/3"), unpublished("gpt-new", "1")},
	).report()
	for _, want := range []string{
		"**published-removed** (1 unpublished rates changed)",
		"| gpt-5.5 | standard | standard | 0 | input | $5.25 | $0.35 |",
		"| gpt-5.5 | standard | standard | 0 | output | $42 | — |",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("report missing %q:\n%s", want, got)
		}
	}
}

// TestDumpRoundTrips requires a dump of this checkout's book to load and to
// compare equal to itself, which is the sync's no-op path.
func TestDumpRoundTrips(t *testing.T) {
	var out bytes.Buffer
	if err := run([]string{"dump"}, &out); err != nil {
		t.Fatal(err)
	}
	var rs []row
	if err := json.Unmarshal(out.Bytes(), &rs); err != nil || len(rs) == 0 {
		t.Fatalf("dump decoded to %d rows: %v", len(rs), err)
	}
	path := filepath.Join(t.TempDir(), "book.json")
	if err := os.WriteFile(path, out.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	var class bytes.Buffer
	if err := run([]string{"class", path, path}, &class); err != nil || class.String() != "none\n" {
		t.Errorf("class of a book against itself = %q, %v", class.String(), err)
	}
}

func moved(r row) row {
	r.Provider = "azure"
	return r
}

// TestLoadRejectsMalformedCosts requires a book whose cost is missing or not
// an exact rational to fail loading with an error naming the file, never to
// reach the comparison or the report.
func TestLoadRejectsMalformedCosts(t *testing.T) {
	for name, book := range map[string]string{
		"malformed": `[{"provider":"openai","model":"gpt-5.5","tier":"standard","component":"input","cost":"five"}]`,
		"missing":   `[{"provider":"openai","model":"gpt-5.5","tier":"standard","component":"input"}]`,
		"null":      `[{"provider":"openai","model":"gpt-5.5","tier":"standard","component":"input","cost":null}]`,
	} {
		path := filepath.Join(t.TempDir(), "book.json")
		if err := os.WriteFile(path, []byte(book), 0o600); err != nil {
			t.Fatal(err)
		}
		if err := run([]string{"report", path, path}, new(bytes.Buffer)); err == nil || !strings.Contains(err.Error(), path) {
			t.Errorf("%s cost: run = %v, want an error naming %s", name, err, path)
		}
	}
}
