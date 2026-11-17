package listprice

import (
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"sync"
	"testing"

	"github.com/prassoai/llmcost"
)

// initBudget is the most this package may allocate at init. Building the
// price book allocates about 26 MB, so a budget this small fails the moment
// it is built at init again.
const initBudget = 64 << 10

// TestInitIsCheap requires importing this package to cost almost nothing at
// start: every program that links it pays its init on every run, whether or
// not it prices anything, and murmur-vm runs on every git call on an agent
// VM. The price book is built on first use instead.
func TestInitIsCheap(t *testing.T) {
	out := fresh(t, "^$", "GODEBUG=inittrace=1")
	m := regexp.MustCompile(`(?m)^init github\.com/prassoai/llmcost/listprice @\S+ ms, (\S+) ms clock, (\d+) bytes, (\d+) allocs$`).FindStringSubmatch(out)
	if m == nil {
		t.Fatalf("no inittrace line for listprice in:\n%s", out)
	}
	if bytes, _ := strconv.Atoi(m[2]); bytes > initBudget {
		t.Errorf("listprice init allocated %d bytes in %s ms (%s allocs), over the %d-byte budget: is the price book built at init?", bytes, m[1], m[3], initBudget)
	}
}

// TestFirstCallPrices requires the first calls a process makes — however
// many race to build the book — to price exactly as a book built up front
// does. It prices in a fresh process, so its calls are the first.
func TestFirstCallPrices(t *testing.T) {
	if os.Getenv("LISTPRICE_FRESH") == "" {
		fresh(t, "^TestFirstCallPrices$", "LISTPRICE_FRESH=1")
		return
	}
	usages := map[string]llmcost.Usage{
		"claude-opus-5": llmcost.ClaudeUsage{InputTokens: 12_345, CacheReadInputTokens: 50_000, OutputTokens: 2_000},
		"gpt-5.5":       llmcost.OpenAIUsage{InputTokens: 300_000, CachedInputTokens: 100_000, OutputTokens: 9_000},
	}
	got := make([]map[string]Price, 8)
	var wg sync.WaitGroup
	for i := range got {
		wg.Go(func() {
			got[i] = map[string]Price{}
			for model, u := range usages {
				got[i][model], _ = Of(model, u)
			}
		})
	}
	wg.Wait()
	b := newBook(published, overrides)
	for model, u := range usages {
		want, ok := b.of(model, u)
		if !ok {
			t.Fatalf("%s does not price; the fixture must", model)
		}
		for i, g := range got {
			if g[model] != want {
				t.Errorf("caller %d: first Of(%s) = %+v, want %+v", i, model, g[model], want)
			}
		}
	}
}

// fresh runs the tests matching run in a new process of this test binary,
// with env added, and returns its combined output. A failing run fails t.
func fresh(t *testing.T, run string, env ...string) string {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run="+run, "-test.count=1", "-test.v")
	cmd.Env = append(os.Environ(), env...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("fresh process for %s: %v\n%s", run, err, out)
	}
	return string(out)
}
