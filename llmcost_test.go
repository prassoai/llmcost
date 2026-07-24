package llmcost

import (
	"encoding/json"
	"math/big"
	"slices"
	"testing"
)

// Fixtures mirror real LiteLLM entries at the vendored commit, inlined so the
// exact-math tests are independent of the vendored data — a weekly price sync
// must never turn these red. Values are the providers' published prices.

// sonnet45Spec mirrors claude-sonnet-4-5: $3/M input, $0.30/M cache read,
// $3.75/M 5-minute cache write (1.25×), $6/M 1-hour cache write (2×), $15/M
// output, with the 1M-context-beta premium tier above 200k prompt tokens
// ($6 / $0.60 / $7.50 / $12 / $22.50).
const sonnet45Spec = `{
	"input_cost_per_token": 3e-06,
	"cache_read_input_token_cost": 3e-07,
	"cache_creation_input_token_cost": 3.75e-06,
	"cache_creation_input_token_cost_above_1hr": 6e-06,
	"output_cost_per_token": 1.5e-05,
	"input_cost_per_token_above_200k_tokens": 6e-06,
	"cache_read_input_token_cost_above_200k_tokens": 6e-07,
	"cache_creation_input_token_cost_above_200k_tokens": 7.5e-06,
	"cache_creation_input_token_cost_above_1hr_above_200k_tokens": 1.2e-05,
	"output_cost_per_token_above_200k_tokens": 2.25e-05,
	"litellm_provider": "anthropic"
}`

// gpt54Spec mirrors gpt-5.4: OpenAI-style — cache reads but no cache
// creation, tiered above 272k prompt tokens, with the 1.1× regional-
// processing uplift OpenAI charges on its eu./us. data-residency hosts.
const gpt54Spec = `{
	"input_cost_per_token": 2.5e-06,
	"cache_read_input_token_cost": 2.5e-07,
	"output_cost_per_token": 1.5e-05,
	"input_cost_per_token_above_272k_tokens": 5e-06,
	"cache_read_input_token_cost_above_272k_tokens": 5e-07,
	"output_cost_per_token_above_272k_tokens": 2.25e-05,
	"cache_read_input_token_cost_priority": 5e-07,
	"input_cost_per_token_above_272k_tokens_priority": 1e-05,
	"regional_processing_uplift_multiplier_eu": 1.1,
	"regional_processing_uplift_multiplier_us": 1.1,
	"litellm_provider": "openai"
}`

// opus46Spec mirrors claude-opus-4-6: $5/M input, $0.50/M cache read,
// $6.25/M 5-minute cache write, $10/M 1-hour cache write, $25/M output, and
// LiteLLM's provider_specific_entry price multipliers — 6× for fast mode,
// 1.1× for us-pinned inference — which scale uncached input and output only.
const opus46Spec = `{
	"input_cost_per_token": 5e-06,
	"cache_read_input_token_cost": 5e-07,
	"cache_creation_input_token_cost": 6.25e-06,
	"cache_creation_input_token_cost_above_1hr": 1e-05,
	"output_cost_per_token": 2.5e-05,
	"provider_specific_entry": {"us": 1.1, "fast": 6.0},
	"litellm_provider": "anthropic"
}`

func mustParse(t *testing.T, spec string) Rates {
	t.Helper()
	return mustParseTiers(t, spec)[TierStandard]
}

func mustParseTiers(t *testing.T, spec string) map[ServiceTier]Rates {
	t.Helper()
	m, ok := parseModel(json.RawMessage(spec))
	if !ok {
		t.Fatal("fixture spec did not parse as priceable")
	}
	return m
}

// price runs the full internal pipeline — normalization, premium resolution,
// costing — against fixture rates: exactly what Cost does after the table
// lookup.
func price(r Rates, u Usage) (Nls, bool) {
	p, ok := u.premium(r)
	if !ok {
		return 0, false
	}
	return cost(r, u.disjoint(), p)
}

// TestExactCost encodes the core requirement: a response is priced as
// Σ rate × tokens over all four components — uncached input, cache reads at
// the real cache-read rate (0.1× input), cache writes at the real
// cache-creation rate (1.25× input), output — computed exactly, with a total
// landing on an nls boundary returned as-is. 1000×3e-6 + 30000×3e-7 +
// 2000×3.75e-6 + 100×1.5e-5 = $0.021 = exactly 2100 nls.
func TestExactCost(t *testing.T) {
	got, ok := cost(mustParse(t, sonnet45Spec), components{input: 1000, cacheRead: 30000, cacheCreation: 2000, output: 100}, premium{})
	if !ok || got != 2100 {
		t.Fatalf("cost = %d, %v; want 2100, true", got, ok)
	}
}

// TestTieredPricing encodes the context-window tier requirement, with the
// semantics verified against both LiteLLM's cost calculator and Anthropic's
// long-context billing: the tier is selected by the request's TOTAL prompt
// size (uncached input + cache reads + cache writes), the threshold is
// STRICT (exactly 200k bills at base rates), and once exceeded the ENTIRE
// request — every token, not just the excess — bills at the tier's rates.
func TestTieredPricing(t *testing.T) {
	r := mustParse(t, sonnet45Spec)

	// Exactly at the threshold: prompt = 150000 + 50000 = 200000, NOT above.
	// Base rates: 150000×3e-6 + 50000×3e-7 + 1000×1.5e-5 = $0.48 = 48000 nls.
	if got, ok := cost(r, components{input: 150000, cacheRead: 50000, output: 1000}, premium{}); !ok || got != 48000 {
		t.Fatalf("at threshold: cost = %d, %v; want 48000, true", got, ok)
	}

	// One token more: prompt = 200001 > 200000 — the whole request re-rates.
	// 150001×6e-6 + 50000×6e-7 + 1000×2.25e-5 = $0.952506 → ceil = 95251 nls.
	// Marginal-only billing would give ~48000; whole-request is nearly 2×.
	if got, ok := cost(r, components{input: 150001, cacheRead: 50000, output: 1000}, premium{}); !ok || got != 95251 {
		t.Fatalf("above threshold: cost = %d, %v; want 95251, true", got, ok)
	}
}

// TestTierInheritsBaseRates encodes per-component tier resolution: a
// component upstream leaves untiered bills at the base rate even when the
// request is in the premium tier (LiteLLM's calculator falls back per
// component the same way).
func TestTierInheritsBaseRates(t *testing.T) {
	r := mustParse(t, `{
		"input_cost_per_token": 1e-06,
		"cache_read_input_token_cost": 1e-07,
		"output_cost_per_token": 2e-06,
		"input_cost_per_token_above_128k_tokens": 2e-06
	}`)
	// prompt = 200000 > 128000 → tiered input, but output has no tier rate:
	// 200000×2e-6 + 1000×2e-6 (base) = $0.402 = 40200 nls.
	if got, ok := cost(r, components{input: 200000, output: 1000}, premium{}); !ok || got != 40200 {
		t.Fatalf("cost = %d, %v; want 40200, true", got, ok)
	}
}

// TestTierOnlyCacheCreation encodes the Gemini shape: no base cache-creation
// rate, but the premium tier defines one. Cache writes must fail below the
// threshold (unpriced component) and price above it.
func TestTierOnlyCacheCreation(t *testing.T) {
	r := mustParse(t, `{
		"input_cost_per_token": 1.25e-06,
		"cache_read_input_token_cost": 1.25e-07,
		"output_cost_per_token": 1e-05,
		"input_cost_per_token_above_200k_tokens": 2.5e-06,
		"cache_read_input_token_cost_above_200k_tokens": 2.5e-07,
		"cache_creation_input_token_cost_above_200k_tokens": 2.5e-07,
		"output_cost_per_token_above_200k_tokens": 1.5e-05
	}`)
	if _, ok := cost(r, components{input: 1000, cacheCreation: 500}, premium{}); ok {
		t.Fatal("cache writes priced below threshold despite no base cache-creation rate")
	}
	if _, ok := cost(r, components{input: 250000, cacheCreation: 500}, premium{}); !ok {
		t.Fatal("cache writes not priced above threshold despite tiered cache-creation rate")
	}
}

// TestCacheWriteTTLSplit encodes that cache writes price by TTL: the 1-hour
// subset bills at the 1h rate (2× input) and the remainder at the 5-minute
// rate (1.25× input). 1500 5m-writes ×3.75e-6 + 500 1h-writes ×6e-6 =
// $0.008625 = exactly 862.5 nls → ceil 863; billing all 2000 at the 5m rate
// would give 750, at the 1h rate 1200. Cost with ClaudeUsage takes the API's raw shape —
// total writes plus the 1h subset — and splits internally.
func TestCacheWriteTTLSplit(t *testing.T) {
	got, ok := cost(mustParse(t, sonnet45Spec), components{cacheCreation: 1500, cacheCreation1h: 500}, premium{})
	if !ok || got != 863 {
		t.Fatalf("cost = %d, %v; want 863, true", got, ok)
	}
	viaAPI, ok2 := Cost("claude-sonnet-4-5", ClaudeUsage{CacheCreationInputTokens: 2000, CacheCreation1hInputTokens: 500})
	direct, ok3 := cost(table()["claude-sonnet-4-5"][TierStandard], components{cacheCreation: 1500, cacheCreation1h: 500}, premium{})
	if !ok2 || !ok3 || viaAPI != direct {
		t.Fatalf("Cost(ClaudeUsage) = %d, %v; internal = %d, %v; want equal and ok", viaAPI, ok2, direct, ok3)
	}
}

// TestCacheWrite1hTiered encodes that the 1-hour write rate participates in
// context-window tiers (cache_creation_input_token_cost_above_1hr_above_200k_tokens),
// and that 1h writes count toward the tier threshold: 200001 1h-write tokens
// alone exceed 200k, so the whole request bills at the tiered 1h rate.
// 200001×1.2e-5 = $2.400012 → 240001.2 nls → ceil 240002.
func TestCacheWrite1hTiered(t *testing.T) {
	got, ok := cost(mustParse(t, sonnet45Spec), components{cacheCreation1h: 200001}, premium{})
	if !ok || got != 240002 {
		t.Fatalf("cost = %d, %v; want 240002, true", got, ok)
	}
}

// TestNoCacheRateFallback encodes the no-silent-fallback requirement: a model
// without a cache-read (or cache-creation) rate must FAIL when usage reports
// those tokens — never bill them at the full input rate, never bill zero.
// Zero counts on the unpriced component still price fine.
func TestNoCacheRateFallback(t *testing.T) {
	r := mustParse(t, `{"input_cost_per_token": 1e-06, "output_cost_per_token": 2e-06}`)
	if _, ok := cost(r, components{input: 100, cacheRead: 1}, premium{}); ok {
		t.Fatal("cache reads priced despite model having no cache-read rate")
	}
	if _, ok := cost(r, components{input: 100, cacheCreation: 1}, premium{}); ok {
		t.Fatal("cache writes priced despite model having no cache-creation rate")
	}
	if _, ok := cost(r, components{input: 100, cacheCreation1h: 1}, premium{}); ok {
		t.Fatal("1h cache writes priced despite model having no 1h cache-creation rate")
	}
	// A model priced only for 5m writes must not bill 1h writes at the 5m rate.
	fiveMinOnly := mustParse(t, `{"input_cost_per_token": 1e-06, "cache_creation_input_token_cost": 1.25e-06, "output_cost_per_token": 2e-06}`)
	if _, ok := cost(fiveMinOnly, components{cacheCreation1h: 1}, premium{}); ok {
		t.Fatal("1h cache writes priced despite model having only a 5m cache-creation rate")
	}
	if got, ok := cost(r, components{input: 100}, premium{}); !ok || got != 10 {
		t.Fatalf("cost without cache usage = %d, %v; want 10, true", got, ok)
	}
	// OpenAI models lack cache-creation by design; reads-only must price.
	if _, ok := cost(mustParse(t, gpt54Spec), components{input: 600, cacheRead: 400}, premium{}); !ok {
		t.Fatal("OpenAI-shaped model failed to price cache reads")
	}
}

// TestExplicitZeroCacheRateIsFree encodes the absent-vs-zero distinction: an
// upstream rate explicitly listed as 0 is a real free tier (DeepSeek lists
// cache writes at 0 because it genuinely doesn't bill them; gateway-hosted
// OpenAI entries do the same) and prices as $0, while an ABSENT rate is
// unpriced and fails — see TestNoCacheRateFallback. Collapsing the two would
// turn correctly-priced free usage into false failures. Consumers needing a
// stricter bar for the models they bill should assert those models' rates in
// their own tests.
func TestExplicitZeroCacheRateIsFree(t *testing.T) {
	r := mustParse(t, `{"input_cost_per_token": 1e-06, "output_cost_per_token": 2e-06, "cache_creation_input_token_cost": 0.0}`)
	if got, ok := cost(r, components{input: 100, cacheCreation: 5000}, premium{}); !ok || got != 10 {
		t.Fatalf("cost = %d, %v; want 10, true (free cache writes bill $0, input still bills)", got, ok)
	}
}

// TestOpenAINormalization encodes the OpenAI provider convention: raw
// InputTokens INCLUDES the cached subset, and the module — not the caller —
// splits it. 1000 total input with 400 cached on gpt-5.4 rates:
// 600×2.5e-6 + 400×2.5e-7 + 100×1.5e-5 = $0.0031 = 310 nls. Billing all 1000
// at the input rate (double-counting) would give 360.
func TestOpenAINormalization(t *testing.T) {
	got, ok := cost(mustParse(t, gpt54Spec), components{input: 600, cacheRead: 400, output: 100}, premium{})
	if !ok || got != 310 {
		t.Fatalf("cost = %d, %v; want 310, true", got, ok)
	}
	// The exported entry point must produce the same normalization from raw counts.
	viaAPI, ok2 := Cost("gpt-5.4", OpenAIUsage{InputTokens: 1000, CachedInputTokens: 400, OutputTokens: 100})
	direct, ok3 := cost(table()["gpt-5.4"][TierStandard], components{input: 600, cacheRead: 400, output: 100}, premium{})
	if !ok2 || !ok3 || viaAPI != direct {
		t.Fatalf("Cost(OpenAIUsage) = %d, %v; internal = %d, %v; want equal and ok", viaAPI, ok2, direct, ok3)
	}
}

// TestOpenAITierCountsCachedTokens encodes that OpenAI's tier threshold is
// judged on the FULL reported InputTokens (cached included): 272001 input of
// which 272000 cached exceeds the 272k tier even though only 1 token is
// uncached. 1×5e-6 + 272000×5e-7 = $0.136005 → ceil = 13601 nls.
func TestOpenAITierCountsCachedTokens(t *testing.T) {
	got, ok := cost(mustParse(t, gpt54Spec), components{input: 1, cacheRead: 272000}, premium{})
	if !ok || got != 13601 {
		t.Fatalf("cost = %d, %v; want 13601, true", got, ok)
	}
}

// gpt56Spec mirrors gpt-5.6-sol: OpenAI-style with a cache-CREATION rate
// (1.25× uncached input), the first OpenAI family to bill cache writes, plus
// the 272k context tier with a tiered cache-creation rate. No 1-hour TTL —
// OpenAI has a single cache-write bucket.
const gpt56Spec = `{
	"input_cost_per_token": 5e-06,
	"cache_read_input_token_cost": 5e-07,
	"cache_creation_input_token_cost": 6.25e-06,
	"output_cost_per_token": 3e-05,
	"input_cost_per_token_above_272k_tokens": 1e-05,
	"cache_read_input_token_cost_above_272k_tokens": 1e-06,
	"cache_creation_input_token_cost_above_272k_tokens": 1.25e-05,
	"output_cost_per_token_above_272k_tokens": 4.5e-05,
	"litellm_provider": "openai"
}`

// TestOpenAICacheWriteNormalization encodes GPT-5.6 cache-write billing: the
// cache_write_tokens subset of input bills at the cache-creation rate (1.25×
// uncached input), disjoint from both plain input and cache reads. Raw usage
// InputTokens=1175 splits into 500 plain + 50 read + 625 write:
// 500×5e-6 + 50×5e-7 + 625×6.25e-6 + 3000×3e-5 = $0.09643125 → ceil 9644 nls.
func TestOpenAICacheWriteNormalization(t *testing.T) {
	r := mustParse(t, gpt56Spec)
	got, ok := cost(r, components{input: 500, cacheRead: 50, cacheCreation: 625, output: 3000}, premium{})
	if !ok || got != 9644 {
		t.Fatalf("cost = %d, %v; want 9644, true", got, ok)
	}
	// The exported path must split raw OpenAI counts the same way: cache-write
	// is subtracted out of plain input (not double-billed) and priced at the
	// cache-creation rate.
	viaAPI, ok2 := price(r, OpenAIUsage{InputTokens: 1175, CachedInputTokens: 50, CacheWriteTokens: 625, OutputTokens: 3000})
	if !ok2 || viaAPI != got {
		t.Fatalf("price(OpenAIUsage) = %d, %v; want %d, true", viaAPI, ok2, got)
	}
}

// TestOpenAICacheWriteCountsTowardTier pins that OpenAI cache-write tokens
// count toward the 272k context-window threshold. OpenAI's long-context rule
// applies the higher rate to "requests above 272K input tokens", and
// input_tokens is the full total of which cache-write is a subset — so a
// cache-write token that pushes the total past 272k re-prices the ENTIRE
// request. (This is deliberately the opposite of back's
// TestCacheWriteDoesNotFlipOpenAITier; the OpenAI docs' definition governs.)
func TestOpenAICacheWriteCountsTowardTier(t *testing.T) {
	r := mustParse(t, gpt56Spec)
	// 272000 total → base tier: 272000×5e-6 = $1.36 = 136000 nls.
	if got, ok := cost(r, components{input: 272000}, premium{}); !ok || got != 136000 {
		t.Fatalf("at threshold: cost = %d, %v; want 136000, true", got, ok)
	}
	// One cache-write token makes the total 272001 (> 272000) → above tier,
	// whole request re-priced: 272000×1e-5 + 1×1.25e-5 = $2.7200125 → ceil 272002.
	if got, ok := cost(r, components{input: 272000, cacheCreation: 1}, premium{}); !ok || got != 272002 {
		t.Fatalf("above threshold via cache-write: cost = %d, %v; want 272002, true", got, ok)
	}
}

// TestOpenAICacheWriteExceedsInputPanics encodes the disjointness invariant:
// cached + cache-write are subsets of input_tokens, so their sum exceeding it
// is a malformed response and panics rather than yielding negative plain input.
func TestOpenAICacheWriteExceedsInputPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic when CachedInputTokens+CacheWriteTokens exceeds InputTokens")
		}
	}()
	OpenAIUsage{InputTokens: 100, CachedInputTokens: 60, CacheWriteTokens: 50}.disjoint()
}

// gpt55Spec models gpt-5.5's real standard/flex/priority rates (flex is
// exactly half, priority exactly double) plus a priority context-window tier
// in the gpt-5.1-codex-era shape (input and cache-read tiered, output
// deliberately NOT) so tier-within-tier resolution and intra-service-tier
// inheritance are both exercised.
const gpt55Spec = `{
	"input_cost_per_token": 5e-06,
	"cache_read_input_token_cost": 5e-07,
	"output_cost_per_token": 3e-05,
	"input_cost_per_token_above_272k_tokens": 1e-05,
	"cache_read_input_token_cost_above_272k_tokens": 1e-06,
	"output_cost_per_token_above_272k_tokens": 4.5e-05,
	"input_cost_per_token_flex": 2.5e-06,
	"cache_read_input_token_cost_flex": 2.5e-07,
	"output_cost_per_token_flex": 1.5e-05,
	"input_cost_per_token_priority": 1e-05,
	"cache_read_input_token_cost_priority": 1e-06,
	"output_cost_per_token_priority": 6e-05,
	"input_cost_per_token_above_272k_tokens_priority": 2e-05,
	"cache_read_input_token_cost_above_272k_tokens_priority": 2e-06,
	"litellm_provider": "openai"
}`

// TestServiceTierCost encodes the service-tier requirement: the same usage
// prices at each tier's own rates. On gpt-5.5's published prices, 600
// uncached input + 400 cache reads + 100 output:
//
//	standard: 600×5e-6  + 400×5e-7   + 100×3e-5   = $0.0062 = 620 nls
//	flex:     600×2.5e-6 + 400×2.5e-7 + 100×1.5e-5 = $0.0031 = 310 nls (half)
//	priority: 600×1e-5  + 400×1e-6   + 100×6e-5   = $0.0124 = 1240 nls (double)
func TestServiceTierCost(t *testing.T) {
	tiers := mustParseTiers(t, gpt55Spec)
	c := components{input: 600, cacheRead: 400, output: 100}
	for tier, want := range map[ServiceTier]Nls{TierStandard: 620, TierFlex: 310, TierPriority: 1240} {
		r, ok := tiers[tier]
		if !ok {
			t.Fatalf("%s tier did not parse", tier)
		}
		if got, ok := cost(r, c, premium{}); !ok || got != want {
			t.Errorf("%s: cost = %d, %v; want %d, true", tier, got, ok, want)
		}
	}
}

// TestPriorityContextTier encodes tier-within-tier: the priority service
// tier has its own context-window tiers (*_above_272k_tokens_priority) with
// the same strict-threshold, whole-request semantics — and a component the
// context tier leaves untiered inherits the PRIORITY base, never standard's.
//
//	at 272000 (not above): 272000×1e-5 + 1000×6e-5 = $2.78 = 278000 nls
//	at 272001 (above):     272001×2e-5 + 1000×6e-5 = $5.50002 = 550002 nls
//	                       (output inherits priority base 6e-5; inheriting the
//	                       standard ctx-tier 4.5e-5 would give 548502, the
//	                       standard base 3e-5 would give 547002)
func TestPriorityContextTier(t *testing.T) {
	r := mustParseTiers(t, gpt55Spec)[TierPriority]
	if got, ok := cost(r, components{input: 272000, output: 1000}, premium{}); !ok || got != 278000 {
		t.Fatalf("at threshold: cost = %d, %v; want 278000, true", got, ok)
	}
	if got, ok := cost(r, components{input: 272001, output: 1000}, premium{}); !ok || got != 550002 {
		t.Fatalf("above threshold: cost = %d, %v; want 550002, true", got, ok)
	}
}

// TestServiceTierNoCrossFallback encodes the no-cross-tier-fallback
// requirement, both wholesale and per component:
//
//   - A model with only stray tier-variant fields but no priceable tier base
//     (gpt54Spec carries cache_read_priority and a priority context tier but
//     no priority input/output) has NO priority entry — those fragments must
//     never resolve against standard rates.
//   - A tier missing one component (flex without a flex cache-read) fails
//     usage reporting that component instead of borrowing standard's rate,
//     which would overbill flex cache reads 2×.
func TestServiceTierNoCrossFallback(t *testing.T) {
	tiers := mustParseTiers(t, gpt54Spec)
	if _, ok := tiers[TierPriority]; ok {
		t.Error("priority tier resolved from fragments (no priority input/output rates)")
	}
	if _, ok := tiers[TierFlex]; ok {
		t.Error("flex tier resolved despite no flex fields at all")
	}

	partialFlex := mustParseTiers(t, `{
		"input_cost_per_token": 1e-06,
		"cache_read_input_token_cost": 1e-07,
		"output_cost_per_token": 2e-06,
		"input_cost_per_token_flex": 5e-07,
		"output_cost_per_token_flex": 1e-06
	}`)
	flex, ok := partialFlex[TierFlex]
	if !ok {
		t.Fatal("flex tier with its own input+output rates did not parse")
	}
	if _, ok := cost(flex, components{input: 100, cacheRead: 1}, premium{}); ok {
		t.Error("flex cache reads priced despite no flex cache-read rate")
	}
	// 100×5e-7 + 50×1e-6 = $0.0001 = exactly 10 nls: the tier's own rates apply.
	if got, ok := cost(flex, components{input: 100, output: 50}, premium{}); !ok || got != 10 {
		t.Errorf("flex without cache usage = %d, %v; want 10, true", got, ok)
	}
}

// TestUnknownServiceTier encodes the fail-loud tier contract on both
// entrances. On [OpenAIUsage.ServiceTier] the zero value bills standard,
// but any unrecognized non-empty value — arbitrary strings, consumer
// vocabulary that was never mapped, case variants — returns ok=false
// rather than silently pricing at standard rates. [RatesFor] takes only
// the ServiceTier constants: "" is not a constant there and fails like
// any other unrecognized value.
func TestUnknownServiceTier(t *testing.T) {
	for _, tier := range []ServiceTier{"turbo", "fast", "Standard", "STANDARD"} {
		if _, ok := Cost("gpt-5.4", OpenAIUsage{InputTokens: 1, ServiceTier: tier}); ok {
			t.Errorf("Cost(ServiceTier: %q) resolved; want ok=false", tier)
		}
	}
	if _, ok := Cost("gpt-5.4", OpenAIUsage{InputTokens: 1}); !ok {
		t.Error("Cost with zero-value ServiceTier failed; want ok=true (zero value = standard)")
	}
	for _, tier := range []ServiceTier{"", "turbo", "fast", "Standard", "STANDARD"} {
		if _, ok := RatesFor("gpt-5.4", tier); ok {
			t.Errorf("RatesFor(%q) resolved; want ok=false", tier)
		}
	}
}

// TestClaudeFastMode encodes Anthropic fast-mode pricing (LiteLLM's
// provider_specific_entry.fast): a speed:"fast" request bills uncached input
// and output at the fast multiplier (6× on opus-4-6) while cache reads and
// writes bill UNSCALED — Anthropic does not premium-price cache traffic, and
// LiteLLM's calculator excludes cache costs from the multiplier the same
// way. Standard: 1000×5e-6 + 30000×5e-7 + 2000×6.25e-6 + 100×2.5e-5 =
// $0.035 = 3500 nls. Fast: only input and output scale ×6 → 0.03 + 0.015 +
// 0.0125 + 0.015 = $0.0725 = 7250 nls (scaling cache too would give $0.21).
func TestClaudeFastMode(t *testing.T) {
	r := mustParse(t, opus46Spec)
	u := ClaudeUsage{InputTokens: 1000, CacheReadInputTokens: 30000, CacheCreationInputTokens: 2000, OutputTokens: 100}
	if got, ok := price(r, u); !ok || got != 3500 {
		t.Fatalf("standard = %d, %v; want 3500, true", got, ok)
	}
	u.Speed = "fast"
	if got, ok := price(r, u); !ok || got != 7250 {
		t.Fatalf("fast = %d, %v; want 7250, true", got, ok)
	}
	// The exported entry point resolves the same premium from the same raw usage.
	viaAPI, ok := Cost("claude-opus-4-6", u)
	direct, ok2 := price(table()["claude-opus-4-6"][TierStandard], u)
	if !ok || !ok2 || viaAPI != direct {
		t.Fatalf("Cost(fast) = %d, %v; internal = %d, %v; want equal and ok", viaAPI, ok, direct, ok2)
	}
}

// TestClaudeInferenceGeo encodes regional-inference pricing
// (usage.inference_geo): a response pinned to a region bills uncached input
// and output at the geo multiplier (1.1× for "us" on opus-4-6), matched
// case-insensitively as in LiteLLM; the API's unpinned values — "global",
// "not_available", and absent — never carry a premium. (1000×5e-6 +
// 100×2.5e-5) × 1.1 = $0.00825 = 825 nls; unpinned $0.0075 = 750 nls.
func TestClaudeInferenceGeo(t *testing.T) {
	r := mustParse(t, opus46Spec)
	for geo, want := range map[string]Nls{"us": 825, "US": 825, "": 750, "global": 750, "not_available": 750} {
		got, ok := price(r, ClaudeUsage{InputTokens: 1000, OutputTokens: 100, InferenceGeo: geo})
		if !ok || got != want {
			t.Errorf("geo %q = %d, %v; want %d, true", geo, got, ok, want)
		}
	}
}

// TestClaudeFastGeoCompose encodes that the fast and geo premiums compose
// multiplicatively, as in LiteLLM: fast us-pinned opus-4-6 bills 6 × 1.1 =
// 6.6×. (1000×5e-6 + 100×2.5e-5) × 6.6 = $0.0495 = 4950 nls.
func TestClaudeFastGeoCompose(t *testing.T) {
	got, ok := price(mustParse(t, opus46Spec), ClaudeUsage{InputTokens: 1000, OutputTokens: 100, Speed: "fast", InferenceGeo: "us"})
	if !ok || got != 4950 {
		t.Fatalf("fast+us = %d, %v; want 4950, true", got, ok)
	}
}

// TestUnpricedModeFails encodes the no-silent-standard-rate requirement: a
// mode the model has no multiplier for must FAIL, never bill unscaled —
// fast premiums reach 6×, so billing standard rates on a data lag is a 6×
// underbill. Fast on a model without fast pricing fails, a pinned geo
// without a factor fails, and an unrecognized speed value fails. (LiteLLM
// bills all three at standard rates — exactly the underbilling class this
// module refuses.) The unpremiumed request values — Speed "" and "standard",
// unpinned geos — price normally.
func TestUnpricedModeFails(t *testing.T) {
	noMult := mustParse(t, sonnet45Spec) // no provider_specific_entry
	if _, ok := price(noMult, ClaudeUsage{InputTokens: 1, Speed: "fast"}); ok {
		t.Error("fast priced despite model having no fast multiplier")
	}
	if _, ok := price(noMult, ClaudeUsage{InputTokens: 1, InferenceGeo: "us"}); ok {
		t.Error("us-pinned inference priced despite model having no geo multiplier")
	}
	r := mustParse(t, opus46Spec)
	if _, ok := price(r, ClaudeUsage{InputTokens: 1, InferenceGeo: "eu"}); ok {
		t.Error("eu-pinned inference priced despite model having only a us multiplier")
	}
	if _, ok := price(r, ClaudeUsage{InputTokens: 1, Speed: "turbo"}); ok {
		t.Error("unrecognized speed priced")
	}
	for _, speed := range []string{"", "standard"} {
		if got, ok := price(r, ClaudeUsage{InputTokens: 1000, Speed: speed}); !ok || got != 500 {
			t.Errorf("speed %q = %d, %v; want 500, true", speed, got, ok)
		}
	}
}

// TestOpenAIRegionalUplift encodes OpenAI data-residency pricing
// (regional_processing_uplift_multiplier_eu/us): a request served by a
// regionalized host bills EVERY component — cache reads included — at the
// uplift, unlike Anthropic's cache-exempt multipliers. (600×2.5e-6 +
// 400×2.5e-7 + 100×1.5e-5) × 1.1 = $0.00341 = 341 nls; the global host
// bills $0.0031 = 310 nls. A region the model has no uplift for bills
// standard rates — OpenAI publishes the uplift per model, residency is a
// transport fact that holds for every model, and LiteLLM prices absent
// uplifts the same way (see the DataResidency field doc).
func TestOpenAIRegionalUplift(t *testing.T) {
	r := mustParse(t, gpt54Spec)
	for region, want := range map[string]Nls{"eu": 341, "EU": 341, "us": 341, "": 310, "jp": 310} {
		got, ok := price(r, OpenAIUsage{InputTokens: 1000, CachedInputTokens: 400, OutputTokens: 100, DataResidency: region})
		if !ok || got != want {
			t.Errorf("residency %q = %d, %v; want %d, true", region, got, ok, want)
		}
	}
	// The exported entry point resolves the same premium from the same raw usage.
	u := OpenAIUsage{InputTokens: 1000, CachedInputTokens: 400, OutputTokens: 100, DataResidency: "eu"}
	viaAPI, ok := Cost("gpt-5.4", u)
	direct, ok2 := price(table()["gpt-5.4"][TierStandard], u)
	if !ok || !ok2 || viaAPI != direct {
		t.Fatalf("Cost(eu) = %d, %v; internal = %d, %v; want equal and ok", viaAPI, ok, direct, ok2)
	}
}

// TestUpliftScalesTierRates encodes multiplier × tier composition: premiums
// scale the TIER-RESOLVED rates, while tier selection stays a pure function
// of token counts. 272001 prompt tokens exceed gpt-5.4's 272k tier; served
// from the eu host: (1×5e-6 + 272000×5e-7) × 1.1 = $0.1496055 → 14960.55
// nls → ceil 14961.
func TestUpliftScalesTierRates(t *testing.T) {
	got, ok := price(mustParse(t, gpt54Spec), OpenAIUsage{InputTokens: 272001, CachedInputTokens: 272000, DataResidency: "eu"})
	if !ok || got != 14961 {
		t.Fatalf("cost = %d, %v; want 14961, true", got, ok)
	}
}

// TestCeilingRounding encodes the rounding rule: the final total — and only
// the final total — is ceiling-rounded. Any non-zero usage costs at least
// 1 nls: a single cache-read token at $3e-7 is 0.03 nls, never free. And
// sub-nls tokens accumulate exactly: 400 cache reads at 0.03 nls each are
// exactly 12 nls — per-token flooring would truncate to 0, per-token ceiling
// would inflate to 400.
func TestCeilingRounding(t *testing.T) {
	r := mustParse(t, sonnet45Spec)
	if got, ok := cost(r, components{cacheRead: 1}, premium{}); !ok || got != 1 {
		t.Fatalf("1 cache-read token = %d, %v; want 1 (ceil of 0.03)", got, ok)
	}
	if got, ok := cost(r, components{cacheRead: 400}, premium{}); !ok || got != 12 {
		t.Fatalf("400 cache-read tokens = %d, %v; want exactly 12", got, ok)
	}
	if got, ok := cost(r, components{}, premium{}); !ok || got != 0 {
		t.Fatalf("zero usage = %d, %v; want 0, true (ceiling never invents cost)", got, ok)
	}
}

// TestZeroUsageIsFree encodes the same at the exported surface.
func TestZeroUsageIsFree(t *testing.T) {
	if got, ok := Cost("claude-opus-4-8", ClaudeUsage{}); !ok || got != 0 {
		t.Fatalf("Cost(zero) = %d, %v; want 0, true", got, ok)
	}
	if got, ok := Cost("gpt-5", OpenAIUsage{}); !ok || got != 0 {
		t.Fatalf("Cost(zero) = %d, %v; want 0, true", got, ok)
	}
}

// TestUnknownModel encodes the fail-loud contract: a model the module cannot
// price returns ok=false — callers must never mistake "unknown" for "free".
// LiteLLM's "sample_spec" documentation row and its zero-rate entries must
// not resolve either.
func TestUnknownModel(t *testing.T) {
	for _, model := range []string{"no-such-model", "sample_spec", ""} {
		if _, ok := Cost(model, ClaudeUsage{InputTokens: 1}); ok {
			t.Errorf("Cost(%q) resolved; want ok=false", model)
		}
		if _, ok := Cost(model, OpenAIUsage{InputTokens: 1}); ok {
			t.Errorf("Cost(%q) resolved; want ok=false", model)
		}
		if _, ok := RatesFor(model, TierStandard); ok {
			t.Errorf("RatesFor(%q) resolved; want ok=false", model)
		}
	}
}

// TestTableInvariants encodes structural guarantees over every parsed model
// and service tier: the standard tier always present (it anchors table
// membership), only known service tiers stored, priceable base rates, and
// context-window tiers strictly ascending with positive thresholds and
// priceable rates. A violation means parseModel accepted garbage.
func TestTableInvariants(t *testing.T) {
	for model, tiers := range table() {
		if _, ok := tiers[TierStandard]; !ok {
			t.Errorf("%s: no standard tier in table", model)
		}
		for tier, r := range tiers {
			if _, ok := serviceSuffixes[tier]; !ok {
				t.Errorf("%s: unknown service tier %q in table", model, tier)
			}
			if !r.Base.priceable() {
				t.Errorf("%s/%s: unpriceable base rates in table", model, tier)
			}
			for i, ctx := range r.Tiers {
				if ctx.AbovePromptTokens <= 0 || (i > 0 && ctx.AbovePromptTokens <= r.Tiers[i-1].AbovePromptTokens) {
					t.Errorf("%s/%s: tier thresholds not strictly ascending and positive", model, tier)
				}
				if !ctx.priceable() {
					t.Errorf("%s/%s: unpriceable tier rates in table", model, tier)
				}
			}
			if r.Fast != nil && r.Fast.Sign() <= 0 {
				t.Errorf("%s/%s: non-positive fast multiplier", model, tier)
			}
			for _, m := range []map[string]*big.Rat{r.Geo, r.RegionalUplift} {
				for key, f := range m {
					if f == nil || f.Sign() <= 0 {
						t.Errorf("%s/%s: non-positive multiplier %q", model, tier, key)
					}
				}
			}
		}
	}
}

// TestVendoredDataCanaries pins known shapes of the vendored data: a handful
// of stable LiteLLM keys must resolve, and the long-context tiers known to
// exist — claude-sonnet-4-5's 200k (the 1M-context beta) and gpt-5.4's 272k —
// must be parsed. If this fails after a sync, either upstream restructured
// its schema (fix parseModel) or the canary models' pricing genuinely
// vanished (verify before merging). Consumers separately test that every
// model id THEY bill resolves, so a dropped model they depend on fails their
// build, not silently bills zero.
func TestVendoredDataCanaries(t *testing.T) {
	for _, model := range []string{"claude-opus-5", "claude-opus-4-8", "claude-haiku-4-5", "gpt-5.4", "gpt-4o", "codex-mini-latest"} {
		if _, ok := RatesFor(model, TierStandard); !ok {
			t.Errorf("canary %s no longer resolves", model)
		}
	}
	for model, want := range map[string]int64{"claude-sonnet-4-5": 200000, "gpt-5.4": 272000} {
		r, ok := RatesFor(model, TierStandard)
		if !ok {
			t.Errorf("%s no longer resolves", model)
			continue
		}
		if !slices.ContainsFunc(r.Tiers, func(tier Tier) bool { return tier.AbovePromptTokens == want }) {
			t.Errorf("%s: no %d tier parsed from vendored data (tiers: %+v)", model, want, r.Tiers)
		}
	}
	// Service-tier canaries: known tier shapes at the vendored commit. gpt-5.5
	// publishes both flex and priority; gpt-5.3-codex publishes priority but
	// no flex — a sync that grows it flex rates should update this canary, a
	// sync that drops gpt-5.5's tiers demands scrutiny before merging.
	for model, tiers := range map[string]map[ServiceTier]bool{
		"gpt-5.5":       {TierFlex: true, TierPriority: true},
		"gpt-5.4-mini":  {TierFlex: true, TierPriority: true},
		"gpt-5.3-codex": {TierFlex: false, TierPriority: true},
	} {
		for tier, want := range tiers {
			if _, ok := RatesFor(model, tier); ok != want {
				t.Errorf("%s at %s: resolves=%v, want %v", model, tier, ok, want)
			}
		}
	}
	// Multiplier canaries: opus-4-8's fast/us factors and gpt-5.4's regional
	// uplifts must parse. If these vanish after a sync, fast-mode and pinned-
	// geo usage starts failing (never underbilling) and residency usage
	// starts billing standard — verify against upstream before merging.
	if r, ok := RatesFor("claude-opus-4-8", TierStandard); !ok || r.Fast == nil || r.Fast.Cmp(big.NewRat(2, 1)) != 0 || r.Geo["us"] == nil {
		t.Errorf("claude-opus-4-8 multipliers = fast %v, geo %v; want fast 2 and a us factor", r.Fast, r.Geo)
	}
	// claude-opus-5 launched 2026-07-24 at $5/M input, $25/M output — the same
	// rates as opus-4-8 (Anthropic's announcement; corroborated by the LiteLLM
	// entry vendored here). Pin the exact base rates and the fast/us multipliers
	// so a sync that renames the key or shifts the price fails loudly rather
	// than silently underbilling a model we serve as the default Opus.
	if r, ok := RatesFor("claude-opus-5", TierStandard); !ok ||
		r.Base.Input.Cmp(big.NewRat(5, 1_000_000)) != 0 ||
		r.Base.Output.Cmp(big.NewRat(25, 1_000_000)) != 0 ||
		r.Fast == nil || r.Fast.Cmp(big.NewRat(2, 1)) != 0 || r.Geo["us"] == nil {
		t.Errorf("claude-opus-5 rates = %+v (fast %v, geo %v); want $5/M in, $25/M out, fast 2, a us factor", r.Base, r.Fast, r.Geo)
	}
	if r, ok := RatesFor("gpt-5.4", TierStandard); !ok || r.RegionalUplift["eu"] == nil || r.RegionalUplift["us"] == nil {
		t.Errorf("gpt-5.4 regional uplifts = %v; want eu and us factors", r.RegionalUplift)
	}
}

// TestCostMatchesRatesFor encodes that the exported views never disagree:
// Cost prices exactly what RatesFor-derived math predicts — for
// EVERY model and service tier in the table, including across context-tier
// boundaries and on models that fail (unpriced components must fail
// identically). The ClaudeUsage shape covers the standard tier (Claude has
// no tier field and always bills standard); the OpenAIUsage shape covers
// every tier via [OpenAIUsage.ServiceTier].
func TestCostMatchesRatesFor(t *testing.T) {
	for model, tiers := range table() {
		for tier, r := range tiers {
			for _, c := range []components{
				{input: 3117, cacheRead: 41775, cacheCreation: 2048, cacheCreation1h: 512, output: 977},
				{input: 300000, cacheRead: 41775, output: 977}, // above any 200k/272k tier
			} {
				want, wantOK := cost(r, c, premium{})
				if tier == TierStandard {
					got, gotOK := Cost(model, ClaudeUsage{
						InputTokens:                c.input,
						CacheReadInputTokens:       c.cacheRead,
						CacheCreationInputTokens:   c.cacheCreation + c.cacheCreation1h, // raw API total
						CacheCreation1hInputTokens: c.cacheCreation1h,
						OutputTokens:               c.output,
					})
					if got != want || gotOK != wantOK {
						t.Errorf("%s/%s %+v: Cost = %d, %v; RatesFor math = %d, %v", model, tier, c, got, gotOK, want, wantOK)
					}
				}
				if c.cacheCreation == 0 && c.cacheCreation1h == 0 {
					gotOA, okOA := Cost(model, OpenAIUsage{InputTokens: c.input + c.cacheRead, CachedInputTokens: c.cacheRead, OutputTokens: c.output, ServiceTier: tier})
					if gotOA != want || okOA != wantOK {
						t.Errorf("%s/%s %+v: Cost(OpenAIUsage) = %d, %v; RatesFor math = %d, %v", model, tier, c, gotOA, okOA, want, wantOK)
					}
				}
			}
		}
	}
}

// TestRatesForReturnsCopies encodes that the shared rate table is immutable:
// a caller mutating the rats RatesFor returned — base, tier, or multiplier —
// must not corrupt later costs of the same model.
func TestRatesForReturnsCopies(t *testing.T) {
	for _, tc := range []struct {
		model string
		u     Usage
	}{
		{"claude-sonnet-4-5", ClaudeUsage{InputTokens: 250000}},                                  // premium-tier rats exercised
		{"claude-opus-4-8", ClaudeUsage{InputTokens: 1000, Speed: "fast", InferenceGeo: "us"}},   // fast and geo rats exercised
		{"gpt-5.4", OpenAIUsage{InputTokens: 1000, CachedInputTokens: 400, DataResidency: "eu"}}, // uplift rat exercised
	} {
		before, _ := Cost(tc.model, tc.u)
		r, _ := RatesFor(tc.model, TierStandard)
		all := []TierRates{r.Base}
		for _, tier := range r.Tiers {
			all = append(all, tier.TierRates)
		}
		for _, tr := range all {
			for _, rat := range []*big.Rat{tr.Input, tr.CacheRead, tr.CacheCreation, tr.CacheCreation1h, tr.Output} {
				if rat != nil {
					rat.SetInt64(999)
				}
			}
		}
		if r.Fast != nil {
			r.Fast.SetInt64(999)
		}
		for _, m := range []map[string]*big.Rat{r.Geo, r.RegionalUplift} {
			for _, rat := range m {
				rat.SetInt64(999)
			}
		}
		if after, _ := Cost(tc.model, tc.u); after != before {
			t.Fatalf("%s: mutating RatesFor result changed Cost: %d -> %d", tc.model, before, after)
		}
	}
}

// TestInvalidUsagePanics encodes that impossible raw counts are caller bugs,
// rejected loudly rather than priced as negative or double-counted bills:
// negative counts on either provider shape, and OpenAI cached > input
// (cached is a subset of input).
func TestInvalidUsagePanics(t *testing.T) {
	mustPanic := func(name string, f func()) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Errorf("%s did not panic", name)
			}
		}()
		f()
	}
	mustPanic("claude negative input", func() { Cost("claude-opus-4-8", ClaudeUsage{InputTokens: -1}) })
	mustPanic("claude negative cache read", func() { Cost("claude-opus-4-8", ClaudeUsage{CacheReadInputTokens: -1}) })
	mustPanic("claude negative cache write", func() { Cost("claude-opus-4-8", ClaudeUsage{CacheCreationInputTokens: -1}) })
	mustPanic("claude negative 1h cache write", func() {
		Cost("claude-opus-4-8", ClaudeUsage{CacheCreationInputTokens: 1, CacheCreation1hInputTokens: -1})
	})
	mustPanic("claude 1h writes exceed total writes", func() {
		Cost("claude-opus-4-8", ClaudeUsage{CacheCreationInputTokens: 10, CacheCreation1hInputTokens: 11})
	})
	mustPanic("claude negative output", func() { Cost("claude-opus-4-8", ClaudeUsage{OutputTokens: -1}) })
	mustPanic("openai negative input", func() { Cost("gpt-5", OpenAIUsage{InputTokens: -1}) })
	mustPanic("openai negative cached", func() { Cost("gpt-5", OpenAIUsage{CachedInputTokens: -1}) })
	mustPanic("openai negative output", func() { Cost("gpt-5", OpenAIUsage{OutputTokens: -1}) })
	mustPanic("openai cached exceeds input", func() { Cost("gpt-5", OpenAIUsage{InputTokens: 10, CachedInputTokens: 11}) })
	// Usage validation must not be masked by the model lookup: impossible
	// counts panic even when the model is unknown.
	mustPanic("unknown model, negative input", func() { Cost("no-such-model", ClaudeUsage{InputTokens: -1}) })
}

// TestRatParsesDecimalLiteralsExactly encodes the no-float64 requirement:
// rates come out of the JSON as exact rationals of their decimal literals.
// 2.5e-7 is exactly 1/4,000,000 — a value float64 cannot represent.
func TestRatParsesDecimalLiteralsExactly(t *testing.T) {
	fields := map[string]json.RawMessage{"x": json.RawMessage("2.5e-7")}
	if r := ratField(fields, "x"); r == nil || r.Cmp(big.NewRat(1, 4_000_000)) != 0 {
		t.Fatalf("ratField(2.5e-7) = %v, want exactly 1/4000000", r)
	}
	if r := ratField(fields, "absent"); r != nil {
		t.Fatalf("ratField(absent) = %v, want nil", r)
	}
}
