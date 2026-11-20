// Package listprice is the inference price book: what a model call is
// charged, and the rate card that explains it.
//
// A call's list price is its provider cost — what [llmcost] prices for the
// same usage — plus [MarkupBasisPoints], rounded up to a whole [llmcost.Nls]:
//
//	price = ceil(cost × (10000 + MarkupBasisPoints) / 10000)
//
// A call's cost and price are all-in — tokens plus any provider-hosted web
// search fees — and [Price] carries the web search part of each separately.
//
// [Of] prices a call; [Markup] re-derives a price from a recorded cost.
// [Priceable] answers before the call whether a model, service tier and
// speed have a list price at all, so a caller can refuse instead of serving
// a call it cannot charge for.
//
// # Sources
//
// A price comes from the vendored LiteLLM rates ([SourceLLMCost]), from a
// curated override restating rates LiteLLM lacks or gets wrong
// ([SourceOverride]), or — for response-only dimensions such as an inference
// geo, a 1-hour cache write, or web searches the model has no rate for — from
// the rest of the call's rates with that dimension dropped ([SourceDegraded]),
// because a call that already happened is never free.
// Request-time modes (model, service tier, fast mode) never degrade.
//
// # Published models
//
// GPT-6.2 Astra, Astra Law, and Sol publish standard rates; Astra and Sol
// also publish priority rates. Unpublished service tiers remain unpriceable.
//
// The book publishes a curated set of models by name, each mapped to the
// llmcost key that prices it. [Book] enumerates the rate card — published
// models first, then every other first-party key llmcost prices — as exact
// USD per million tokens. A model has one list price whichever provider
// serves it, set by the cost of its pricing provider ([Rate.PricedBy]). [Costs]
// enumerates what each provider charges for the same models, so a price can
// be read beside every cost it could be served at.
//
// # Versioning
//
// [CurrentVersion] identifies the price book a binary carries: this module's
// version and the LiteLLM commit its rates were vendored from. Callers record
// it beside every price so a charge can be traced to the rates behind it.
package listprice
