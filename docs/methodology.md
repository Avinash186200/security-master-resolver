# Methodology

The rules I followed so the numbers mean something, and the places where they could go wrong.

## 1. Every number comes from a file

`python -m secmaster.cli bench` writes `results/universe.json`, `resolution.json`, `seeds.json`, `backtest.json`, `asof.json` and the trained combined scorer `scorer_combined.json`, then renders `results/RESULTS.md` from the JSON. The README tables are copied from `RESULTS.md`.

**Seeds.** Training: 1, 2, 3. Development: 100 (the detailed tables) and 101-104. Held-out: 200-209, added after an independent review and never looked at while building the generator, the features or the thresholds. The README headline is the held-out mean. `seeds.json` has both sets.

`tests/test_bench.py::test_committed_results_reproduce` reruns the full benchmark in CI and checks every field against the committed JSON (floats to 1e-6 relative). Nothing in the benchmark reads the clock or the network.

**Library versions.** The results were produced on Python 3.11 with numpy 2.4.4, pandas 3.0.2, scikit-learn 1.9.0 and httpx 0.28.1 (full list in `constraints.txt`). CI installs with `pip install -c constraints.txt -e ".[dev]"`. I ran the test suite, including the reproduction test, on Windows and in a `python:3.11-slim` Linux container. Other library versions could move the logistic fit in the last digits and flip a pair that sits on the threshold, so the reproduction test is only promised for these versions.

## 2. The synthetic universe

`universe.py` simulates 10 x 252 business days from 2012-01-02. The default has 400 companies at the start and about 30 listings a year (initial public offerings), plus spin-off children.

**Entities, ids and segments.** Every company has a permanent id (`E0001`...). The vendor keys data by a security id (`V` + six random digits, so the id says nothing about the company or the order of events). Each id has one or more segments, each with a ticker, a vendor-spelled name and an exchange.

**Events and what they do to ids.**

| event | rate (default) | vendor ids | prices |
|---|---|---|---|
| id reissue | 5% per company-year | old id ends, new id starts, same ticker, new name spelling; 15% change exchange; 30% overlap 2-8 days | continuous; during an overlap the new id's close differs by about 0.1% and its volume is 30% |
| rename | 4% per company-year | 60% new id with new ticker and new name, 40% same id with a new segment | continuous |
| merger | 12% of companies are targets | target's id ends; 40% of acquirers are reissued under a combined name (50% with a new ticker) | target jumps 15-35% sixty days before the close, then is pinned to the deal |
| spin-off | 6% of companies | a new company starts with the parent's first name word and often a ticker prefix; 40% of parents are reissued on the day | parent drops by the spun fraction (10-35%), not adjusted; the child starts at a share of the parent's value |
| delist and relist | 4% of companies | id ends, 40-300 days later a new id starts; 50% try to reclaim the old ticker; 60% change exchange | the company's value keeps moving while delisted; volume is about half after relisting |
| bankruptcy | 10% | id ends, ticker freed | about -1% a day over the last 120 days |
| ticker reuse | 30% of new ticker assignments | a freed ticker goes to a different company. 40% of reuse attempts first look for a ticker freed 1-5 business days earlier (`ticker_reuse_quick`); otherwise, or if there is none, one freed 20-400 days earlier (`ticker_reuse`) | independent price |
| exchange transfer | 2% per company-year | new segment, same id | none |

Events for one company are at least 60 days apart and at least 70 days after its first trading day, so every link has a 60-day price window before it.

**Prices.** Daily log returns are `alpha + beta * market + sigma * noise`, with beta 0.6-1.4 and sigma 1.0-2.5% per company. Volume is lognormal around a company-level mean.

**Ground truth.** The generator writes the true company of every id, the seams (old id, new id, event) and the hard negatives: a ticker reused by a different company (quick or slow), a merger target against the acquirer's new id, a spin-off child against the parent's id before the spin.

## 3. The resolver

**Blocking.** For each id A, candidates B start between 10 days before and 300 days after A's last day, start after A starts and end after A ends. If B starts more than 5 days after A ends, A's last ticker must equal B's first ticker or they must share a name token (first four letters after normalisation). On seed 100 this gives 6,885 candidate pairs. They contain all 470 true links (`seams_in_candidates` in `resolution.json`) and 500 same-company pairs in all (`same_entity_pairs_in_candidates`); the other 30 skip an id (A to C where the chain is A to B to C).

**Features.** All pairs use A's last segment and B's first segment.

| feature | definition |
|---|---|
| gap, log gap, overlap, short gap | business days from A's last day to B's first; negative when both trade |
| ticker equal, ticker similarity | exact and character similarity |
| name similarity | names lower-cased, '&' to 'and', punctuation and legal suffixes removed; the larger of a sorted-token sequence ratio and a token Jaccard that counts two tokens as equal at 0.8 character similarity (handles one-letter typos) |
| exchange equal | |
| abs z | `log(B first close / A last close) / (sigma_A * sqrt(gap))`, sigma_A = std of A's last 60 log returns; for overlaps, the median log price ratio over the overlap / sigma_A; capped at 10 |
| volume ratio | log of B's median volume over its first 20 days / A's over its last 20 days |
| volatility ratio | log of B's first-20-day volatility / A's last-20-day volatility |
| overlap correlation | correlation of daily returns while both trade (4+ days), else 0 with a missing flag |

**Scorers.** A rule baseline (same ticker, gap of at most 7 days, |z| at most 3, no one-to-one step), and logistic regressions (standardised features, C = 1) on identifier features, price features, or both. The logistic models are fitted on all candidates from seeds 1, 2 and 3, labelled 1 when both ids belong to the same company. That includes skip pairs: linking one still puts the ids in the right chain, and one-to-one matching usually prefers the direct seam because its price evidence is stronger. Seeds 100-104 and 200-209 are never used for fitting.

**Decision.** Score at least 0.5 is a link. Pairs scoring 0.2-0.8 go to the adjudicator if one is given; its verdict replaces the score (`0.5 + confidence/2` for "same", `0.5 - confidence/2` for "different"). Accepted pairs are taken in descending score; a pair is kept only if A has no successor and B no predecessor yet. Union-find over kept links gives chains; resolved ids are numbered by each chain's first trading day.

**known_from.** The features read B's first 20 days, so a link is decided 20 business days after B starts. The mapping table stores that as `known_from` for B's rows. Queries can filter on it, and the as-known backtest uses a link only from that date.

## 4. Metrics

- **Pairwise precision / recall / F1.** Over all pairs of ids that are in the same predicted chain vs the same true company.
- **Chain purity.** For each predicted chain, the share of its ids that belong to its most common true company, weighted by size. **Completeness** is the same the other way round. Both include single-id companies, so they start high.
- **Link precision and recall.** Precision: accepted links whose two ids belong to one company (skip links count as correct). Recall: true seams accepted as direct links, over all seams; skip links do not count, so recall cannot be inflated by them. `resolution.json` reports `seam_links` and `skip_links` separately.
- **Exact chains.** The share of true companies with 2+ ids that are rebuilt exactly.
- **Errors by event type.** A true link counts as recovered if both ids end in one chain (even through another link). A look-alike counts as wrongly merged if both ids end in one chain; look-alikes that fail blocking are counted separately (`not_candidates` in the JSON). Other false links are accepted links between different companies that are not registered look-alikes.

## 5. The LLM adjudicator

The adjudicator sees an evidence card per pair: both tickers, names, exchanges, last and first trade dates, the gap, the percentage price change and its z, the volume ratio and the overlap correlation. It does not see ids, other candidates or ground truth. It replies with JSON `{"same_entity": bool, "confidence": 0..1, "reason": "..."}`; prose replies are parsed for "same company" / "different company"; anything else is a parse failure and the model score stands.

The committed results use a deterministic mock behind the real httpx adapter. Its rules: same ticker +1; name similarity at least 0.8 +1.5, at least 0.5 +0.5, below 0.3 -1; same exchange +0.3; overlap correlation above 0.9 +1.5; |z| below 3 +1, above 8 -1.5; a same-ticker, same-name drop of 8-45% within 2 days +1.5 (reads as a distribution); volume ratio outside 0.25-4 -0.5; gap over 60 days -0.3; "same" if the total is at least 1.5. Then, chosen by a hash of the card, 8% of answers are flipped against the mock's own rule and given confidence 0.85 (to imitate confident mistakes), and 2% come back as prose. These numbers describe the mock only.

## 6. The downstream backtest

`backtest.py` builds price series under six keyings (see the module docstring); the resolver keyings run for both the combined and the rule resolver, final and as-known, so the table has eight rows. It runs two monthly strategies: equal weight in every eligible series, and equal weight in the top fifth by 12-1 month momentum. A series is eligible at a month end t if it has a price at t, t-21 and t-252 (forward fill up to 5 days). Daily returns from t+1 to the next month end; a held series that stops trading goes to cash for the rest of the month.

**Leakage checks.** Positions at t use prices up to t. `tests/test_backtest.py` scales all prices after a date and checks that every return before the preceding rebalance is unchanged. The as-known run rebuilds chains at every rebalance from links with `known_from <= t`.

**Join and break counts.** A wrong-company join is a day-to-day step inside one series where the supplying company changes. A broken history is a step inside one true company's history where the series changes. The as-known rows report counts on the final mapping.

## 7. What could still be wrong

- The generator and the scorer were written by the same person; the generator's events are the ones I thought of.
- The training and test seeds share all generator settings, so the test is in-distribution. The held-out seeds protect against my choices overfitting seed 100, not against the generator being easier than real data.
- The ambiguous band and the threshold were set before scoring, but I chose the feature list after looking at seed-100 feature distributions while building the generator. The held-out seeds 200-209 give a combined F1 of 0.976 (sd 0.008, range 0.966-0.988), and seed 100 gives 0.975, so seed 100 sits at the held-out mean rather than at its high end. Before quick ticker reuse was added to the generator, the review measured 0.978 on the same held-out seeds.
