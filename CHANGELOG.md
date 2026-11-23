# Changelog

All notable changes are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
uses [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-11-23

### Added

- Synthetic universe generator: companies with permanent ids, vendor security ids with ticker / name / exchange segments, daily close and volume; id reissues (with optional overlap), renames with and without a new id, mergers (acquirer reissued under a combined name), spin-offs (parent reissued, unadjusted price drop), delist and relist, bankruptcies, ticker reuse by a different company (including within 1-5 days of the ticker being freed), exchange transfers, vendor name spelling noise. Ground truth for every id, every true link and every look-alike pair.
- Name normalisation and fuzzy similarity (suffix spellings, case, '&', one-letter typos).
- Candidate blocking by date window, ticker and name token; identifier and price-continuity features (return jump in sigmas, volume ratio, volatility ratio, overlap return correlation).
- Rule baseline and logistic scorers on identifier, price and combined features, trained on separate seeds.
- Resolver: ambiguous band, one-to-one matching, union-find chains, stable resolved ids, point-in-time mapping table with `valid_from`, `valid_to` and `known_from`.
- Optional LLM adjudicator through a vendor-neutral httpx `/chat/completions` adapter with retry, a verdict parser, and a deterministic mock used in tests and the committed results.
- Evaluation: pairwise precision / recall / F1, chain purity and completeness, exact chains, errors by event type, development seeds (100-104) and ten held-out seeds (200-209).
- Downstream backtest: equal-weight and momentum strategies on truth-, resolver- (final and as-known), vendor-id-, ticker- and current-ticker-keyed histories, with wrong-company join and broken-history counts.
- SQLite output with as-of ticker queries (ids not yet linked come back as provisional entities) and entity-history queries; CLI (`generate`, `resolve`, `asof`, `history`, `bench`, `report`).
- Tests including a full rerun against the committed results, `constraints.txt` with exact library versions, CI on ubuntu (Python 3.11) with a secret scan.
