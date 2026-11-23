# Walkthrough

## 1. Offline benchmark (what CI runs)

```bash
pip install -c constraints.txt -e ".[dev]"
pytest -q           # about 55 s; includes a full rerun checked against results/*.json
make bench          # about 50 s; writes results/
```

`make bench` generates training universes (seeds 1-3) and fits the three logistic scorers, resolves the test universe (seed 100) with all five methods, repeats the scoring on seeds 101-104 and on the held-out seeds 200-209, runs the downstream backtest, writes the SQLite table and runs the as-of examples.

## 2. One universe, step by step in Python

```python
from secmaster.universe import UniverseConfig, generate_universe
from secmaster.pipeline import candidates_for, train_scorers, training_set, resolve_with
from secmaster.evaluate import evaluate
from secmaster.resolver import mapping_table
from secmaster.store import write_sqlite, as_of_ticker

uni = generate_universe(UniverseConfig(seed=100))
print(uni.stats()["seams"])                 # true links by event type

scorers = train_scorers(training_set(UniverseConfig()))   # seeds 1-3, never seed 100
cands, first = candidates_for(uni)          # blocking + features, one row per pair
res = resolve_with("combined", uni, cands, first, scorers)

m = evaluate(res, uni, cands)
print(m["pairwise_f1"], m["by_event"]["delist_relist"])

table = mapping_table(res, uni.master, uni.vids, uni.dates)
db = write_sqlite(table, "out/security_master.sqlite")
print(as_of_ticker(db, "OVSK", "2017-01-09"))
```

`cands` has the evidence for every pair (`name_sim`, `ticker_eq`, `z`, `volume_ratio`, `overlap_corr`, ...), which is the first place to look when a link is wrong.

## 3. Command line on vendor-shaped files

```bash
secmaster generate --seed 100 --out data
```

writes the vendor view (`master.csv`: vid, ticker, name, exchange, valid_from, valid_to; `prices.npz`: close and volume per vid and date) and, in separate files, the ground truth (`truth.csv`, `truth_seams.csv`, `truth_hard_negatives.csv`, `events.csv`). `resolve` reads only the vendor view:

```bash
secmaster resolve --data data --db out/security_master.sqlite --model results/scorer_combined.json
secmaster asof    --db out/security_master.sqlite --ticker OVSK --date 2017-01-09
secmaster asof    --db out/security_master.sqlite --ticker ISS --date 2012-08-20 --known-on 2012-08-20
secmaster history --db out/security_master.sqlite --id R00005 --known-on 2012-08-30
```

Without `--model`, `resolve` fits the combined scorer on seeds 1-3 first (a few seconds). To run your own data through it, write the same two files: one row per (id, ticker segment) in `master.csv`, and a close and volume matrix in `prices.npz`.

## 4. A real LLM on the ambiguous pairs

```bash
export SECMASTER_LLM_BASE_URL=http://localhost:8000/v1   # any /chat/completions server
export SECMASTER_LLM_MODEL=your-model-name
export SECMASTER_LLM_API_KEY=...                         # only if the server needs one
secmaster resolve --data data --db out/sm_llm.sqlite --model results/scorer_combined.json --llm openai
```

The command prints how many pairs were sent, how many replies could not be parsed and how many decisions flipped. To score it against ground truth, use the Python route in section 2 with a custom adjudicator:

```python
from secmaster.adjudicator import LLMAdjudicator, OpenAICompatibleChat
res = resolve_with("combined_llm", uni, cands, first, scorers,
                   adjudicator=LLMAdjudicator(OpenAICompatibleChat()))
print(evaluate(res, uni, cands)["pairwise_f1"])
```

On seed 100 that is 72 calls. The bar to beat is the combined model's own accuracy on those pairs, 0.694, and its F1 without the adjudicator, 0.975 (0.976 on the held-out seeds 200-209).

## 5. Changing the universe

Every rate is a field of `UniverseConfig` (`p_ticker_reuse`, `p_delist_relist`, `p_overlap`, ...). For example, to see how the resolver copes with more relistings:

```python
cfg = UniverseConfig(seed=100, p_delist_relist=0.2)
```

If you change the defaults, rerun `make bench` and commit the new results; the reproduction test compares against the committed JSON.
