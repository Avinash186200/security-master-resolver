# Contributing

Thanks for taking a look. Changes are judged on whether they keep the repo offline, point-in-time and honest about its numbers.

## Ground rules

1. **Everything in CI runs offline.** Tests and `make bench` must pass with no network and no API key. A real LLM is only ever reached through `--llm openai` or a custom adapter, never from a test.
2. **Numbers come from results.** Any number in a doc must be in a `results/*.json` file written by the benchmark. If you change code that affects results, rerun `make bench` and commit the new JSON and `RESULTS.md` with the change. CI fails if the committed JSON does not match a fresh run.
3. **Never fit on the test seeds.** Scorers are trained on seeds 1-3. Seed 100 and the stability seeds 101-104 are for scoring only. If you add a setting that needs tuning, tune it on the training seeds and say so in the docs.
4. **Point-in-time only.** Features may read at most `POST_WINDOW` days of the new series, and every link carries a `known_from` date. A new strategy needs a test showing that later prices cannot change earlier returns.
5. **Seeds stay fixed.** New generators take a seed.
6. **No secrets.** LLM settings come from environment variables only; `.env` is git-ignored.

## Set up

```bash
python -m venv .venv && source .venv/bin/activate
pip install -c constraints.txt -e ".[dev]"
pytest -q
```

## Adding an event type

1. Generate it in `universe.py`, and register its seams or look-alike pairs so the evaluation can score it.
2. Add it to `LINK_EVENTS` or `HARD_NEGATIVES` so it gets a row in the error breakdown.
3. Add a test that the default universe produces it.
4. Rerun the benchmark and update the docs from the new JSON.

## Adding a feature or scorer

Add the feature in `candidates.py` (and to a feature list), check it does not read past `POST_WINDOW` (there is a test for this), then rerun the benchmark.
