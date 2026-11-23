"""Command line: generate a universe, resolve it into SQLite, query it as of a date.

  python -m secmaster.cli generate --seed 100 --out data
  python -m secmaster.cli resolve  --data data --db out/security_master.sqlite [--llm mock|openai]
  python -m secmaster.cli asof     --db out/security_master.sqlite --ticker ABC --date 2016-03-31
  python -m secmaster.cli history  --db out/security_master.sqlite --id R00012 [--known-on 2015-01-01]
  python -m secmaster.cli bench    --out results
  python -m secmaster.cli report   --out results
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .universe import Universe, UniverseConfig, generate_universe


def save_universe(uni: Universe, out: Path) -> None:
    """Vendor view (master + prices) and, in separate files, the ground truth."""
    out.mkdir(parents=True, exist_ok=True)
    uni.master.drop(columns=["from_idx", "to_idx"]).to_csv(out / "master.csv", index=False)
    np.savez_compressed(out / "prices.npz", close=uni.close, volume=uni.volume,
                        vids=np.array(uni.vids, dtype=str),
                        dates=np.array(uni.dates.strftime("%Y-%m-%d"), dtype=str))
    uni.truth.to_csv(out / "truth.csv", index=False)
    uni.seams.to_csv(out / "truth_seams.csv", index=False)
    uni.hard_negatives.to_csv(out / "truth_hard_negatives.csv", index=False)
    uni.events.to_csv(out / "events.csv", index=False)
    (out / "config.json").write_text(json.dumps(asdict(uni.config), indent=1))


def load_vendor_data(path: Path):
    """(master with integer day columns, vids, dates, close, volume). Truth is not read."""
    z = np.load(path / "prices.npz", allow_pickle=False)
    dates = pd.DatetimeIndex(pd.to_datetime(z["dates"]))
    vids = [str(v) for v in z["vids"]]
    master = pd.read_csv(path / "master.csv", parse_dates=["valid_from", "valid_to"])
    pos = pd.Series(np.arange(len(dates)), index=dates)
    master["from_idx"] = pos[master.valid_from].to_numpy()
    master["to_idx"] = pos[master.valid_to].to_numpy()
    return master, vids, dates, z["close"], z["volume"]


def _print(df: pd.DataFrame) -> None:
    if df.empty:
        print("(no rows)")
    else:
        print(df.to_string(index=False))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="secmaster")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="write a synthetic universe")
    g.add_argument("--seed", type=int, default=100)
    g.add_argument("--n-initial", type=int, default=400)
    g.add_argument("--years", type=int, default=10)
    g.add_argument("--out", default="data")
    r = sub.add_parser("resolve", help="resolve vendor ids into a point-in-time table")
    r.add_argument("--data", default="data")
    r.add_argument("--db", default="out/security_master.sqlite")
    r.add_argument("--model", default=None, help="scorer JSON (default: train on seeds 1-3)")
    r.add_argument("--llm", choices=["none", "mock", "openai"], default="none",
                   help="adjudicate ambiguous pairs; 'openai' = any /chat/completions server")
    a = sub.add_parser("asof", help="which entity had this ticker on this date")
    a.add_argument("--db", required=True)
    a.add_argument("--ticker", required=True)
    a.add_argument("--date", required=True)
    a.add_argument("--known-on", default="9999-12-31")
    h = sub.add_parser("history", help="all rows of one resolved entity")
    h.add_argument("--db", required=True)
    h.add_argument("--id", required=True)
    h.add_argument("--known-on", default="9999-12-31")
    b = sub.add_parser("bench", help="full benchmark into results/")
    b.add_argument("--out", default="results")
    rp = sub.add_parser("report", help="re-render RESULTS.md from the JSON files")
    rp.add_argument("--out", default="results")
    args = p.parse_args(argv)
    try:
        return _run(args)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


def _run(args) -> int:

    if args.cmd == "generate":
        uni = generate_universe(UniverseConfig(seed=args.seed, n_initial=args.n_initial,
                                               years=args.years))
        save_universe(uni, Path(args.out))
        print(json.dumps(uni.stats(), indent=1))
    elif args.cmd == "resolve":
        from .adjudicator import LLMAdjudicator, OpenAICompatibleChat, mock_adjudicator
        from .candidates import build_candidates
        from .pipeline import train_scorers, training_set
        from .resolver import mapping_table, resolve
        from .scorer import LogisticScorer
        from .store import write_sqlite

        master, vids, dates, close, volume = load_vendor_data(Path(args.data))
        if args.model:
            scorer = LogisticScorer.load(args.model)
        else:
            scorer = train_scorers(training_set(UniverseConfig()))["combined"]
        cands, summ = build_candidates(master, vids, close, volume)
        adjudicator = None
        if args.llm == "mock":
            adjudicator = mock_adjudicator()
        elif args.llm == "openai":
            adjudicator = LLMAdjudicator(OpenAICompatibleChat())
        res = resolve(cands, scorer.predict(cands), len(vids), np.array([s.first for s in summ]),
                      len(dates), "combined", adjudicator=adjudicator, dates=dates)
        table = mapping_table(res, master, vids, dates)
        write_sqlite(table, args.db)
        print(json.dumps({"candidates": int(len(cands)), "links": int(len(res.links)),
                          "resolved_entities": int(table.resolved_id.nunique()),
                          "rows": int(len(table)), "db": str(args.db),
                          **({"adjudication": res.adjudication} if res.adjudication else {})},
                         indent=1))
    elif args.cmd == "asof":
        from .store import as_of_ticker
        _print(as_of_ticker(args.db, args.ticker, args.date, args.known_on))
    elif args.cmd == "history":
        from .store import entity_history
        _print(entity_history(args.db, args.id, args.known_on))
    elif args.cmd == "bench":
        from .bench import run
        run(args.out)
        print((Path(args.out) / "RESULTS.md").read_text())
    elif args.cmd == "report":
        from .bench import render
        (Path(args.out) / "RESULTS.md").write_text(render(args.out), newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
