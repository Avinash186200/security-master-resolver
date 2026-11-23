"""The full benchmark: writes results/*.json and renders results/RESULTS.md from them."""
from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import downstream_backtest
from .candidates import label_candidates
from .evaluate import evaluate
from .pipeline import (EXTRA_TEST_SEEDS, HOLDOUT_SEEDS, METHODS, TEST_SEED, TRAIN_SEEDS, candidates_for,
                       resolve_with, train_scorers, training_set)
from .resolver import ResolverConfig, mapping_table
from .store import as_of_ticker, entity_history, write_sqlite
from .universe import UniverseConfig, generate_universe

SUMMARY_KEYS = ["pairwise_precision", "pairwise_recall", "pairwise_f1", "chain_purity",
                "chain_completeness", "exact_chains"]


def _r(x, nd=4):
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, dict):
        return {k: _r(v, nd) for k, v in x.items()}
    if isinstance(x, list):
        return [_r(v, nd) for v in x]
    return x


def _band_analysis(res, cands, y, scores, cfg: ResolverConfig) -> dict:
    """On the ambiguous pairs: how often the model threshold vs the adjudicator is right."""
    idx = sorted(res.adjudicated)
    if not idx:
        return {"n": 0}
    yy = y[idx]
    model = (scores[idx] >= cfg.threshold).astype(int)
    llm = (np.array([res.adjudicated[i] for i in idx]) >= cfg.threshold).astype(int)
    return {"n": len(idx), "true_links": int(yy.sum()),
            "model_threshold_accuracy": float((model == yy).mean()),
            "adjudicator_accuracy": float((llm == yy).mean()),
            "adjudicator_fixed": int(((model != yy) & (llm == yy)).sum()),
            "adjudicator_broke": int(((model == yy) & (llm != yy)).sum())}


def _asof_example(table, uni, path) -> dict:
    """A reused ticker queried on a date inside each holder's life, and one entity history."""
    db = write_sqlite(table, path)
    perm = uni.perm_of
    first_tk = uni.master.groupby("vid").ticker.first()
    hn = uni.hard_negatives
    ex = None
    for a, b in zip(hn.old_vid[hn.kind == "ticker_reuse"], hn.new_vid[hn.kind == "ticker_reuse"]):
        if perm[a] != perm[b] and ((table.vid == a) & (table.ticker == first_tk[b])).any():
            ex = (a, b, first_tk[b])
            break
    out = {}
    if ex:
        a, b, tk = ex
        ra = table[(table.vid == a) & (table.ticker == tk)].iloc[0]
        rb = table[(table.vid == b) & (table.ticker == tk)].iloc[0]
        d1 = str(ra.valid_to.date())
        mid = rb.valid_from + (rb.valid_to - rb.valid_from) / 2
        d2 = str(pd.offsets.BDay().rollback(mid.normalize()).date())
        out["ticker"] = tk
        out["queries"] = []
        for d in (d1, d2):
            q = as_of_ticker(db, tk, d)
            out["queries"].append({"as_of": d, "rows": q.to_dict(orient="records")})
    sizes = table.groupby("resolved_id").vid.nunique()
    rid = sizes[sizes >= 3].index[0] if (sizes >= 3).any() else sizes.idxmax()
    h = entity_history(db, rid)
    out["entity"] = {"resolved_id": rid, "rows": h.to_dict(orient="records")}
    later = h[h.vid != h.vid.iloc[0]]
    k = later.known_from.min() if len(later) else h.known_from.iloc[0]
    early = str(np.datetime64(k) - np.timedelta64(1, "D"))
    out["entity_known_on"] = {"known_on": early,
                              "rows": entity_history(db, rid, early).to_dict(orient="records")}
    return out


def run(out_dir, base: UniverseConfig | None = None, train_seeds=TRAIN_SEEDS,
        test_seed=TEST_SEED, extra_seeds=EXTRA_TEST_SEEDS, holdout_seeds=HOLDOUT_SEEDS,
        db_path=None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = base or UniverseConfig()
    cfg = ResolverConfig()

    train = training_set(base, train_seeds)
    scorers = train_scorers(train)

    # ---------------------------------------------------------------- test universe
    uni = generate_universe(replace(base, seed=test_seed))
    cands, first = candidates_for(uni)
    y = label_candidates(cands, uni.truth)
    resolutions, metrics = {}, {}
    for m in METHODS:
        resolutions[m] = resolve_with(m, uni, cands, first, scorers, cfg=cfg)
        metrics[m] = evaluate(resolutions[m], uni, cands)
    band = _band_analysis(resolutions["combined_llm_mock"], cands, y,
                          scorers["combined"].predict(cands), cfg)
    resolution = {
        "test_seed": test_seed, "train_seeds": list(train_seeds),
        "train_candidates": int(len(train)), "train_true_links": int(train.y.sum()),
        "resolver_config": {"threshold": cfg.threshold, "band": list(cfg.band)},
        "methods": metrics, "llm_band": band,
        "combined_coefficients": dict(zip(scorers["combined"].features,
                                          scorers["combined"].model.coef_[0].tolist())),
    }
    universe = {"config": asdict(replace(base, seed=test_seed)), "test": uni.stats()}

    # ---------------------------------------------------------------- more seeds
    def seed_metrics(seed_list):
        per_seed = {m: [] for m in METHODS}
        for s in seed_list:
            if s == test_seed:
                for m in METHODS:
                    per_seed[m].append({k: metrics[m][k] for k in SUMMARY_KEYS})
                continue
            u2 = generate_universe(replace(base, seed=s))
            c2, f2 = candidates_for(u2)
            for m in METHODS:
                e = evaluate(resolve_with(m, u2, c2, f2, scorers, cfg=cfg), u2, c2)
                per_seed[m].append({k: e[k] for k in SUMMARY_KEYS})
        res = {"seeds": list(seed_list), "methods": {}}
        for m, rows in per_seed.items():
            res["methods"][m] = {k: {"mean": float(np.mean([r[k] for r in rows])),
                                     "sd": float(np.std([r[k] for r in rows], ddof=1))
                                     if len(rows) > 1 else 0.0,
                                     "min": float(np.min([r[k] for r in rows])),
                                     "max": float(np.max([r[k] for r in rows]))}
                                 for k in SUMMARY_KEYS}
        return res

    seeds = {"development": seed_metrics((test_seed,) + tuple(extra_seeds)),
             "holdout": seed_metrics(tuple(holdout_seeds))}

    # ---------------------------------------------------------------- downstream
    backtest = downstream_backtest(uni, {"combined": resolutions["combined"],
                                         "rule": resolutions["rule"]})

    # ---------------------------------------------------------------- table + as-of
    table = mapping_table(resolutions["combined"], uni.master, uni.vids, uni.dates)
    asof = _asof_example(table, uni, db_path or (out / "security_master.sqlite"))
    asof["table_rows"] = int(len(table))
    asof["resolved_entities"] = int(table.resolved_id.nunique())

    results = {"universe": universe, "resolution": resolution, "seeds": seeds,
               "backtest": backtest, "asof": asof}
    for name, obj in results.items():
        (out / f"{name}.json").write_text(json.dumps(_r(obj), indent=1, default=str) + "\n",
                                          newline="\n")
    scorers["combined"].save(out / "scorer_combined.json")
    (out / "RESULTS.md").write_text(render(out), newline="\n")
    return results


# ==================================================================== RESULTS.md

def _pct(x, nd=1):
    return "n/a" if x is None else f"{100 * x:.{nd}f}%"


def _f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def render(out_dir) -> str:
    out = Path(out_dir)
    U = json.loads((out / "universe.json").read_text())
    R = json.loads((out / "resolution.json").read_text())
    S = json.loads((out / "seeds.json").read_text())
    B = json.loads((out / "backtest.json").read_text())
    A = json.loads((out / "asof.json").read_text())
    L = ["# Results", "",
         "All numbers come from synthetic data generated by this repo "
         "(`python -m secmaster.cli bench`).", ""]

    st = U["test"]
    L += ["## Test universe", "",
          f"Seed {R['test_seed']}: {st['days']} business days ({st['first_date']} to "
          f"{st['last_date']}), {st['entities']} companies, {st['vids']} vendor ids, "
          f"{st['entities_with_2plus_vids']} companies with 2+ ids. Candidates after blocking: "
          f"{R['methods']['combined']['candidates']}, containing "
          f"{R['methods']['combined']['seams_in_candidates']} of {sum(st['seams'].values())} "
          f"true links (seams) and {R['methods']['combined']['same_entity_pairs_in_candidates']} "
          "same-company pairs in all (the rest skip an id).", "",
          "| true links (seams) | n | look-alike pairs | n |", "|---|---|---|---|"]
    seams, hn = list(st["seams"].items()), list(st["hard_negatives"].items())
    for i in range(max(len(seams), len(hn))):
        a = seams[i] if i < len(seams) else ("", "")
        b = hn[i] if i < len(hn) else ("", "")
        L.append(f"| {a[0]} | {a[1]} | {b[0]} | {b[1]} |")
    L += ["", f"Scorers trained on seeds {R['train_seeds']} ({R['train_candidates']} candidate "
          f"pairs, {R['train_true_links']} true links). Threshold {R['resolver_config']['threshold']}, "
          f"ambiguous band {R['resolver_config']['band']}.", ""]

    L += ["## Resolution quality (test seed)", "",
          "| method | candidates | links accepted | link precision (same company) | "
          "link recall (direct seams) | pairwise P | "
          "pairwise R | pairwise F1 | chain purity | chain completeness | exact chains |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, e in R["methods"].items():
        L.append(f"| {m} | {e['candidates']} | {e['accepted_links']} ({e['skip_links']} skip) | "
                 f"{_f(e['link_precision'])} | "
                 f"{_f(e['link_recall'])} | {_f(e['pairwise_precision'])} | "
                 f"{_f(e['pairwise_recall'])} | **{_f(e['pairwise_f1'])}** | "
                 f"{_f(e['chain_purity'])} | {_f(e['chain_completeness'])} | "
                 f"{_f(e['exact_chains'])} |")

    L += ["", "## Errors by event type (test seed)", "",
          "True links: share recovered (both ids end up in one chain). Look-alikes: share "
          "wrongly merged, among those that pass blocking.", "",
          "| event | kind | n | " + " | ".join(R["methods"]) + " |",
          "|---|---|---|" + "---|" * len(R["methods"])]
    first = next(iter(R["methods"].values()))["by_event"]
    for ev, d in first.items():
        if ev == "other_false_links":
            continue
        cells = []
        for m, e in R["methods"].items():
            x = e["by_event"][ev]
            cells.append(_pct(x["recall"]) if d["type"] == "true link"
                         else _pct(x["false_merge_rate"]))
        L.append(f"| {ev} | {'recovered' if d['type'] == 'true link' else 'false merge'} | "
                 f"{d['n']} | " + " | ".join(cells) + " |")
    L.append("| other false links (coincidences) | count | | " + " | ".join(
        str(e["by_event"]["other_false_links"]["n"]) for e in R["methods"].values()) + " |")

    b = R["llm_band"]
    adj = R["methods"]["combined_llm_mock"].get("adjudication", {})
    L += ["", "## LLM adjudicator (deterministic mock) on the ambiguous band", ""]
    if b.get("n"):
        L += [f"{b['n']} pairs fell in the band ({b['true_links']} true links). "
              f"Parse failures: {adj.get('parse_failures', 0)}.", "",
              "| decision on the band | accuracy |", "|---|---|",
              f"| combined model, threshold {R['resolver_config']['threshold']} | "
              f"{_f(b['model_threshold_accuracy'])} |",
              f"| mock adjudicator | {_f(b['adjudicator_accuracy'])} |", "",
              f"The adjudicator fixed {b['adjudicator_fixed']} model decisions and broke "
              f"{b['adjudicator_broke']}."]

    for key, title in (("development", "Development seeds (seed 100 was looked at while "
                                        "choosing features)"),
                       ("holdout", "Held-out seeds (not looked at while building)")):
        block = S[key]
        if not block["seeds"]:
            continue
        L += ["", f"## {title}: {block['seeds']}", "",
              "| method | pairwise F1 mean (sd) | min | max | purity mean | completeness mean | "
              "exact chains mean |", "|---|---|---|---|---|---|---|"]
        for m, d in block["methods"].items():
            f1 = d["pairwise_f1"]
            L.append(f"| {m} | {_f(f1['mean'])} ({_f(f1['sd'])}) | {_f(f1['min'])} | "
                     f"{_f(f1['max'])} | {_f(d['chain_purity']['mean'])} | "
                     f"{_f(d['chain_completeness']['mean'])} | {_f(d['exact_chains']['mean'])} |")

    L += ["", "## Downstream backtest (test seed)", "",
          "Annualised; errors are against the truth-keyed history. EW = equal weight in every "
          "eligible series, MOM = top fifth by 12-1 momentum, monthly.", "",
          "| history keyed by | EW return | EW error | EW tracking error | MOM return | MOM error | "
          "MOM tracking error | avg names (EW) | wrong-company joins | broken histories |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for name, row in B.items():
        if name.startswith("_"):
            continue
        ew, mo = row["EW"], row["MOM"]
        L.append(f"| {name} | {_pct(ew['ann_return'], 2)} | "
                 f"{_pct(ew.get('return_error_vs_truth'), 2) if name != 'truth' else '-'} | "
                 f"{_pct(ew.get('tracking_error_vs_truth'), 2) if name != 'truth' else '-'} | "
                 f"{_pct(mo['ann_return'], 2)} | "
                 f"{_pct(mo.get('return_error_vs_truth'), 2) if name != 'truth' else '-'} | "
                 f"{_pct(mo.get('tracking_error_vs_truth'), 2) if name != 'truth' else '-'} | "
                 f"{ew['avg_names']:.0f} | {row['false_joins']} | {row['broken_histories']} |")
    L += ["", f"Current ticker list used by `ticker_current`: {B['_notes']['current_tickers']} "
          "tickers. Join and break counts for the as-known rows use the final mapping.", ""]

    L += ["## As-of query example", "",
          f"Mapping table: {A['table_rows']} rows, {A['resolved_entities']} resolved entities.", ""]
    if "ticker" in A:
        L.append(f"Ticker `{A['ticker']}` was reused by a different company:")
        L.append("")
        L.append("| as of | resolved_id | vid | name | valid_from | valid_to |")
        L.append("|---|---|---|---|---|---|")
        for q in A["queries"]:
            for r in q["rows"]:
                L.append(f"| {q['as_of']} | {r['resolved_id']} | {r['vid']} | {r['name']} | "
                         f"{r['valid_from']} | {r['valid_to']} |")
    L += ["", f"History of {A['entity']['resolved_id']}:", "",
          "| vid | ticker | name | exchange | valid_from | valid_to | known_from |",
          "|---|---|---|---|---|---|---|"]
    for r in A["entity"]["rows"]:
        L.append(f"| {r['vid']} | {r['ticker']} | {r['name']} | {r['exchange']} | "
                 f"{r['valid_from']} | {r['valid_to']} | {r['known_from']} |")
    L += ["", f"Same query with `known_on = {A['entity_known_on']['known_on']}` returns "
          f"{len(A['entity_known_on']['rows'])} of {len(A['entity']['rows'])} rows: the later "
          "ids are not yet linked on that date.", ""]
    return "\n".join(L)
