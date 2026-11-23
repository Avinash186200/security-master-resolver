"""Train scorers on training universes, then resolve a universe with every method."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .adjudicator import LLMAdjudicator, mock_adjudicator
from .candidates import build_candidates, label_candidates
from .resolver import Resolution, ResolverConfig, resolve
from .scorer import FEATURE_SETS, LogisticScorer, rule_scores
from .universe import Universe, UniverseConfig, generate_universe

TRAIN_SEEDS = (1, 2, 3)
TEST_SEED = 100
EXTRA_TEST_SEEDS = (101, 102, 103, 104)
# Never looked at while building the generator, the features or the thresholds.
HOLDOUT_SEEDS = tuple(range(200, 210))
METHODS = ["rule", "ids_only", "price_only", "combined", "combined_llm_mock"]


def candidates_for(uni: Universe) -> tuple[pd.DataFrame, np.ndarray]:
    cands, summ = build_candidates(uni.master, uni.vids, uni.close, uni.volume)
    return cands, np.array([s.first for s in summ])


def training_set(base: UniverseConfig, seeds=TRAIN_SEEDS) -> pd.DataFrame:
    parts = []
    for s in seeds:
        uni = generate_universe(replace(base, seed=s))
        c, _ = candidates_for(uni)
        c["y"] = label_candidates(c, uni.truth)
        c["train_seed"] = s
        parts.append(c)
    return pd.concat(parts, ignore_index=True)


def train_scorers(train: pd.DataFrame) -> dict[str, LogisticScorer]:
    return {k: LogisticScorer(k, f).fit(train, train.y.to_numpy()) for k, f in FEATURE_SETS.items()}


def resolve_with(method: str, uni: Universe, cands: pd.DataFrame, first: np.ndarray,
                 scorers: dict, adjudicator: LLMAdjudicator | None = None,
                 cfg: ResolverConfig | None = None) -> Resolution:
    n, T = len(uni.vids), len(uni.dates)
    if method == "rule":
        return resolve(cands, rule_scores(cands), n, first, T, method,
                       ResolverConfig(one_to_one=False))
    if method == "combined_llm_mock":
        return resolve(cands, scorers["combined"].predict(cands), n, first, T, method, cfg,
                       adjudicator=adjudicator or mock_adjudicator(), dates=uni.dates)
    if method == "combined_llm":
        return resolve(cands, scorers["combined"].predict(cands), n, first, T, method, cfg,
                       adjudicator=adjudicator, dates=uni.dates)
    return resolve(cands, scorers[method].predict(cands), n, first, T, method, cfg)
