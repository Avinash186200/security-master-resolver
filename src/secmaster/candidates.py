"""Candidate pairs and their evidence.

A candidate is an ordered pair (u, v) where vid v starts near or after the end of vid u
and could be its continuation. Blocking keeps the pair count small:

* v starts between 10 business days before u ends and `max_gap` days after it,
  v starts after u starts and ends after u ends;
* if the gap is more than `short_gap` days, the pair must also share a ticker or a
  name token (first four letters), because price evidence is weak over long gaps.

Features use u's last 60 days and v's first `POST_WINDOW` days. A link decided from
these features is therefore only knowable `POST_WINDOW` business days after v starts;
the mapping table records that as `known_from`.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .names import name_similarity, ticker_similarity, token_keys

PRE_WINDOW = 60
POST_WINDOW = 20
MAX_OVERLAP = 10


@dataclass
class VidSummary:
    vid: str
    idx: int
    first: int
    last: int
    first_ticker: str
    last_ticker: str
    first_name: str
    last_name: str
    first_exchange: str
    last_exchange: str


def summarize(master: pd.DataFrame, vids: list[str]) -> list[VidSummary]:
    g = master.sort_values(["vid", "from_idx"]).groupby("vid", sort=False)
    first, last = g.first(), g.last()
    out = []
    for i, v in enumerate(vids):
        f, l_ = first.loc[v], last.loc[v]
        out.append(VidSummary(v, i, int(f.from_idx), int(l_.to_idx), f.ticker, l_.ticker,
                              f["name"], l_["name"], f.exchange, l_.exchange))
    return out


def generate_candidates(summ: list[VidSummary], max_gap: int = 300,
                        short_gap: int = 5) -> list[tuple[int, int]]:
    order = sorted(range(len(summ)), key=lambda i: summ[i].first)
    starts = [summ[i].first for i in order]
    keys = [token_keys(s.first_name) for s in summ]
    last_keys = [token_keys(s.last_name) for s in summ]
    pairs = []
    for u in summ:
        lo = bisect_left(starts, u.last - MAX_OVERLAP)
        hi = bisect_right(starts, u.last + max_gap)
        for k in range(lo, hi):
            v = summ[order[k]]
            if v.idx == u.idx or v.first <= u.first or v.last <= u.last:
                continue
            gap = v.first - u.last
            if gap > short_gap and u.last_ticker != v.first_ticker \
                    and not (last_keys[u.idx] & keys[v.idx]):
                continue
            pairs.append((u.idx, v.idx))
    return pairs


FEATURE_NAMES = [
    "gap", "log_gap", "overlap", "short_gap", "ticker_eq", "ticker_sim", "name_sim",
    "exchange_eq", "abs_z", "jump_pct", "abs_log_volume_ratio", "has_overlap_corr",
    "overlap_corr", "abs_log_vol_ratio",
]
ID_FEATURES = ["gap", "log_gap", "overlap", "short_gap", "ticker_eq", "ticker_sim", "name_sim",
               "exchange_eq"]
PRICE_FEATURES = ["gap", "log_gap", "overlap", "short_gap", "abs_z", "abs_log_volume_ratio",
                  "has_overlap_corr", "overlap_corr", "abs_log_vol_ratio"]
COMBINED_FEATURES = [f for f in FEATURE_NAMES if f != "jump_pct"]


def _logret(x: np.ndarray) -> np.ndarray:
    x = x[np.isfinite(x)]
    return np.diff(np.log(x)) if len(x) > 1 else np.array([])


def pair_features(u: VidSummary, v: VidSummary, close: np.ndarray, volume: np.ndarray) -> dict:
    gap = v.first - u.last
    pu = close[u.idx, max(u.first, u.last - PRE_WINDOW):u.last + 1]
    ru = _logret(pu)
    sigma = max(float(np.std(ru[-PRE_WINDOW:])) if len(ru) >= 10 else 0.02, 0.003)
    pv = close[v.idx, v.first:min(v.last, v.first + POST_WINDOW) + 1]
    rv = _logret(pv)
    corr = np.nan
    if gap <= 0:  # both trade on days v.first .. u.last
        a, b = close[u.idx, v.first:u.last + 1], close[v.idx, v.first:u.last + 1]
        jump = float(np.median(np.log(b / a)))
        z = jump / sigma
        if len(a) >= 4:
            ra, rb = np.diff(np.log(a)), np.diff(np.log(b))
            if np.std(ra) > 0 and np.std(rb) > 0:
                corr = float(np.corrcoef(ra, rb)[0, 1])
    else:
        jump = float(np.log(close[v.idx, v.first] / close[u.idx, u.last]))
        z = jump / (sigma * np.sqrt(gap))
    vu = np.nanmedian(volume[u.idx, max(u.first, u.last - POST_WINDOW):u.last + 1])
    vv = np.nanmedian(volume[v.idx, v.first:min(v.last, v.first + POST_WINDOW) + 1])
    sv = float(np.std(rv)) if len(rv) >= 5 else sigma
    sig20 = max(float(np.std(ru[-POST_WINDOW:])) if len(ru) >= 5 else sigma, 0.003)
    return {
        "gap": float(np.clip(gap, -MAX_OVERLAP, 300)),
        "log_gap": float(np.log1p(max(gap, 0))),
        "overlap": float(gap <= 0),
        "short_gap": float(gap <= 5),
        "ticker_eq": float(u.last_ticker == v.first_ticker),
        "ticker_sim": ticker_similarity(u.last_ticker, v.first_ticker),
        "name_sim": name_similarity(u.last_name, v.first_name),
        "exchange_eq": float(u.last_exchange == v.first_exchange),
        "abs_z": float(min(abs(z), 10.0)),
        "z": float(z),
        "jump_pct": float(np.expm1(jump)),
        "abs_log_volume_ratio": float(min(abs(np.log(max(vv, 1.0) / max(vu, 1.0))), 5.0)),
        "volume_ratio": float(max(vv, 1.0) / max(vu, 1.0)),
        "has_overlap_corr": float(np.isfinite(corr)),
        "overlap_corr": float(corr) if np.isfinite(corr) else 0.0,
        "abs_log_vol_ratio": float(min(abs(np.log(max(sv, 1e-4) / sig20)), 3.0)),
    }


def build_candidates(master: pd.DataFrame, vids: list[str], close: np.ndarray,
                     volume: np.ndarray, max_gap: int = 300) -> tuple[pd.DataFrame, list[VidSummary]]:
    """One row per candidate pair with identifiers, dates and every feature."""
    summ = summarize(master, vids)
    rows = []
    for ui, vi in generate_candidates(summ, max_gap=max_gap):
        u, v = summ[ui], summ[vi]
        f = pair_features(u, v, close, volume)
        f.update(u=ui, v=vi, old_vid=u.vid, new_vid=v.vid, old_last=u.last, new_first=v.first,
                 old_ticker=u.last_ticker, new_ticker=v.first_ticker, old_name=u.last_name,
                 new_name=v.first_name, old_exchange=u.last_exchange,
                 new_exchange=v.first_exchange, raw_gap=v.first - u.last)
        rows.append(f)
    cols = ["u", "v", "old_vid", "new_vid", "old_last", "new_first", "raw_gap", "old_ticker",
            "new_ticker", "old_name", "new_name", "old_exchange", "new_exchange", "z",
            "volume_ratio"] + FEATURE_NAMES
    return pd.DataFrame(rows, columns=cols), summ


def label_candidates(cands: pd.DataFrame, truth: pd.DataFrame) -> np.ndarray:
    perm = dict(zip(truth.vid, truth.perm_id))
    return np.array([perm[a] == perm[b] for a, b in zip(cands.old_vid, cands.new_vid)], dtype=int)
