"""Downstream check: the same two strategies on differently keyed price histories.

A *scheme* says which price series each vendor row belongs to:

* truth             ground-truth permanent entity (the reference)
* resolved          the resolver's entity chains, final mapping
* resolved_as_known the resolver's chains, using at each rebalance only links whose
                    `known_from` date has passed (no look-ahead in the mapping itself)
* vid_keyed         one series per vendor id; every reissue looks like a delisting
* ticker_keyed      one series per ticker string, every company that ever used it
* ticker_current    ticker_keyed, but the universe is today's ticker list: the history
                    a data team gets if it pulls "all current tickers" from a ticker-keyed
                    store. Dead companies are missing: survivorship, which is look-ahead
                    in the universe (it uses the knowledge that a company survives).

When two rows of one series trade on the same day, the vid that started first wins.

Strategies, rebalanced at each month end from the end of the first year:
* EW   equal weight in every eligible series
* MOM  equal weight in the top fifth by 12-1 month momentum
A series is eligible at t if it trades at t, at t-21 and at t-252 (forward fill of up
to 5 days). A held series that stops trading goes to cash for the rest of the month.
Positions at t use prices up to t only; returns are earned from t+1.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .resolver import UnionFind
from .universe import Universe

LOOKBACK, SKIP, FFILL = 252, 21, 5


def key_matrix(active: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    return np.where(active, mapping[:, None], -1)


def ticker_key_matrix(uni: Universe) -> np.ndarray:
    codes = {t: i for i, t in enumerate(sorted(uni.master.ticker.unique()))}
    K = np.full(uni.close.shape, -1, dtype=np.int64)
    vix = uni.vid_index
    for row in uni.master.itertuples(index=False):
        K[vix[row.vid], row.from_idx:row.to_idx + 1] = codes[row.ticker]
    K[~np.isfinite(uni.close)] = -1
    return K


def build_series(close: np.ndarray, K: np.ndarray, rank: np.ndarray, lo: int, hi: int):
    """(S, SRC, keys): price per (series, day) in [lo, hi) and the vid that supplied it."""
    sub = K[:, lo:hi]
    v_idx, j_idx = np.nonzero(sub >= 0)
    keys = sub[v_idx, j_idx]
    order = np.lexsort((rank[v_idx], j_idx, keys))
    keys, j_idx, v_idx = keys[order], j_idx[order], v_idx[order]
    first = np.ones(len(keys), dtype=bool)
    first[1:] = (keys[1:] != keys[:-1]) | (j_idx[1:] != j_idx[:-1])
    keys, j_idx, v_idx = keys[first], j_idx[first], v_idx[first]
    ukeys, row = np.unique(keys, return_inverse=True)
    S = np.full((len(ukeys), hi - lo), np.nan)
    SRC = np.full((len(ukeys), hi - lo), -1, dtype=np.int64)
    S[row, j_idx] = close[v_idx, lo + j_idx]
    SRC[row, j_idx] = v_idx
    return S, SRC, ukeys


def _ffill(S: np.ndarray, limit: int) -> np.ndarray:
    return pd.DataFrame(S).T.ffill(limit=limit).T.to_numpy()


def _start_rank(close: np.ndarray) -> np.ndarray:
    """Rank of each vid by its first trading day (earlier start = lower rank)."""
    first = np.argmax(np.isfinite(close), axis=1)
    return np.argsort(np.argsort(first, kind="stable"), kind="stable")


def month_ends(dates: pd.DatetimeIndex) -> np.ndarray:
    s = pd.Series(np.arange(len(dates)), index=dates)
    return s.groupby([dates.year, dates.month]).max().to_numpy()


def run_strategies(uni: Universe, keys_at: Callable[[int], np.ndarray],
                   allowed_keys: np.ndarray | None = None) -> dict:
    close = uni.close
    T = close.shape[1]
    rank = _start_rank(close)
    reb = month_ends(uni.dates)
    reb = reb[reb >= LOOKBACK]
    daily = {"EW": np.zeros(T), "MOM": np.zeros(T)}
    names = {"EW": [], "MOM": []}
    for t, t_next in zip(reb[:-1], reb[1:]):
        K = keys_at(int(t))
        lo, hi = t - LOOKBACK, t_next + 1
        S, _, keys = build_series(close, K, rank, lo, hi)
        Sf = _ffill(S, FFILL)
        j = t - lo
        elig = np.isfinite(Sf[:, j]) & np.isfinite(Sf[:, 0]) & np.isfinite(Sf[:, j - SKIP])
        if allowed_keys is not None:
            elig &= np.isin(keys, allowed_keys)
        idx = np.where(elig)[0]
        if len(idx) == 0:
            continue
        r = Sf[idx, j + 1:] / Sf[idx, j:-1] - 1.0
        alive = np.cumprod(np.isfinite(r), axis=1).astype(bool)
        r0 = np.where(alive, r, 0.0)
        mom = Sf[idx, j - SKIP] / Sf[idx, 0] - 1.0
        n_top = max(1, int(round(0.2 * len(idx))))
        top = np.lexsort((keys[idx], -mom))[:n_top]
        daily["EW"][t + 1:t_next + 1] = r0.mean(axis=0)
        daily["MOM"][t + 1:t_next + 1] = r0[top].mean(axis=0)
        names["EW"].append(len(idx))
        names["MOM"].append(n_top)
    span = slice(int(reb[0]) + 1, int(reb[-1]) + 1)
    return {k: (daily[k][span], float(np.mean(names[k]))) for k in daily}


def series_stats(uni: Universe, K: np.ndarray, K_truth: np.ndarray) -> dict:
    """False joins: day-to-day steps inside one series where the supplying company changes.
    Broken histories: steps inside one true company's history where the series changes."""
    close = uni.close
    T = close.shape[1]
    rank = _start_rank(close)
    perm_codes = pd.factorize(uni.truth.perm_id)[0]
    _, SRC, _ = build_series(close, K, rank, 0, T)
    false_joins = 0
    for row in SRC:
        src = row[row >= 0]
        p = perm_codes[src]
        false_joins += int((p[1:] != p[:-1]).sum())
    _, SRC_t, _ = build_series(close, K_truth, rank, 0, T)
    breaks = 0
    for row in SRC_t:
        days = np.where(row >= 0)[0]
        k = K[row[days], days]
        breaks += int((k[1:] != k[:-1]).sum())
    return {"false_joins": false_joins, "broken_histories": breaks}


def _summary(r: np.ndarray, ref: np.ndarray | None, avg_names: float) -> dict:
    ann = float(r.mean() * 252)
    vol = float(r.std(ddof=1) * np.sqrt(252))
    out = {"ann_return": ann, "ann_vol": vol, "sharpe": ann / vol if vol > 0 else 0.0,
           "avg_names": avg_names}
    if ref is not None:
        d = r - ref
        out["return_error_vs_truth"] = float(d.mean() * 252)
        out["tracking_error_vs_truth"] = float(d.std(ddof=1) * np.sqrt(252))
    return out


def downstream_backtest(uni: Universe, resolutions: dict) -> dict:
    """resolutions: name -> Resolution. Each gets a final and (if it has links) an as-known run."""
    active = np.isfinite(uni.close)
    n = len(uni.vids)
    perm_codes = pd.factorize(uni.truth.perm_id)[0]
    K_truth = key_matrix(active, perm_codes)
    K_vid = key_matrix(active, np.arange(n))
    K_tick = ticker_key_matrix(uni)
    current = np.unique(K_tick[:, -1][K_tick[:, -1] >= 0])

    schemes: list[tuple[str, Callable, np.ndarray | None, np.ndarray]] = [
        ("truth", lambda t: K_truth, None, K_truth)]
    for name, res in resolutions.items():
        K_res = key_matrix(active, np.asarray(res.cluster))
        schemes.append((f"resolved_{name}", lambda t, K=K_res: K, None, K_res))
        links = res.links

        def as_known(t, links=links):
            uf = UnionFind(n)
            for u, v, k in zip(links.u, links.v, links.known_from_idx):
                if k <= t:
                    uf.union(int(u), int(v))
            return key_matrix(active, uf.labels())
        schemes.append((f"resolved_{name}_as_known", as_known, None, K_res))
    schemes += [("vid_keyed", lambda t: K_vid, None, K_vid),
                ("ticker_keyed", lambda t: K_tick, None, K_tick),
                ("ticker_current", lambda t: K_tick, current, K_tick)]

    runs = {name: run_strategies(uni, fn, allowed) for name, fn, allowed, _ in schemes}
    out = {}
    for name, _, _, K_final in schemes:
        row = {s: _summary(runs[name][s][0], None if name == "truth" else runs["truth"][s][0],
                           runs[name][s][1]) for s in ("EW", "MOM")}
        row.update(series_stats(uni, K_final, K_truth))
        out[name] = row
    out["_notes"] = {
        "current_tickers": int(len(current)),
        "as_known_series_stats": "computed on the final mapping",
    }
    return out
