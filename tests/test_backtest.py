import copy

import numpy as np
import pandas as pd

from secmaster.backtest import (build_series, downstream_backtest, key_matrix, month_ends,
                                run_strategies, ticker_key_matrix, LOOKBACK)
from secmaster.pipeline import resolve_with


def test_build_series_first_started_wins():
    close = np.array([[1.0, 2.0, 3.0, np.nan], [np.nan, 20.0, 30.0, 40.0]])
    K = np.where(np.isfinite(close), 0, -1)
    S, SRC, keys = build_series(close, K, np.array([0, 1]), 0, 4)
    assert keys.tolist() == [0]
    assert S[0].tolist() == [1.0, 2.0, 3.0, 40.0]
    assert SRC[0].tolist() == [0, 0, 0, 1]


def test_schemes_on_small_universe(uni, cands, scorers):
    c, first = cands
    res = resolve_with("combined", uni, c, first, scorers)
    out = downstream_backtest(uni, {"combined": res})
    assert out["truth"]["false_joins"] == 0 and out["truth"]["broken_histories"] == 0
    assert out["vid_keyed"]["broken_histories"] == len(uni.seams)
    assert out["ticker_current"]["EW"]["avg_names"] < out["ticker_keyed"]["EW"]["avg_names"]
    assert out["resolved_combined"]["EW"]["tracking_error_vs_truth"] < \
        out["vid_keyed"]["EW"]["tracking_error_vs_truth"]


def test_returns_before_a_date_ignore_prices_after_it(uni):
    """Scaling every price after day `cut` must not change any return earned before the
    rebalance preceding `cut`: positions use past prices only."""
    active = np.isfinite(uni.close)
    K = key_matrix(active, pd.factorize(uni.truth.perm_id)[0])
    base = run_strategies(uni, lambda t: K)
    reb = month_ends(uni.dates)
    reb = reb[reb >= LOOKBACK]
    cut = int(reb[-4]) + 3
    u2 = copy.copy(uni)
    u2.close = uni.close.copy()
    rng = np.random.default_rng(3)
    u2.close[:, cut:] *= np.exp(rng.normal(0, 0.2, size=(u2.close.shape[0], 1)))
    alt = run_strategies(u2, lambda t: K)
    n_safe = int(reb[-4]) - int(reb[0])  # daily returns up to that rebalance
    for s in ("EW", "MOM"):
        assert np.allclose(base[s][0][:n_safe], alt[s][0][:n_safe])
    assert not np.allclose(base["EW"][0], alt["EW"][0])


def test_ticker_keys_follow_segments(uni):
    K = ticker_key_matrix(uni)
    assert ((K >= 0) == np.isfinite(uni.close)).all()
