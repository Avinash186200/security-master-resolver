import numpy as np

from secmaster.candidates import POST_WINDOW, VidSummary, label_candidates, pair_features


def _toy(gap, jump=0.0, n=200):
    T = 2 * n
    close = np.full((2, T), np.nan)
    volume = np.full((2, T), np.nan)
    rng = np.random.default_rng(0)
    p = 50 * np.exp(np.cumsum(0.01 * rng.standard_normal(T)))
    end_u = n - 1
    start_v = end_u + gap
    close[0, :end_u + 1] = p[:end_u + 1]
    close[1, start_v:] = p[start_v:] * np.exp(jump)
    volume[0, :end_u + 1] = 1e5
    volume[1, start_v:] = 1e5
    u = VidSummary("A", 0, 0, end_u, "ABC", "ABC", "Granite Energy Corp", "Granite Energy Corp",
                   "XNYS", "XNYS")
    v = VidSummary("B", 1, start_v, T - 1, "ABC", "ABC", "GRANITE ENERGY CORPORATION",
                   "Granite Energy Corp", "XNYS", "XNYS")
    return u, v, close, volume


def test_continuous_seam_has_small_z():
    u, v, c, vol = _toy(gap=1)
    f = pair_features(u, v, c, vol)
    assert f["abs_z"] < 3 and f["ticker_eq"] == 1 and f["name_sim"] > 0.9
    assert f["abs_log_volume_ratio"] < 1e-9


def test_price_jump_gives_large_z():
    u, v, c, vol = _toy(gap=1, jump=np.log(0.7))
    f = pair_features(u, v, c, vol)
    assert f["abs_z"] > 8
    assert abs(f["jump_pct"] - (-0.3)) < 0.05


def test_overlap_correlation():
    u, v, c, vol = _toy(gap=-5)
    f = pair_features(u, v, c, vol)
    assert f["overlap"] == 1 and f["has_overlap_corr"] == 1
    assert f["overlap_corr"] > 0.99 and f["abs_z"] < 1e-6


def test_features_do_not_read_beyond_post_window():
    u, v, c, vol = _toy(gap=1)
    f1 = pair_features(u, v, c, vol)
    c2 = c.copy()
    c2[1, v.first + POST_WINDOW + 1:] *= 3.0  # change prices after the window
    vol2 = vol.copy()
    vol2[1, v.first + POST_WINDOW + 1:] *= 3.0
    assert pair_features(u, v, c2, vol2) == f1


def test_candidates_cover_seams(uni, cands):
    c, _ = cands
    pairs = set(zip(c.old_vid, c.new_vid))
    covered = [(a, b) in pairs for a, b in zip(uni.seams.old_vid, uni.seams.new_vid)]
    assert all(covered)
    y = label_candidates(c, uni.truth)
    assert 0 < y.sum() < len(y)
