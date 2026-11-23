import numpy as np

from secmaster.universe import generate_universe


def test_deterministic(small_cfg, uni):
    again = generate_universe(small_cfg)
    assert again.vids == uni.vids
    assert np.array_equal(np.nan_to_num(again.close), np.nan_to_num(uni.close))
    assert again.master.equals(uni.master)


def test_every_vid_has_one_entity_and_prices(uni):
    assert uni.truth.vid.is_unique
    assert set(uni.truth.vid) == set(uni.vids)
    active = np.isfinite(uni.close)
    assert active.any(axis=1).all()
    for row in uni.master.itertuples():
        i = uni.vid_index[row.vid]
        assert active[i, row.from_idx:row.to_idx + 1].all()
        assert active[i].sum() >= row.to_idx - row.from_idx + 1


def test_vid_segments_are_contiguous(uni):
    for _, g in uni.master.groupby("vid"):
        g = g.sort_values("from_idx")
        assert (g.from_idx.to_numpy()[1:] == g.to_idx.to_numpy()[:-1] + 1).all()


def test_seams_link_same_entity_in_time_order(uni):
    perm, vix = uni.perm_of, uni.vid_index
    first = np.argmax(np.isfinite(uni.close), axis=1)
    for a, b, _ in uni.seams.itertuples(index=False):
        assert perm[a] == perm[b]
        assert first[vix[b]] > first[vix[a]]


def test_hard_negatives_are_different_entities(uni):
    perm = uni.perm_of
    assert len(uni.hard_negatives) > 0
    for a, b, _ in uni.hard_negatives.itertuples(index=False):
        assert perm[a] != perm[b]


def test_ticker_held_by_one_company_per_day(uni):
    """Two different companies never trade under one ticker on the same day."""
    perm = uni.perm_of
    for ticker, g in uni.master.groupby("ticker"):
        rows = sorted((r.from_idx, r.to_idx, perm[r.vid]) for r in g.itertuples())
        for i, (a0, a1, pa) in enumerate(rows):
            for b0, b1, pb in rows[i + 1:]:
                if pa != pb:
                    assert b0 > a1 or a0 > b1, f"{ticker} held by two companies on one day"


def test_default_universe_has_every_event_type():
    st = generate_universe().stats()
    for ev in ["id_reissue", "rename_reissue", "merger_acquirer_reissue",
               "spinoff_parent_reissue", "delist_relist"]:
        assert st["seams"].get(ev, 0) > 0
    for kind in ["ticker_reuse", "ticker_reuse_quick", "merger_target", "spinoff_child"]:
        assert st["hard_negatives"].get(kind, 0) > 0
