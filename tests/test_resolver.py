import numpy as np
import pandas as pd

from secmaster.evaluate import evaluate
from secmaster.pipeline import resolve_with
from secmaster.resolver import ResolverConfig, UnionFind, mapping_table, resolve


def test_union_find_chains():
    uf = UnionFind(6)
    uf.union(0, 1)
    uf.union(1, 2)
    uf.union(4, 5)
    lab = uf.labels()
    assert lab[0] == lab[1] == lab[2] and lab[4] == lab[5]
    assert len(set(lab)) == 3
    assert not uf.union(2, 0)


def _cands(pairs):
    return pd.DataFrame({"u": [p[0] for p in pairs], "v": [p[1] for p in pairs],
                         "old_vid": [f"V{p[0]}" for p in pairs],
                         "new_vid": [f"V{p[1]}" for p in pairs],
                         "new_first": [10 * p[1] for p in pairs]})


def test_one_to_one_keeps_best_predecessor():
    # The acquirer (0) and the target (1) both point at the new id (2).
    c = _cands([(0, 2), (1, 2)])
    res = resolve(c, np.array([0.9, 0.7]), 3, np.array([0, 0, 20]), 100, "t")
    assert list(zip(res.links.old_vid, res.links.new_vid)) == [("V0", "V2")]
    assert res.cluster[0] == res.cluster[2] != res.cluster[1]


def test_without_one_to_one_both_merge():
    c = _cands([(0, 2), (1, 2)])
    res = resolve(c, np.array([0.9, 0.7]), 3, np.array([0, 0, 20]), 100, "t",
                  ResolverConfig(one_to_one=False))
    assert len(set(res.cluster)) == 1


def test_resolved_ids_are_stable_and_ordered():
    c = _cands([(0, 1)])
    res = resolve(c, np.array([0.9]), 3, np.array([0, 15, 5]), 100, "t")
    assert res.resolved_id == ["R00001", "R00001", "R00002"]


def test_combined_beats_rule_on_small_universe(uni, cands, scorers):
    c, first = cands
    rule = evaluate(resolve_with("rule", uni, c, first, scorers), uni, c)
    comb = evaluate(resolve_with("combined", uni, c, first, scorers), uni, c)
    assert comb["pairwise_f1"] > rule["pairwise_f1"]
    assert comb["pairwise_f1"] > 0.9


def test_mapping_table_point_in_time(uni, cands, scorers):
    c, first = cands
    res = resolve_with("combined", uni, c, first, scorers)
    t = mapping_table(res, uni.master, uni.vids, uni.dates)
    assert len(t) == len(uni.master)
    assert (t.valid_from <= t.valid_to).all()
    assert (t.known_from >= t.valid_from).all()
    linked = set(res.links.new_vid)
    later = t[t.vid.isin(linked)].groupby("vid").agg(vf=("valid_from", "min"),
                                                     kf=("known_from", "min"))
    assert (later.kf > later.vf).all()  # a continuation is known only after its window
    assert t.groupby("resolved_id").vid.nunique().max() >= 2
