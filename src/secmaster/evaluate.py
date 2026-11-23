"""Scoring a resolution against the generator's ground truth."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .universe import HARD_NEGATIVES, LINK_EVENTS, Universe


def _pairs(sizes) -> int:
    return int(sum(n * (n - 1) // 2 for n in sizes))


def cluster_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """Pairwise precision/recall/F1 over all vid pairs, chain purity and completeness,
    and the share of multi-vid true entities rebuilt exactly."""
    pred, true = np.asarray(pred), np.asarray(true)
    n = len(pred)
    joint = Counter(zip(pred.tolist(), true.tolist()))
    pc, tc = Counter(pred.tolist()), Counter(true.tolist())
    tp = _pairs(joint.values())
    pp, tp_all = _pairs(pc.values()), _pairs(tc.values())
    precision = tp / pp if pp else 1.0
    recall = tp / tp_all if tp_all else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    best_t: dict = {}
    best_p: dict = {}
    for (p, t), c in joint.items():
        best_t[p] = max(best_t.get(p, 0), c)
        best_p[t] = max(best_p.get(t, 0), c)
    purity = sum(best_t.values()) / n
    completeness = sum(best_p.values()) / n
    multi = [t for t, c in tc.items() if c > 1]
    exact = 0
    for t in multi:
        ps = {p for (p, tt) in joint if tt == t}
        if len(ps) == 1 and pc[next(iter(ps))] == tc[t]:
            exact += 1
    return {"pairwise_precision": precision, "pairwise_recall": recall, "pairwise_f1": f1,
            "true_pairs": tp_all, "predicted_pairs": pp, "correct_pairs": tp,
            "chain_purity": purity, "chain_completeness": completeness,
            "multi_vid_entities": len(multi),
            "exact_chains": exact / len(multi) if multi else 1.0}


def evaluate(res, uni: Universe, cands: pd.DataFrame) -> dict:
    vix = uni.vid_index
    perm = uni.perm_of
    true = np.array([perm[v] for v in uni.vids])
    out = cluster_metrics(res.cluster, true)
    links = res.links
    # A link is "correct" if both ids belong to one company. That includes skip links
    # (A -> C where the true chain is A -> B -> C): they put ids in the right chain.
    # Link recall counts only direct seams, so a skip link does not inflate it.
    ok = np.array([perm[a] == perm[b] for a, b in zip(links.old_vid, links.new_vid)], dtype=bool)
    seam_set = set(zip(uni.seams.old_vid, uni.seams.new_vid))
    direct = np.array([(a, b) in seam_set for a, b in zip(links.old_vid, links.new_vid)],
                      dtype=bool)
    cand_pairs = list(zip(cands.old_vid, cands.new_vid))
    out.update({
        "candidates": int(len(cands)),
        "seams_in_candidates": int(sum(p in seam_set for p in cand_pairs)),
        "same_entity_pairs_in_candidates": int(sum(perm[a] == perm[b] for a, b in cand_pairs)),
        "accepted_links": int(len(links)), "correct_links": int(ok.sum()),
        "seam_links": int(direct.sum()), "skip_links": int((ok & ~direct).sum()),
        "link_precision": float(ok.mean()) if len(ok) else 1.0,
        "link_recall": float(direct.sum() / len(uni.seams)) if len(uni.seams) else 1.0,
    })
    if res.adjudication:
        out["adjudication"] = res.adjudication
    out["by_event"] = error_breakdown(res, uni, cands)
    return out


def error_breakdown(res, uni: Universe, cands: pd.DataFrame) -> dict:
    """Seams: was each true continuation recovered (both vids in one chain)?
    Hard negatives: was each look-alike pair kept apart? Only hard negatives that pass
    blocking (are candidates) are scored; the rest are counted as `not_candidates`."""
    vix = uni.vid_index
    lab = res.cluster
    cand_set = set(zip(cands.old_vid, cands.new_vid))
    out = {}
    for ev in LINK_EVENTS:
        s = uni.seams[uni.seams.event == ev]
        rec = [lab[vix[a]] == lab[vix[b]] for a, b in zip(s.old_vid, s.new_vid)]
        out[ev] = {"type": "true link", "n": int(len(s)), "recovered": int(sum(rec)),
                   "recall": float(np.mean(rec)) if rec else None}
    for kind in HARD_NEGATIVES:
        h = uni.hard_negatives[uni.hard_negatives.kind == kind]
        inc = [(a, b) in cand_set for a, b in zip(h.old_vid, h.new_vid)]
        hh = h[inc]
        merged = [lab[vix[a]] == lab[vix[b]] for a, b in zip(hh.old_vid, hh.new_vid)]
        out[kind] = {"type": "look-alike", "n": int(len(hh)),
                     "not_candidates": int(len(h) - len(hh)),
                     "wrongly_merged": int(sum(merged)),
                     "false_merge_rate": float(np.mean(merged)) if merged else None}
    perm = uni.perm_of
    hn = set(zip(uni.hard_negatives.old_vid, uni.hard_negatives.new_vid))
    other = [(a, b) for a, b in zip(res.links.old_vid, res.links.new_vid)
             if perm[a] != perm[b] and (a, b) not in hn]
    out["other_false_links"] = {"type": "coincidence", "n": len(other)}
    return out
