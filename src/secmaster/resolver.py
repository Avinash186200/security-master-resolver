"""From scored candidate pairs to entity chains and a point-in-time mapping table.

1. Score every candidate pair (rule or logistic model).
2. Pairs with a score inside the ambiguous band go to the adjudicator, if one is given.
3. One-to-one: each vid gets at most one successor and one predecessor. Accepted pairs
   are taken in descending score order (this settles a merger, where the acquirer's and
   the target's old ids both point at the new id). The rule baseline skips this step.
4. Union-find over accepted links gives entity chains.
5. The mapping table has one row per (vid, ticker segment) with the resolved entity,
   `valid_from` / `valid_to`, and `known_from`: the first date the link that put this
   vid in its chain could have been decided (POST_WINDOW business days after the vid
   starts, because the features read that many days of the new series).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .adjudicator import LLMAdjudicator, evidence_card
from .candidates import POST_WINDOW


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True

    def labels(self) -> np.ndarray:
        return np.array([self.find(i) for i in range(len(self.parent))])


@dataclass
class Resolution:
    method: str
    links: pd.DataFrame            # accepted links: old_vid, new_vid, score, source, known_from_idx
    cluster: np.ndarray            # cluster label per vid index
    resolved_id: list[str]         # resolved entity id per vid index
    adjudication: dict = field(default_factory=dict)
    adjudicated: dict = field(default_factory=dict)   # candidate row -> score after the LLM


@dataclass
class ResolverConfig:
    threshold: float = 0.5
    band: tuple[float, float] = (0.2, 0.8)
    one_to_one: bool = True


def resolve(cands: pd.DataFrame, scores: np.ndarray, n_vids: int, first_idx: np.ndarray,
            n_days: int, method: str, cfg: ResolverConfig | None = None,
            adjudicator: LLMAdjudicator | None = None, dates=None) -> Resolution:
    cfg = cfg or ResolverConfig()
    s = np.asarray(scores, dtype=float).copy()
    source = np.array(["model"] * len(s), dtype=object)
    adj, adjudicated = {}, {}
    if adjudicator is not None and len(s):
        lo, hi = cfg.band
        amb = np.where((s >= lo) & (s <= hi))[0]
        flips_to_yes = flips_to_no = 0
        for i in amb:
            same, conf, _ = adjudicator.judge(evidence_card(cands.iloc[i], dates))
            if same is None:
                continue
            new = 0.5 + 0.5 * conf if same else 0.5 - 0.5 * conf
            new = min(max(new, 0.0), 1.0)
            if (new >= cfg.threshold) != (s[i] >= cfg.threshold):
                flips_to_yes += int(new >= cfg.threshold)
                flips_to_no += int(new < cfg.threshold)
            s[i] = new if new != cfg.threshold else new + 1e-9
            source[i] = "llm"
            adjudicated[int(i)] = float(s[i])
        adj = {"ambiguous_pairs": int(len(amb)), "calls": adjudicator.calls,
               "parse_failures": adjudicator.parse_failures,
               "flipped_to_link": flips_to_yes, "flipped_to_no_link": flips_to_no}

    acc = np.where(s >= cfg.threshold)[0]
    acc = acc[np.lexsort((cands.v.to_numpy()[acc], cands.u.to_numpy()[acc], -s[acc]))]
    has_succ, has_pred = set(), set()
    keep = []
    for i in acc:
        u, v = int(cands.u.iat[i]), int(cands.v.iat[i])
        if cfg.one_to_one and (u in has_succ or v in has_pred):
            continue
        has_succ.add(u)
        has_pred.add(v)
        keep.append(i)

    uf = UnionFind(n_vids)
    for i in keep:
        uf.union(int(cands.u.iat[i]), int(cands.v.iat[i]))
    labels = uf.labels()

    # Stable resolved ids: clusters numbered by their earliest first trading day.
    first_of = {}
    for i, lab in enumerate(labels):
        key = (int(first_idx[i]), i)
        if lab not in first_of or key < first_of[lab]:
            first_of[lab] = key
    order = sorted(first_of, key=lambda lab: first_of[lab])
    rid = {lab: f"R{k + 1:05d}" for k, lab in enumerate(order)}

    k = cands.iloc[keep] if keep else cands.iloc[0:0]
    links = pd.DataFrame({
        "old_vid": k.old_vid.to_numpy(), "new_vid": k.new_vid.to_numpy(),
        "u": k.u.to_numpy(), "v": k.v.to_numpy(),
        "score": s[keep] if keep else np.zeros(0),
        "source": source[keep] if keep else np.zeros(0, dtype=object),
        "known_from_idx": np.minimum(k.new_first.to_numpy() + POST_WINDOW, n_days - 1),
    })
    return Resolution(method, links, labels, [rid[lab] for lab in labels], adj, adjudicated)


def mapping_table(res: Resolution, master: pd.DataFrame, vids: list[str],
                  dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Point-in-time security master: one row per (vid, ticker segment)."""
    vix = {v: i for i, v in enumerate(vids)}
    known = dict(zip(res.links.new_vid, res.links.known_from_idx))
    m = master.copy()
    m.insert(0, "resolved_id", [res.resolved_id[vix[v]] for v in m.vid])
    first = m.groupby("vid").from_idx.transform("min")
    kidx = np.array([known.get(v, f) for v, f in zip(m.vid, first)])
    m["known_from"] = dates[np.maximum(kidx, m.from_idx.to_numpy())]
    m = m.sort_values(["resolved_id", "from_idx", "vid"]).reset_index(drop=True)
    return m[["resolved_id", "vid", "ticker", "name", "exchange", "valid_from", "valid_to",
              "known_from"]]
