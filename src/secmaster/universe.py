"""Synthetic equity universe with corporate events and known ground truth.

Every company is a permanent entity (`perm_id`). The data vendor keys its data by a
vendor security id (`vid`). A vid carries one or more (ticker, name, exchange)
segments and a daily close and volume series. Some events keep the vid, some end it
and start a new one for the same company, and some start a new company that looks
like an old one. The generator records which vids belong to which entity, which vid
continues which (seams), and which look-alike pairs are not the same company (hard
negatives), so a resolver can be scored exactly.

Events
------
link events (same company, new vid)  id_reissue, rename_reissue, merger_acquirer_reissue,
                                     spinoff_parent_reissue, delist_relist
same vid, new segment                rename, exchange_transfer
new companies                        ipo, spinoff (the child)
terminations                         bankrupt, acquired (merger target), taken_private
look-alikes (hard negatives)         ticker_reuse (freed 20-400 days before), ticker_reuse_quick
                                     (freed 1-5 days before), merger_target, spinoff_child
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .names import SECOND_WORDS, CompanyName, make_ticker, random_name, render_variant

EXCHANGES = ["XNYS", "XNAS", "XASE"]
EXCHANGE_P = [0.4, 0.5, 0.1]
LINK_EVENTS = ["id_reissue", "rename_reissue", "merger_acquirer_reissue",
               "spinoff_parent_reissue", "delist_relist"]
HARD_NEGATIVES = ["ticker_reuse", "ticker_reuse_quick", "merger_target", "spinoff_child"]


@dataclass
class UniverseConfig:
    seed: int = 100
    start: str = "2012-01-02"
    years: int = 10
    n_initial: int = 400
    ipo_per_year: float = 30.0
    p_bankrupt: float = 0.10
    p_acquired: float = 0.12
    p_acquirer_reissue: float = 0.4
    p_spinoff: float = 0.06
    p_spin_parent_reissue: float = 0.4
    reissue_rate: float = 0.05        # per entity-year
    rename_rate: float = 0.04         # per entity-year
    p_rename_reissue: float = 0.6
    p_delist_relist: float = 0.04
    exchange_transfer_rate: float = 0.02
    p_ticker_reuse: float = 0.3
    p_quick_reuse: float = 0.4        # of reuses: take a ticker freed 1-5 days ago if one exists
    p_overlap: float = 0.3            # share of reissues where both vids trade for a few days


@dataclass
class _Ent:
    perm: str
    start: int
    end: int
    fate: str
    origin: str
    name: CompanyName
    parent: str | None = None
    events: dict = field(default_factory=dict)       # day -> (kind, info dict)
    blocked: list = field(default_factory=list)      # (from, to) days with no other events
    announce: int | None = None                      # merger announcement day (targets)


@dataclass
class Universe:
    config: UniverseConfig
    dates: pd.DatetimeIndex
    vids: list[str]
    master: pd.DataFrame          # vid, ticker, name, exchange, valid_from, valid_to, from_idx, to_idx
    close: np.ndarray             # (n_vids, n_days), NaN when the vid does not trade
    volume: np.ndarray            # (n_vids, n_days)
    truth: pd.DataFrame           # vid, perm_id
    seams: pd.DataFrame           # old_vid, new_vid, event
    hard_negatives: pd.DataFrame  # old_vid, new_vid, kind
    events: pd.DataFrame          # date, event, perm_id, old_vid, new_vid, other_perm_id
    entities: pd.DataFrame        # perm_id, name, start, end, fate, origin

    @property
    def vid_index(self) -> dict[str, int]:
        return {v: i for i, v in enumerate(self.vids)}

    @property
    def perm_of(self) -> dict[str, str]:
        return dict(zip(self.truth.vid, self.truth.perm_id))

    def stats(self) -> dict:
        ev = self.events.event.value_counts().to_dict()
        multi = self.truth.groupby("perm_id").size()
        return {
            "days": int(len(self.dates)),
            "first_date": str(self.dates[0].date()),
            "last_date": str(self.dates[-1].date()),
            "entities": int(self.truth.perm_id.nunique()),
            "vids": int(len(self.vids)),
            "entities_with_2plus_vids": int((multi > 1).sum()),
            "segments": int(len(self.master)),
            "events": {k: int(ev.get(k, 0)) for k in sorted(ev)},
            "seams": {k: int(v) for k, v in self.seams.event.value_counts().sort_index().items()},
            "hard_negatives": {k: int(v) for k, v in
                               self.hard_negatives.kind.value_counts().sort_index().items()},
        }


class _Tickers:
    """Who holds which ticker, and which tickers are free to be reused."""

    def __init__(self, rng: np.random.Generator, p_reuse: float, p_quick: float = 0.0):
        self.rng, self.p_reuse, self.p_quick = rng, p_reuse, p_quick
        self.active: dict[str, str] = {}
        self.ever: set[str] = set()
        self.freed: list[tuple[str, int, str]] = []   # ticker, day freed, last vid

    def assign(self, day: int, perm: str, words, prefix: str = "") -> tuple[str, tuple | None]:
        """(ticker, (vid that last held it, look-alike kind) if reused, else None)."""
        if self.rng.random() < self.p_reuse:
            pool = []
            if self.rng.random() < self.p_quick:  # a new company grabs a just-freed ticker
                pool = [i for i, (t, d, _) in enumerate(self.freed)
                        if t not in self.active and day - 5 <= d <= day - 1]
            if not pool:
                pool = [i for i, (t, d, _) in enumerate(self.freed)
                        if t not in self.active and day - 400 <= d <= day - 20]
            if pool:
                i = pool[int(self.rng.integers(0, len(pool)))]
                t, freed_day, last_vid = self.freed.pop(i)
                self.active[t] = perm
                kind = "ticker_reuse_quick" if day - freed_day <= 5 else "ticker_reuse"
                return t, (last_vid, kind)
        t = make_ticker(tuple(words), self.rng, self.ever, prefix=prefix)
        self.ever.add(t)
        self.active[t] = perm
        return t, None

    def reclaim(self, ticker: str, perm: str) -> bool:
        if ticker in self.active:
            return False
        self.freed = [f for f in self.freed if f[0] != ticker]
        self.active[ticker] = perm
        return True

    def free(self, ticker: str, day: int, last_vid: str) -> None:
        if self.active.get(ticker) is not None:
            del self.active[ticker]
            self.freed.append((ticker, day, last_vid))


def _free(ent: _Ent, d: int, spacing: int = 60) -> bool:
    if d < ent.start + 70 or d > ent.end - 40:
        return False
    if any(abs(e - d) < spacing for e in ent.events):
        return False
    return not any(a - spacing <= d <= b + spacing for a, b in ent.blocked)


def _pick_day(ent: _Ent, rng, lo: int, hi: int, tries: int = 30) -> int | None:
    lo, hi = max(lo, ent.start + 70), min(hi, ent.end - 40)
    if hi <= lo:
        return None
    for _ in range(tries):
        d = int(rng.integers(lo, hi))
        if _free(ent, d):
            return d
    return None


def _draw_fate(rng, cfg: UniverseConfig, start: int, T: int) -> tuple[int, str]:
    if start + 330 >= T - 30:
        return T - 1, "alive"
    u = rng.random()
    if u < cfg.p_bankrupt:
        return int(rng.integers(start + 300, T - 30)), "bankrupt"
    if u < cfg.p_bankrupt + cfg.p_acquired:
        return int(rng.integers(start + 300, T - 30)), "acquired"
    return T - 1, "alive"


def generate_universe(cfg: UniverseConfig | None = None) -> Universe:
    cfg = cfg or UniverseConfig()
    rng = np.random.default_rng(cfg.seed)
    dates = pd.bdate_range(cfg.start, periods=cfg.years * 252)
    T = len(dates)

    # ------------------------------------------------------------ entities and fates
    ents: list[_Ent] = []

    def new_ent(start: int, origin: str, name: CompanyName, parent: str | None = None) -> _Ent:
        end, fate = _draw_fate(rng, cfg, start, T)
        e = _Ent(f"E{len(ents) + 1:04d}", start, end, fate, origin, name, parent)
        ents.append(e)
        return e

    for _ in range(cfg.n_initial):
        new_ent(0, "initial", random_name(rng))
    for _ in range(int(rng.poisson(cfg.ipo_per_year * cfg.years))):
        new_ent(int(rng.integers(60, T - 250)), "ipo", random_name(rng))

    # ------------------------------------------------------------ spin-offs
    for p in list(ents):
        if rng.random() >= cfg.p_spinoff:
            continue
        d = _pick_day(p, rng, p.start + 150, p.end - 330)
        if d is None:
            continue
        second = str(rng.choice([w for w in SECOND_WORDS if w != p.name.words[1]]))
        child = new_ent(d, "spinoff", CompanyName((p.name.words[0], second),
                                                  str(rng.choice(["Inc", "Corp", "Holdings"]))),
                        parent=p.perm)
        p.events[d] = ("spinoff", {"child": child.perm, "f": float(rng.uniform(0.10, 0.35)),
                                   "reissue": bool(rng.random() < cfg.p_spin_parent_reissue)})

    # ------------------------------------------------------------ mergers
    for t in ents:
        if t.fate != "acquired":
            continue
        d = t.end
        t.announce = max(t.start + 5, d - 60)
        cands = [a for a in ents if a is not t and a.start < d - 150 and a.end > d + 150
                 and _free(a, d + 1)]
        if not cands:
            t.fate = "taken_private"
            continue
        a = cands[int(rng.integers(0, len(cands)))]
        a.events[d + 1] = ("merger", {"target": t.perm,
                                      "reissue": bool(rng.random() < cfg.p_acquirer_reissue),
                                      "new_ticker": bool(rng.random() < 0.5)})

    # ------------------------------------------------------------ delist and relist
    for e in ents:
        if rng.random() >= cfg.p_delist_relist:
            continue
        d1 = _pick_day(e, rng, e.start + 150, e.end - 450)
        if d1 is None:
            continue
        d2 = d1 + int(rng.integers(40, 300))
        if d2 + 120 >= e.end or any(d1 - 60 <= x <= d2 + 60 for x in e.events):
            continue
        e.events[d1] = ("delist", {"relist": d2})
        e.events[d2] = ("relist", {})
        e.blocked.append((d1, d2))

    # ------------------------------------------------------------ renames, reissues, transfers
    for e in ents:
        years = (e.end - e.start) / 252
        for kind, rate in (("rename", cfg.rename_rate), ("id_reissue", cfg.reissue_rate),
                           ("exchange_transfer", cfg.exchange_transfer_rate)):
            for _ in range(int(rng.poisson(rate * years))):
                d = _pick_day(e, rng, e.start, e.end)
                if d is None:
                    continue
                info: dict = {}
                if kind == "rename":
                    info["reissue"] = bool(rng.random() < cfg.p_rename_reissue)
                if kind == "id_reissue" or info.get("reissue"):
                    info["overlap"] = int(rng.integers(2, 9)) if rng.random() < cfg.p_overlap else 0
                e.events[d] = (kind, info)

    # ------------------------------------------------------------ listings (chronological)
    by_perm = {e.perm: e for e in ents}
    actions = []  # (day, order, perm, kind, info)
    for e in ents:
        actions.append((e.start, 1, e.perm, "list", {}))
        if e.fate != "alive":
            actions.append((e.end, 0, e.perm, "end", {}))
        for d, (kind, info) in e.events.items():
            actions.append((d, 0 if kind == "delist" else 1, e.perm, kind, info))
    actions.sort(key=lambda a: (a[0], a[1], a[2]))

    tick = _Tickers(rng, cfg.p_ticker_reuse, cfg.p_quick_reuse)
    vid_codes = rng.choice(np.arange(100000, 1000000), size=len(ents) * 4, replace=False)
    listings: dict[str, dict] = {}
    state: dict[str, dict] = {}
    seams, hard_neg, events = [], [], []

    def open_vid(perm: str, day: int, ticker: str, name: CompanyName, exch: str) -> str:
        vid = f"V{vid_codes[len(listings)]}"
        listings[vid] = {"perm": perm, "start": day, "end": None,
                         "segments": [[day, ticker, render_variant(name, rng), exch]]}
        state[perm].update(vid=vid, ticker=ticker, name=name, exch=exch)
        return vid

    def close_vid(vid: str, day: int) -> None:
        listings[vid]["end"] = day

    def new_segment(perm: str, day: int, ticker: str, name: CompanyName, exch: str) -> None:
        st = state[perm]
        seg = listings[st["vid"]]["segments"]
        text = seg[-1][2] if name == st["name"] else render_variant(name, rng)
        seg.append([day, ticker, text, exch])
        st.update(ticker=ticker, name=name, exch=exch)

    def other_exchange(exch: str) -> str:
        return str(rng.choice([x for x in EXCHANGES if x != exch]))

    def reissue(perm: str, day: int, overlap: int, ticker: str, name: CompanyName, exch: str,
                kind: str, reuse_from: tuple | None = None) -> tuple[str, str]:
        st = state[perm]
        old, old_ticker = st["vid"], st["ticker"]
        close_vid(old, day - 1 + overlap)
        if ticker != old_ticker:
            tick.free(old_ticker, day - 1 + overlap, old)
        new = open_vid(perm, day, ticker, name, exch)
        seams.append((old, new, kind))
        if reuse_from is not None:
            hard_neg.append((reuse_from[0], new, reuse_from[1]))
        events.append((day, kind, perm, old, new, None))
        return old, new

    for day, _, perm, kind, info in actions:
        e = by_perm[perm]
        if kind == "list":
            state[perm] = {}
            prefix = ""
            if e.origin == "spinoff" and rng.random() < 0.6:
                prefix = state[e.parent]["ticker"][:2]
            ticker, reuse_from = tick.assign(day, perm, e.name.words, prefix=prefix)
            vid = open_vid(perm, day, ticker, e.name, str(rng.choice(EXCHANGES, p=EXCHANGE_P)))
            if reuse_from is not None:
                hard_neg.append((reuse_from[0], vid, reuse_from[1]))
            events.append((day, "initial" if e.origin == "initial" else e.origin, perm,
                           None, vid, e.parent))
        elif kind == "end":
            st = state[perm]
            close_vid(st["vid"], day)
            tick.free(st["ticker"], day, st["vid"])
            events.append((day, e.fate, perm, st["vid"], None, None))
        elif kind == "id_reissue":
            st = state[perm]
            exch = other_exchange(st["exch"]) if rng.random() < 0.15 else st["exch"]
            reissue(perm, day, info["overlap"], st["ticker"], st["name"], exch, "id_reissue")
        elif kind == "rename":
            st = state[perm]
            if rng.random() < 0.5:
                name = random_name(rng)
            else:
                name = CompanyName((st["name"].words[0],
                                    str(rng.choice([w for w in SECOND_WORDS
                                                    if w != st["name"].words[1]]))),
                                   st["name"].suffix)
            ticker, reuse_from = tick.assign(day, perm, name.words)
            if info["reissue"]:
                reissue(perm, day, info["overlap"], ticker, name, st["exch"], "rename_reissue",
                        reuse_from)
            else:
                tick.free(st["ticker"], day - 1, st["vid"])
                new_segment(perm, day, ticker, name, st["exch"])
                events.append((day, "rename", perm, st["vid"], st["vid"], None))
        elif kind == "exchange_transfer":
            st = state[perm]
            new_segment(perm, day, st["ticker"], st["name"], other_exchange(st["exch"]))
            events.append((day, "exchange_transfer", perm, st["vid"], st["vid"], None))
        elif kind == "merger":
            st, tgt = state[perm], by_perm[info["target"]]
            tgt_vid = state[tgt.perm]["vid"]
            events.append((day, "merger", perm, None, None, tgt.perm))
            if info["reissue"]:
                w1, w2 = e.name.words[0], tgt.name.words[0]
                name = CompanyName((w1, w2) if w1 != w2 else e.name.words, "Holdings")
                if info["new_ticker"]:
                    ticker, reuse_from = tick.assign(day, perm, name.words)
                else:
                    ticker, reuse_from = st["ticker"], None
                _, new = reissue(perm, day, 0, ticker, name, st["exch"],
                                 "merger_acquirer_reissue", reuse_from)
                hard_neg.append((tgt_vid, new, "merger_target"))
        elif kind == "spinoff":
            events.append((day, "spinoff_parent", perm, None, None, info["child"]))
            if info["reissue"]:
                st = state[perm]
                old, _ = reissue(perm, day, 0, st["ticker"], st["name"], st["exch"],
                                 "spinoff_parent_reissue")
            else:
                old = state[perm]["vid"]
            info["parent_vid_before"] = old
        elif kind == "delist":
            st = state[perm]
            close_vid(st["vid"], day - 1)
            tick.free(st["ticker"], day - 1, st["vid"])
            events.append((day, "delist", perm, st["vid"], None, None))
            st["delisted_vid"] = st["vid"]
        elif kind == "relist":
            st = state[perm]
            old = st.pop("delisted_vid")
            if rng.random() < 0.5 and tick.reclaim(st["ticker"], perm):
                ticker, reuse_from = st["ticker"], None
            else:
                ticker, reuse_from = tick.assign(day, perm, st["name"].words)
            exch = other_exchange(st["exch"]) if rng.random() < 0.6 else st["exch"]
            new = open_vid(perm, day, ticker, st["name"], exch)
            seams.append((old, new, "delist_relist"))
            if reuse_from is not None:
                hard_neg.append((reuse_from[0], new, reuse_from[1]))
            events.append((day, "delist_relist", perm, old, new, None))

    for e in ents:  # spin-off children look like their parent
        for d, (kind, info) in e.events.items():
            if kind == "spinoff" and "parent_vid_before" in info:
                child_vid = next(v for v, L in listings.items()
                                 if L["perm"] == info["child"] and L["start"] == d)
                hard_neg.append((info["parent_vid_before"], child_vid, "spinoff_child"))
    for L in listings.values():
        if L["end"] is None:
            L["end"] = T - 1

    # ------------------------------------------------------------ prices and volumes
    m = rng.normal(0.0003, 0.010, T)
    logp: dict[str, np.ndarray] = {}
    logv_base: dict[str, float] = {}
    for e in ents:  # parents come before their spin-off children
        n = e.end - e.start + 1
        beta, sig = rng.uniform(0.6, 1.4), rng.uniform(0.010, 0.025)
        r = rng.normal(0.0, 0.0002) + beta * m[e.start:e.end + 1] + sig * rng.standard_normal(n)
        if e.fate == "bankrupt":
            k = min(120, n - 1)
            r[-k:] += -0.010 + 0.5 * sig * rng.standard_normal(k)
        if e.fate in ("acquired", "taken_private"):
            a = max(1, n - 60)
            r[a] += rng.uniform(0.15, 0.35)
            r[a + 1:] = 0.05 * m[e.start + a + 1:e.end + 1] + 0.003 * rng.standard_normal(n - a - 1)
        for d, (kind, info) in e.events.items():
            if kind == "spinoff":
                r[d - e.start] += np.log(1.0 - info["f"])
            if kind == "delist":
                a, b = d - e.start, info["relist"] - e.start
                r[a:b] += -0.001 + 0.5 * sig * rng.standard_normal(b - a)
        if e.origin == "spinoff":
            parent = by_perm[e.parent]
            spin = next(i for d, (k, i) in parent.events.items() if k == "spinoff"
                        and i["child"] == e.perm)
            p_prev = np.exp(logp[parent.perm][e.start - 1 - parent.start])
            p0 = np.log(p_prev * spin["f"] * rng.uniform(0.5, 2.0))
        elif e.origin == "ipo":
            p0 = np.log(rng.uniform(10, 40))
        else:
            p0 = np.log(rng.uniform(5, 150))
        r[0] = 0.0
        logp[e.perm] = p0 + np.cumsum(r)
        logv_base[e.perm] = rng.normal(12.5, 1.0)

    vids = sorted(listings, key=lambda v: (listings[v]["start"], v))
    close = np.full((len(vids), T), np.nan)
    volume = np.full((len(vids), T), np.nan)
    prev_end: dict[str, int] = {}
    relisted: set[str] = {s[1] for s in seams if s[2] == "delist_relist"}
    shift: dict[str, float] = {}
    for i, v in enumerate(vids):
        L = listings[v]
        e = by_perm[L["perm"]]
        a, b = L["start"], L["end"]
        if v in relisted:
            shift[e.perm] = shift.get(e.perm, 0.0) - 0.7
        px = np.exp(logp[e.perm][a - e.start:b - e.start + 1])
        lv = logv_base[e.perm] + shift.get(e.perm, 0.0) + 0.35 * rng.standard_normal(b - a + 1)
        ov = prev_end.get(e.perm, -1) - a + 1  # days this vid overlaps the previous one
        if ov > 0:
            px[:ov] *= np.exp(rng.normal(0, 0.001, ov))
            lv[:ov] += np.log(0.3)
        close[i, a:b + 1] = np.round(px, 4)
        volume[i, a:b + 1] = np.round(np.exp(lv))
        prev_end[e.perm] = b

    # ------------------------------------------------------------ tables
    rows = []
    for v in vids:
        L = listings[v]
        segs = L["segments"]
        for k, (d, ticker, name, exch) in enumerate(segs):
            to = segs[k + 1][0] - 1 if k + 1 < len(segs) else L["end"]
            rows.append((v, ticker, name, exch, d, to))
    master = pd.DataFrame(rows, columns=["vid", "ticker", "name", "exchange", "from_idx", "to_idx"])
    master["valid_from"] = dates[master.from_idx.to_numpy()]
    master["valid_to"] = dates[master.to_idx.to_numpy()]
    master = master[["vid", "ticker", "name", "exchange", "valid_from", "valid_to",
                     "from_idx", "to_idx"]]

    truth = pd.DataFrame({"vid": vids, "perm_id": [listings[v]["perm"] for v in vids]})
    ev = pd.DataFrame(events, columns=["day", "event", "perm_id", "old_vid", "new_vid",
                                       "other_perm_id"])
    ev.insert(0, "date", dates[ev.day.to_numpy()])
    ev = ev.drop(columns="day").sort_values(["date", "perm_id", "event"]).reset_index(drop=True)
    ent_df = pd.DataFrame({
        "perm_id": [e.perm for e in ents], "name": [e.name.canonical() for e in ents],
        "start": dates[[e.start for e in ents]], "end": dates[[e.end for e in ents]],
        "fate": [e.fate for e in ents], "origin": [e.origin for e in ents]})
    return Universe(cfg, dates, vids, master, close, volume, truth,
                    pd.DataFrame(seams, columns=["old_vid", "new_vid", "event"]),
                    pd.DataFrame(hard_neg, columns=["old_vid", "new_vid", "kind"]),
                    ev, ent_df)
