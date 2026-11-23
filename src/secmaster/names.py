"""Company names: generation, vendor-style spelling noise, normalisation and similarity.

Names are built from two generic word lists, so two different companies can share a
word ("Granite Energy Corp" and "Granite Foods Inc"). That is on purpose: it makes
name similarity a useful but imperfect signal, as it is with real vendor data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

import numpy as np

FIRST_WORDS = [
    "Northern", "Summit", "Granite", "Bluewater", "Silverline", "Harbor", "Prairie", "Atlas",
    "Crescent", "Pioneer", "Cedar", "Meridian", "Beacon", "Liberty", "Evergreen", "Coastal",
    "Redwood", "Ironwood", "Lakeside", "Keystone", "Orion", "Falcon", "Sterling", "Highland",
    "Riverbend", "Copperleaf", "Westgate", "Eastbrook", "Polaris", "Juniper", "Tamarack",
    "Sequoia", "Bayview", "Stonebridge", "Clearpoint", "Brightwater", "Oakmont", "Pinecrest",
    "Sunrise", "Vantage",
]
SECOND_WORDS = [
    "Systems", "Energy", "Foods", "Logistics", "Health", "Bancorp", "Materials", "Software",
    "Networks", "Biosciences", "Industries", "Robotics", "Water", "Minerals", "Pharma",
    "Retail", "Media", "Semiconductor", "Insurance", "Realty", "Aerospace", "Chemicals",
    "Devices", "Analytics", "Apparel", "Brands", "Capital", "Marine", "Power", "Telecom",
]
SUFFIXES = ["Inc", "Corp", "Holdings", "Group", "Co", "Ltd"]

SUFFIX_VARIANTS = {
    "Inc": ["Inc", "Inc.", "Incorporated"],
    "Corp": ["Corp", "Corp.", "Corporation"],
    "Holdings": ["Holdings", "Hldgs", "Holdings Inc"],
    "Group": ["Group", "Grp", "Group Inc"],
    "Co": ["Co", "Co.", "Company"],
    "Ltd": ["Ltd", "Ltd.", "Limited"],
}

# Words that carry no identity: legal suffixes and their spellings.
_STOP = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "holdings", "hldgs",
    "group", "grp", "ltd", "limited", "plc", "the", "and",
}


@dataclass(frozen=True)
class CompanyName:
    words: tuple[str, ...]
    suffix: str

    def canonical(self) -> str:
        return " ".join(self.words) + " " + self.suffix


def random_name(rng: np.random.Generator) -> CompanyName:
    return CompanyName((str(rng.choice(FIRST_WORDS)), str(rng.choice(SECOND_WORDS))),
                       str(rng.choice(SUFFIXES)))


def _typo(word: str, rng: np.random.Generator) -> str:
    if len(word) < 5:
        return word
    i = int(rng.integers(1, len(word) - 2))
    if rng.random() < 0.5:  # swap two neighbouring letters
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return word[:i] + word[i + 1:]  # drop a letter


def render_variant(name: CompanyName, rng: np.random.Generator, typo_rate: float = 0.08,
                   upper_rate: float = 0.15) -> str:
    """One vendor spelling of a name: suffix form, optional typo, optional upper case."""
    words = list(name.words)
    if rng.random() < typo_rate:
        k = int(rng.integers(0, len(words)))
        words[k] = _typo(words[k], rng)
    variants = SUFFIX_VARIANTS[name.suffix]
    text = " ".join(words) + " " + variants[int(rng.integers(0, len(variants)))]
    if rng.random() < upper_rate:
        text = text.upper()
    return text


def normalize_name(text: str) -> list[str]:
    """Lower case, '&' to 'and', punctuation removed, legal suffixes dropped."""
    s = text.lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return [t for t in s.split() if t and t not in _STOP]


def _soft_jaccard(ta: list[str], tb: list[str], tol: float = 0.8) -> float:
    """Token Jaccard where two tokens match if their character similarity is >= tol."""
    used = set()
    hits = 0
    for a in ta:
        for j, b in enumerate(tb):
            if j not in used and (a == b or SequenceMatcher(None, a, b).ratio() >= tol):
                used.add(j)
                hits += 1
                break
    union = len(ta) + len(tb) - hits
    return hits / union if union else 0.0


def name_similarity(a: str, b: str) -> float:
    """0..1. Robust to suffix spelling, case, '&', word order and one-letter typos."""
    ta, tb = normalize_name(a), normalize_name(b)
    if not ta or not tb:
        return 0.0
    seq = SequenceMatcher(None, " ".join(sorted(ta)), " ".join(sorted(tb))).ratio()
    return max(seq, _soft_jaccard(ta, tb))


def token_keys(text: str) -> set[str]:
    """Cheap blocking keys: the first four letters of each normalised token."""
    return {t[:4] for t in normalize_name(text)}


def ticker_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio() if a and b else 0.0


_LETTERS = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))


def make_ticker(words: tuple[str, ...], rng: np.random.Generator, taken: set[str],
                prefix: str = "") -> str:
    """A new ticker not in `taken`: initials of the name plus random letters, or random."""
    for _ in range(200):
        if prefix:
            base = prefix
        elif rng.random() < 0.6:
            base = "".join(w[0] for w in words).upper()
        else:
            base = ""
        n = int(rng.integers(3, 5))
        extra = "".join(rng.choice(_LETTERS, size=max(n - len(base), 1)))
        t = (base + extra)[:4]
        if len(t) >= 2 and t not in taken:
            return t
    raise RuntimeError("could not find a free ticker")  # pragma: no cover
