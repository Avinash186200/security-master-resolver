import numpy as np

from secmaster.names import (CompanyName, make_ticker, name_similarity, normalize_name,
                             render_variant)


def test_normalize_drops_suffixes_and_punctuation():
    assert normalize_name("GRANITE ENERGY CORP.") == ["granite", "energy"]
    assert normalize_name("Granite & Energy Holdings Inc") == ["granite", "energy"]


def test_variants_score_high_and_different_companies_lower():
    rng = np.random.default_rng(0)
    name = CompanyName(("Granite", "Energy"), "Corp")
    for _ in range(50):
        assert name_similarity(name.canonical(), render_variant(name, rng)) >= 0.8
    assert name_similarity("Granite Energy Corp", "Granite Foods Inc") < 0.8
    assert name_similarity("Granite Energy Corp", "Polaris Robotics Ltd") < 0.4


def test_typo_still_matches():
    assert name_similarity("Meridian Logistics Inc", "Meridain Logistics Inc.") > 0.85


def test_empty_name_is_zero():
    assert name_similarity("", "Granite Energy") == 0.0
    assert name_similarity("Inc", "Corp") == 0.0


def test_make_ticker_avoids_taken():
    rng = np.random.default_rng(1)
    taken = set()
    for _ in range(300):
        t = make_ticker(("Granite", "Energy"), rng, taken)
        assert t not in taken and 2 <= len(t) <= 4 and t.isupper()
        taken.add(t)
