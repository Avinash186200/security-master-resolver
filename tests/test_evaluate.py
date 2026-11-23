import pytest

from secmaster.evaluate import cluster_metrics


def test_perfect():
    m = cluster_metrics([0, 0, 1, 2], ["a", "a", "b", "c"])
    assert m["pairwise_f1"] == 1.0 and m["chain_purity"] == 1.0 and m["exact_chains"] == 1.0


def test_known_values():
    # truth: {0,1,2} {3}; predicted: {0,1} {2,3}
    m = cluster_metrics([0, 0, 1, 1], ["a", "a", "a", "b"])
    assert m["true_pairs"] == 3 and m["predicted_pairs"] == 2 and m["correct_pairs"] == 1
    assert m["pairwise_precision"] == pytest.approx(0.5)
    assert m["pairwise_recall"] == pytest.approx(1 / 3)
    assert m["chain_purity"] == pytest.approx(3 / 4)
    assert m["chain_completeness"] == pytest.approx(3 / 4)
    assert m["exact_chains"] == 0.0


def test_all_singletons():
    m = cluster_metrics([0, 1, 2], ["a", "b", "c"])
    assert m["pairwise_precision"] == 1.0 and m["pairwise_recall"] == 1.0
