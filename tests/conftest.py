import pytest

from secmaster.pipeline import candidates_for, train_scorers, training_set
from secmaster.universe import UniverseConfig, generate_universe

SMALL = UniverseConfig(seed=7, n_initial=90, years=5, ipo_per_year=12)


@pytest.fixture(scope="session")
def small_cfg():
    return SMALL


@pytest.fixture(scope="session")
def uni():
    return generate_universe(SMALL)


@pytest.fixture(scope="session")
def cands(uni):
    return candidates_for(uni)


@pytest.fixture(scope="session")
def scorers():
    return train_scorers(training_set(SMALL, seeds=(1, 2)))
