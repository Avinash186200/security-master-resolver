import json
import math
from pathlib import Path

import pytest

from secmaster.bench import run

ROOT = Path(__file__).resolve().parents[1]


def test_small_bench_end_to_end(tmp_path, small_cfg):
    out = run(tmp_path, base=small_cfg, train_seeds=(1, 2), test_seed=7, extra_seeds=(8,),
              holdout_seeds=(9,))
    assert set(out["resolution"]["methods"]) == {"rule", "ids_only", "price_only", "combined",
                                                 "combined_llm_mock"}
    assert (tmp_path / "RESULTS.md").read_text().startswith("# Results")
    assert (tmp_path / "security_master.sqlite").exists()


def _diff(a, b, path="$"):
    """Differences between two JSON trees; floats match to 1e-6 relative."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(a.keys() | b.keys()):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: missing")
            else:
                out += _diff(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _diff(x, y, f"{path}[{i}]")]
    if isinstance(a, float) or isinstance(b, float):
        ok = isinstance(a, (int, float)) and isinstance(b, (int, float)) and \
            math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)
        return [] if ok else [f"{path}: {a} != {b}"]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("full")
    run(out)
    return out


@pytest.mark.parametrize("name", ["universe", "resolution", "seeds", "backtest", "asof"])
def test_committed_results_reproduce(full_run, name):
    fresh = json.loads((full_run / f"{name}.json").read_text())
    committed = json.loads((ROOT / "results" / f"{name}.json").read_text())
    diffs = _diff(fresh, committed)
    assert not diffs, "\n".join(diffs[:20])
