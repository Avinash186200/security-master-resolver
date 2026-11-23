import json

from secmaster.cli import load_vendor_data, main
from secmaster.store import ticker_history


def test_generate_resolve_asof_history(tmp_path, capsys, scorers):
    data = tmp_path / "data"
    assert main(["generate", "--seed", "9", "--n-initial", "60", "--years", "4",
                 "--out", str(data)]) == 0
    capsys.readouterr()
    master, vids, dates, close, volume = load_vendor_data(data)
    assert close.shape == (len(vids), len(dates))
    model = tmp_path / "scorer.json"
    scorers["combined"].save(model)
    db = tmp_path / "sm.sqlite"
    assert main(["resolve", "--data", str(data), "--db", str(db), "--model", str(model),
                 "--llm", "mock"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["rows"] == len(master)
    t = master.ticker.iloc[0]
    d = str(master.valid_from.iloc[0].date())
    assert main(["asof", "--db", str(db), "--ticker", t, "--date", d]) == 0
    assert t in capsys.readouterr().out
    rid = ticker_history(db, t).resolved_id.iloc[0]
    assert main(["history", "--db", str(db), "--id", rid]) == 0
    assert rid in capsys.readouterr().out


def test_saved_scorer_round_trips(tmp_path, cands, scorers):
    import numpy as np
    from secmaster.scorer import LogisticScorer
    c, _ = cands
    p = tmp_path / "s.json"
    scorers["combined"].save(p)
    assert np.allclose(LogisticScorer.load(p).predict(c), scorers["combined"].predict(c))


def test_missing_db_is_a_clear_error(tmp_path, capsys):
    assert main(["asof", "--db", str(tmp_path / "none.sqlite"), "--ticker", "ABC",
                 "--date", "2015-01-02"]) == 2
    assert "run `secmaster resolve` first" in capsys.readouterr().err
