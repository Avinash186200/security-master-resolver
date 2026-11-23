import pandas as pd

from secmaster.store import as_of_ticker, entity_history, ticker_history, write_sqlite


def _table():
    d = pd.Timestamp
    return pd.DataFrame([
        ("R1", "V1", "ABC", "Old Co", "XNYS", d("2015-01-01"), d("2016-06-30"), d("2015-01-01")),
        ("R1", "V2", "ABD", "Old Co New Name", "XNYS", d("2016-07-01"), d("2018-12-31"),
         d("2016-07-29")),
        ("R2", "V3", "ABC", "Someone Else", "XNAS", d("2017-03-01"), d("2020-12-31"),
         d("2017-03-01")),
    ], columns=["resolved_id", "vid", "ticker", "name", "exchange", "valid_from", "valid_to",
                "known_from"])


def test_as_of_ticker_reuse(tmp_path):
    db = write_sqlite(_table(), tmp_path / "sm.sqlite")
    assert as_of_ticker(db, "ABC", "2015-06-30").resolved_id.tolist() == ["R1"]
    assert as_of_ticker(db, "ABC", "2016-12-30").empty
    assert as_of_ticker(db, "ABC", "2019-01-02").resolved_id.tolist() == ["R2"]
    assert len(ticker_history(db, "ABC")) == 2


def test_entity_history_known_on(tmp_path):
    db = write_sqlite(_table(), tmp_path / "sm.sqlite")
    assert entity_history(db, "R1").vid.tolist() == ["V1", "V2"]
    assert entity_history(db, "R1", known_on="2016-07-15").vid.tolist() == ["V1"]


def test_rewrite_replaces(tmp_path):
    p = tmp_path / "sm.sqlite"
    write_sqlite(_table(), p)
    write_sqlite(_table().iloc[:1], p)
    assert len(ticker_history(p, "ABC")) == 1


def test_as_of_ticker_returns_unlinked_ids(tmp_path):
    """V2 trades from 2016-07-01 but its link is only knowable on 2016-07-29: before that,
    it comes back as its own provisional entity, not hidden."""
    db = write_sqlite(_table(), tmp_path / "sm.sqlite")
    q = as_of_ticker(db, "ABD", "2016-07-15", known_on="2016-07-15")
    assert q.vid.tolist() == ["V2"]
    assert q.resolved_id.tolist() == ["V2"] and q.linked.tolist() == [0]
    q = as_of_ticker(db, "ABD", "2016-07-15", known_on="2016-08-01")
    assert q.resolved_id.tolist() == ["R1"] and q.linked.tolist() == [1]
    # a row that had not started trading by known_on is not returned
    assert as_of_ticker(db, "ABD", "2016-07-15", known_on="2016-06-01").empty
