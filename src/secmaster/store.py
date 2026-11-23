"""SQLite output and point-in-time (as-of) queries.

Table `security_master`, one row per (vid, ticker segment):

  resolved_id  permanent id assigned by the resolver
  vid          vendor security id
  ticker, name, exchange
  valid_from, valid_to   dates the row describes (inclusive, ISO text)
  known_from             first date the row's entity link could have been decided

Dates are stored as ISO 'YYYY-MM-DD' text, so string comparison is date comparison.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE security_master (
    resolved_id TEXT NOT NULL,
    vid         TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    name        TEXT NOT NULL,
    exchange    TEXT NOT NULL,
    valid_from  TEXT NOT NULL,
    valid_to    TEXT NOT NULL,
    known_from  TEXT NOT NULL
);
CREATE INDEX ix_sm_ticker ON security_master (ticker, valid_from, valid_to);
CREATE INDEX ix_sm_entity ON security_master (resolved_id, valid_from);
"""

# A row whose link was not yet knowable on `known_on` is still returned (the vendor id
# traded), as its own provisional entity: resolved_id = vid and linked = 0.
AS_OF_TICKER = """
SELECT CASE WHEN known_from <= :known_on THEN resolved_id ELSE vid END AS resolved_id,
       vid, ticker, name, exchange, valid_from, valid_to, known_from,
       CASE WHEN known_from <= :known_on THEN 1 ELSE 0 END AS linked
FROM security_master
WHERE ticker = :ticker
  AND valid_from <= :as_of AND valid_to >= :as_of
  AND valid_from <= :known_on
ORDER BY valid_from
"""

ENTITY_HISTORY = """
SELECT resolved_id, vid, ticker, name, exchange, valid_from, valid_to, known_from
FROM security_master
WHERE resolved_id = :resolved_id AND known_from <= :known_on
ORDER BY valid_from, vid
"""

TICKER_HISTORY = """
SELECT resolved_id, vid, ticker, name, exchange, valid_from, valid_to, known_from
FROM security_master
WHERE ticker = :ticker
ORDER BY valid_from
"""


def write_sqlite(table: pd.DataFrame, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    t = table.copy()
    for c in ("valid_from", "valid_to", "known_from"):
        t[c] = pd.to_datetime(t[c]).dt.strftime("%Y-%m-%d")
    with closing(sqlite3.connect(path)) as con, con:
        con.executescript(SCHEMA)
        con.executemany(
            "INSERT INTO security_master VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            t[["resolved_id", "vid", "ticker", "name", "exchange", "valid_from", "valid_to",
               "known_from"]].itertuples(index=False, name=None))
    return path


def _query(path, sql: str, params: dict) -> pd.DataFrame:
    if not Path(path).is_file():
        raise FileNotFoundError(f"no database at {path}; run `secmaster resolve` first")
    with closing(sqlite3.connect(path)) as con:
        return pd.read_sql_query(sql, con, params=params)


def as_of_ticker(path, ticker: str, as_of: str, known_on: str = "9999-12-31") -> pd.DataFrame:
    """Which entity traded under `ticker` on `as_of`, using links known by `known_on`.

    Rows that had started trading by `known_on` but whose link was not yet knowable come
    back unlinked: resolved_id is the vendor id itself and `linked` is 0."""
    return _query(path, AS_OF_TICKER, {"ticker": ticker, "as_of": as_of, "known_on": known_on})


def entity_history(path, resolved_id: str, known_on: str = "9999-12-31") -> pd.DataFrame:
    """Rows linked to `resolved_id` by `known_on`. Ids not yet linked are left out here;
    `as_of_ticker` returns them as provisional entities."""
    return _query(path, ENTITY_HISTORY, {"resolved_id": resolved_id, "known_on": known_on})


def ticker_history(path, ticker: str) -> pd.DataFrame:
    return _query(path, TICKER_HISTORY, {"ticker": ticker})
