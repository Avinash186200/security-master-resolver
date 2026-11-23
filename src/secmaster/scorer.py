"""Pair scorers: a classic rule and logistic models on three feature sets.

The logistic models are trained on separate synthetic universes (different seeds from
the one they are scored on), so no test-universe label is used to fit weights or to
choose thresholds.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .candidates import COMBINED_FEATURES, ID_FEATURES, PRICE_FEATURES

FEATURE_SETS = {"ids_only": ID_FEATURES, "price_only": PRICE_FEATURES,
                "combined": COMBINED_FEATURES}


def rule_scores(cands: pd.DataFrame, max_gap: int = 7, max_abs_z: float = 3.0) -> np.ndarray:
    """The textbook rule: same ticker, new id starts within a week, price jump under 3 sigma.
    Returns 1.0 for accepted pairs and 0.0 otherwise."""
    ok = (cands.ticker_eq == 1) & (cands.raw_gap <= max_gap) & (cands.abs_z <= max_abs_z)
    return ok.to_numpy(dtype=float)


@dataclass
class LogisticScorer:
    name: str
    features: list[str]
    C: float = 1.0
    scaler: StandardScaler = field(default_factory=StandardScaler)
    model: LogisticRegression | None = None

    def fit(self, cands: pd.DataFrame, y: np.ndarray) -> "LogisticScorer":
        X = self.scaler.fit_transform(cands[self.features].to_numpy(dtype=float))
        self.model = LogisticRegression(C=self.C, max_iter=2000).fit(X, y)
        return self

    def predict(self, cands: pd.DataFrame) -> np.ndarray:
        if len(cands) == 0:
            return np.zeros(0)
        X = self.scaler.transform(cands[self.features].to_numpy(dtype=float))
        return self.model.predict_proba(X)[:, 1]

    def to_dict(self) -> dict:
        return {"name": self.name, "features": self.features, "C": self.C,
                "mean": self.scaler.mean_.tolist(), "scale": self.scaler.scale_.tolist(),
                "coef": self.model.coef_[0].tolist(), "intercept": float(self.model.intercept_[0])}

    @classmethod
    def from_dict(cls, d: dict) -> "LogisticScorer":
        s = cls(d["name"], list(d["features"]), d["C"])
        s.scaler.mean_ = np.array(d["mean"])
        s.scaler.scale_ = np.array(d["scale"])
        s.scaler.var_ = s.scaler.scale_ ** 2
        s.scaler.n_features_in_ = len(d["features"])
        m = LogisticRegression()
        m.coef_ = np.array([d["coef"]])
        m.intercept_ = np.array([d["intercept"]])
        m.classes_ = np.array([0, 1])
        m.n_features_in_ = len(d["features"])
        s.model = m
        return s

    def save(self, path) -> None:
        with open(path, "w", newline="\n") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def load(cls, path) -> "LogisticScorer":
        with open(path) as f:
            return cls.from_dict(json.load(f))
