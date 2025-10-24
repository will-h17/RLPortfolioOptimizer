from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

class FitOnTrainScaler:
    """
    Z-score per column using *train-only* stats.
    Works with pandas Index or MultiIndex columns.
    """
    def __init__(self):
        self.mu_: pd.Series | None = None
        self.sigma_: pd.Series | None = None

    def fit(self, X_train: pd.DataFrame) -> "FitOnTrainScaler":
        mu = X_train.mean(axis=0)
        sigma = X_train.std(axis=0).replace(0.0, 1.0)
        self.mu_, self.sigma_ = mu, sigma
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.mu_ is not None and self.sigma_ is not None, "Call fit() first"
        # align by columns (handles missing/reordered)
        mu = self.mu_.reindex(X.columns)
        sigma = self.sigma_.reindex(X.columns).replace(0.0, 1.0)
        return (X - mu) / sigma

    # simple artifact I/O (JSON + parquet are both fine; JSON keeps it light)
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mu": self.mu_.to_dict() if isinstance(self.mu_.index, pd.Index) else dict(self.mu_),
            "sigma": self.sigma_.to_dict() if isinstance(self.sigma_.index, pd.Index) else dict(self.sigma_),
            "columns": list(map(str, self.mu_.index)),
            "multiindex": isinstance(self.mu_.index, pd.MultiIndex),
        }
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: str | Path, columns) -> "FitOnTrainScaler":
        path = Path(path)
        payload = json.loads(path.read_text())
        scaler = cls()
        # rebuild index (MultiIndex or simple Index) from provided columns at runtime
        idx = columns
        scaler.mu_ = pd.Series(payload["mu"]).reindex(idx)
        scaler.sigma_ = pd.Series(payload["sigma"]).reindex(idx).replace(0.0, 1.0)
        return scaler
