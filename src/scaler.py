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

        # Ensure scaler was fitted
        if getattr(self, "mu_", None) is None or getattr(self, "sigma_", None) is None:
            raise RuntimeError("Scaler not fitted. Call fit(X_train) before save().")

        # JSON keys must be strings; MultiIndex / tuple keys are converted to strings here.
        mu_dict_raw = self.mu_.to_dict()
        sigma_dict_raw = self.sigma_.to_dict()
        mu_dict = {str(k): float(v) for k, v in mu_dict_raw.items()}
        sigma_dict = {str(k): float(v) for k, v in sigma_dict_raw.items()}

        payload = {
            "mu": mu_dict,
            "sigma": sigma_dict,
            "columns": list(map(str, self.mu_.index)),
            "multiindex": isinstance(self.mu_.index, pd.MultiIndex),
        }
        path.write_text(json.dumps(payload))

    

    @classmethod
    def load(cls, path: str | Path, columns=None) -> "FitOnTrainScaler":
        path = Path(path)
        payload = json.loads(path.read_text())

        mu_dict = payload.get("mu", {})
        sigma_dict = payload.get("sigma", {})
        saved_columns = payload.get("columns", [])
        multiindex = payload.get("multiindex", False)

        # Prefer caller-provided columns, otherwise use saved columns
        cols = columns if columns is not None else saved_columns

        # Rebuild index (if MultiIndex, parse tuple-like strings back to tuples)
        if multiindex:
            import ast
            parsed_cols = [ast.literal_eval(c) if isinstance(c, str) else c for c in cols]
            index = pd.MultiIndex.from_tuples(parsed_cols)
        else:
            index = pd.Index(cols)

        # Recreate Series aligned to the reconstructed index; saved keys are stringified
        mu = pd.Series([float(mu_dict.get(str(c), 0.0)) for c in cols], index=index)
        sigma = pd.Series([float(sigma_dict.get(str(c), 1.0)) for c in cols], index=index)

        scaler = cls()
        scaler.mu_ = mu
        scaler.sigma_ = sigma
        return scaler
