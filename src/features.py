"""
Feature engineering for price data.

Outputs a wide table with a 2-level or flattened column index: (asset, feature_name).
Index aligns to input prices. Rows with NaNs from rolling windows are dropped.
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional
import numpy as np
import pandas as pd

# ---------- Paths ----------
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
PROC_DIR = DATA_DIR / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


def make_features(
    prices: pd.DataFrame,
    lookbacks: Iterable[int] = (5, 20),
    out_path: Path | str = PROC_DIR / "features.parquet",
) -> pd.DataFrame:
    """Compute returns, log-returns, momentum, rolling vol, and EMA trend per asset."""
    rets = prices.pct_change()

    feats = {}
    for col in prices.columns:
        r = rets[col]
        feats[(col, "ret_1")] = r
        feats[(col, "logret_1")] = np.log1p(r)
        for lb in lookbacks:
            feats[(col, f"mom_{lb}")] = prices[col].pct_change(lb)
            feats[(col, f"vol_{lb}")] = r.rolling(lb).std()
            ema = prices[col].ewm(span=lb, adjust=False).mean()
            feats[(col, f"ema_{lb}")] = ema.pct_change()

    X = pd.DataFrame(feats, index=prices.index).dropna(how="any")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    X.to_parquet(out_path)
    print(f"[features] wrote {out_path}")
    return X


# --------------------------- CLI --------------------------- #
if __name__ == "__main__":
    prices_path = PROC_DIR / "prices_adj.parquet"
    if not prices_path.exists():
        raise SystemExit(f"Missing {prices_path}. Run src/data.py first.")
    prices = pd.read_parquet(prices_path)
    make_features(prices)
