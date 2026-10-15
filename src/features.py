"""
Feature engineering for price data.

Outputs a wide table with a 2-level or flattened column index: (asset, feature_name).
Index aligns to input prices. Rows with NaNs from rolling windows are dropped.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from src.paths import FEATURES_PATH, PRICES_PATH


def make_features(
    prices: pd.DataFrame,
    lookbacks: Iterable[int] = (5, 20),
    out_path: Path | str = FEATURES_PATH,
) -> pd.DataFrame:
    """
    Compute returns, log-returns, momentum, rolling vol, and EMA trend per asset.
    
    Optimizations:
    - Cache 1-day returns to avoid recomputation
    - Use cumulative returns for momentum (more efficient than repeated pct_change)
    """
    # Precompute 1-day returns once (used for ret_1, logret_1, and vol features)
    rets = prices.pct_change()
    
    # Precompute cumulative returns for efficient momentum calculation
    # (1 + rets).cumprod() gives cumulative returns, then we can compute momentum via ratios
    cumrets = (1 + rets).cumprod()

    feats = {}
    for col in prices.columns:
        r = rets[col]
        feats[(col, "ret_1")] = r
        feats[(col, "logret_1")] = np.log1p(r)
        
        for lb in lookbacks:
            # Momentum: use cumulative returns ratio (more efficient than pct_change)
            # cumrets[t] / cumrets[t-lb] - 1 = (1 + ret[t-lb+1]) * ... * (1 + ret[t]) - 1
            # This is equivalent to pct_change(lb) but avoids recomputing from prices
            cumret_col = cumrets[col]
            feats[(col, f"mom_{lb}")] = cumret_col / cumret_col.shift(lb) - 1.0
            
            # Volatility: use cached 1-day returns
            feats[(col, f"vol_{lb}")] = r.rolling(lb).std()
            
            # EMA: still need to compute from prices (no good way to cache)
            ema = prices[col].ewm(span=lb, adjust=False).mean()
            feats[(col, f"ema_{lb}")] = ema.pct_change()

    X = pd.DataFrame(feats, index=prices.index).dropna(how="any")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    X.to_parquet(out_path)
    print(f"[features] wrote {out_path}")
    return X


# CLI
if __name__ == "__main__":
    if not PRICES_PATH.exists():
        raise SystemExit(f"Missing {PRICES_PATH}. Run: python -m src.data")
    prices = pd.read_parquet(PRICES_PATH)
    make_features(prices)
