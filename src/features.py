"""
Feature engineering for price data.

Starting simple and transparent:
- Daily simple returns and log-returns
- Momentum over short/medium windows
- Rolling volatility over short/medium windows
- EMA-based trend signal (percent change of EMA)

The output is a wide table with a 2-level column index: (asset, feature_name).
Index is aligned to the input price index. We drop rows with NaNs from rolling windows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

def make_features(prices: pd.DataFrame,
                  lookbacks: Iterable[int] = (5, 20),
                  out_path: str | Path = "data/processed/features.parquet"
                  ) -> pd.DataFrame:
    # Basic daily returns
    rets = prices.pct_change()

    feats = {}
    for col in prices.columns:
        r = rets[col]
        feats[(col, "ret_1")] = r
        feats[(col, "logret_1")] = np.log1p(r)

        for lb in lookbacks:
            # Momentum over lookback window; shift handled by pct_change
            feats[(col, f"mom_{lb}")] = prices[col].pct_change(lb)

            # Rolling volatility of daily returns
            feats[(col, f"vol_{lb}")] = r.rolling(lb).std()

            # EMA trend: compute EMA, then take daily pct change of the EMA itself
            ema = prices[col].ewm(span=lb, adjust=False).mean()
            feats[(col, f"ema_{lb}")] = ema.pct_change()

    X = pd.DataFrame(feats, index=prices.index)

    # Drop rows with NaNs from rolling windows
    X = X.dropna(how="any")

    # Persist to disk for the RL agent to use
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    X.to_parquet(out_path)

    return X