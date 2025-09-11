"""
Data loading and preprocessing utilities.

1) Download and cache Alpha Vantage "TIME_SERIES_DAILY_ADJUSTED" CSV for each symbol
2) Load cached CVSs, align by date, keep *adjusted close* for total return consistency
3) Fill missing data by forward-filling, then drop NaNs
4) Persist a clean price matrix (Parquet) under data/processed/."""

import os

def _get_api_key():
    """Retrieve Alpha Vantage API key from environment variable."""
    key = explicit_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not key:
        raise RuntimeError(
            "Alpha Vantage API key not found. Please set the ALPHAVANTAGE_API_KEY environment variable."
        )
    return key
