"""
Data loading and preprocessing utilities.

1) Download and cache Alpha Vantage "TIME_SERIES_DAILY_ADJUSTED" CSV for each symbol
2) Load cached CVSs, align by date, keep *adjusted close* for total return consistency
3) Fill missing data by forward-filling, then drop NaNs
4) Persist a clean price matrix (Parquet) under data/processed/."""

import os

from typing import Optional

def _get_api_key(explicit_key: Optional[str] = None) -> str:
    """Retrieve Alpha Vantage API key from environment variable."""
    key = explicit_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not key:
        raise RuntimeError(
            "Alpha Vantage API key not found. Please set the ALPHAVANTAGE_API_KEY environment variable."
        )
    return key

def av_daily_adjusted():
    """
    Download alpha vantage DAILY_ADJUSTED for one symbol and cache to CSV
    """
    pass

def download_universe():
    """
    Download and cache all symbols in the universe
    """
    pass

def load_and_merge():
    """
    Load all cached CSVs, align by date, keep adjusted close
    """
    pass