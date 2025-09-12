"""
Data loading and preprocessing utilities.

1) Download and cache Alpha Vantage "TIME_SERIES_DAILY_ADJUSTED" CSV for each symbol
2) Load cached CVSs, align by date, keep *adjusted close* for total return consistency
3) Fill missing data by forward-filling, then drop NaNs
4) Persist a clean price matrix (Parquet) under data/processed/."""

import os
import requests

from typing import Optional
from pathlib import Path

RAW_DIR = os.path.join("data", "raw")
PROC_DIR = os.path.join("data", "processed")
ALPHA_BASE = "https://www.alphavantage.co/query"

def _get_api_key(explicit_key: Optional[str] = None) -> str:
    """Retrieve Alpha Vantage API key from environment variable."""
    key = explicit_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not key:
        raise RuntimeError(
            "Alpha Vantage API key not found. Please set the ALPHAVANTAGE_API_KEY environment variable."
        )
    return key

def av_daily_adjusted(
        symbol: str,
        api_key: Optional[str] = None,
        outdir: Path | str = RAW_DIR,
        force: bool = False,
        timeout: int = 30,) -> Path:
    """
    Download alpha vantage DAILY_ADJUSTED for one symbol and cache to CSV
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{symbol}_daily_adjusted.csv"

    if path.exists() and not force:
        print(f"File {path} already exists. Skipping download.")
        return path
    
    key = _get_api_key(api_key)
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "apikey": key,
        "outputsize": "full",  #full history
        "datatype": "csv",  #csv format
    }

    resp = requests.get(ALPHA_BASE, params=params, timeout=timeout)
    resp.raise_for_status()

    # Check for API call frequency limit message
    text_head = resp.text[:128].lower()
    if "thank you for using alpha vantage" in text_head and "please visit" in text_head:
        # API call frequency exceeded
        raise RuntimeError("API call frequency exceeded. Please try again later.")
    with open(path, "wb") as f:
        f.write(resp.content)

    return path

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