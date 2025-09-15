"""
Data loading and preprocessing utilities.

1) Download and cache Alpha Vantage "TIME_SERIES_DAILY_ADJUSTED" CSV for each symbol
2) Load cached CVSs, align by date, keep *adjusted close* for total return consistency
3) Fill missing data by forward-filling, then drop NaNs
4) Persist a clean price matrix (Parquet) under data/processed/."""
from __future__ import annotations

import os
import time
import pandas as pd
from typing import Optional, List
from pathlib import Path

import requests

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

def download_universe(symbols: List[str], sleep_sec: float = 12.5, force: bool = False, api_key: Optional[str] = None) -> List[Path]:
    """
    Download and cache all symbols in the universe
    """
    paths: List[Path] = []
    for i, symbol in enumerate(symbols):
        p = av_daily_adjusted(symbol, api_key=api_key, force=force)
        paths.append(p)
        # Sleep between requests, not after the last one
        if i < len(symbols) - 1:
            time.sleep(sleep_sec)
    return paths

def load_and_merge(symbols: List[str],
                   rawdir: Path | str = RAW_DIR,
                   processed: Path | str = PROC_DIR,
                   out_name: str = "prices_adj.parquet",
                   forward_fill: bool = True) -> pd.DataFrame:
    """
    Load all cached CSVs, align by date, keep adjusted close
    """
    rawdir = Path(rawdir)
    processed = Path(processed)
    processed.mkdir(parents=True, exist_ok=True)

    dfs = []
    for sym in symbols:
        path = rawdir / f"{sym}_daily_adjusted.csv"
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist. Please download the data first.")
        df = pd.read_csv(path, parse_dates=["timestamp"])
        # Keep the columns we trust downstream
        df = df.rename(
            columns={"timestamp": "date", "adjusted_close": sym}
        )[["date", sym]].sort_values("date")
        dfs.append(df)

    # Outer join to keep all dates
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="date", how="outer")

    merged = merged.sort_values("date").set_index("date")
    if forward_fill:
        merged = merged.ffill()

    # Drop any remaining NaNs (e.g. leading NaNs)
    merged = merged.dropna(how="any")

    out_path = processed / out_name
    merged.to_parquet(out_path)

    return merged