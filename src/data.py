"""
Data loading and preprocessing utilities (using yfinance).

Pipeline:
1) Download daily adjusted close prices for each symbol via yfinance.
2) Cache CSVs under data/raw.
3) Load cached CSVs, align by date, keep adjusted close.
4) Forward-fill gaps, then drop any leading NaNs.
5) Persist clean price matrix under data/processed/prices_adj.parquet.
"""

from __future__ import annotations
from pathlib import Path
from typing import List
import pandas as pd
import yfinance as yf

# ---------- Paths ----------
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)


def download_universe(symbols: List[str], force: bool = False) -> List[Path]:
    """
    Download daily adjusted data for all symbols and cache to CSV.
    """
    paths: List[Path] = []
    for sym in symbols:
        path = RAW_DIR / f"{sym}_daily_adjusted.csv"
        if path.exists() and not force:
            print(f"[yfinance] cached {path.name}")
            paths.append(path)
            continue

        print(f"[yfinance] downloading {sym}...")
        df = yf.download(sym, period="max", auto_adjust=True)  # auto_adjust=True gives adjusted close
        df = df.reset_index().rename(columns={"Date": "date"})
        df = df[["date", "Close"]].rename(columns={"Close": sym})

        df.to_csv(path, index=False)
        print(f"[yfinance] wrote {path}")
        paths.append(path)
    return paths


def load_and_merge(symbols: List[str], out_name: str = "prices_adj.parquet") -> pd.DataFrame:
    dfs = []
    for sym in symbols:
        path = RAW_DIR / f"{sym}_daily_adjusted.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run download_universe first.")
        df = pd.read_csv(path, parse_dates=["date"])
        # Ensure numeric
        df[sym] = pd.to_numeric(df[sym], errors="coerce")
        dfs.append(df)

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="date", how="outer")

    merged = merged.sort_values("date").set_index("date").ffill().dropna(how="any")

    out_path = PROC_DIR / out_name
    merged.to_parquet(out_path)
    print(f"[data] wrote {out_path}")
    return merged



# CLI test
if __name__ == "__main__":
    SYMS = ["SPY", "QQQ", "TLT"]  # Example universe
    download_universe(SYMS)
    load_and_merge(SYMS)