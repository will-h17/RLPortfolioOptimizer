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

import pandas as pd
import yfinance as yf

from src.paths import PROC_DIR, RAW_DIR


def download_universe(symbols: list[str], force: bool = False) -> list[Path]:
    """
    Download daily adjusted data for all symbols and cache to CSV.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for sym in symbols:
        path = RAW_DIR / f"{sym}_daily_adjusted.csv"
        if path.exists() and not force:
            print(f"[yfinance] cached {path.name}")
            paths.append(path)
            continue

        print(f"[yfinance] downloading {sym}...")
        df = yf.download(sym, period="max", auto_adjust=True)  # auto_adjust=True gives adjusted close
        if df is None or df.empty:
            raise RuntimeError(f"yfinance returned no data for {sym!r} (bad symbol or network problem)")
        # yfinance >= 0.2.48 returns (field, ticker) columns even for one ticker; writing those to CSV
        # adds a second header row that would be read back as an undated price row, so keep only
        # the dates and the adjusted close values
        close = df["Close"].to_numpy().reshape(len(df), -1)[:, 0]
        df = pd.DataFrame({"date": df.index, sym: close})

        df.to_csv(path, index=False)
        print(f"[yfinance] wrote {path}")
        paths.append(path)
    return paths


def load_and_merge(symbols: list[str], out_name: str = "prices_adj.parquet") -> pd.DataFrame:
    dfs = []
    for sym in symbols:
        path = RAW_DIR / f"{sym}_daily_adjusted.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run download_universe first.")
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        # Drop rows without a valid date (e.g. the extra ticker header row in CSVs cached by
        # older versions of this script)
        bad = df["date"].isna()
        if bool(bad.any()):
            print(f"[data] {path.name}: dropping {int(bad.sum())} row(s) without a valid date")
            df = df.loc[~bad]
        # Ensure numeric
        df[sym] = pd.to_numeric(df[sym], errors="coerce")
        dfs.append(df)

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="date", how="outer")

    merged = merged.sort_values("date").set_index("date").ffill().dropna(how="any")

    out_path = PROC_DIR / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path)
    print(f"[data] wrote {out_path}")
    return merged



# CLI test
if __name__ == "__main__":
    SYMS = ["SPY", "QQQ", "TLT"]  # Example universe
    download_universe(SYMS)
    load_and_merge(SYMS)