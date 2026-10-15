"""
Data alignment, date splitting, and dataset preparation utilities
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pandas as pd

from src.paths import resolve


def align_dataframes(
    prices: pd.DataFrame,
    feats: pd.DataFrame,
    dropna: bool = True,
    verbose: bool = False,
    context: str | None = None,
    show_trimming: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align prices and features DataFrames on common timestamps.
    
    Args:
        prices: Price DataFrame with DatetimeIndex
        feats: Features DataFrame with DatetimeIndex
        dropna: If True, drop rows with any NaN in features before alignment
        verbose: If True, print alignment information
        context: Optional context string for error messages
        show_trimming: If True, print message when data is trimmed
    
    Returns:
        Tuple of aligned (prices, feats) DataFrames
    
    Raises:
        ValueError: If no common timestamps exist after alignment
    """
    # Optionally drop NaNs from features
    if dropna:
        feats = feats.dropna(how="any")
    
    # Find common indices
    common = prices.index.intersection(feats.index)
    
    # Check for empty intersection
    if len(common) == 0:
        error_msg = "No common timestamps between prices and features"
        if dropna:
            error_msg += " after dropping NaNs in features"
        if context:
            error_msg += f" ({context})"
        if dropna:
            error_msg += ". Check your warm-up trimming and feature construction."
        else:
            error_msg += "."
        raise ValueError(error_msg)
    
    # Show trimming information if requested
    if show_trimming:
        if len(common) < len(prices.index) or len(common) < len(feats.index):
            print(f"[align] trimming: prices {len(prices)}→{len(common)}, features {len(feats)}→{len(common)}")
    
    # Align both dataframes
    prices_aligned = prices.loc[common]
    feats_aligned = feats.loc[common]
    
    # Print verbose information if requested
    if verbose and len(common) > 0:
        print(f"[data] usable rows: {len(common)} from {cast(pd.Timestamp, common.min()).date()} to "
              f"{cast(pd.Timestamp, common.max()).date()}")
    
    return prices_aligned, feats_aligned


def align_after_load(prices: pd.DataFrame, feats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    1) Drop rows in features that are not fully ready (NaNs from rolling/shift).
    2) Intersect indices and return aligned (prices, feats) on the common dates.
    
    This is a convenience wrapper around align_dataframes() for backward compatibility.
    """
    return align_dataframes(prices, feats, dropna=True, verbose=True)


# Rows dropped from the start of the aligned data before splitting. Features are already
# NaN-free after align_after_load; this extra burn-in is kept so splits match earlier runs.
MAX_LOOKBACK = 60  # days
WARMUP = MAX_LOOKBACK + 1


@dataclass
class DatasetSplits:
    """Aligned prices/features for the whole usable range and each date split (raw, unscaled features)."""
    prices: pd.DataFrame
    feats: pd.DataFrame
    train: tuple[pd.DataFrame, pd.DataFrame]
    val: tuple[pd.DataFrame, pd.DataFrame]
    test: tuple[pd.DataFrame, pd.DataFrame]


def validate_split_dates(
    idx: pd.DatetimeIndex,
    train_end: str,
    val_end: str,
    test_end: str,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """
    Parse split end dates and check they are ordered and leave every split non-empty.
    Raises ValueError with the data range instead of silently adjusting the split.
    """
    if len(idx) == 0:
        raise ValueError("Cannot split: index is empty")
    t_end, v_end, te_end = (cast(pd.Timestamp, pd.Timestamp(d)) for d in (train_end, val_end, test_end))
    first, last = cast(pd.Timestamp, idx.min()), cast(pd.Timestamp, idx.max())
    if not (t_end < v_end < te_end):
        raise ValueError(f"Split dates must satisfy train_end < val_end < test_end; "
                         f"got {t_end.date()}, {v_end.date()}, {te_end.date()}")

    counts = {
        "train": int((idx <= t_end).sum()),
        "val": int(((idx > t_end) & (idx <= v_end)).sum()),
        "test": int(((idx > v_end) & (idx <= te_end)).sum()),
    }
    empty = [name for name, n in counts.items() if n == 0]
    if empty:
        raise ValueError(f"Empty {', '.join(empty)} split for dates train_end={t_end.date()}, "
                         f"val_end={v_end.date()}, test_end={te_end.date()}; "
                         f"usable data spans {first.date()} to {last.date()}")
    if te_end > last:
        print(f"[dates] note: test_end={te_end.date()} is after the last data row ({last.date()})")
    return t_end, v_end, te_end


def prepare_dataset(cfg) -> DatasetSplits:
    """
    Load processed prices/features from cfg.paths, drop the warm-up rows, align them, and split
    by cfg.dates (inclusive end dates). Used by training, HPO and the cost sweep so all of
    them see exactly the same rows.
    """
    prices = pd.read_parquet(resolve(cfg.paths.prices))
    feats = pd.read_parquet(resolve(cfg.paths.features))
    for name, df in (("prices", prices), ("features", feats)):
        if df.index.isna().any():
            raise ValueError(f"{name} has {int(df.index.isna().sum())} row(s) without a date. Rebuild the "
                             f"processed data: python -m src.data && python -m src.features")
        if not (df.index.is_monotonic_increasing and df.index.is_unique):
            raise ValueError(f"{name} index must be sorted and unique")

    prices, feats = align_after_load(prices, feats)
    prices, feats = prices.iloc[WARMUP:], feats.iloc[WARMUP:]

    idx = pd.DatetimeIndex(prices.index)
    t_end, v_end, te_end = validate_split_dates(idx, cfg.dates.train_end, cfg.dates.val_end, cfg.dates.test_end)
    masks = {
        "train": idx <= t_end,
        "val": (idx > t_end) & (idx <= v_end),
        "test": (idx > v_end) & (idx <= te_end),
    }
    splits = {name: (prices.loc[m], feats.loc[m]) for name, m in masks.items()}

    for name, (P, _) in splits.items():
        print(f"[split] {name:<5} {len(P):>5} rows  {P.index.min().date()} → {P.index.max().date()}")

    return DatasetSplits(prices=prices, feats=feats, **splits)
