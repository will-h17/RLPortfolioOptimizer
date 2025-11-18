"""
Data alignment and date utilities to avoid code redundancy.
"""
from __future__ import annotations
from typing import Tuple
import pandas as pd


def align_after_load(prices: pd.DataFrame, feats: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    1) Drop rows in features that are not fully ready (NaNs from rolling/shift).
    2) Intersect indices and return aligned (prices, feats) on the common dates.
    """
    feats = feats.dropna(how="any")  # keep only rows where all features exist
    common = prices.index.intersection(feats.index)
    if len(common) == 0:
        raise ValueError(
            "No common timestamps after dropping NaNs in features. "
            "Check your warm-up trimming and feature construction."
        )
    prices = prices.loc[common]
    feats = feats.loc[common]
    print(f"[data] usable rows: {len(common)} from {common.min().date()} to {common.max().date()}")
    return prices, feats


def clamp_dates_to_index(
    idx: pd.DatetimeIndex, 
    train_end: str, 
    val_end: str, 
    test_end: str
) -> Tuple[str, str, str]:
    """
    Clamp date strings to live within idx[min..max] and enforce ordering train_end <= val_end <= test_end.
    Returns ISO strings we can pass to date_slices.
    """
    idx = pd.to_datetime(idx).tz_localize(None)
    idx = pd.DatetimeIndex(idx.unique()).sort_values()

    dmin, dmax = idx.min().normalize(), idx.max().normalize()

    t_end = pd.to_datetime(train_end).normalize()
    v_end = pd.to_datetime(val_end).normalize()
    te_end = pd.to_datetime(test_end).normalize()

    # clamp into bounds
    t_end = min(max(t_end, dmin), dmax)
    v_end = min(max(v_end, dmin), dmax)
    te_end = min(max(te_end, dmin), dmax)

    # enforce nondecreasing order; fallback if needed
    if not (t_end <= v_end <= te_end):
        n = len(idx)
        i_tr = max(1, int(0.70 * n) - 1)
        i_va = max(i_tr + 1, int(0.85 * n) - 1)
        t_end, v_end, te_end = idx[i_tr].normalize(), idx[i_va].normalize(), idx[-1].normalize()

    # use searchsorted (works on sorted index) instead of get_indexer(..., method="bfill")
    i_tr_end = int(idx.searchsorted(t_end, side="left"))
    i_va_end = int(idx.searchsorted(v_end, side="left"))
    i_te_end = int(idx.searchsorted(te_end, side="left"))

    # ensure segments are non-empty; fallback if necessary
    if not (0 <= i_tr_end < len(idx) and i_tr_end < i_va_end < i_te_end):
        n = len(idx)
        i_tr = max(1, int(0.70 * n) - 1)
        i_va = max(i_tr + 1, int(0.85 * n) - 1)
        t_end, v_end, te_end = idx[i_tr].normalize(), idx[i_va].normalize(), idx[-1].normalize()

    return str(t_end.date()), str(v_end.date()), str(te_end.date())

