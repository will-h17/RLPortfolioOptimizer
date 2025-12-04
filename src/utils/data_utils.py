"""
Data alignment and date utilities
"""
from __future__ import annotations
from typing import Tuple, Optional
import pandas as pd


def align_dataframes(
    prices: pd.DataFrame,
    feats: pd.DataFrame,
    dropna: bool = True,
    verbose: bool = False,
    context: Optional[str] = None,
    show_trimming: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        print(f"[data] usable rows: {len(common)} from {common.min().date()} to {common.max().date()}")
    
    return prices_aligned, feats_aligned


def align_after_load(prices: pd.DataFrame, feats: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    1) Drop rows in features that are not fully ready (NaNs from rolling/shift).
    2) Intersect indices and return aligned (prices, feats) on the common dates.
    
    This is a convenience wrapper around align_dataframes() for backward compatibility.
    """
    return align_dataframes(prices, feats, dropna=True, verbose=True)


def clamp_dates_to_index(
    idx: pd.DatetimeIndex, 
    train_end: str, 
    val_end: str, 
    test_end: str
) -> Tuple[str, str, str]:
    """
    Clamp date strings to live within idx[min..max] and enforce ordering train_end <= val_end <= test_end.
    Returns ISO strings to be passed to date_slices.
    """
    idx = pd.to_datetime(idx).tz_localize(None)
    idx = pd.DatetimeIndex(idx.unique()).sort_values()

    if len(idx) == 0:
        raise ValueError("Cannot clamp dates: index is empty")
    
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
        if n < 3:
            raise ValueError(f"Not enough data points ({n}) for train/val/test split. Need at least 3 rows.")
        i_tr = max(1, min(int(0.70 * n) - 1, n - 3))  # Ensure valid index
        i_va = max(i_tr + 1, min(int(0.85 * n) - 1, n - 2))  # Ensure valid index
        i_te = n - 1  # Use last index
        
        # Get dates from index with bounds checking and NaT handling
        if 0 <= i_tr < len(idx):
            t_end = idx[i_tr]
            if pd.notna(t_end):
                t_end = t_end.normalize()
            else:
                t_end = dmin
        else:
            t_end = dmin
            
        if 0 <= i_va < len(idx):
            v_end = idx[i_va]
            if pd.notna(v_end):
                v_end = v_end.normalize()
            else:
                v_end = idx[min(i_tr + 1, n - 1)].normalize() if i_tr + 1 < n else dmax
        else:
            v_end = idx[min(i_tr + 1, n - 1)].normalize() if i_tr + 1 < n else dmax
            
        if 0 <= i_te < len(idx):
            te_end = idx[i_te]
            if pd.notna(te_end):
                te_end = te_end.normalize()
            else:
                te_end = dmax
        else:
            te_end = dmax

    # use searchsorted (works on sorted index)
    i_tr_end = int(idx.searchsorted(t_end, side="left"))
    i_va_end = int(idx.searchsorted(v_end, side="left"))
    i_te_end = int(idx.searchsorted(te_end, side="left"))

    # ensure segments are non-empty; fallback if necessary
    if not (0 <= i_tr_end < len(idx) and i_tr_end < i_va_end < i_te_end <= len(idx)):
        n = len(idx)
        if n < 3:
            raise ValueError(f"Not enough data points ({n}) for train/val/test split. Need at least 3 rows.")
        i_tr = max(1, min(int(0.70 * n) - 1, n - 3))  # Ensure valid index
        i_va = max(i_tr + 1, min(int(0.85 * n) - 1, n - 2))  # Ensure valid index
        i_te = n - 1  # Use last index
        
        # Safely get dates from index with bounds checking and NaT handling
        if 0 <= i_tr < len(idx):
            t_end_val = idx[i_tr]
            t_end = t_end_val.normalize() if pd.notna(t_end_val) else dmin
        else:
            t_end = dmin
            
        if 0 <= i_va < len(idx):
            v_end_val = idx[i_va]
            v_end = v_end_val.normalize() if pd.notna(v_end_val) else idx[min(i_tr + 1, n - 1)].normalize()
        else:
            v_end = idx[min(i_tr + 1, n - 1)].normalize() if i_tr + 1 < n else dmax
            
        if 0 <= i_te < len(idx):
            te_end_val = idx[i_te]
            te_end = te_end_val.normalize() if pd.notna(te_end_val) else dmax
        else:
            te_end = dmax

    # Final validation: ensure all dates are valid (not NaT)
    if pd.isna(t_end) or pd.isna(v_end) or pd.isna(te_end):
        # Last resort: use simple split
        n = len(idx)
        if n < 3:
            raise ValueError(f"Not enough data points ({n}) for train/val/test split.")
        i_tr = max(1, n // 3)
        i_va = max(i_tr + 1, 2 * n // 3)
        i_te = n - 1
        
        # Get dates with NaT checking
        t_end_val = idx[i_tr] if i_tr < len(idx) else idx[0]
        v_end_val = idx[i_va] if i_va < len(idx) else idx[min(i_tr + 1, n - 1)]
        te_end_val = idx[i_te] if i_te < len(idx) else idx[-1]
        
        t_end = t_end_val.normalize() if pd.notna(t_end_val) else dmin
        v_end = v_end_val.normalize() if pd.notna(v_end_val) else dmax
        te_end = te_end_val.normalize() if pd.notna(te_end_val) else dmax

    return str(t_end.date()), str(v_end.date()), str(te_end.date())

