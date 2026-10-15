"""
Portfolio performance metrics utilities.

Provides standardized implementations of:
- Sharpe ratio (annualized)
- Maximum drawdown
- Portfolio turnover
"""

from __future__ import annotations

import numpy as np


def sharpe_ratio(returns: np.ndarray, freq: int = 252) -> float:
    """
    Calculate annualized Sharpe ratio from daily returns.
    
    Args:
        returns: Array of daily returns
        freq: Trading days per year (default 252)
    
    Returns:
        Annualized Sharpe ratio (0.0 if insufficient data or zero std)
    """
    if len(returns) < 2:
        return 0.0
    mu = returns.mean()
    sd = returns.std()
    if sd == 0 or not np.isfinite(mu) or not np.isfinite(sd):
        return 0.0
    return float(np.sqrt(freq) * mu / (sd + 1e-12))


def max_drawdown(equity_curve: np.ndarray, return_positive: bool = False) -> float:
    """
    Calculate maximum drawdown from equity curve.
    
    Args:
        equity_curve: Array of portfolio values over time
        return_positive: If True, return positive value (absolute drawdown).
                        If False, return negative value (e.g., -0.23 for -23%)
    
    Returns:
        Maximum drawdown as a float. Negative by default (e.g., -0.23 means -23%).
        Positive if return_positive=True (e.g., 0.23 means 23%).
    """
    if len(equity_curve) < 2:
        return 0.0
    
    peak = np.maximum.accumulate(equity_curve)
    dd = equity_curve / peak - 1.0
    mdd = float(dd.min())
    
    if return_positive:
        return abs(mdd)
    return mdd


def turnover(weights: np.ndarray | list[np.ndarray]) -> float:
    """
    Calculate average portfolio turnover from weight changes.
    
    Args:
        weights: Either:
                - 2D numpy array of shape [T, N] where T=time, N=assets
                - List of 1D numpy arrays, each representing weights at a time step
    
    Returns:
        Average L1 turnover per period (fraction of portfolio turned over)
    """
    # Handle list of arrays (stack them)
    if isinstance(weights, list):
        if len(weights) < 2:
            return 0.0
        W = np.vstack(weights)  # [T, N]
    else:
        W = np.asarray(weights)
        if len(W) < 2:
            return 0.0
        # Ensure 2D
        if W.ndim == 1:
            W = W.reshape(1, -1)
    
    # Calculate L1 change per step, then average
    per_step = np.abs(np.diff(W, axis=0)).sum(axis=1)
    return float(per_step.mean())



def cagr(returns: np.ndarray, freq: int = 252) -> float:
    """Compound annual growth rate from per-period returns."""
    returns = np.asarray(returns, dtype=float)
    if len(returns) == 0:
        return 0.0
    growth = float(np.prod(1.0 + returns))
    return growth ** (freq / len(returns)) - 1.0 if growth > 0 else -1.0


def summarize_returns(returns: np.ndarray, trades: np.ndarray | None = None, freq: int = 252) -> dict:
    """
    Standard performance summary for a series of per-period net returns. `trades` is the L1 size of
    each period's trade (info["turnover"] from the env); its mean is reported as turnover.
    """
    returns = np.asarray(returns, dtype=float)
    equity = np.cumprod(1.0 + returns)
    return {
        "final_value": float(equity[-1]) if len(equity) else 1.0,
        "cagr": cagr(returns, freq),
        "ann_vol": float(returns.std() * np.sqrt(freq)) if len(returns) > 1 else 0.0,
        "sharpe": sharpe_ratio(returns, freq),
        "max_drawdown": max_drawdown(np.concatenate([[1.0], equity])),  # negative, from the starting value
        "turnover": float(np.mean(trades)) if trades is not None and len(trades) else float("nan"),
        "num_steps": int(len(returns)),
    }
