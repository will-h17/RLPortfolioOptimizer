from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd

# -------------------------------
# Cost model (fee + slippage + 1/2 spread)
# -------------------------------

@dataclass
class CostCfg:
    enabled: bool = False
    # legacy flat cost (fraction, e.g. 0.0005 for 5 bps) used only when enabled=False
    flat_fraction: float = 0.0

    # component bps when enabled=True
    fee_bps: float = 0.0
    slippage_bps: float = 0.0
    spread_type: str = "fixed"        # "fixed" | "vol20"
    spread_fixed_bps: float = 0.0
    spread_k_vol_to_bps: float = 8000.0
    spread_vol_col_suffix: str = "_s20"  # e.g., columns like "AAPL_s20" with daily rolling std

def _per_asset_cost_bps(
    t_loc: pd.Timestamp,
    asset_cols: List[str],
    features: Optional[pd.DataFrame],
    cfg: CostCfg
) -> np.ndarray:
    """
    Return per-asset bps cost for crossing once at time t (fee + slippage + half-spread).
    If cfg.enabled=False, returns vector equivalent to flat_fraction.
    """
    n = len(asset_cols)
    if not cfg.enabled:
        flat_bps = float(cfg.flat_fraction) * 10_000.0
        return np.full(n, flat_bps, dtype=np.float64)

    fee = float(cfg.fee_bps)
    slp = float(cfg.slippage_bps)

    # default: fixed spread or 0
    spread_bps = np.full(n, float(cfg.spread_fixed_bps), dtype=np.float64)

    if cfg.spread_type == "vol20" and features is not None and t_loc in features.index:
        suf = cfg.spread_vol_col_suffix
        vol_cols = [f"{c}{suf}" for c in asset_cols]
        if all(c in features.columns for c in vol_cols):
            row = features.loc[t_loc, vol_cols]
            vol = row.to_numpy(dtype=np.float64)  # daily vol ~ 0.01
            spread_bps = cfg.spread_k_vol_to_bps * vol  # bps
        # else: keep fixed/zero fallback

    return fee + slp + 0.5 * spread_bps


# -------------------------------
# Metrics helpers
# -------------------------------

def _sharpe_daily(rets: np.ndarray) -> float:
    if len(rets) < 2:
        return 0.0
    mu, sd = rets.mean(), rets.std()
    return float(np.sqrt(252.0) * mu / (sd + 1e-12))

def _max_drawdown(equity: np.ndarray) -> float:
    if len(equity) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(dd.min())

def _avg_turnover(weights_seq: List[np.ndarray]) -> float:
    if len(weights_seq) < 2:
        return 0.0
    W = np.vstack(weights_seq)  # [T, N]
    per_step = np.abs(np.diff(W, axis=0)).sum(axis=1)
    return float(per_step.mean())


# -------------------------------
# Rebalancing / engine utilities
# -------------------------------

def _price_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().fillna(0.0)

def _apply_cost(
    w_prev: np.ndarray,
    w_new: np.ndarray,
    t_loc: pd.Timestamp,
    asset_cols: List[str],
    features: Optional[pd.DataFrame],
    cfg: CostCfg
) -> float:
    """Return cost as a fraction of portfolio value due to rebalancing at t."""
    dw = np.abs(w_new - w_prev)
    bps = _per_asset_cost_bps(t_loc, asset_cols, features, cfg)
    return float(np.dot(bps, dw)) / 10_000.0

def _run_backtest(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,   # index aligned to prices.index, columns=assets, rows sum to <= 1 (long-only)
    cost_cfg: CostCfg,
    features: Optional[pd.DataFrame] = None,
    band: float = 0.0               # trade only if |w_target - w_current| L1 > band
) -> Dict[str, object]:
    """
    Execute daily portfolio using provided target weights (piecewise-constant between rebalances).
    Applies costs only when we trade (change weights).
    """
    assets = list(prices.columns)
    R = _price_to_returns(prices).to_numpy(dtype=np.float64)
    dates = prices.index

    # start flat; first target will trade into position (cost applies)
    w = np.zeros(len(assets), dtype=np.float64)
    w_hist, rets = [], []

    # forward-fill target weights to daily grid
    T = len(dates)
    TW = target_weights.reindex(dates).ffill().fillna(0.0).to_numpy(dtype=np.float64)

    for t in range(T):
        # portfolio return using *previous* weights
        port_ret_gross = float(np.dot(w, R[t, :]))
        cost = 0.0

        # after earning return at t, decide if we rebalance *for next day*
        w_target = TW[t, :].copy()
        # normalize safety (avoid drift in targets)
        s = w_target.sum()
        if s > 0:
            w_target /= s

        # trade only if band is exceeded (L1 distance)
        if np.sum(np.abs(w_target - w)) > band + 1e-12:
            cost = _apply_cost(w, w_target, dates[t], assets, features, cost_cfg)
            w = w_target

        port_ret_net = port_ret_gross - cost
        rets.append(port_ret_net)
        w_hist.append(w.copy())

    rets = np.asarray(rets, dtype=np.float64)
    equity = (1.0 + rets).cumprod()
    metrics = {
        "sharpe": _sharpe_daily(rets),
        "max_drawdown": _max_drawdown(equity),
        "turnover": _avg_turnover(w_hist),
        "num_steps": int(len(rets)),
    }
    return {
        "metrics": metrics,
        "equity": equity,
        "weights": np.vstack(w_hist) if w_hist else np.zeros((0, len(assets))),
        "dates": dates,
        "assets": assets,
        "returns": rets,
    }


# -------------------------------
# Baseline 1: Vol-targeted Equal-Weight
# -------------------------------

def vt_equal_weight(
    prices: pd.DataFrame,
    window: int = 60,
    target_vol: float = 0.10,      # annualized (e.g., 10%)
    lev_cap: float = 1.0,
    rebalance: str = "W-FRI"       # Pandas offset alias
) -> pd.DataFrame:
    """
    Produce target weights DataFrame (index: rebal dates, columns: assets).
    Equal-weight by capital, then scale to hit target_vol (cap leverage).
    """
    assets = prices.columns
    rebal_idx = prices.resample(rebalance).last().index.intersection(prices.index)
    # daily returns matrix
    R = _price_to_returns(prices)

    # rolling covariance (use simple diagonal approx for speed/stability)
    vol = R.rolling(window).std() * np.sqrt(252.0)  # annualized vol per asset

    rows = []
    for t in rebal_idx:
        if t not in vol.index:
            continue
        # base EW
        w = np.full(len(assets), 1.0 / len(assets), dtype=np.float64)
        # portfolio vol using diagonal approx
        v = vol.loc[t, assets].to_numpy(dtype=np.float64)
        port_vol = float(np.sqrt((w ** 2 * v ** 2).sum()))
        lev = np.clip(target_vol / (port_vol + 1e-12), 0.0, lev_cap)
        w = w * lev
        # renormalize to sum≤1 (long-only, leftover is cash)
        s = w.sum()
        if s > 1.0:
            w = w / s
        rows.append(pd.Series(w, index=assets, name=t))

    return pd.DataFrame(rows)


# -------------------------------
# Baseline 2: Inverse-Variance Portfolio (IVP)
# -------------------------------

def ivp_inverse_variance(
    prices: pd.DataFrame,
    window: int = 60,
    rebalance: str = "W-FRI"
) -> pd.DataFrame:
    """Weights ∝ 1/variance (diagonal covariance)."""
    assets = prices.columns
    rebal_idx = prices.resample(rebalance).last().index.intersection(prices.index)
    var = _price_to_returns(prices).rolling(window).var()

    rows = []
    for t in rebal_idx:
        if t not in var.index:
            continue
        inv_var = 1.0 / (var.loc[t, assets].to_numpy(dtype=np.float64) + 1e-12)
        w = inv_var / inv_var.sum()
        rows.append(pd.Series(w, index=assets, name=t))
    return pd.DataFrame(rows)


# -------------------------------
# Baseline 3: Naive Cross-Sectional Momentum (12-1 style)
# -------------------------------

def momentum_xs(
    prices: pd.DataFrame,
    lookback_days: int = 252,   # ~12m
    skip_days: int = 21,        # skip most-recent month
    top_frac: float = 0.3,      # long top 30%
    rebalance: str = "M"        # month-end
) -> pd.DataFrame:
    """
    Cross-sectional momentum: rank by trailing (lookback - skip) return,
    long top fraction equally, others 0 (long-only).
    """
    assets = prices.columns
    rebal_idx = prices.resample(rebalance).last().index.intersection(prices.index)

    # precompute trailing return
    # R_{t} = P_{t} / P_{t-L} - 1 (skip most-recent 'skip_days')
    trailing = prices / prices.shift(lookback_days) - 1.0
    if skip_days > 0:
        trailing = trailing.shift(skip_days)

    rows = []
    k = max(1, int(np.ceil(len(assets) * top_frac)))
    for t in rebal_idx:
        if t not in trailing.index:
            continue
        scores = trailing.loc[t, assets].to_numpy(dtype=np.float64)
        # handle missing: set to -inf to avoid selection
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        # pick top-k
        top_idx = np.argsort(scores)[-k:]
        w = np.zeros(len(assets), dtype=np.float64)
        w[top_idx] = 1.0 / k
        rows.append(pd.Series(w, index=assets, name=t))
    return pd.DataFrame(rows)
