"""
backtest.py — Evaluate trading strategies (RL agent & baselines).

- Aligns prices & features on common dates
- Backtesting loop passes (features_t, w_prev, t) into policy_fn
- Baselines: Buy & Hold, Equal Weight
- Metrics: Sharpe, Max Drawdown, Turnover
"""

from __future__ import annotations
from typing import Callable, Dict, Any
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from pathlib import Path

# Local imports
from src.scaler import FitOnTrainScaler

# Baselines module (new)
from src.baselines import (
    CostCfg,
    vt_equal_weight,
    ivp_inverse_variance,
    momentum_xs,
    _run_backtest,  # engine to execute with costs
)

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
ART  = Path(__file__).resolve().parent.parent / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------
# Metrics (legacy helpers used by simple loop + RL eval)
# --------------------------------------------------------------------------------------
def sharpe_ratio(returns: np.ndarray, freq: int = 252) -> float:
    if returns.std() == 0:
        return 0.0
    return np.sqrt(freq) * returns.mean() / (returns.std() + 1e-12)

def max_drawdown(equity_curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    return float(-dd.min()) if dd.size else 0.0

def turnover(weights: np.ndarray) -> float:
    if len(weights) < 2:
        return 0.0
    return float(np.abs(np.diff(weights, axis=0)).sum(axis=1).mean())

# --------------------------------------------------------------------------------------
# Simple backtest engine (for Buy&Hold / Equal-Weight baselines)
# --------------------------------------------------------------------------------------
def _align(prices: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inner-join on dates so we never index out of bounds."""
    idx = prices.index.intersection(features.index)
    if idx.empty:
        raise ValueError("prices and features have disjoint indices; cannot backtest.")
    return prices.loc[idx], features.loc[idx]

def run_backtest(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    policy_fn: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
    start_value: float = 1.0,
) -> Dict[str, Any]:
    """
    Simulate portfolio evolution using a policy function (no explicit cost model here).
    """
    prices, features = _align(prices, features)
    T, N = prices.shape
    rets_assets = prices.pct_change().fillna(0.0).to_numpy()  # (T, N)

    equity = np.empty(T, dtype=np.float64)
    equity[0] = start_value

    weights = np.empty((T, N), dtype=np.float64)
    w_prev = np.ones(N) / N  # start equal
    weights[0] = w_prev

    for t in range(1, T):
        feats_t = features.iloc[t].to_numpy(dtype=np.float64)

        # Ask policy for today's target weights given features_t and previous weights
        w = policy_fn(feats_t, w_prev, t)
        w = np.asarray(w, dtype=np.float64).ravel()
        if w.size != N:
            raise ValueError(f"policy_fn returned {w.size} weights but N={N}")
        s = w.sum()
        if s <= 0:
            w = np.ones(N) / N
        else:
            w = np.maximum(w, 0.0) / s  # simple normalization/clip

        weights[t] = w

        # Realize today's return
        port_ret = float(np.dot(w, rets_assets[t]))
        equity[t] = equity[t - 1] * (1.0 + port_ret)
        w_prev = w

    # Metrics
    port_rets = pd.Series(equity).pct_change().dropna().to_numpy()
    metrics = {
        "final_value": float(equity[-1]),
        "sharpe": sharpe_ratio(port_rets),
        "max_drawdown": max_drawdown(equity),
        "turnover": turnover(weights),
    }

    return {"equity_curve": equity, "weights": weights, "metrics": metrics}

# --------------------------------------------------------------------------------------
# Simple baselines (legacy): Buy & Hold, Equal Weight
# --------------------------------------------------------------------------------------
def buy_and_hold(prices: pd.DataFrame, features: pd.DataFrame) -> Dict[str, Any]:
    N = prices.shape[1]
    w0 = np.ones(N) / N
    def policy_fn(features_t, w_prev, t):
        return w0  # never rebalance
    return run_backtest(prices, features, policy_fn)

def equal_weight(prices: pd.DataFrame, features: pd.DataFrame) -> Dict[str, Any]:
    N = prices.shape[1]
    def policy_fn(features_t, w_prev, t):
        return np.ones(N) / N  # rebalance daily to equal
    return run_backtest(prices, features, policy_fn)

# --------------------------------------------------------------------------------------
# Evaluate an SB3 PPO model in your custom env
# --------------------------------------------------------------------------------------
def evaluate_sb3_model(prices: pd.DataFrame, features: pd.DataFrame, model_path: str) -> Dict[str, Any]:
    from src.env import PortfolioEnv, EnvConfig
    env = PortfolioEnv(prices, features, EnvConfig())
    model = PPO.load(model_path, device="cpu")

    equity = [1.0]
    weights = []
    state, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(state, deterministic=True)
        state, reward, done, trunc, info = env.step(action)
        weights.append(info["weights"])
        equity.append(info["value"])

    equity = np.array(equity[1:])
    weights = np.array(weights)
    rets_port = pd.Series(equity).pct_change().dropna().to_numpy()
    metrics = {
        "final_value": float(equity[-1]),
        "sharpe": sharpe_ratio(rets_port),
        "max_drawdown": max_drawdown(equity),
        "turnover": turnover(weights),
    }
    return {"equity_curve": equity, "weights": weights, "metrics": metrics}

# --------------------------------------------------------------------------------------
# Main / CLI
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, glob

    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="", help="Path to a run folder under artifacts/ (contains scaler.json, models/)")
    args = ap.parse_args()

    # Resolve run_dir: use provided path or auto-pick latest artifacts/*__*
    if args.run_dir:
        RUN = Path(args.run_dir).resolve()
    else:
        runs = sorted((ART).glob("*__*"))
        if not runs:
            raise FileNotFoundError(
                "No run folders found under artifacts/. "
                "Run training first: python -m src.train --config configs/rlppo.yaml"
            )
        RUN = runs[-1]

    scaler_path = RUN / "scaler.json"
    best_model_path = RUN / "models" / "best" / "best_model.zip"
    final_model_path = RUN / "models" / "ppo_dirichlet_sb3.zip"  # fallback if you saved here

    if not scaler_path.exists():
        raise FileNotFoundError(f"scaler.json not found at: {scaler_path}\n"
                                f"Did you run train.py? Expected inside the run folder.")

    # Load data
    prices = pd.read_parquet(PROC / "prices_adj.parquet")
    feats   = pd.read_parquet(PROC / "features.parquet")

    # Load train-fitted scaler
    from src.scaler import FitOnTrainScaler  # ensure package import
    scaler = FitOnTrainScaler.load(scaler_path, columns=feats.columns)
    feats_z = scaler.transform(feats)

    # Align once (defensive)
    idx = prices.index.intersection(feats.index)
    prices = prices.loc[idx]
    feats   = feats.loc[idx]
    feats_z = feats_z.loc[idx]

    # ---------------- Legacy baselines (no explicit costs here) ----------------
    bh = buy_and_hold(prices, feats_z)
    ew = equal_weight(prices, feats_z)
    print("Buy & Hold:", bh["metrics"])
    print("Equal Weight:", ew["metrics"])

    # ---------------- RL Agent (if available) ---------------------------------
    rl_zip = best_model_path if best_model_path.exists() else final_model_path
    try:
        rl = evaluate_sb3_model(prices, feats_z, str(rl_zip))
        print("RL Agent (SB3):", rl["metrics"])
    except Exception as e:
        print("RL Agent (SB3): skipped or failed ->", repr(e))

    # =============================================================================
    # NEW: Robust baselines with cost model (VT-EW, IVP, Momentum)
    # =============================================================================

    # 1) Cost config.
    from src.baselines import CostCfg, vt_equal_weight, ivp_inverse_variance, momentum_xs, _run_backtest
    cost_cfg = CostCfg(
        enabled=True,
        fee_bps=0.0,
        slippage_bps=0.0,
        spread_type="fixed",          # change to "vol20" if your RAW features have *_s20
        spread_fixed_bps=0.0,
        spread_k_vol_to_bps=8000.0,
        spread_vol_col_suffix="_s20",
    )

    # If you want volatility-based spread and your RAW features include *_s20:
    # features_for_cost = feats.loc[prices.index]
    features_for_cost = None

    # 2) Build targets for each baseline
    targets_vtew = vt_equal_weight(prices, window=60, target_vol=0.10, lev_cap=1.0, rebalance="W-FRI")
    targets_ivp  = ivp_inverse_variance(prices, window=60, rebalance="W-FRI")
    targets_mom  = momentum_xs(prices, lookback_days=252, skip_days=21, top_frac=0.3, rebalance="M")

    # 3) Run the cost-aware backtests (band throttles trading for VT-EW/IVP)
    res_vtew = _run_backtest(prices, targets_vtew, cost_cfg, features=features_for_cost, band=0.0025)
    res_ivp  = _run_backtest(prices, targets_ivp,  cost_cfg, features=features_for_cost, band=0.0025)
    res_mom  = _run_backtest(prices, targets_mom,  cost_cfg, features=features_for_cost, band=0.0)

    # 4) Comparison table
    def _row(name, r):
        m = r["metrics"]
        return dict(
            baseline=name,
            sharpe=m["sharpe"],
            max_drawdown=m["max_drawdown"],
            turnover=m["turnover"],
            steps=m["num_steps"],
        )

    rows = [
        _row("VT-EW (10% target)", res_vtew),
        _row("IVP", res_ivp),
        _row("Momentum (12-1, top 30%)", res_mom),
    ]
    df_baselines = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    print("\n=== Cost-Aware Baselines (full sample) ===")
    print(df_baselines.to_string(index=False))

    # 5) Save alongside the run artifacts
    out_csv = Path("artifacts") / "baseline_results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_baselines.to_csv(out_csv, index=False)
    print(f"Saved baseline results to {out_csv.resolve()}")
