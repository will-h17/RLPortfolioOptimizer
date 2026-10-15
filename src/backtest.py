"""
backtest.py — Evaluate trading strategies (RL agent & baselines).

- Aligns prices & features on common dates
- Backtesting loop passes (features_t, w_prev, t) into policy_fn
- Baselines: Buy & Hold, Equal Weight
- Metrics: Sharpe, Max Drawdown, Turnover
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# Baselines module
from src.baselines import (
    CostCfg,
    _run_backtest,  # engine to execute with costs
    ivp_inverse_variance,
    momentum_xs,
    vt_equal_weight,
)
from src.env import EnvConfig, PortfolioEnv, load_run_env_config
from src.paths import ARTIFACTS_DIR, FEATURES_PATH, PRICES_PATH
from src.runlog import latest_run, resolve_run_model
from src.scaler import FitOnTrainScaler
from src.utils.data_utils import align_dataframes
from src.utils.metrics import max_drawdown, sharpe_ratio, turnover


# Simple backtest engine (for Buy&Hold / Equal-Weight baselines)
def run_backtest(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    policy_fn: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
    start_value: float = 1.0,
) -> dict[str, Any]:
    """
    Simulate portfolio evolution using a policy function (no transaction costs).

    At each day t the policy sees features_t and the current (drifted) holdings, and
    returns target weights that earn the *next* day's return P[t+1]/P[t] - 1.
    Turnover is measured as |target - drifted holdings|, i.e. what was actually traded.
    """
    prices, features = align_dataframes(prices, features, dropna=False, context="backtest")
    T, N = prices.shape
    rets_assets = prices.pct_change().fillna(0.0).to_numpy()  # row t = P[t]/P[t-1] - 1

    equity = np.empty(T, dtype=np.float64)
    equity[0] = start_value

    weights = np.empty((T, N), dtype=np.float64)
    trades = np.zeros(T, dtype=np.float64)
    w_held = np.ones(N) / N  # start equal
    weights[0] = w_held

    for t in range(T - 1):
        feats_t = features.iloc[t].to_numpy(dtype=np.float64)

        # Ask policy for target weights given information up to t and current holdings
        w = policy_fn(feats_t, w_held, t)
        w = np.maximum(np.asarray(w, dtype=np.float64).ravel(), 0.0)
        if w.size != N:
            raise ValueError(f"policy_fn returned {w.size} weights but N={N}")
        s = w.sum()
        w = np.ones(N) / N if s <= 0 else w / s
        trades[t] = np.abs(w - w_held).sum()

        # Realize the next day's return, then let holdings drift with prices
        r_next = rets_assets[t + 1]
        port_ret = float(np.dot(w, r_next))
        equity[t + 1] = equity[t] * (1.0 + port_ret)
        w_held = w * (1.0 + r_next) / (1.0 + port_ret)
        weights[t + 1] = w_held

    # Metrics
    port_rets = pd.Series(equity).pct_change().dropna().to_numpy()
    metrics = {
        "final_value": float(equity[-1]),
        "sharpe": sharpe_ratio(port_rets),
        "max_drawdown": max_drawdown(equity, return_positive=True),  # Return positive value for consistency
        "turnover": float(trades[1:T - 1].mean()) if T > 2 else 0.0,  # excludes the initial allocation
    }

    return {"equity_curve": equity, "weights": weights, "metrics": metrics}

# Simple baselines (legacy): Buy & Hold, Equal Weight
def buy_and_hold(prices: pd.DataFrame, features: pd.DataFrame) -> dict[str, Any]:
    def policy_fn(features_t, w_held, t):
        return w_held  # start equal-weight, never rebalance; weights drift with prices
    return run_backtest(prices, features, policy_fn)

def equal_weight(prices: pd.DataFrame, features: pd.DataFrame) -> dict[str, Any]:
    N = prices.shape[1]
    def policy_fn(features_t, w_held, t):
        return np.ones(N) / N  # rebalance daily to equal
    return run_backtest(prices, features, policy_fn)

def rollout_policy(model, env: PortfolioEnv) -> dict[str, np.ndarray]:
    """
    Run one deterministic episode of `model` in `env` (a raw PortfolioEnv, not a VecEnv).
    Returns per-step net returns, post-trade weights, trade sizes (L1), and portfolio values.
    """
    obs, _ = env.reset()
    rets, weights, trades, values = [], [], [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rets.append(info["ret"])
        weights.append(info["weights"])
        trades.append(info["turnover"])
        values.append(info["value"])
    return {"returns": np.asarray(rets), "weights": np.asarray(weights),
            "trades": np.asarray(trades), "values": np.asarray(values)}


# Evaluate an SB3 PPO model in the custom env
def evaluate_sb3_model(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    model_path: str,
    env_cfg: EnvConfig,
) -> dict[str, Any]:
    """
    Roll out a trained model deterministically over the whole period. env_cfg must be the config
    the model was trained with (see src.env.load_run_env_config) so costs, cash and observations
    match; training-only episode settings (random windows) are switched off.
    """
    env = PortfolioEnv(prices, features, env_cfg.for_evaluation())
    model = PPO.load(model_path, device="cpu")
    res = rollout_policy(model, env)

    equity, weights = res["values"], res["weights"]
    metrics = {
        "final_value": float(equity[-1]),
        "sharpe": sharpe_ratio(res["returns"]),
        "max_drawdown": max_drawdown(equity, return_positive=True),  # Return positive value for consistency
        "turnover": turnover(weights),
    }
    return {"equity_curve": equity, "weights": weights, "returns": res["returns"], "metrics": metrics}

# Main CLI entry point
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="", help="Path to a run folder under artifacts/ (contains scaler.json, models/)")
    args = ap.parse_args()

    # Resolve run_dir: use provided path or auto-pick the latest completed run
    RUN = Path(args.run_dir).resolve() if args.run_dir else latest_run(ARTIFACTS_DIR)
    scaler_path = RUN / "scaler.json"

    if not scaler_path.exists():
        raise FileNotFoundError(f"scaler.json not found at: {scaler_path}\n"
                                f"Did you run train.py? Expected inside the run folder.")

    # Load data
    prices = pd.read_parquet(PRICES_PATH)
    feats   = pd.read_parquet(FEATURES_PATH)

    # Load train-fitted scaler
    from src.scaler import FitOnTrainScaler  # ensure package import
    scaler = FitOnTrainScaler.load(scaler_path, columns=feats.columns)
    feats_z = scaler.transform(feats)

    # Align once (defensive)
    idx = prices.index.intersection(feats.index)
    prices = prices.loc[idx]
    feats   = feats.loc[idx]
    feats_z = feats_z.loc[idx]

    # ---------------- Legacy baselines ----------------
    bh = buy_and_hold(prices, feats_z)
    ew = equal_weight(prices, feats_z)
    print("Buy & Hold:", bh["metrics"])
    print("Equal Weight:", ew["metrics"])

    # ---------------- RL Agent ---------------------------------
    try:
        rl = evaluate_sb3_model(prices, feats_z, str(resolve_run_model(RUN)), load_run_env_config(RUN))
        print("RL Agent (SB3):", rl["metrics"])
    except Exception as e:
        print("RL Agent (SB3): skipped or failed ->", repr(e))

    # 1) Cost config.
    from src.baselines import CostCfg, _run_backtest, ivp_inverse_variance, momentum_xs, vt_equal_weight
    cost_cfg = CostCfg(
        enabled=True,
        fee_bps=0.0,
        slippage_bps=0.0,
        spread_type="fixed",          # change to "vol20" if your RAW features have *_s20
        spread_fixed_bps=0.0,
        spread_k_vol_to_bps=8000.0,
        spread_vol_col_suffix="_s20",
    )

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
    out_csv = ARTIFACTS_DIR / "baseline_results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_baselines.to_csv(out_csv, index=False)
    print(f"Saved baseline results to {out_csv.resolve()}")
