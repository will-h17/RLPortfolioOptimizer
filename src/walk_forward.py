"""
walk_forward.py — Walk-forward evaluation across yearly test windows and multiple seeds.

For each test year Y (from --first-test-year to the end of the data):
  train = everything before the validation window (expanding)
  val   = the --val-months before Y (model selection by validation Sharpe)
  test  = calendar year Y
an agent is trained per seed with the given config, and evaluated on the test year alongside
baselines (equal weight, buy & hold, inverse variance) run through the *same* PortfolioEnv, so
costs, cash yield and trade timing are identical for every strategy.

Outputs in artifacts/walk_forward/<name>/:
  runs.csv     one row per (fold, strategy, seed) with performance metrics
  returns.csv  dated daily net returns for every row in runs.csv
  summary.csv  per-strategy mean/std across folds, plus full-period stitched results
  runs/        the training run folders
Re-running with the same --name skips (fold, seed) pairs that already finished.

Usage:
  python -m src.walk_forward --config configs/rlppo_v2.yaml --seeds 0 1 2 --first-test-year 2016
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from src.config_loader import DatesCfg, load_config
from src.env import EnvConfig, PortfolioEnv, build_env_config
from src.paths import ARTIFACTS_DIR, resolve
from src.train import train_once
from src.utils.data_utils import WARMUP, align_after_load, prepare_dataset
from src.utils.metrics import summarize_returns

IVP_WINDOW = 60       # trailing days for inverse-variance weights
IVP_REBALANCE = 5     # rebalance every 5 trading days (weekly)
MIN_TEST_ROWS = 60    # skip a final partial year shorter than this


# ---------------- folds ----------------

def make_folds(index: pd.DatetimeIndex, first_test_year: int, val_months: int = 6) -> list[dict[str, str]]:
    """Yearly test windows from first_test_year onward, with the preceding val_months as validation."""
    folds = []
    last = cast(pd.Timestamp, index.max())
    for year in range(first_test_year, last.year + 1):
        test_start = cast(pd.Timestamp, pd.Timestamp(year, 1, 1))
        if ((index >= test_start) & (index <= pd.Timestamp(year, 12, 31))).sum() < MIN_TEST_ROWS:
            continue
        val_start = test_start - pd.DateOffset(months=val_months)
        folds.append({
            "fold": str(year),
            "train_end": str((val_start - pd.Timedelta(days=1)).date()),
            "val_end": str((test_start - pd.Timedelta(days=1)).date()),
            "test_end": f"{year}-12-31",
        })
    if not folds:
        raise ValueError(f"No test years with >= {MIN_TEST_ROWS} rows from {first_test_year} "
                         f"(data ends {last.date()})")
    return folds


def _usable_index(cfg) -> pd.DatetimeIndex:
    prices = pd.read_parquet(resolve(cfg.paths.prices))
    feats = pd.read_parquet(resolve(cfg.paths.features))
    prices, _ = align_after_load(prices, feats)
    return pd.DatetimeIndex(prices.index[WARMUP:])


# ---------------- strategies run through the env ----------------

def run_in_env(prices: pd.DataFrame, feats: pd.DataFrame, env_cfg: EnvConfig,
               policy: Callable[[PortfolioEnv, int], np.ndarray]) -> dict[str, np.ndarray]:
    """Run a rule-based policy(env, step) -> target weights for one full episode."""
    env = PortfolioEnv(prices, feats, env_cfg)
    env.reset()
    rets, trades = [], []
    step, done = 0, False
    while not done:
        _, _, terminated, truncated, info = env.step(policy(env, step))
        done = terminated or truncated
        rets.append(info["ret"])
        trades.append(info["turnover"])
        step += 1
    return {"returns": np.asarray(rets), "trades": np.asarray(trades)}


def baseline_policies(all_prices: pd.DataFrame) -> dict[str, Callable[[PortfolioEnv, int], np.ndarray]]:
    """Baselines over the risky assets (no cash). all_prices supplies history before the test window."""
    log_returns = cast(pd.DataFrame, np.log(all_prices)).diff()
    log_var = cast(pd.DataFrame, log_returns.rolling(IVP_WINDOW).var())

    def risky_mask(env):
        return np.array([a != "__CASH__" for a in env.assets])

    def equal_weight(env, step):
        m = risky_mask(env)
        return m / m.sum()  # rebalanced every day

    def buy_and_hold(env, step):
        return equal_weight(env, step) if step == 0 else env.drifted_weights()

    def inverse_variance(env, step):
        if step % IVP_REBALANCE:
            return env.drifted_weights()
        date = env.prices.index[env.t]  # information up to today's close only
        var = log_var.loc[date].reindex(env.assets).to_numpy()
        w = np.where(risky_mask(env) & np.isfinite(var) & (var > 0), 1.0 / var, 0.0)
        return w / w.sum() if w.sum() > 0 else equal_weight(env, step)

    return {"Equal weight": equal_weight, "Buy & hold": buy_and_hold, "Inverse variance": inverse_variance}


# ---------------- driver ----------------

def _rows_for(fold, strategy, seed, returns, trades, dates, run_dir=""):
    metrics = summarize_returns(returns, trades)
    row = {"fold": fold, "strategy": strategy, "seed": seed, "run_dir": run_dir, **metrics}
    rets = pd.DataFrame({"fold": fold, "strategy": strategy, "seed": seed, "date": dates, "ret": returns})
    return row, rets


def _append_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def summarize(runs: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """
    Per strategy: mean/std of fold metrics (RL seeds first averaged within each fold), the share of
    folds beating equal weight on Sharpe, and Sharpe/CAGR/max drawdown of the stitched out-of-sample
    return series (per seed for RL, then mean/std across seeds).
    """
    per_fold = runs.groupby(["strategy", "fold"], as_index=False)[["sharpe", "cagr", "max_drawdown", "turnover"]].mean()
    ew = cast(pd.DataFrame, per_fold[per_fold["strategy"] == "Equal weight"]).set_index("fold")["sharpe"]

    out = []
    for strategy, g in per_fold.groupby("strategy"):
        g = g.set_index("fold")
        stitched = [summarize_returns(cast(pd.DataFrame, r).sort_values(by="date")["ret"].to_numpy())
                    for _, r in returns[returns["strategy"] == strategy].groupby("seed")]
        out.append({
            "strategy": strategy,
            "folds": len(g),
            "seeds": runs.loc[runs["strategy"] == strategy, "seed"].nunique(),
            "fold_sharpe_mean": g["sharpe"].mean(),
            "fold_sharpe_std": g["sharpe"].std(),
            "fold_cagr_mean": g["cagr"].mean(),
            "fold_max_dd_mean": g["max_drawdown"].mean(),
            "turnover_mean": g["turnover"].mean(),
            "beats_ew_sharpe": float((g["sharpe"] > ew.reindex(g.index)).mean()),
            "oos_sharpe": np.mean([s["sharpe"] for s in stitched]),
            "oos_sharpe_seed_std": np.std([s["sharpe"] for s in stitched]),
            "oos_cagr": np.mean([s["cagr"] for s in stitched]),
            "oos_max_dd": np.mean([s["max_drawdown"] for s in stitched]),
        })
    return pd.DataFrame(out).sort_values("oos_sharpe", ascending=False)


def main():
    ap = argparse.ArgumentParser(description="Walk-forward evaluation over yearly test windows and seeds")
    ap.add_argument("--config", required=True, help="Training config (dates are overridden per fold)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--first-test-year", type=int, default=2016)
    ap.add_argument("--val-months", type=int, default=6)
    ap.add_argument("--timesteps", type=int, default=None, help="Override train.total_timesteps per run")
    ap.add_argument("--name", default=None, help="Output folder name (default: config run_name); reuse to resume")
    args = ap.parse_args()

    base_cfg = load_config(args.config)
    if args.timesteps:
        base_cfg.train.total_timesteps = args.timesteps
    name = args.name or base_cfg.run_name
    out_dir = ARTIFACTS_DIR / "walk_forward" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_csv, returns_csv = out_dir / "runs.csv", out_dir / "returns.csv"

    done = set()
    if runs_csv.exists():
        prev = pd.read_csv(runs_csv, dtype={"fold": str})
        done = set(zip(prev["fold"], prev["strategy"], prev["seed"], strict=True))
        print(f"[wf] resuming: {len(done)} results already in {runs_csv}")

    folds = make_folds(_usable_index(base_cfg), args.first_test_year, args.val_months)
    print(f"[wf] {len(folds)} folds x {len(args.seeds)} seeds -> {out_dir}")

    for f in folds:
        cfg = deepcopy(base_cfg)
        cfg.dates = DatesCfg(train_end=f["train_end"], val_end=f["val_end"], test_end=f["test_end"])
        cfg.paths.out_root = str(out_dir / "runs")
        print(f"\n[wf] fold {f['fold']}: train ≤ {f['train_end']}, val ≤ {f['val_end']}, test ≤ {f['test_end']}")

        data = prepare_dataset(cfg)
        P_te, X_te = data.test
        dates = P_te.index[2:]  # step t earns the return into row t+1, for t = 1 .. T-2
        env_cfg = build_env_config(cfg).for_evaluation()

        # Baselines (deterministic: computed once per fold, recorded with seed -1)
        for strategy, policy in baseline_policies(data.prices).items():
            if (f["fold"], strategy, -1) in done:
                continue
            res = run_in_env(P_te, X_te, env_cfg, policy)
            row, rets = _rows_for(f["fold"], strategy, -1, res["returns"], res["trades"], dates)
            _append_csv(runs_csv, pd.DataFrame([row]))
            _append_csv(returns_csv, rets)

        for seed in args.seeds:
            if (f["fold"], "RL", seed) in done:
                continue
            cfg.seed = seed
            cfg.run_name = f"{base_cfg.run_name}__wf{f['fold']}_s{seed}"
            run_dir = train_once(cfg)
            returns = np.load(run_dir / "eval" / "test_returns.npy")
            trades = np.load(run_dir / "eval" / "test_trades.npy")
            assert len(returns) == len(dates), (len(returns), len(dates))
            row, rets = _rows_for(f["fold"], "RL", seed, returns, trades, dates, str(run_dir))
            _append_csv(runs_csv, pd.DataFrame([row]))
            _append_csv(returns_csv, rets)
            print(f"[wf] fold {f['fold']} seed {seed}: test Sharpe {row['sharpe']:.3f}")

    runs = pd.read_csv(runs_csv, dtype={"fold": str})
    returns = pd.read_csv(returns_csv, dtype={"fold": str}, parse_dates=["date"])
    summary = summarize(runs, returns)
    summary.to_csv(out_dir / "summary.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== Walk-forward summary (out-of-sample) ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n[wf] wrote {out_dir}")


if __name__ == "__main__":
    main()
