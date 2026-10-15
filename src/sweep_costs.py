from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.backtest import rollout_policy
from src.config_loader import load_config
from src.env import PortfolioEnv, load_run_env_config
from src.runlog import resolve_run_model
from src.scaler import FitOnTrainScaler
from src.utils.data_utils import prepare_dataset
from src.utils.metrics import max_drawdown, sharpe_ratio, turnover


def _metrics(rets: np.ndarray, weights_seq: list[np.ndarray]) -> dict:
    eq = (1.0 + rets).cumprod()
    return {
        "sharpe": sharpe_ratio(rets),
        "max_drawdown": max_drawdown(eq),  # Returns negative (e.g., -0.23 for -23%)
        "turnover": turnover(weights_seq),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML/JSON config")
    ap.add_argument("--run_dir", required=True, help="Run folder that contains models/best/best_model.zip and scaler.json")
    ap.add_argument("--out_csv", default="cost_sweep.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_dir = Path(args.run_dir)
    model_zip = resolve_run_model(run_dir)
    scaler_path = run_dir / "scaler.json"

    # Load & split exactly like train.py does
    P_te, X_te = prepare_dataset(cfg).test

    # Load scaler from the run and transform TEST features
    # Scale TEST features with the train-fitted scaler from this run
    scaler = FitOnTrainScaler.load(scaler_path, columns=X_te.columns)
    X_tez = scaler.transform(X_te)

    # Final defensive alignment (indices should already match, but be explicit)
    common = P_te.index.intersection(X_tez.index)
    P_te   = P_te.loc[common]
    X_tez  = X_tez.loc[common]
    print(f"[sweep] final test rows after align: {len(common)}")

    # Load policy
    best = PPO.load(str(model_zip))

    # Start from the env the model was trained in (observations, cash), then override costs only
    train_env_cfg = load_run_env_config(run_dir).for_evaluation()

    rows = []
    # Sweep total *base* basis points from 0 to 10 (inclusive)
    for base_bps in range(0, 11):
        # Cost model with fee+slippage=0 and a fixed spread; per-unit-turnover cost is half the spread
        env_cfg = replace(
            train_env_cfg,
            cost_enabled=True,
            fee_bps=0.0,
            slippage_bps=0.0,
            spread_type="fixed",
            spread_fixed_bps=float(base_bps),
            seed=cfg.seed + base_bps,
        )

        print(f"[env] P_te {P_te.shape}, X_tez {X_tez.shape}, equal index? {P_te.index.equals(X_tez.index)}")
        res = rollout_policy(best, PortfolioEnv(P_te, X_tez, env_cfg))
        rets = res["returns"]
        m = _metrics(rets, list(res["weights"]))
        rows.append({
            "base_bps": base_bps,
            "sharpe": m["sharpe"],
            "max_drawdown": m["max_drawdown"],
            "turnover": m["turnover"],
            "num_steps": len(rets),
        })
        print(f"[sweep] bps={base_bps:2d}  Sharpe={m['sharpe']:.3f}  MDD={m['max_drawdown']:.1%}  Turnover={m['turnover']:.3f}")

    df = pd.DataFrame(rows).sort_values("base_bps")
    out_csv = Path(args.out_csv)
    df.to_csv(out_csv, index=False)
    print(f"[sweep] wrote {out_csv.resolve()}")

if __name__ == "__main__":
    main()
