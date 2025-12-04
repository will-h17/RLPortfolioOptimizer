from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.config_loader import load_config
from src.runlog import RunRecorder
from src.scaler import FitOnTrainScaler
from src.env import EnvConfig, make_env
from src.splits import date_slices
from src.utils.data_utils import align_after_load, clamp_dates_to_index
from src.utils.metrics import sharpe_ratio, max_drawdown, turnover

def _load_data(cfg):
    prices = pd.read_parquet(cfg.paths.prices)
    feats  = pd.read_parquet(cfg.paths.features)

    prices, feats = align_after_load(prices, feats)

    # clamp to usable range and do mask-based split, same as train.py
    t_end_str, v_end_str, te_end_str = clamp_dates_to_index(feats.index, cfg.dates.train_end, cfg.dates.val_end, cfg.dates.test_end)
    t_end = pd.to_datetime(t_end_str)
    v_end = pd.to_datetime(v_end_str)
    te_end = pd.to_datetime(te_end_str)
    print(f"[dates] using train_end={t_end.date()}, val_end={v_end.date()}, test_end={te_end.date()}")

    idx = prices.index
    m_train = (idx <= t_end)
    m_val   = (idx > t_end) & (idx <= v_end)
    m_test  = (idx > v_end) & (idx <= te_end)

    P_te = prices.loc[m_test]
    X_te = feats.loc[m_test]

    print(f"[split] test={len(P_te)}  span {P_te.index.min().date() if len(P_te) else '—'} → {P_te.index.max().date() if len(P_te) else '—'}")
    if len(P_te) == 0:
        raise ValueError("Empty TEST split after clamping; adjust your dates.")

    # also return the full (aligned) raw features so we can pass vol columns to cost model if needed
    return prices, feats, P_te, X_te


def _rollout(best_model, env):
    import numpy as np
    obs = env.reset()
    done = [False]
    rets = []
    weights = []
    steps = 0

    while True:
        action, _ = best_model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        # Always record reward as the per-step net return
        rets.append(float(np.asarray(reward).ravel()[0]))

        # Record weights if env exposes them
        if info and "weights" in info[0]:
            weights.append(np.asarray(info[0]["weights"], dtype=float))

        steps += 1
        if done[0]:
            break

    return np.asarray(rets, dtype=float), weights


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
    best_zip = run_dir / "models" / "best" / "best_model.zip"
    scaler_path = run_dir / "scaler.json"

    # Load & split like train.py does
    prices_all, feats_all, P_te, X_te = _load_data(cfg)

    # Load scaler from the run and transform TEST features
    from src.scaler import FitOnTrainScaler
    # Scale TEST features with the train-fitted scaler from this run
    scaler = FitOnTrainScaler.load(scaler_path, columns=X_te.columns)
    X_tez = scaler.transform(X_te)

    # Final defensive alignment (indices should already match, but be explicit)
    common = P_te.index.intersection(X_tez.index)
    P_te   = P_te.loc[common]
    X_tez  = X_tez.loc[common]
    print(f"[sweep] final test rows after align: {len(common)}")

    # Load policy
    best = PPO.load(str(best_zip))

    rows = []
    # Sweep total *base* basis points from 0 to 10 (inclusive)
    for base_bps in range(0, 11):
        # Build EnvConfig: enable cost model with fee+slippage=0 and spread fixed at base_bps
        env_cfg = EnvConfig(
            cost_enabled=True,
            fee_bps=0.0,
            slippage_bps=0.0,
            spread_type="fixed",
            spread_fixed_bps=float(base_bps),
            # spread_type="vol20", # uncomment to show how spread widens with volatility
            # spread_k_vol_to_bps=base_bps * 100.0,  # scale k with base_bps
            # spread_vol_col_suffix=cfg.cost_model.spread.vol_col_suffix if hasattr(cfg, "cost_model") else "_s20",
            include_cash=cfg.env.include_cash,
            cash_rate_annual=getattr(cfg.env, "cash_rate_annual", 0.0),
            seed=cfg.seed + base_bps,
        )

        print(f"[env] P_te {P_te.shape}, X_tez {X_tez.shape}, equal index? {P_te.index.equals(X_tez.index)}")
        env = DummyVecEnv([make_env(P_te, X_tez, env_cfg, seed=cfg.seed + 100 + base_bps)])
        rets, weights = _rollout(best, env)
        m = _metrics(rets, weights)
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
