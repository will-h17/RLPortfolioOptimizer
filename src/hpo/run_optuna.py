from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner

from src.config_loader import load_config
from src.train import train_once
from src.utils.data_utils import align_after_load, clamp_dates_to_index
from src.scaler import FitOnTrainScaler
from src.backtest import evaluate_sb3_model



# Search space (uses current config)
def suggest_params(trial: optuna.trial.Trial, base_cfg):
    cfg = deepcopy(base_cfg)

    # PPO core knobs (SB3)
    cfg.ppo.learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    cfg.ppo.clip_range    = trial.suggest_float("clip_range",    0.05, 0.30)
    cfg.ppo.ent_coef      = trial.suggest_float("ent_coef",      1e-4, 1e-1, log=True)
    cfg.ppo.gae_lambda    = trial.suggest_float("gae_lambda",    0.80, 0.99)
    cfg.ppo.gamma         = trial.suggest_float("gamma",         0.90, 0.999)

    # Optional reward shaping knobs
    if hasattr(cfg, "reward"):
        if hasattr(cfg.reward, "turnover_penalty"):
            cfg.reward.turnover_penalty = trial.suggest_float("turnover_penalty", 0.0, 0.6)
        if hasattr(cfg.reward, "volatility_penalty"):
            cfg.reward.volatility_penalty = trial.suggest_float("volatility_penalty", 0.0, 0.6)
        if hasattr(cfg.reward, "drawdown_penalty"):
            cfg.reward.drawdown_penalty = trial.suggest_float("drawdown_penalty", 0.0, 0.6)

    # Trial budget: shorten timesteps for exploration
    if hasattr(cfg, "train") and hasattr(cfg.train, "total_timesteps"):
        
        trial_ts = getattr(getattr(cfg, "hpo", object()), "trial_timesteps", None)
        cfg.train.total_timesteps = int(trial_ts) if trial_ts else 300_000

    # Distinguish run folders by trial number
    base = getattr(cfg, "run_name", "run")
    cfg.run_name = f"{base}__t{trial.number:03d}"

    return cfg


# Validation scoring (Sharpe)
def eval_on_validation(run_dir: Path, cfg) -> float:
    """
    Recreate the validation slice exactly as in train.py, transform with the TRAIN scaler
    saved in run_dir, then evaluate best checkpoint with your existing function.
    """
    # Load raw data paths from cfg
    prices = pd.read_parquet(Path(cfg.paths.prices))
    feats  = pd.read_parquet(Path(cfg.paths.features))

    # Apply the same warmup we use in train.py
    MAX_LOOKBACK = 60
    WARMUP = MAX_LOOKBACK + 1
    prices = prices.iloc[WARMUP:].copy()
    feats  = feats.iloc[WARMUP:].copy()

    # Align and clamp dates
    prices, feats = align_after_load(prices, feats)
    t_end, v_end, te_end = clamp_dates_to_index(
        pd.DatetimeIndex(feats.index),
        cfg.dates.train_end, cfg.dates.val_end, cfg.dates.test_end
    )

    idx = prices.index
    t_end_ts = pd.to_datetime(t_end)
    v_end_ts = pd.to_datetime(v_end)

    m_train = (idx <= t_end_ts)
    m_val   = (idx > t_end_ts) & (idx <= v_end_ts)

    P_tr, X_tr = prices.loc[m_train], feats.loc[m_train]
    P_va, X_va = prices.loc[m_val],   feats.loc[m_val]

    # Load the TRAIN-fitted scaler from this run and transform VAL
    scaler_path = Path(run_dir) / "scaler.json"
    scaler = FitOnTrainScaler.load(scaler_path, columns=feats.columns)
    X_vaz = scaler.transform(X_va)

    # Best checkpoint chosen by EvalCallback
    best_zip = Path(run_dir) / "models" / "best" / "best_model.zip"
    if not best_zip.exists():
        # Fallback: a final model path if you ever save one there
        best_zip = next((Path(run_dir) / "models").glob("*final*.zip"), None)
        if best_zip is None:
            raise FileNotFoundError(f"No best model found under {run_dir}/models")

    res = evaluate_sb3_model(P_va, X_vaz, str(best_zip))
    return float(res["metrics"]["sharpe"])


# Optuna objective
def objective(trial: optuna.trial.Trial, base_cfg):
    cfg_trial = suggest_params(trial, base_cfg)

    # Run a normal training job using your exact pipeline
    run_dir = train_once(cfg_trial)

    # Score on the validation slice (not the test slice!)
    val_sharpe = eval_on_validation(run_dir, cfg_trial)

    # Report so pruners can act if you later add intermediate steps
    trial.report(val_sharpe, step=1)
    return val_sharpe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to your YAML/JSON config")
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--study-name", default="ppo_hpo")
    ap.add_argument("--storage", default=None, help='e.g. "sqlite:///hpo.db" (optional)')
    args = ap.parse_args()

    base_cfg = load_config(args.config)

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        pruner=MedianPruner(n_warmup_steps=1),
    )
    study.optimize(lambda t: objective(t, base_cfg), n_trials=args.n_trials, gc_after_trial=True)

    print("\n=== HPO complete ===")
    print("Best Sharpe:", study.best_value)
    print("Best params:", study.best_trial.params)


if __name__ == "__main__":
    main()
