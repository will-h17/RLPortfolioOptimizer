# src/train.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# from src.utils.logging import make_loggers
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.agent import make_sb3_ppo
from src.backtest import rollout_policy
from src.callbacks import ValSharpeCallback
from src.config_loader import load_config
from src.env import PortfolioEnv, build_env_config, make_env
from src.paths import resolve
from src.repro import collect_versions, set_global_seed
from src.runlog import RunRecorder
from src.scaler import FitOnTrainScaler
from src.utils.data_utils import prepare_dataset
from src.utils.logging import make_logger
from src.utils.metrics import max_drawdown, sharpe_ratio, turnover

# make helpers importable by HPO (build_env_config re-exported for compatibility):
__all__ = ["train_once", "build_env_config"]


# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML/JSON config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    train_once(cfg)

def train_once(cfg) -> Path:
    """
    Runs one full training job using the existing logic in main().
    Returns the run directory containing models/, scaler.json, eval/, metrics.json, etc.
    """
    # Set global seed (handles random, numpy, torch, SB3, and deterministic algorithms)
    set_global_seed(cfg.seed, deterministic=True)

    # Load, align and split (shared with HPO and the cost sweep); validated before creating the run folder
    data = prepare_dataset(cfg)
    (P_tr, X_tr), (P_va, X_va), (P_te, X_te) = data.train, data.val, data.test

    # Record versions & seed in the run folder
    versions = collect_versions(cfg.seed)
    rec = RunRecorder(run_name=cfg.run_name, out_root=str(resolve(cfg.paths.out_root)), cfg_obj=cfg)
    (rec.path("versions.json")).write_text(json.dumps(versions, indent=2))

    # Fit scaler on TRAIN only; save under run dir; transform all splits
    scaler = FitOnTrainScaler().fit(X_tr)
    scaler.save(rec.path("scaler.json"))
    X_trz, X_vaz, X_tez = scaler.transform(X_tr), scaler.transform(X_va), scaler.transform(X_te)

    # Build EnvConfig safely and vec envs
    env_cfg  = build_env_config(cfg)
    env_cfg.save(rec.path("env_config.json"))  # evaluation rebuilds the exact same env from this
    
    # Verify training data is reasonable
    if len(P_tr) < 100:
        raise ValueError(f"Training data too small: {len(P_tr)} rows. Check date splits.")
    
    # Training envs: n_envs copies, each seeded differently (random episode windows differ)
    n_envs = cfg.train.n_envs
    episode = f"random {env_cfg.max_steps}-day windows" if env_cfg.random_start else f"~{len(P_tr) - 2} steps (whole split)"
    print(f"[env] {n_envs} training env(s) on {len(P_tr)} rows; episodes: {episode}")
    env = DummyVecEnv([make_env(P_tr, X_trz, env_cfg, seed=cfg.seed + i) for i in range(n_envs)])
    if cfg.ppo.normalize_reward:
        # Daily returns are ~1e-3; scaling rewards by a running std gives the value function a usable
        # signal. Observations are already z-scored, so only rewards are normalized (nothing to save
        # for evaluation).
        env = VecNormalize(env, norm_obs=False, norm_reward=True, gamma=cfg.ppo.gamma)

    # Validation and test envs: one deterministic pass over the whole split
    eval_env_cfg = env_cfg.for_evaluation()
    val_env = PortfolioEnv(P_va, X_vaz, eval_env_cfg)

    # PPO kwargs (SB3) pulled from config — only SB3 parameters here
    ppo_kwargs = {
        "gamma":       cfg.ppo.gamma,
        "gae_lambda":  cfg.ppo.gae_lambda,
        "n_steps":     cfg.ppo.n_steps,  # per env; each update uses n_steps * n_envs samples
        "batch_size":  cfg.ppo.batch_size,
        "n_epochs":    cfg.ppo.n_epochs,
        "clip_range":  cfg.ppo.clip_range,
        "ent_coef":    cfg.ppo.ent_coef,
    }

    # Build model (make_sb3_ppo forwards **ppo_kwargs to SB3.PPO)
    model = make_sb3_ppo(
        env,
        learning_rate=cfg.ppo.learning_rate,
        policy_hidden_sizes=cfg.ppo.policy_hidden_sizes,
        alpha_floor=cfg.ppo.alpha_floor,
        **ppo_kwargs,
        seed=cfg.seed
    )
    # Training stats (losses, KL, validation Sharpe, ...) for every update -> logs/progress.csv
    model.set_logger(make_logger(str(rec.path("logs"))))

    # Callbacks writing into this run's folder. CheckpointCallback counts vec-env steps, so divide
    # by n_envs to keep checkpoint_freq in total timesteps.
    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, cfg.train.checkpoint_freq // n_envs),
        save_path=str(rec.path("models")),
        name_prefix=cfg.run_name,
    )
    best_model_zip = rec.path("models", "best", "best_model.zip")
    val_cb = ValSharpeCallback(
        val_env,
        eval_freq=cfg.train.eval_freq,
        best_model_path=best_model_zip,
        log_path=rec.path("eval", "val_sharpe.csv"),
    )

    # Train with progress printing
    rollout_size = cfg.ppo.n_steps * n_envs
    print(f"[train] Starting PPO training for {cfg.train.total_timesteps:,} timesteps")
    print(f"[train] Samples per update: {rollout_size:,} ({cfg.ppo.n_steps} steps x {n_envs} envs)")
    print(f"[train] Expected updates: ~{cfg.train.total_timesteps // rollout_size}")
    print(f"[train] Model selection: validation Sharpe every {cfg.train.eval_freq:,} steps")
    print(f"[train] Checkpoint frequency: every {cfg.train.checkpoint_freq:,} steps")

    model.learn(total_timesteps=cfg.train.total_timesteps, callback=[checkpoint_cb, val_cb], progress_bar=True)
    print(f"[train] Training completed. Best validation Sharpe: {val_cb.best_sharpe:.3f}")

    final_model_zip = rec.path("models", f"{cfg.run_name}_final.zip")
    model.save(str(final_model_zip))

    print(f"[train] Loading best model from: {best_model_zip}")
    best_model = PPO.load(str(best_model_zip), device="cpu")

    # Roll out deterministically once on TEST (same env config, TRAIN-fitted scaler)
    test = rollout_policy(best_model, PortfolioEnv(P_te, X_tez, eval_env_cfg))
    rets, equity, step_weights = test["returns"], test["values"], list(test["weights"])

    # Compute test metrics (annualized Sharpe, Max Drawdown, Turnover)
    test_sharpe = sharpe_ratio(rets) if len(rets) > 1 else 0.0
    test_mdd    = max_drawdown(equity) if len(equity) > 1 else 0.0  # Returns negative (e.g., -0.23 for -23%)
    test_tov    = turnover(step_weights)

    # Save arrays for inspection
    np.save(rec.path("eval", "test_equity.npy"), equity)
    np.save(rec.path("eval", "test_returns.npy"), rets)
    np.save(rec.path("eval", "test_trades.npy"), test["trades"])  # L1 trade size per step
    if len(step_weights):
        np.save(rec.path("eval", "test_weights.npy"), np.vstack(step_weights))

    # Write metrics.json to the run folder
    metrics = {
        "split": "test",
        "timesteps": cfg.train.total_timesteps,
        "transaction_cost_bps": cfg.env.transaction_cost_bps,
        "results": {
            "sharpe": test_sharpe,
            "max_drawdown": test_mdd,      # negative for drawdown (e.g., -0.23)
            "turnover": test_tov,
            "num_steps": int(len(rets)),
            "best_val_sharpe": val_cb.best_sharpe,
        },
        "artifacts": {
            "best_model": str(best_model_zip),
            "final_model": str(final_model_zip),
            "scaler": str(rec.path("scaler.json")),
            "returns_npy": str(rec.path("eval", "test_returns.npy")),
            "equity_npy": str(rec.path("eval", "test_equity.npy")),
            "weights_npy": str(rec.path("eval", "test_weights.npy")),
        }
    }
    rec.save_metrics(metrics)

    print(f"[OK] Run recorded in: {rec.run_dir}")
    print("Saved: config.json, env_config.json, commit.txt, scaler.json, models/, eval/, metrics.json")
    # At the end (run path already printed)
    return Path(rec.run_dir)


if __name__ == "__main__":
    main()
