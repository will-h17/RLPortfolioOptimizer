from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import json
from dataclasses import fields as dataclass_fields

from src.backtest import evaluate_sb3_model, run_backtest
from src.config_loader import load_config
from src.splits import date_slices
from src.scaler import FitOnTrainScaler
from src.runlog import RunRecorder
from src.env import PortfolioEnv, EnvConfig, make_env
from src.agent import make_sb3_ppo
from src.repro import set_global_seed, collect_versions
from src.utils.data_utils import align_after_load, clamp_dates_to_index, align_dataframes
from src.utils.metrics import sharpe_ratio, max_drawdown, turnover
# from src.utils.logging import make_loggers

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# make helpers importable by HPO:
__all__ = ["train_once"]


def build_env_config(cfg) -> EnvConfig:
    """
    Construct EnvConfig safely: only pass fields that actually exist on EnvConfig.
    Prevents 'no parameter named ...' errors when your config has extras.
    """
    tc = cfg.env.transaction_cost_bps / 10_000.0  # bps -> proportion

    # candidates from config (include all you *might* support)
    candidates = {
        "transaction_cost": tc,
        "include_cash": getattr(cfg.env, "include_cash", True),
        "cash_rate_annual": getattr(cfg.env, "cash_rate_annual", 0.0),
        "normalize_obs_weights": getattr(cfg.env, "normalize_obs_weights", False),
        
        # Advanced reward shaping for profitability
        "reward_vol_scaling": getattr(cfg.env, "reward_vol_scaling", False),
        "reward_sharpe_bonus": getattr(cfg.env, "reward_sharpe_bonus", 0.0),
        "reward_vol_window": getattr(cfg.env, "reward_vol_window", 20),
        "reward_return_bonus": getattr(cfg.env, "reward_return_bonus", 0.0),
        "reward_return_threshold": getattr(cfg.env, "reward_return_threshold", 0.0),
        "reward_compound_bonus": getattr(cfg.env, "reward_compound_bonus", 0.0),

        # Reward shaping knobs (only used if EnvConfig defines them)
        "return_weight": getattr(cfg.reward, "return_weight", 1.0),
        "turnover_penalty": getattr(cfg.reward, "turnover_penalty", 0.0),
        "drawdown_penalty": getattr(cfg.reward, "drawdown_penalty", 0.0),
        "volatility_penalty": getattr(cfg.reward, "volatility_penalty", 0.0),

        "seed": cfg.seed,
    }

    # Cost model parameters (if cost_model section exists)
    cost_model = getattr(cfg, "cost_model", None)
    if cost_model is not None:
        spread = getattr(cost_model, "spread", {})
        candidates.update({
            "cost_enabled": getattr(cost_model, "enabled", False),
            "fee_bps": getattr(cost_model, "fee_bps", 0.0),
            "slippage_bps": getattr(cost_model, "slippage_bps", 0.0),
            "spread_type": spread.get("type", "fixed") if isinstance(spread, dict) else getattr(spread, "type", "fixed"),
            "spread_fixed_bps": spread.get("fixed_bps", 0.0) if isinstance(spread, dict) else getattr(spread, "fixed_bps", 0.0),
            "spread_k_vol_to_bps": spread.get("k_vol_to_bps", 8000.0) if isinstance(spread, dict) else getattr(spread, "k_vol_to_bps", 8000.0),
            "spread_vol_col_suffix": spread.get("vol_col_suffix", "_s20") if isinstance(spread, dict) else getattr(spread, "vol_col_suffix", "_s20"),
        })

    allowed = {f.name for f in dataclass_fields(EnvConfig)}
    filtered = {k: v for k, v in candidates.items() if k in allowed}
    return EnvConfig(**filtered)






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

    # Paths
    prices_path   = Path(cfg.paths.prices)
    features_path = Path(cfg.paths.features)
    out_root      = Path(cfg.paths.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Record versions & seed in the run folder
    versions = collect_versions(cfg.seed)
    rec = RunRecorder(run_name=cfg.run_name, out_root=str(out_root), cfg_obj=cfg)
    (rec.path("versions.json")).write_text(json.dumps(versions, indent=2))

    # Load data (aligned index)
    prices = pd.read_parquet(prices_path)
    feats  = pd.read_parquet(features_path)
    assert prices.index.equals(prices.index.unique().sort_values())  # sanity
    assert feats.index.equals(feats.index.unique().sort_values())
    
    MAX_LOOKBACK = 60  # days
    WARMUP = MAX_LOOKBACK + 1
    
    # Drop the first WARMUP rows from BOTH prices and feats (keeps indices aligned)
    prices = prices.iloc[WARMUP:].copy()
    feats  = feats.iloc[WARMUP:].copy()

    prices, feats = align_after_load(prices, feats)

    # clamp/validate dates
    t_end, v_end, te_end = clamp_dates_to_index(
        pd.DatetimeIndex(feats.index),  # feats/prices share same index now
        cfg.dates.train_end,
        cfg.dates.val_end,
        cfg.dates.test_end,
    )
    print(f"[dates] using train_end={t_end}, val_end={v_end}, test_end={te_end}")

    # Time split (leak-proof)
    idx = prices.index  # same as feats.index now
    t_end_ts  = pd.to_datetime(t_end)
    v_end_ts  = pd.to_datetime(v_end)
    te_end_ts = pd.to_datetime(te_end)

    # Inclusive end-points; strict “>” for next segment’s start
    m_train = (idx <= t_end_ts)
    m_val   = (idx > t_end_ts) & (idx <= v_end_ts)
    m_test  = (idx > v_end_ts) & (idx <= te_end_ts)

    P_tr, X_tr = prices.loc[m_train], feats.loc[m_train]
    P_va, X_va = prices.loc[m_val],   feats.loc[m_val]
    P_te, X_te = prices.loc[m_test],  feats.loc[m_test]

    print(f"[split] train={len(P_tr)}  val={len(P_va)}  test={len(P_te)}")
    print(f"[split] train span {P_tr.index.min().date()} → {P_tr.index.max().date() if len(P_tr) else '—'}")
    print(f"[split] val   span {P_va.index.min().date() if len(P_va) else '—'} → {P_va.index.max().date() if len(P_va) else '—'}")
    print(f"[split] test  span {P_te.index.min().date() if len(P_te) else '—'} → {P_te.index.max().date() if len(P_te) else '—'}")

    # Fit scaler on TRAIN only; save under run dir; transform all splits
    scaler = FitOnTrainScaler().fit(X_tr)
    scaler.save(rec.path("scaler.json"))
    X_trz, X_vaz, X_tez = scaler.transform(X_tr), scaler.transform(X_va), scaler.transform(X_te)

    # Align splits (no NaN dropping needed as features are already clean from align_after_load)
    P_tr, X_tr = align_dataframes(P_tr, X_tr, dropna=False, show_trimming=True, context="train split")
    P_va, X_va = align_dataframes(P_va, X_va, dropna=False, show_trimming=True, context="validation split")
    P_te, X_te = align_dataframes(P_te, X_te, dropna=False, show_trimming=True, context="test split")

    # Build EnvConfig safely and vec envs
    env_cfg  = build_env_config(cfg)
    
    # Verify training data is reasonable
    if len(P_tr) < 100:
        raise ValueError(f"Training data too small: {len(P_tr)} rows. Check date splits.")
    
    print(f"[env] Creating training environment with {len(P_tr)} data points")
    env      = DummyVecEnv([make_env(P_tr, X_trz, env_cfg, seed=cfg.seed)])
    
    # Test that environment works correctly (sample action from action space)
    test_obs = env.reset()
    test_action = [env.action_space.sample() for _ in range(env.num_envs)]
    test_obs, test_reward, test_done, test_info = env.step(test_action)
    print(f"[env] Environment test: obs_shape={test_obs.shape}, reward={test_reward[0]:.6f}, done={test_done[0]}")
    if test_done[0]:
        print(f"[env] WARNING: Environment done after 1 step! This will cause training issues.")
    env.reset()  # Reset after test
    
    eval_env = DummyVecEnv([make_env(P_va, X_vaz, env_cfg, seed=cfg.seed + 1)])

    # PPO kwargs (SB3) pulled from config — only SB3 parameters here
    ppo_kwargs = {
        "gamma":       cfg.ppo.gamma,
        "gae_lambda":  cfg.ppo.gae_lambda,
        "n_steps":     cfg.ppo.n_steps,
        "batch_size":  cfg.ppo.batch_size,
        "clip_range":  cfg.ppo.clip_range,
        "ent_coef":    cfg.ppo.ent_coef,
        # add more SB3 knobs here if included in config:
        # "vf_coef": cfg.ppo.vf_coef,
        # "max_grad_norm": cfg.ppo.max_grad_norm,
    }

    # Build model (make_sb3_ppo forwards **ppo_kwargs to SB3.PPO)
    model = make_sb3_ppo(
        env,
        learning_rate=cfg.ppo.learning_rate,
        policy_hidden_sizes=cfg.ppo.policy_hidden_sizes,
        **ppo_kwargs,
        seed=cfg.seed
    )

    # Callbacks writing into this run's folder
    models_dir = rec.path("models")
    best_dir   = rec.path("models", "best")
    eval_dir   = rec.path("eval")

    checkpoint_cb = CheckpointCallback(
        save_freq=cfg.train.checkpoint_freq,
        save_path=str(models_dir),
        name_prefix=cfg.run_name,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(eval_dir),
        eval_freq=cfg.train.eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=1,  # Explicitly set to avoid issues
        verbose=1,  # Show evaluation progress
    )

    # Train with progress printing
    print(f"[train] Starting PPO training for {cfg.train.total_timesteps:,} timesteps")
    print(f"[train] n_steps per update: {cfg.ppo.n_steps}")
    print(f"[train] Training data: {len(P_tr)} rows")
    print(f"[train] Episode length: ~{len(P_tr) - 2} steps")
    print(f"[train] Expected updates: ~{cfg.train.total_timesteps // cfg.ppo.n_steps}")
    print(f"[train] Eval frequency: every {cfg.train.eval_freq:,} steps")
    print(f"[train] Checkpoint frequency: every {cfg.train.checkpoint_freq:,} steps")
    
    try:
        model.learn(total_timesteps=cfg.train.total_timesteps, callback=[checkpoint_cb, eval_cb], progress_bar=True)
        print(f"[train] Training completed successfully! Total timesteps: {cfg.train.total_timesteps:,}")
    except Exception as e:
        print(f"[train] ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Load the best checkpoint chosen by EvalCallback
    best_model_zip = rec.path("models", "best", "best_model.zip")
    if not best_model_zip.exists():
        # Fallback: try to find any model
        model_zips = list(rec.path("models").glob("*.zip"))
        if model_zips:
            best_model_zip = model_zips[0]
            print(f"[train] Warning: best_model.zip not found, using {best_model_zip.name}")
        else:
            raise FileNotFoundError(f"No model found in {rec.path('models')}. Training may have failed.")
    
    print(f"[train] Loading best model from: {best_model_zip}")
    best_model = PPO.load(str(best_model_zip))

    # Build a TEST env using the *same* env config and the saved TRAIN scaler
    test_env = DummyVecEnv([make_env(P_te, X_tez, env_cfg, seed=cfg.seed + 2)])

    # Roll out deterministically once to collect returns/weights
    obs = test_env.reset()
    done = [False]
    step_rewards = []
    step_weights = []

    while not done[0]:
        action, _ = best_model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)

    # Common pattern we’ve used: info[0]["ret"] = per-step portfolio return (after costs),
    # and info[0]["weights"] = current portfolio weights vector.
        if info and "ret" in info[0]:
            step_rewards.append(float(info[0]["ret"]))
        if info and "weights" in info[0]:
        # ensure it's a 1D numpy array
            step_weights.append(np.asarray(info[0]["weights"], dtype=float))

    # Compute test metrics (annualized Sharpe, Max Drawdown, Turnover)
    rets = np.array(step_rewards, dtype=float)
    equity = (1.0 + rets).cumprod()

    test_sharpe = sharpe_ratio(rets) if len(rets) > 1 else 0.0
    test_mdd    = max_drawdown(equity) if len(equity) > 1 else 0.0  # Returns negative (e.g., -0.23 for -23%)
    test_tov    = turnover(step_weights)

    # Save arrays for inspection
    np.save(rec.path("eval", "test_equity.npy"), equity)
    np.save(rec.path("eval", "test_returns.npy"), rets)
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
        },
        "artifacts": {
            "best_model": str(best_model_zip),
            "final_model": str(rec.path("models", f"{cfg.run_name}_final.zip")),
            "scaler": str(rec.path("scaler.json")),
            "returns_npy": str(rec.path("eval", "test_returns.npy")),
            "equity_npy": str(rec.path("eval", "test_equity.npy")),
            "weights_npy": str(rec.path("eval", "test_weights.npy")),
        }
    }
    rec.save_metrics(metrics)

    print(f"[OK] Run recorded in: {rec.run_dir}")
    print("Saved: config.json, commit.txt, scaler.json, models/, eval/, metrics.json")
    # At the end (run path already printed)
    return Path(rec.run_dir)


if __name__ == "__main__":
    main()
