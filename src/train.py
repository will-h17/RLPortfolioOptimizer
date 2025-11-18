from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import json
from dataclasses import fields as dataclass_fields

from src.backtest import evaluate_sb3_model, run_backtest
from src.config_loader import load_config
from src.splits import date_slices
from src.scaler import FitOnTrainScaler
from src.runlog import RunRecorder
from src.env import PortfolioEnv, EnvConfig
from src.agent import make_sb3_ppo
from src.repro import set_global_seed, collect_versions
from src.utils.data_utils import align_after_load, clamp_dates_to_index
# from src.utils.logging import make_loggers

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed as sb3_set_seed

# make helpers importable by HPO:
__all__ = ["train_once", "_align_after_load", "_clamp_dates_to_index"]


# Utilities
def set_global_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


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
        
        # Advanced reward shaping for higher Sharpe
        "reward_vol_scaling": getattr(cfg.env, "reward_vol_scaling", False),
        "reward_sharpe_bonus": getattr(cfg.env, "reward_sharpe_bonus", 0.0),
        "reward_vol_window": getattr(cfg.env, "reward_vol_window", 20),

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


def make_env(prices=None, features=None, env_cfg: EnvConfig = None, seed: int = 0):
    """
    Factory that returns an env-initializer callable.

    This accepts the original signature make_env(prices, features, env_cfg, seed)
    but also has defaults so calls that omit arguments (e.g. make_env(cfg))
    will not raise a static "missing arguments" error; instead a clear runtime
    ValueError will be raised if required pieces are not provided.
    """
    def _init():
        if prices is None or features is None or env_cfg is None:
            raise ValueError("make_env requires (prices, features, env_cfg, seed); called with missing or None arguments.")
        env = PortfolioEnv(prices=prices, features=features, config=env_cfg)
        if hasattr(env, "seed"):
            env.seed(seed)
        # gymnasium-style (optional)
        try:
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
        except Exception:
            pass
        return Monitor(env)
    return _init





def _align_split(P: pd.DataFrame, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = P.index.intersection(X.index)
    if len(common) == 0:
        raise ValueError(
            "No overlap between prices and features for this split. "
            "This usually means your date cutoffs fell outside the usable index. "
            "Check the printed [data] usable rows and adjust your config dates."
        )
    if len(common) < len(P.index) or len(common) < len(X.index):
        print(f"[align] trimming split: prices {len(P)}→{len(common)}, features {len(X)}→{len(common)}")
    return P.loc[common], X.loc[common]




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
    set_global_seed(cfg.seed)
    sb3_set_seed(cfg.seed)  # SB3’s own RNGs

    try:
        import torch as th
        th.use_deterministic_algorithms(True)
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False
    except Exception:
        pass

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

    # Make a unique run folder + record config/commit hash
    rec = RunRecorder(run_name=cfg.run_name, out_root=str(out_root), cfg_obj=cfg)

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

    P_tr, X_tr = _align_split(P_tr, X_tr)
    P_va, X_va = _align_split(P_va, X_va)
    P_te, X_te = _align_split(P_te, X_te)

    # Build EnvConfig safely and vec envs
    env_cfg  = build_env_config(cfg)
    env      = DummyVecEnv([make_env(P_tr, X_trz, env_cfg, seed=cfg.seed)])
    eval_env = DummyVecEnv([make_env(P_va, X_vaz, env_cfg, seed=cfg.seed + 1)])

    # PPO kwargs (SB3) pulled from config — only SB3 parameters here
    ppo_kwargs = {
        "gamma":       cfg.ppo.gamma,
        "gae_lambda":  cfg.ppo.gae_lambda,
        "n_steps":     cfg.ppo.n_steps,
        "batch_size":  cfg.ppo.batch_size,
        "clip_range":  cfg.ppo.clip_range,
        "ent_coef":    cfg.ppo.ent_coef,
        # add more SB3 knobs here if you include them in your config:
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
    )

    # Train
    model.learn(total_timesteps=cfg.train.total_timesteps, callback=[checkpoint_cb, eval_cb])

    # 1) Load the best checkpoint chosen by EvalCallback
    best_model_zip = rec.path("models", "best", "best_model.zip")
    best_model = PPO.load(str(best_model_zip))

    # 2) Build a TEST env using the *same* env config and the saved TRAIN scaler
    test_env = DummyVecEnv([make_env(P_te, X_tez, env_cfg, seed=cfg.seed + 2)])

    # 3) Roll out deterministically once to collect returns/weights
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

    # 4) Compute test metrics (annualized Sharpe, Max Drawdown, Turnover)
    rets = np.array(step_rewards, dtype=float)
    equity = (1.0 + rets).cumprod()

    def _sharpe_daily(x: np.ndarray) -> float:
        mu = x.mean()
        sig = x.std()
        return float(np.sqrt(252.0) * mu / (sig + 1e-12))

    def _max_drawdown(eq: np.ndarray) -> float:
    # returns negative number (e.g., -0.23 for -23%)
        peak = np.maximum.accumulate(eq)
        dd = eq / peak - 1.0
        return float(dd.min())

    def _avg_turnover(weights_seq: list[np.ndarray]) -> float:
        if len(weights_seq) < 2:
            return 0.0
        W = np.vstack(weights_seq)  # shape [T, N]
        # L1 change per step, then average across steps
        per_step = np.abs(np.diff(W, axis=0)).sum(axis=1)
        return float(per_step.mean())

    test_sharpe = _sharpe_daily(rets) if len(rets) > 1 else 0.0
    test_mdd    = _max_drawdown(equity) if len(equity) > 1 else 0.0
    test_tov    = _avg_turnover(step_weights)

    # 5) Save arrays for inspection (optional)
    np.save(rec.path("eval", "test_equity.npy"), equity)
    np.save(rec.path("eval", "test_returns.npy"), rets)
    if len(step_weights):
        np.save(rec.path("eval", "test_weights.npy"), np.vstack(step_weights))

    # 6) Write metrics.json to the run folder
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
    #
    # At the end (run path already printed), just:
    return Path(rec.run_dir)


if __name__ == "__main__":
    main()
