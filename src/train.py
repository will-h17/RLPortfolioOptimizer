# src/train.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import json
from dataclasses import fields as dataclass_fields

from src.config_loader import load_config
from src.splits import date_slices
from src.scaler import FitOnTrainScaler
from src.runlog import RunRecorder
from src.env import PortfolioEnv, EnvConfig
from src.agent import make_sb3_ppo
from src.repro import set_global_seed, collect_versions

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed as sb3_set_seed

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

        # Reward shaping knobs (only used if EnvConfig defines them)
        "return_weight": getattr(cfg.reward, "return_weight", 1.0),
        "turnover_penalty": getattr(cfg.reward, "turnover_penalty", 0.0),
        "drawdown_penalty": getattr(cfg.reward, "drawdown_penalty", 0.0),
        "volatility_penalty": getattr(cfg.reward, "volatility_penalty", 0.0),

        "seed": cfg.seed,
    }

    allowed = {f.name for f in dataclass_fields(EnvConfig)}
    filtered = {k: v for k, v in candidates.items() if k in allowed}
    return EnvConfig(**filtered)


def make_env(prices, features, env_cfg: EnvConfig, seed: int):
    def _init():
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



def _align_after_load(prices: pd.DataFrame, feats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    1) Drop rows in features that are not fully ready (NaNs from rolling/shift).
    2) Intersect indices and return aligned (prices, feats) on the common dates.
    """
    feats = feats.dropna(how="any")  # keep only rows where all features exist
    common = prices.index.intersection(feats.index)
    if len(common) == 0:
        raise ValueError(
            "No common timestamps after dropping NaNs in features. "
            "Check your warm-up trimming and feature construction."
        )
    prices = prices.loc[common]
    feats  = feats.loc[common]
    print(f"[data] usable rows: {len(common)} from {common.min().date()} to {common.max().date()}")
    return prices, feats

def _clamp_dates_to_index(idx: pd.DatetimeIndex, train_end: str, val_end: str, test_end: str) -> tuple[str, str, str]:
    """
    Clamp date strings to live within idx[min..max] and enforce ordering train_end <= val_end <= test_end.
    Returns ISO strings you can pass to date_slices.
    """
    dmin, dmax = idx.min().normalize(), idx.max().normalize()

    t_end  = pd.to_datetime(train_end).normalize()
    v_end  = pd.to_datetime(val_end).normalize()
    te_end = pd.to_datetime(test_end).normalize()

    # clamp to [dmin, dmax]
    t_end  = min(max(t_end,  dmin), dmax)
    v_end  = min(max(v_end,  dmin), dmax)
    te_end = min(max(te_end, dmin), dmax)

    # enforce nondecreasing order
    if not (t_end <= v_end <= te_end):
        # make a simple 70/15/15 split as a safe fallback
        n = len(idx)
        i_tr = max(1, int(0.70 * n) - 1)
        i_va = max(i_tr + 1, int(0.85 * n) - 1)
        t_end, v_end, te_end = idx[i_tr].normalize(), idx[i_va].normalize(), idx[-1].normalize()
        print("[dates] config ordering invalid; using 70/15/15 fallback based on index.")

    # tiny sanity: ensure at least 1 row per segment
    i_tr_end = idx.get_indexer([t_end], method="bfill")[0]
    i_va_end = idx.get_indexer([v_end], method="bfill")[0]
    i_te_end = idx.get_indexer([te_end], method="bfill")[0]
    if not (i_tr_end >= 0 and i_va_end > i_tr_end and i_te_end > i_va_end):
        # fallback again if any segment would be empty
        n = len(idx)
        i_tr = max(1, int(0.70 * n) - 1)
        i_va = max(i_tr + 1, int(0.85 * n) - 1)
        t_end, v_end, te_end = idx[i_tr].normalize(), idx[i_va].normalize(), idx[-1].normalize()
        print("[dates] segments too small; using 70/15/15 fallback based on index.")

    return str(t_end.date()), str(v_end.date()), str(te_end.date())

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




# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML/JSON config")
    args = ap.parse_args()

    # Load config + set seeds
    cfg = load_config(args.config)

    # Repro: global seed + SB3 helper
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

    prices, feats = _align_after_load(prices, feats)

    # --- clamp/validate dates using the *usable* index you just printed ---
    t_end, v_end, te_end = _clamp_dates_to_index(
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
        "gae_lambda":  cfg.ppo.gae_lambda,  # NOTE: gae_lambda (not 'gae_gamma')
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

    # --- IMPORTANT: adapt the keys below to whatever your env puts in info ---
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


if __name__ == "__main__":
    main()
