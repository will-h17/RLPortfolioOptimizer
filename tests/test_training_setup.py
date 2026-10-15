"""Tests for the Phase 3 training setup: episode windows, the Dirichlet floor, and model selection."""
import csv
from typing import cast

import numpy as np
import pandas as pd
import pytest
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agent import DirichletActorCriticPolicy, make_sb3_ppo
from src.callbacks import ValSharpeCallback
from src.env import EnvConfig, PortfolioEnv, make_env


@pytest.fixture
def market():
    """60 business days, 3 assets, random-walk prices with simple features."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=60)
    rets = rng.normal(0.0005, 0.01, size=(60, 3))
    prices = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=pd.Index(["A", "B", "C"]))
    feats = prices.pct_change().fillna(0.0).add_suffix("_r1")
    return prices, feats


def _episode(env, seed=None):
    env.reset(seed=seed)
    start, steps = env.t, 0
    while True:
        steps += 1
        _, _, terminated, truncated, _ = env.step(np.ones(env.n_assets))
        if terminated or truncated:
            return start, steps


def test_random_start_episodes_have_fixed_length_and_vary(market):
    prices, feats = market
    env = PortfolioEnv(prices, feats, EnvConfig(random_start=True, max_steps=10))
    episodes = [_episode(env, seed=s) for s in range(20)]
    assert all(steps == 10 for _, steps in episodes)
    starts = [start for start, _ in episodes]
    assert len(set(starts)) > 1
    assert min(starts) >= 1 and max(starts) + 10 - 1 <= len(prices) - 2


def test_for_evaluation_runs_one_full_episode(market):
    prices, feats = market
    train_cfg = EnvConfig(random_start=True, max_steps=10)
    env = PortfolioEnv(prices, feats, train_cfg.for_evaluation())
    assert _episode(env, seed=3) == (1, len(prices) - 2)


def test_random_start_requires_episode_length(market):
    prices, feats = market
    with pytest.raises(ValueError, match="max_steps"):
        PortfolioEnv(prices, feats, EnvConfig(random_start=True))


def test_alpha_floor_is_saved_with_the_model(market, tmp_path):
    prices, feats = market
    env = DummyVecEnv([make_env(prices, feats, EnvConfig(), seed=0)])
    obs = cast(np.ndarray, env.reset())

    for floor in (1.0, 1e-3):
        model = make_sb3_ppo(env, alpha_floor=floor, n_steps=8, batch_size=8, seed=0)
        model.save(tmp_path / f"m_{floor}.zip")
        loaded = PPO.load(tmp_path / f"m_{floor}.zip", device="cpu")
        policy = cast(DirichletActorCriticPolicy, loaded.policy)
        assert policy.alpha_floor == floor
        dist = policy.get_distribution(th.as_tensor(obs, dtype=th.float32))
        assert dist.alpha is not None and float(dist.alpha.min()) >= floor


def test_model_saved_without_alpha_floor_loads_with_old_default(market, tmp_path):
    # Simulate a model saved before alpha_floor existed: drop it from the saved policy kwargs
    prices, feats = market
    env = DummyVecEnv([make_env(prices, feats, EnvConfig(), seed=0)])
    model = make_sb3_ppo(env, alpha_floor=1.0, n_steps=8, batch_size=8, seed=0)
    model.policy_kwargs.pop("alpha_floor")
    model.save(tmp_path / "old.zip")
    assert PPO.load(tmp_path / "old.zip", device="cpu").policy.alpha_floor == 1e-3


def test_val_sharpe_callback_saves_best_model_and_logs(market, tmp_path):
    prices, feats = market
    cfg = EnvConfig(random_start=True, max_steps=10)
    env = DummyVecEnv([make_env(prices, feats, cfg, seed=i) for i in range(2)])
    model = make_sb3_ppo(env, n_steps=8, batch_size=16, n_epochs=1, seed=0, alpha_floor=1.0)

    cb = ValSharpeCallback(PortfolioEnv(prices, feats, cfg.for_evaluation()), eval_freq=16,
                           best_model_path=tmp_path / "best" / "best_model.zip",
                           log_path=tmp_path / "val.csv", verbose=0)
    model.learn(total_timesteps=48, callback=cb)

    rows = list(csv.DictReader((tmp_path / "val.csv").open()))
    assert [int(r["timesteps"]) for r in rows] == [16, 32, 48]
    assert (tmp_path / "best" / "best_model.zip").exists()
    assert np.isclose(cb.best_sharpe, max(float(r["val_sharpe"]) for r in rows))
