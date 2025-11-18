# tests/test_env_math.py
import numpy as np
import pandas as pd
import pytest

from src.env import PortfolioEnv, EnvConfig

def _weights(*ws):
    """Helper: list/np -> np array normalized to simplex."""
    w = np.array(ws, dtype=float).ravel()
    w = np.maximum(w, 0.0)
    s = w.sum()
    return w / s if s > 0 else np.full_like(w, 1.0 / len(w))

def test_pnl_equals_wprev_dot_ret_minus_costs(tiny_market):
    prices, feats = tiny_market
    cfg = EnvConfig(include_cash=False, transaction_cost=0.0005, seed=123, reward_mode="arith")

    env = PortfolioEnv(prices, feats, EnvConfig(include_cash=False, transaction_cost=0.0, seed=0, reward_mode="arith"))
    obs, _ = env.reset()

    # Start at day 0 with default env weights (often equal weights). We force a trade to known weights at step 0.
    # After reset(), env.t = 1, so:
    # - Step 0 uses _rets[1] which is return from day 1->2
    # - Step 1 uses _rets[2] which is return from day 2->3
    
    # Step 0: set weights for *next* day
    w_tgt = _weights(0.75, 0.25)
    obs, reward, done, trunc, info = env.step(w_tgt)

    # The reward at step 0 should be based on *previous* weights (env's initial), not w_tgt.
    # Next step establishes w_prev = w_tgt. So check step 1 math exactly:
    w_prev = info["weights"]  # weights *after* applying trade at t=0
    # Now step once more to realize PnL with w_prev
    obs, reward1, done, trunc, info1 = env.step(w_prev)

    # Compute expected net using the env-provided components
    gross = float(np.dot(info1["w_prev"], info1["r_t"]))
    expect = gross - info1["cost"]
    assert np.isclose(reward1, expect, atol=1e-12), f"reward1={reward1} vs expect={expect}"
    
    # For step 1, env uses _rets[2] which is return from day 2->3
    # Verify using the actual return vector from info (which is what env used)
    ret_calc = float(np.dot(w_prev, info1["r_t"]))
    assert np.isclose(reward1, ret_calc, atol=1e-12), f"reward1={reward1} vs w_prev·r_t={ret_calc} (using env's r_t)"

def test_action_projection_simplex(tiny_market):
    prices, feats = tiny_market
    cfg = EnvConfig(include_cash=False, transaction_cost=0.0, seed=1)
    env = PortfolioEnv(prices, feats, cfg)
    obs, _ = env.reset()

    # Provide a wild action; env/policy head should project/clip to simplex domain used internally.
    raw = np.array([10.0, -3.0])  # invalid
    obs, reward, done, trunc, info = env.step(raw)
    w = np.asarray(info["weights"], dtype=float)
    assert np.all(w >= -1e-12)
    assert np.isclose(w.sum(), 1.0, atol=1e-6)

def test_costs_apply_only_when_weights_change(tiny_market):
    prices, feats = tiny_market
    # 100 bps to make effect obvious
    cfg = EnvConfig(include_cash=False, transaction_cost=0.01, seed=7)
    env = PortfolioEnv(prices, feats, cfg)
    obs, _ = env.reset()

    # Hold weights constant → no trading cost beyond first rebalance
    obs, reward0, *_ = env.step(np.array([0.5, 0.5], dtype=float))  # pay cost once here
    obs, reward1, *_ = env.step(np.array([0.5, 0.5], dtype=float))  # should NOT pay trading cost again
    # Compute expected: reward1 = w_prev · r_t (no cost)
    w_prev = _weights(0.5, 0.5)
    r = prices.pct_change().iloc[1].to_numpy()
    expect = float(np.dot(w_prev, r))
    assert np.isclose(reward1, expect, atol=1e-12)
