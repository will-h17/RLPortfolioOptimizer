# tests/test_no_lookahead.py
import numpy as np

from src.env import EnvConfig, PortfolioEnv


def test_no_lookahead_rewards_use_previous_weights(tiny_market):
    prices, feats = tiny_market
    env = PortfolioEnv(prices, feats, EnvConfig(include_cash=False, transaction_cost=0.0, seed=0, reward_mode="arith"))
    env.reset()

    # Step 1: choose w := [1, 0]; these weights are held during step 2.
    env.step(np.array([1.0, 0.0]))
    # Step 2: pick something very different to try to "cheat". Over step 2, A returns -1% and B +1%,
    # so using the new action would give +1%; the correct (previous-weights) answer is -1%.
    _, r2, _, _, info2 = env.step(np.array([0.0, 1.0]))

    assert np.allclose(info2["w_prev"], [1.0, 0.0])
    assert np.isclose(r2, -0.01)  # must use weights from step 1 (w_prev)
