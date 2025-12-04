import numpy as np
import pandas as pd
from src.env import PortfolioEnv, EnvConfig

def test_no_lookahead_rewards_use_previous_weights(tiny_market):
    prices, feats = tiny_market
    env = PortfolioEnv(prices, feats, EnvConfig(include_cash=False, transaction_cost=0.0, seed=0, reward_mode="arith"))
    obs, _ = env.reset()

    # At t=0, choose weights w0 := [1, 0]. The reward for the *next* step (day 1)
    # must be r_A[1], not influenced by any action chosen at step 1.
    obs, r0, done, trunc, info0 = env.step(np.array([1.0, 0.0]))     # sets w_prev for day 1 = [1,0]
    # Now at step 1, pick something very different to try to "cheat"
    obs, r1, done, trunc, info1 = env.step(np.array([0.0, 1.0]))

    expected = float(np.dot(info1["w_prev"], info1["r_t"]))  # must use previous weights
    assert np.isclose(r1, expected, atol=1e-12)