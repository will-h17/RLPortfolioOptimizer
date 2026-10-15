import numpy as np
import pytest

from src.env import EnvConfig, PortfolioEnv

# tiny_market (see conftest.py), per-step asset returns the env uses at step t
# (_rets[t] = P[t+1]/P[t] - 1; episodes start at t=1):
#   t=1: A +2%, B  0%
#   t=2: A -1%, B +1%
#   t=3: A  0%, B +2%
#   t=4: A +3%, B  0%


def _weights(*ws):
    """Helper: list/np -> np array normalized to simplex.
    Ensures weights are non-negative and sum to 1."""
    w = np.array(ws, dtype=float).ravel()
    w = np.maximum(w, 0.0)  # ensure non-negative
    s = w.sum()
    return w / s if s > 0 else np.full_like(w, 1.0 / len(w))  # ensure sum to 1


# Trading from equal weight to [1, 0] at step 1: holdings first drift with that step's
# returns (A +2%, B 0%) to [0.51, 0.50] / 1.01, so the trade size is 2 * 0.50 / 1.01
DRIFTED_EW_STEP1 = np.array([0.51, 0.50]) / 1.01
TURNOVER_EW_TO_A = float(np.abs(np.array([1.0, 0.0]) - DRIFTED_EW_STEP1).sum())


def _env(prices, feats, **kw):
    kw.setdefault("include_cash", False)
    return PortfolioEnv(prices, feats, EnvConfig(**kw))


def test_pnl_equals_wprev_dot_ret_minus_costs(tiny_market):
    prices, feats = tiny_market
    env = _env(prices, feats, transaction_cost=0.0005)
    env.reset()

    # Step 1: hold equal weight (initial), trade from the drifted weights to [1, 0]
    _, r1, _, _, info1 = env.step(_weights(1, 0))
    assert np.allclose(info1["w_drift"], DRIFTED_EW_STEP1)
    assert np.isclose(info1["turnover"], TURNOVER_EW_TO_A)
    assert np.isclose(info1["cost"], 0.0005 * TURNOVER_EW_TO_A)
    assert np.isclose(r1, 0.5 * 0.02 + 0.5 * 0.0 - 0.0005 * TURNOVER_EW_TO_A)

    # Step 2: hold [1, 0] (does not drift), trade to [0, 1]. Turnover = 2.0
    _, r2, _, _, info2 = env.step(_weights(0, 1))
    assert np.isclose(info2["cost"], 0.0005 * 2.0)
    assert np.isclose(r2, 1.0 * -0.01 - 0.001)
    assert np.isclose(info2["ret"], r2)  # no shaping configured -> reward == net return


def test_action_projection_simplex(tiny_market):
    prices, feats = tiny_market
    env = _env(prices, feats, seed=1)
    env.reset()

    # Provide a wild action; env/policy head should project/clip to simplex domain used internally.
    raw = np.array([10.0, -3.0])  # invalid
    _, _, _, _, info = env.step(raw)
    w = np.asarray(info["weights"], dtype=float)
    assert np.all(w >= -1e-12)
    assert np.isclose(w.sum(), 1.0, atol=1e-6)


def test_costs_apply_only_when_weights_change(tiny_market):
    prices, feats = tiny_market
    # 100 basis points (1%) to make effect obvious
    env = _env(prices, feats, transaction_cost=0.01)
    env.reset()

    _, _, _, _, info0 = env.step(_weights(1, 0))  # trade from equal weight: pays cost
    assert np.isclose(info0["cost"], 0.01 * TURNOVER_EW_TO_A)

    _, reward1, _, _, info1 = env.step(_weights(1, 0))  # same weights: no cost
    assert info1["cost"] == 0.0
    assert np.isclose(reward1, -0.01)  # 100% in A over t=2


def test_cost_model_fixed_spread(tiny_market):
    prices, feats = tiny_market
    # per-unit-turnover cost = fee + slippage + half spread = 1 + 1 + 4/2 = 4 bps
    env = _env(prices, feats, cost_enabled=True, fee_bps=1.0, slippage_bps=1.0,
               spread_type="fixed", spread_fixed_bps=4.0)
    env.reset()
    _, r, _, _, info = env.step(_weights(1, 0))
    assert np.isclose(info["cost"], 4e-4 * TURNOVER_EW_TO_A)
    assert np.isclose(r, 0.01 - 4e-4 * TURNOVER_EW_TO_A)


def test_cost_model_vol20_spread_uses_raw_price_vol(tiny_market):
    prices, feats = tiny_market
    env = _env(prices, feats, cost_enabled=True, spread_type="vol20", spread_k_vol_to_bps=10_000.0)
    env.reset()
    _, _, _, _, info = env.step(_weights(1, 0))  # trade at t=1 from the drifted equal weights

    # At t=1 there is one return per asset, so the first estimate (t=2, two returns) is backfilled
    vol_a = np.std([0.01, 0.02], ddof=1)
    vol_b = np.std([-0.01, 0.00], ddof=1)
    half_spread_bps = 0.5 * 10_000.0 * np.array([vol_a, vol_b])
    dw = np.abs(np.array([1.0, 0.0]) - DRIFTED_EW_STEP1)
    assert np.isclose(info["cost"], np.dot(half_spread_bps, dw) / 10_000.0)
    assert info["cost"] > 0


def test_cost_model_has_effect_vs_disabled(tiny_market):
    prices, feats = tiny_market
    rewards = {}
    for enabled in (False, True):
        env = _env(prices, feats, cost_enabled=enabled, spread_type="fixed", spread_fixed_bps=10.0)
        env.reset()
        rewards[enabled] = env.step(_weights(1, 0))[1]
    assert rewards[True] < rewards[False]


def test_info_ret_excludes_reward_shaping(tiny_market):
    prices, feats = tiny_market
    env = _env(prices, feats, reward_return_bonus=1.0)
    env.reset()
    _, reward, _, _, info = env.step(_weights(1, 0))
    assert np.isclose(info["ret"], 0.01)                  # true net return
    assert np.isclose(reward, 0.01 * (1.0 + 1.0 * 0.01))  # shaped reward


def test_episode_uses_all_returns_and_truncates(tiny_market):
    prices, feats = tiny_market
    env = _env(prices, feats)
    env.reset()

    steps = []
    while True:
        _, r, terminated, truncated, _ = env.step(_weights(1, 0))
        steps.append(r)
        if terminated or truncated:
            break

    assert len(steps) == len(prices) - 2  # t = 1 .. T-2
    assert truncated and not terminated   # running out of data is a time limit
    assert np.isclose(steps[-1], 0.03)    # last step earns A's final return (P5/P4 - 1)


def test_holding_a_constant_mix_pays_rebalancing_cost(tiny_market):
    prices, feats = tiny_market
    env = _env(prices, feats, transaction_cost=0.01)
    env.reset()
    # Starting at equal weight, A outperforms on step 1, so restoring 50/50 requires a trade
    _, _, _, _, info = env.step(_weights(1, 1))
    expected_turnover = float(np.abs(0.5 - DRIFTED_EW_STEP1).sum())
    assert np.isclose(info["turnover"], expected_turnover) and expected_turnover > 0
    assert np.isclose(info["cost"], 0.01 * expected_turnover)


def test_log_reward_mode_keeps_arithmetic_net_return(tiny_market):
    prices, feats = tiny_market
    env = _env(prices, feats, reward_mode="log", transaction_cost=0.001)
    env.reset()
    _, reward, _, _, info = env.step(_weights(1, 0))
    net = 0.01 - 0.001 * TURNOVER_EW_TO_A
    assert np.isclose(info["ret"], net)            # performance metrics stay arithmetic
    assert np.isclose(info["value"], 1.0 + net)
    assert np.isclose(reward, np.log1p(net))       # only the reward is in log space


def test_unknown_reward_mode_is_rejected(tiny_market):
    prices, feats = tiny_market
    with pytest.raises(ValueError, match="reward_mode"):
        _env(prices, feats, reward_mode="sharpe")
