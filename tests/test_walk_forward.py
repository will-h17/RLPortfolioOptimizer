import numpy as np
import pandas as pd
import pytest

from src.env import EnvConfig
from src.utils.metrics import cagr, summarize_returns
from src.walk_forward import baseline_policies, make_folds, run_in_env, summarize


def test_make_folds_yearly_with_validation_window():
    idx = pd.bdate_range("2010-01-01", "2013-03-29")
    folds = make_folds(idx, first_test_year=2011, val_months=6)
    # 2013 is partial (~64 business days, >= MIN_TEST_ROWS) so it is still a fold
    assert [f["fold"] for f in folds] == ["2011", "2012", "2013"]
    assert folds[0] == {"fold": "2011", "train_end": "2010-06-30", "val_end": "2010-12-31", "test_end": "2011-12-31"}


def test_make_folds_skips_short_final_year():
    idx = pd.bdate_range("2010-01-01", "2012-02-15")  # 2012 has ~33 rows
    assert [f["fold"] for f in make_folds(idx, first_test_year=2011)] == ["2011"]


@pytest.fixture
def market():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2019-01-01", periods=200)
    rets = rng.normal(0.0003, [0.005, 0.01, 0.02], size=(200, 3))
    prices = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=pd.Index(["A", "B", "C"]))
    feats = prices.pct_change().fillna(0.0).add_suffix("_r1")
    return prices, feats


def _run(market, name, **cfg):
    prices, feats = market
    test_p, test_f = prices.iloc[100:], feats.iloc[100:]
    env_cfg = EnvConfig(include_cash=True, transaction_cost=0.001, **cfg)
    return run_in_env(test_p, test_f, env_cfg, baseline_policies(prices)[name])


def test_buy_and_hold_trades_only_once(market):
    res = _run(market, "Buy & hold")
    assert res["trades"][0] > 0  # initial move out of cash into equal-weight risky assets
    assert np.allclose(res["trades"][1:], 0.0)


def test_equal_weight_rebalances_and_pays_for_it(market):
    ew, bh = _run(market, "Equal weight"), _run(market, "Buy & hold")
    assert (ew["trades"][1:] > 0).all()
    assert ew["trades"][1:].sum() > bh["trades"][1:].sum()


def test_inverse_variance_overweights_low_vol_asset(market):
    prices, feats = market
    env_cfg = EnvConfig(include_cash=True)
    from src.env import PortfolioEnv
    env = PortfolioEnv(prices.iloc[100:], feats.iloc[100:], env_cfg)
    env.reset()
    w = baseline_policies(prices)["Inverse variance"](env, 0)
    assert w[env.assets.index("__CASH__")] == 0.0
    assert w[0] > w[1] > w[2] and np.isclose(w.sum(), 1.0)  # vol: A < B < C


def test_summarize_returns_metrics():
    r = np.array([0.01, -0.02, 0.03])
    m = summarize_returns(r, trades=np.array([1.0, 0.0, 0.5]))
    assert np.isclose(m["final_value"], 1.01 * 0.98 * 1.03)
    assert np.isclose(m["cagr"], cagr(r))
    assert np.isclose(m["max_drawdown"], 0.98 - 1.0)
    assert np.isclose(m["turnover"], 0.5)


def test_summary_stitches_folds_and_compares_to_equal_weight():
    dates = pd.bdate_range("2020-01-01", periods=4)
    rows, rets = [], []
    for fold, ds in (("2020a", dates[:2]), ("2020b", dates[2:])):
        for strategy, seed, r in (("Equal weight", -1, 0.001), ("RL", 0, 0.002), ("RL", 1, 0.0)):
            series = np.array([r, -r / 2])
            rows.append({"fold": fold, "strategy": strategy, "seed": seed,
                         **summarize_returns(series, np.zeros(2))})
            rets.append(pd.DataFrame({"fold": fold, "strategy": strategy, "seed": seed, "date": ds, "ret": series}))
    summary = summarize(pd.DataFrame(rows), pd.concat(rets)).set_index("strategy")
    assert summary.loc["RL", "folds"] == 2 and summary.loc["RL", "seeds"] == 2
    assert summary.loc["Equal weight", "beats_ew_sharpe"] == 0.0
    # stitched series for one seed covers all 4 dates
    assert np.isclose(summary.loc["Equal weight", "oos_cagr"], cagr(np.tile([0.001, -0.0005], 2)))
