import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import buy_and_hold, equal_weight
from src.config_loader import load_config
from src.env import EnvConfig, build_env_config, load_run_env_config
from src.scaler import FitOnTrainScaler


def test_buy_and_hold_drifts_and_differs_from_equal_weight(tiny_market):
    prices, feats = tiny_market
    bh = buy_and_hold(prices, feats)
    ew = equal_weight(prices, feats)

    # Buy & hold: start 50/50 and never trade -> final value is the average price relative
    gross = (prices.iloc[-1] / prices.iloc[0]).to_numpy()
    assert np.isclose(bh["metrics"]["final_value"], gross.mean())
    assert np.isclose(bh["metrics"]["turnover"], 0.0)

    # Equal weight rebalances daily: equity compounds the average daily return
    r = prices.pct_change().iloc[1:].to_numpy()
    assert np.isclose(ew["metrics"]["final_value"], np.prod(1.0 + r.mean(axis=1)))
    assert ew["metrics"]["turnover"] > 0.0

    assert not np.isclose(bh["metrics"]["final_value"], ew["metrics"]["final_value"])


def test_scaler_roundtrip_multiindex(tmp_path):
    cols = pd.MultiIndex.from_tuples([("SPY", "ret_1"), ("SPY", "vol_20"), ("TLT", "ret_1")])
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(50, 3)), columns=cols)

    scaler = FitOnTrainScaler().fit(X)
    scaler.save(tmp_path / "scaler.json")
    loaded = FitOnTrainScaler.load(tmp_path / "scaler.json")

    pd.testing.assert_frame_equal(scaler.transform(X), loaded.transform(X))


def test_env_config_roundtrip(tmp_path):
    cfg = EnvConfig(cost_enabled=True, spread_type="vol20", normalize_obs_weights=True, cash_rate_annual=0.02)
    cfg.save(tmp_path / "env_config.json")
    assert load_run_env_config(tmp_path) == cfg


def test_legacy_run_env_config_rebuilt_from_config_json(tmp_path):
    # Runs created before env_config.json existed only have config.json (the dumped AppCfg)
    app_cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "rlppo.yaml")
    (tmp_path / "config.json").write_text(json.dumps(app_cfg.model_dump()))

    env_cfg = load_run_env_config(tmp_path)
    assert env_cfg == build_env_config(app_cfg)
    assert env_cfg.normalize_obs_weights is True
    assert env_cfg.cost_enabled is True and env_cfg.spread_type == "vol20"
    assert env_cfg.cash_rate_annual == 0.02
