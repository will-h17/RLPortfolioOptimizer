from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.paths import PROJECT_ROOT, resolve
from src.runlog import latest_run, resolve_run_model
from src.utils.data_utils import WARMUP, prepare_dataset, validate_split_dates

IDX = pd.bdate_range("2020-01-01", periods=100)


def test_validate_split_dates_accepts_valid_split():
    t, v, te = validate_split_dates(IDX, "2020-02-28", "2020-03-31", "2020-05-15")
    assert t < v < te


@pytest.mark.parametrize("dates, match", [
    (("2020-03-31", "2020-02-28", "2020-05-15"), "train_end < val_end < test_end"),
    (("2019-06-30", "2020-03-31", "2020-05-15"), "Empty train"),
    (("2020-02-28", "2020-03-31", "2021-01-01"), None),  # test_end past data end is allowed
    (("2020-05-29", "2020-06-30", "2020-07-31"), "Empty val, test"),
])
def test_validate_split_dates_rejects_bad_splits(dates, match):
    if match is None:
        validate_split_dates(IDX, *dates)
    else:
        with pytest.raises(ValueError, match=match):
            validate_split_dates(IDX, *dates)


def test_prepare_dataset_splits_are_contiguous_and_disjoint(tmp_path):
    idx = pd.bdate_range("2018-01-01", periods=400)
    prices = pd.DataFrame({"A": np.linspace(100, 140, 400), "B": np.linspace(50, 45, 400)}, index=idx)
    feats = prices.pct_change().dropna().add_suffix("_r1")  # features start one row later than prices
    prices.to_parquet(tmp_path / "p.parquet")
    feats.to_parquet(tmp_path / "f.parquet")

    cfg = SimpleNamespace(
        paths=SimpleNamespace(prices=str(tmp_path / "p.parquet"), features=str(tmp_path / "f.parquet")),
        dates=SimpleNamespace(train_end="2018-12-31", val_end="2019-03-29", test_end="2019-06-28"),
    )
    data = prepare_dataset(cfg)

    assert data.prices.index[0] == feats.index[WARMUP]  # aligned, then warm-up rows dropped
    (P_tr, _), (P_va, _), (P_te, _) = data.train, data.val, data.test
    for P, X in data.train, data.val, data.test:
        assert P.index.equals(X.index) and len(P) > 0
    assert P_tr.index[-1] < P_va.index[0] and P_va.index[-1] < P_te.index[0]
    assert P_te.index[-1] <= pd.Timestamp("2019-06-28")


def test_resolve_is_relative_to_project_root(tmp_path):
    assert resolve("configs/rlppo.yaml") == PROJECT_ROOT / "configs" / "rlppo.yaml"
    assert resolve(tmp_path) == tmp_path


def _make_run(root, name, model):
    run = root / name
    (run / "models" / "best").mkdir(parents=True)
    (run / "config.json").write_text("{}")
    if model:
        (run / "models" / model).write_bytes(b"")
    return run


def test_resolve_run_model_prefers_best_then_final(tmp_path):
    run = _make_run(tmp_path, "20250101-000000__a", "a_final.zip")
    assert resolve_run_model(run).name == "a_final.zip"
    (run / "models" / "best" / "best_model.zip").write_bytes(b"")
    assert resolve_run_model(run).name == "best_model.zip"

    empty = _make_run(tmp_path, "20250102-000000__b", None)
    with pytest.raises(FileNotFoundError):
        resolve_run_model(empty)


def test_latest_run_skips_runs_without_a_model(tmp_path):
    import os
    good = _make_run(tmp_path, "20250101-000000__a", "a_final.zip")
    _make_run(tmp_path, "20250102-000000__a", None)  # aborted run (no model)
    os.utime(good, (1, 1))  # the aborted run is newer
    assert latest_run(tmp_path) == good
    assert latest_run(tmp_path, run_name="a") == good
    with pytest.raises(FileNotFoundError):
        latest_run(tmp_path, run_name="other")


def test_download_universe_writes_flat_csv_from_multiindex_yfinance_frame(tmp_path, monkeypatch):
    """yfinance >= 0.2.48 returns (field, ticker) columns even for one ticker."""
    import src.data as data_mod

    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="Date")
    cols = pd.MultiIndex.from_tuples([("Close", "SPY"), ("Open", "SPY")], names=["Price", "Ticker"])
    frame = pd.DataFrame([[470.0, 469.0], [468.5, 470.1]], index=idx, columns=cols)
    monkeypatch.setattr(data_mod, "RAW_DIR", tmp_path)
    monkeypatch.setattr(data_mod.yf, "download", lambda *a, **k: frame)

    (path,) = data_mod.download_universe(["SPY"])
    lines = path.read_text().splitlines()
    assert lines == ["date,SPY", "2024-01-02,470.0", "2024-01-03,468.5"]  # no extra ticker header row


def test_download_universe_raises_on_empty_download(tmp_path, monkeypatch):
    import src.data as data_mod

    monkeypatch.setattr(data_mod, "RAW_DIR", tmp_path)
    monkeypatch.setattr(data_mod.yf, "download", lambda *a, **k: pd.DataFrame())
    with pytest.raises(RuntimeError, match="no data"):
        data_mod.download_universe(["NOPE"])
