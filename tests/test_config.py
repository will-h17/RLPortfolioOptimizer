import pytest

from src.config_loader import load_config
from src.paths import CONFIGS_DIR

MINIMAL = """
run_name: "t"
seed: 1
paths: {prices: "p.parquet", features: "f.parquet"}
dates: {train_end: "2019-12-31", val_end: "2020-06-30", test_end: "2021-03-31"}
"""


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


@pytest.mark.parametrize("path", sorted(CONFIGS_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_every_training_config_in_repo_loads(path):
    if path.name.startswith("eval_grid"):
        pytest.skip("eval grid configs have their own format")
    load_config(path)


def test_extends_deep_merges_and_child_wins(tmp_path):
    _write(tmp_path, "base.yaml", MINIMAL + "ppo: {learning_rate: 0.001, gamma: 0.9}\n")
    child = _write(tmp_path, "child.yaml", 'extends: base.yaml\nrun_name: "child"\nppo: {gamma: 0.95}\n')
    cfg = load_config(child)
    assert cfg.run_name == "child"
    assert cfg.ppo.gamma == 0.95 and cfg.ppo.learning_rate == 0.001  # merged, not replaced


def test_circular_extends_is_rejected(tmp_path):
    _write(tmp_path, "a.yaml", "extends: b.yaml\n")
    _write(tmp_path, "b.yaml", "extends: a.yaml\n")
    with pytest.raises(ValueError, match="Circular extends"):
        load_config(tmp_path / "a.yaml")


@pytest.mark.parametrize("extra, match", [
    ("hpo: {train_timestep: 5}\n", "train_timestep"),                       # typo'd key
    ("unknown_section: {}\n", "unknown_section"),
    ("env: {reward_mode: sharpe}\n", "reward_mode"),
    ("env: {random_start: true}\n", "episode_length"),
    ("ppo: {n_steps: 64, batch_size: 128}\n", "batch_size"),
    ("ppo: {gamma: 1.5}\n", "gamma"),
    ("cost_model: {spread: {type: wide}}\n", "type"),
])
def test_invalid_values_fail_loudly(tmp_path, extra, match):
    path = _write(tmp_path, "bad.yaml", MINIMAL + extra)
    with pytest.raises(ValueError, match=match):
        load_config(path)


def test_dates_must_be_ordered_and_iso(tmp_path):
    swapped = MINIMAL.replace('val_end: "2020-06-30"', 'val_end: "2019-01-01"')
    with pytest.raises(ValueError, match="train_end < val_end < test_end"):
        load_config(_write(tmp_path, "a.yaml", swapped))
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, "b.yaml", MINIMAL.replace('"2019-12-31"', '"12/31/2019"')))


def test_unquoted_yaml_dates_are_accepted(tmp_path):
    cfg = load_config(_write(tmp_path, "a.yaml", MINIMAL.replace('"2019-12-31"', "2019-12-31")))
    assert cfg.dates.train_end == "2019-12-31"


def test_assignment_is_validated(tmp_path):
    cfg = load_config(_write(tmp_path, "a.yaml", MINIMAL))
    with pytest.raises(ValueError):
        cfg.train.n_envs = 0
