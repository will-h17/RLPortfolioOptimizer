"""
config_loader.py — Validated training configuration.

Configs are YAML (or JSON) files parsed into Pydantic models. Every section rejects unknown keys,
so a typo such as `hpo.train_timestep` fails loudly instead of being silently ignored, and values
are type- and range-checked when loaded.

A config can inherit from another with a top-level `extends:` key (path relative to the file);
nested sections are deep-merged and the child's values win:

    extends: base/common.yaml
    run_name: "my_run"
    ppo:
      learning_rate: 0.0001     # every other ppo.* value comes from the base

Sections:
- PathsCfg: Data file paths (prices, features, output directory)
- DatesCfg: Train/validation/test split end dates (inclusive)
- EnvCfg: Environment settings (cash, reward mode, training episode windows, reward shaping)
- RewardCfg: Reward weights and penalties
- PPOCfg: PPO algorithm hyperparameters
- TrainCfg: Training schedule (timesteps, parallel envs, checkpoint/validation frequency)
- CostModelCfg / SpreadCfg: Transaction cost model
- HPOCfg: Hyperparameter search settings
- AppCfg: Top-level configuration combining all sections
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class _Section(BaseModel):
    # validate_assignment: code that edits a loaded config (HPO, walk-forward) is checked too
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PathsCfg(_Section):
    prices: str
    features: str
    out_root: str = "artifacts"


class DatesCfg(_Section):
    train_end: str
    val_end: str
    test_end: str

    @field_validator("train_end", "val_end", "test_end", mode="before")
    @classmethod
    def _iso_date(cls, v: Any) -> str:
        # Unquoted YAML dates arrive as datetime.date; store every date as an ISO string
        if isinstance(v, dt.date):
            return v.isoformat()
        dt.date.fromisoformat(str(v))  # raises ValueError for anything that is not YYYY-MM-DD
        return str(v)

    @model_validator(mode="after")
    def _ordered(self) -> DatesCfg:
        if not (self.train_end < self.val_end < self.test_end):
            raise ValueError(f"dates must satisfy train_end < val_end < test_end, got "
                             f"{self.train_end}, {self.val_end}, {self.test_end}")
        return self


class EnvCfg(_Section):
    transaction_cost_bps: float = Field(default=0.0, ge=0)  # flat cost, used when cost_model.enabled is false
    include_cash: bool = True
    cash_rate_annual: float = 0.0
    normalize_obs_weights: bool = False
    reward_mode: Literal["arith", "log"] = "arith"  # net return, or log(1 + net return)
    random_start: bool = False         # training episodes start at a random date (needs episode_length)
    episode_length: int | None = Field(default=None, gt=0)  # training episode length in days; None = whole train split
    # Advanced reward shaping for profitability
    reward_vol_scaling: bool = False
    reward_sharpe_bonus: float = 0.0
    reward_vol_window: int = Field(default=20, gt=1)
    reward_return_bonus: float = 0.0  # Bonus multiplier for positive returns
    reward_return_threshold: float = 0.0  # Minimum return to get bonus
    reward_compound_bonus: float = 0.0  # Bonus for cumulative returns

    @model_validator(mode="after")
    def _episode_window(self) -> EnvCfg:
        if self.random_start and self.episode_length is None:
            raise ValueError("env.random_start requires env.episode_length")
        return self


class RewardCfg(_Section):
    return_weight: float = 1.0
    turnover_penalty: float = Field(default=0.0, ge=0)
    drawdown_penalty: float = 0.0    # not implemented in the env yet
    volatility_penalty: float = 0.0  # not implemented in the env yet


class PPOCfg(_Section):
    learning_rate: float = Field(default=3e-4, gt=0)
    gamma: float = Field(default=0.99, gt=0, le=1)
    gae_lambda: float = Field(default=0.95, gt=0, le=1)
    n_steps: int = Field(default=2048, gt=1)      # per env
    batch_size: int = Field(default=256, gt=1)
    clip_range: float = Field(default=0.2, gt=0)
    ent_coef: float = Field(default=0.0, ge=0)
    policy_hidden_sizes: list[int] = Field(default_factory=lambda: [128, 128], min_length=1)
    n_epochs: int = Field(default=10, gt=0)       # SB3 default
    alpha_floor: float = Field(default=1e-3, gt=0)  # minimum Dirichlet concentration; 1.0 keeps the policy off the simplex corners
    normalize_reward: bool = False        # scale rewards with a running std (VecNormalize) during training


class TrainCfg(_Section):
    total_timesteps: int = Field(default=200_000, gt=0)
    n_envs: int = Field(default=1, ge=1)            # parallel training environments
    checkpoint_freq: int = Field(default=50_000, gt=0)
    eval_freq: int = Field(default=20_000, gt=0)    # validation-Sharpe model selection


class HPOCfg(_Section):
    train_timesteps: int = Field(default=300_000, gt=0)  # training budget per Optuna trial


class SpreadCfg(_Section):
    type: Literal["fixed", "vol20"] = "fixed"
    fixed_bps: float = Field(default=0.0, ge=0)     # used when type == "fixed"
    k_vol_to_bps: float = Field(default=8000.0, ge=0)  # vol20: spread in bps = k * trailing 20d daily vol
    vol_col_suffix: str = "_s20"            # deprecated; vol20 is computed from prices


class CostModelCfg(_Section):
    enabled: bool = False
    fee_bps: float = Field(default=0.0, ge=0)
    slippage_bps: float = Field(default=0.0, ge=0)
    spread: SpreadCfg = Field(default_factory=lambda: SpreadCfg())


class AppCfg(_Section):
    run_name: str = Field(min_length=1)
    seed: int
    paths: PathsCfg
    dates: DatesCfg
    env: EnvCfg = Field(default_factory=lambda: EnvCfg())
    reward: RewardCfg = Field(default_factory=lambda: RewardCfg())
    ppo: PPOCfg = Field(default_factory=lambda: PPOCfg())
    train: TrainCfg = Field(default_factory=lambda: TrainCfg())
    cost_model: CostModelCfg = Field(default_factory=lambda: CostModelCfg())
    hpo: HPOCfg = Field(default_factory=lambda: HPOCfg())

    @model_validator(mode="after")
    def _batch_fits_rollout(self) -> AppCfg:
        rollout = self.ppo.n_steps * self.train.n_envs
        if self.ppo.batch_size > rollout:
            raise ValueError(f"ppo.batch_size ({self.ppo.batch_size}) exceeds the samples per update "
                             f"(ppo.n_steps x train.n_envs = {rollout})")
        return self


# ---------------- loading ----------------

def _read(path: Path) -> dict[str, Any]:
    text = path.read_text()
    data = yaml.safe_load(text) if path.suffix.lower() in (".yaml", ".yml") else json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must be a mapping at the top level")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_raw(path: str | Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Read a config file and resolve its `extends:` chain into one merged mapping."""
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f"Circular extends: {' -> '.join(str(p) for p in (*_seen, path))}")
    data = _read(path)
    parent = data.pop("extends", None)
    if parent is None:
        return data
    return _deep_merge(load_raw(path.parent / parent, (*_seen, path)), data)


def load_config(path: str | Path) -> AppCfg:
    raw = load_raw(path)
    try:
        return AppCfg.model_validate(raw)
    except ValidationError as e:
        raise ValueError(f"Invalid config {path}:\n{e}") from e
