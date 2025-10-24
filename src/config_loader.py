from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Any, Dict
import json
import yaml

@dataclass
class PathsCfg:
    prices: str
    features: str
    out_root: str = "artifacts"

@dataclass
class DatesCfg:
    train_end: str
    val_end: str
    test_end: str

@dataclass
class EnvCfg:
    transaction_cost_bps: float = 0.0
    include_cash: bool = True
    cash_rate_annual: float = 0.0

@dataclass
class RewardCfg:
    return_weight: float = 1.0
    turnover_penalty: float = 0.0
    drawdown_penalty: float = 0.0
    volatility_penalty: float = 0.0

@dataclass
class PPOCfg:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    n_steps: int = 2048
    batch_size: int = 256
    clip_range: float = 0.2
    ent_coef: float = 0.0
    policy_hidden_sizes: List[int] = field(default_factory=lambda: [128, 128])

@dataclass
class TrainCfg:
    total_timesteps: int = 200_000
    checkpoint_freq: int = 50_000
    eval_freq: int = 20_000

@dataclass
class AppCfg:
    run_name: str
    seed: int
    paths: PathsCfg
    dates: DatesCfg
    env: EnvCfg
    reward: RewardCfg
    ppo: PPOCfg
    train: TrainCfg

def _load_raw(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    text = p.read_text()
    if p.suffix.lower() in [".yaml", ".yml"]:
        return yaml.safe_load(text)
    return json.loads(text)

def load_config(path: str | Path) -> AppCfg:
    raw = _load_raw(path)
    return AppCfg(
        run_name = raw["run_name"],
        seed     = int(raw["seed"]),
        paths    = PathsCfg(**raw["paths"]),
        dates    = DatesCfg(**raw["dates"]),
        env      = EnvCfg(**raw.get("env", {})),
        reward   = RewardCfg(**raw.get("reward", {})),
        ppo      = PPOCfg(**raw.get("ppo", {})),
        train    = TrainCfg(**raw.get("train", {})),
    )
