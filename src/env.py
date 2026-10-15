"""
env.py — RL environment for portfolio optimization.

State  : concat(features_t, current_portfolio_weights)
Action : target weights (projected to the probability simplex: long-only, sum=1)
Reward : net = w_prev · r_{t->t+1} - cost(|w_new - w_drifted|); reward = net (or log(1+net)),
         plus optional shaping terms. Holdings drift with prices between trades.

info["ret"] is always the true net portfolio return (after costs, before shaping);
use it, not the reward, for any performance metric.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor

from src.paths import FEATURES_PATH, PRICES_PATH

# Rolling window (days) for the "vol20" spread model
SPREAD_VOL_WINDOW = 20


def _project_to_simplex(x: np.ndarray) -> np.ndarray:
    """Euclidean projection to {w >= 0, sum w = 1} via sorting (O(n log n))."""
    v = np.asarray(x, dtype=np.float64).ravel()
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_idx = np.where(u + (1 - cssv) / (np.arange(n) + 1) > 0)[0]
    if rho_idx.size == 0:
        return np.ones(n) / n
    rho = rho_idx[-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    w = np.maximum(v - theta, 0)
    s = w.sum()
    return w if s > 0 else np.ones(n) / n


@dataclass
class EnvConfig:
    transaction_cost: float = 0.0  # fallback (fraction, e.g., 0.0005 for 5 bps)
    include_cash: bool = True
    cash_rate_annual: float = 0.0
    seed: int = 0

    start: int | None = None   # first row index within provided data
    end: int | None = None     # last row index whose forward return is used (inclusive)
    max_steps: int | None = None  # episode length cap (steps); None = run to the end of the data
    random_start: bool = False    # sample each episode's start uniformly (training only; needs max_steps)

    # Reward definition: "arith" (net return, default) or "log" (log(1 + net return))
    reward_mode: str = "arith"
    return_weight: float = 1.0  # <-- MUST be +1.0
    turnover_penalty: float = 0.0
    drawdown_penalty: float = 0.0
    volatility_penalty: float = 0.0

    # Optional dynamic cost model parameters
    cost_enabled: bool = False
    fee_bps: float = 0.0
    slippage_bps: float = 0.0
    spread_type: str = "fixed"     # "fixed" or "vol20"
    spread_fixed_bps: float = 0.0
    spread_k_vol_to_bps: float = 8000.0
    # Deprecated: vol20 is now computed from raw prices inside the env, because the
    # feature matrix the env receives is z-scored and cannot be used as a volatility.
    spread_vol_col_suffix: str = "_s20"

    # Observation normalization (optional, for RL training stability)
    normalize_obs_weights: bool = False  # If True, normalize weights relative to equal-weight

    # Advanced reward shaping for profitability
    reward_vol_scaling: bool = False  # Scale reward by Sharpe-like metric (preserves returns, reduces vol → higher Sharpe AND final value)
    reward_sharpe_bonus: float = 0.0  # Bonus for high rolling Sharpe (0.0 = disabled)
    reward_vol_window: int = 20  # Window for rolling volatility/Sharpe calculation
    
    # Profitability-focused reward shaping
    reward_return_bonus: float = 0.0  # Bonus multiplier for positive returns (encourages profitability)
    reward_return_threshold: float = 0.0  # Minimum return to get bonus (0.0 = any positive return)
    reward_compound_bonus: float = 0.0  # Bonus for cumulative returns (encourages growth)

    @property
    def cash_daily_rate(self) -> float:
        # 252 trading days; adjust we want to use calendar days
        return float(self.cash_rate_annual) / 252.0

    def for_evaluation(self) -> EnvConfig:
        """Same env, but one deterministic episode over the whole data (no random windows)."""
        return replace(self, random_start=False, max_steps=None)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> EnvConfig:
        return cls(**json.loads(Path(path).read_text()))


def build_env_config(cfg) -> EnvConfig:
    """Translate an AppCfg (see config_loader.py) into the EnvConfig used by the environment."""
    spread = cfg.cost_model.spread
    candidates = {
        "transaction_cost": cfg.env.transaction_cost_bps / 10_000.0,  # bps -> proportion
        "include_cash": cfg.env.include_cash,
        "cash_rate_annual": cfg.env.cash_rate_annual,
        "normalize_obs_weights": cfg.env.normalize_obs_weights,
        "reward_mode": cfg.env.reward_mode,
        "random_start": cfg.env.random_start,
        "max_steps": cfg.env.episode_length,

        # Advanced reward shaping for profitability
        "reward_vol_scaling": cfg.env.reward_vol_scaling,
        "reward_sharpe_bonus": cfg.env.reward_sharpe_bonus,
        "reward_vol_window": cfg.env.reward_vol_window,
        "reward_return_bonus": cfg.env.reward_return_bonus,
        "reward_return_threshold": cfg.env.reward_return_threshold,
        "reward_compound_bonus": cfg.env.reward_compound_bonus,

        "return_weight": cfg.reward.return_weight,
        "turnover_penalty": cfg.reward.turnover_penalty,
        "drawdown_penalty": cfg.reward.drawdown_penalty,
        "volatility_penalty": cfg.reward.volatility_penalty,

        "cost_enabled": cfg.cost_model.enabled,
        "fee_bps": cfg.cost_model.fee_bps,
        "slippage_bps": cfg.cost_model.slippage_bps,
        "spread_type": spread.type,
        "spread_fixed_bps": spread.fixed_bps,
        "spread_k_vol_to_bps": spread.k_vol_to_bps,
        "spread_vol_col_suffix": spread.vol_col_suffix,

        "seed": cfg.seed,
    }
    allowed = {f.name for f in dataclass_fields(EnvConfig)}
    return EnvConfig(**{k: v for k, v in candidates.items() if k in allowed})


def load_run_env_config(run_dir: str | Path) -> EnvConfig:
    """
    Return the EnvConfig a run was trained with, so evaluation uses the same env.
    New runs store env_config.json; older runs are rebuilt from their config.json.
    """
    run_dir = Path(run_dir)
    env_json = run_dir / "env_config.json"
    if env_json.exists():
        return EnvConfig.load(env_json)
    cfg_json = run_dir / "config.json"
    if cfg_json.exists():
        from src.config_loader import load_config
        return build_env_config(load_config(cfg_json))
    raise FileNotFoundError(f"No env_config.json or config.json in {run_dir}")


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, prices: pd.DataFrame, features: pd.DataFrame, config: EnvConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or EnvConfig()
        self.config = self.cfg

        # Validate inputs
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("prices index must be DatetimeIndex.")
        assert isinstance(prices, pd.DataFrame) and isinstance(features, pd.DataFrame)
        
        # Align on dates (cast to DatetimeIndex to satisfy type-checkers)
        idx = pd.DatetimeIndex(prices.index).intersection(pd.DatetimeIndex(features.index))
        if idx.empty:
            raise ValueError("prices and features have disjoint indices.")
        self.prices = prices.loc[idx].copy()
        self.features = features.loc[idx].copy()

        if self.prices.isna().any().any() or self.features.isna().any().any():
            raise ValueError("NaNs detected; ensure data.py and features.py produced clean outputs.")

        # Handle cash asset if needed
        if self.cfg.include_cash:
            cash_col = "__CASH__"
            if cash_col in self.prices.columns:
                raise ValueError("__CASH__ already present in prices.")
            self.prices[cash_col] = 1.0  # synthetic price path; returns set below
            # add a trivial feature for cash so shapes stay consistent
            add_col = ("CASH", "bias") if isinstance(self.features.columns, pd.MultiIndex) else "CASH__bias"
            self.features[add_col] = 0.0

        self.assets = list(self.prices.columns)
        self.n_assets = len(self.assets)
        self.asset_cols = self.assets  # alias for compatibility

        # Precompute simple returns P[t+1]/P[t]-1 as numpy array (T-1 x N)
        P = self.prices.to_numpy(dtype=np.float64)
        self._rets = P[1:] / P[:-1] - 1.0
        if self.cfg.include_cash:
            cash_idx = self.assets.index("__CASH__")
            self._rets[:, cash_idx] = float(self.cfg.cash_daily_rate)

        # Precompute features as numpy array for fast indexing (T x F)
        # This avoids DataFrame.iloc calls in _obs() which is called millions of times
        self._features_array = self.features.to_numpy(dtype=np.float32)
        self._n_features = self._features_array.shape[1]

        # Precompute trailing 20d daily vol of each asset from raw prices for the vol20 spread model.
        # Row t uses returns up to and including day t (known when trading at t's close); the
        # first two rows have no estimate yet and are backfilled. Cash has a constant price,
        # so its vol (and spread) is 0.
        self._vol20 = None
        if self.cfg.cost_enabled and self.cfg.spread_type == "vol20":
            vol = self.prices.pct_change().rolling(SPREAD_VOL_WINDOW, min_periods=2).std()
            self._vol20 = vol.bfill().fillna(0.0).to_numpy(dtype=np.float64)

        # Episode bounds
        T = len(self.prices)
        self._start = 0 if self.cfg.start is None else int(self.cfg.start)
        self._end = (T - 2) if self.cfg.end is None else int(self.cfg.end)
        if not (0 <= self._start <= self._end <= T - 2):
            raise ValueError(f"Invalid (start,end)=({self._start},{self._end}) for T={T}")
        if self.cfg.reward_mode not in ("arith", "log"):
            raise ValueError(f"Unknown reward_mode={self.cfg.reward_mode!r}")
        if self.cfg.random_start and not self.cfg.max_steps:
            raise ValueError("random_start requires max_steps (the episode length)")

        # Convention: start at t=1 so the first observation has a prior row of history
        self._t0 = max(self._start + 1, 1)
        self.t = self._t0  # Use self.t consistently (not self._t)
        self._episode_end = self._end  # last t of the current episode (set in reset)

        # Start equal-weight
        self._w = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self._value = 1.0

        # Track portfolio returns for volatility/Sharpe-based reward shaping
        self._portfolio_returns = []  # List of portfolio returns for rolling calculations

        # Done flag MUST exist before first step()
        self.done = False

        # Observation and action spaces
        obs_dim = self._n_features + self.n_assets
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        n_steps = self.cfg.max_steps
        if self.cfg.random_start:
            assert n_steps is not None  # checked in __init__
            latest_start = max(self._t0, self._end - n_steps + 1)
            self.t = int(self.np_random.integers(self._t0, latest_start + 1))
        else:
            self.t = self._t0
        self._episode_end = self._end if n_steps is None else min(self._end, self.t + n_steps - 1)
        self._w = np.ones(self.n_assets, dtype=float) / self.n_assets
        self._value = 1.0
        self._portfolio_returns = []  # Reset return history
        self.done = False
        obs = self._obs()  # observation at time t
        info = {}
        return obs, info
    
    def step(self, action):
        """
        One-step transition with leak-free accounting:
        - Step t earns r_t = P[t+1]/P[t] - 1 on the *previous* weights w_prev, which were
          chosen before observing features[t].
        - Holdings then drift with prices; trading from the drifted weights to the new
          target w_new costs |w_new - w_drift| (so holding a constant mix is not free).
        - New weights become the holdings for the *next* step.
        Reaching the end of the episode is a time limit, so it is reported as truncated.
        """
        if self.done:
            return self._obs(), 0.0, False, True, {}

        # Previous weights and this period's asset returns (arith)
        # Use views instead of copies for performance (only copy when needed for info dict)
        w_prev = self._w
        r_t = self._rets[self.t, :]

        # Gross portfolio return for this step, then let holdings drift with prices
        ret_gross = float(np.dot(w_prev, r_t))
        w_drift = w_prev * (1.0 + r_t) / (1.0 + ret_gross)

        # Project action to simplex of length n_assets
        a = np.asarray(action, dtype=float).ravel()
        w_new = _project_to_simplex(a)

        # Turnover & cost (only when trading); per-asset bps from the cost model,
        # or the flat transaction_cost when the cost model is disabled
        dw = np.abs(w_new - w_drift)
        turnover = float(dw.sum())
        cost = float(np.dot(self._per_asset_cost_bps(self.t), dw)) / 10_000.0 if turnover > 1e-12 else 0.0
        net = ret_gross - cost  # arithmetic net return: what the portfolio actually earned
        base = float(np.log1p(net)) if self.cfg.reward_mode == "log" else net
        rw   = float(getattr(self.config, "return_weight", 1.0))
        pen_turn = float(getattr(self.config, "turnover_penalty", 0.0)) * turnover

        # Track portfolio return for volatility/Sharpe calculations
        self._portfolio_returns.append(net)

        # Advanced reward shaping for profitability
        reward = rw * base - pen_turn
        
        # Return bonus: encourage positive returns (profitability focus)
        return_bonus_coef = float(getattr(self.config, "reward_return_bonus", 0.0))
        return_threshold = float(getattr(self.config, "reward_return_threshold", 0.0))
        if return_bonus_coef > 0.0 and base > return_threshold:
            # Bonus for positive returns - encourages profitability
            # Multiplier: 1.0 + bonus_coef * return (capped at 0.1 to avoid explosion)
            # Example: return_bonus_coef=2.0, base=0.01 → multiplier = 1.0 + 2.0*0.01 = 1.02
            bonus_multiplier = 1.0 + return_bonus_coef * min(base, 0.1)
            reward = reward * bonus_multiplier
        
        # Compound bonus: reward cumulative growth (encourages long-term profitability)
        compound_bonus_coef = float(getattr(self.config, "reward_compound_bonus", 0.0))
        if compound_bonus_coef > 0.0 and self._value > 1.0:
            # Bonus based on portfolio value growth
            growth_factor = max(0.0, (self._value - 1.0) / 10.0)  # Normalize growth
            reward = reward + compound_bonus_coef * growth_factor
        
        # Volatility scaling: scale reward by Sharpe-like metric (preserves returns, reduces vol)
        # This encourages the agent to reduce volatility while maintaining returns → increases Sharpe and final value
        if getattr(self.config, "reward_vol_scaling", False):
            vol_window = int(getattr(self.config, "reward_vol_window", 20))
            # Only apply scaling if we have enough history (avoid issues at episode start)
            if len(self._portfolio_returns) >= vol_window:
                recent_rets = np.array(self._portfolio_returns[-vol_window:], dtype=np.float64)
                rolling_mean = float(np.mean(recent_rets))
                rolling_vol = float(np.std(recent_rets))
                if rolling_vol > 1e-6:
                    # Scale by Sharpe-like metric: mean / (vol + epsilon)
                    # Make it less aggressive to preserve returns
                    # Use a softer scaling that doesn't penalize returns as much
                    sharpe_ratio = rolling_mean / (rolling_vol + 0.01)
                    # Only apply if Sharpe is very low (very bad risk-adjusted returns)
                    # Otherwise, preserve the reward to encourage returns
                    if sharpe_ratio < 0.5:  # Only scale down if Sharpe is very poor
                        sharpe_scale = np.clip(sharpe_ratio / 0.5, 0.5, 1.0)  # Scale between 0.5 and 1.0
                    else:
                        sharpe_scale = 1.0 + 0.1 * min(sharpe_ratio - 0.5, 2.0)  # Slight bonus for good Sharpe
                    sharpe_scale = np.clip(sharpe_scale, 0.5, 2.0)  # Wider range, less aggressive
                    reward = reward * sharpe_scale
            # If we don't have enough history yet, use unscaled reward (normal behavior)
        
        # Sharpe bonus: add bonus for high rolling Sharpe (but smaller to not override returns)
        sharpe_bonus_coef = float(getattr(self.config, "reward_sharpe_bonus", 0.0))
        if sharpe_bonus_coef > 0.0:
            vol_window = int(getattr(self.config, "reward_vol_window", 20))
            if len(self._portfolio_returns) >= vol_window:
                recent_rets = np.array(self._portfolio_returns[-vol_window:], dtype=np.float64)
                rolling_mean = float(np.mean(recent_rets))
                rolling_vol = float(np.std(recent_rets))
                if rolling_vol > 1e-6:
                    rolling_sharpe = rolling_mean / rolling_vol * np.sqrt(252.0)  # Annualized
                    # Smaller bonus to not override return focus
                    sharpe_bonus = sharpe_bonus_coef * 0.1 * np.clip(rolling_sharpe, -5.0, 5.0)
                    reward = reward + sharpe_bonus

        # Advance state
        self._w = w_new
        self.t += 1
        self.done = self.t > self._episode_end  # last step uses _rets[_episode_end]

        obs = self._obs()
        # Update portfolio value for tracking
        self._value = self._value * (1.0 + net)
        
        # Only copy arrays that are returned in info (to avoid reference issues)
        info = {
            "weights": w_new.copy(),  # New weights, copy for safety
            "prev_weights": w_prev.copy(),  # Previous weights, copy for safety
            "w_prev": w_prev.copy(),   # alias expected by tests
            "w_drift": w_drift.copy(),  # holdings just before trading
            "r_t": r_t.copy(),  # Returns, copy for safety
            "ret_gross": float(ret_gross),
            "cost": float(cost),
            "ret": float(net),         # true net return (after costs, before reward shaping)
            "ret_net": float(net),     # alias of "ret"
            "reward": float(reward),   # shaped reward the agent was trained on
            "turnover": float(turnover),
            "value": float(self._value),  # Current portfolio value
        }
        return obs, reward, False, self.done, info

    def drifted_weights(self) -> np.ndarray:
        """
        Holdings after this step's price move, i.e. the target that means "don't trade".
        For passive baselines (buy & hold, hold between rebalances); it is not an observation.
        """
        r_t = self._rets[self.t]
        return self._w * (1.0 + r_t) / (1.0 + float(np.dot(self._w, r_t)))

    def _obs(self) -> np.ndarray:
        # Use precomputed numpy array instead of DataFrame.iloc for ~10x speedup
        feats = self._features_array[self.t, :]
        
        # Optionally normalize weights component (relative to equal-weight)
        # This can help with RL training stability by keeping observation scale consistent
        if getattr(self.cfg, "normalize_obs_weights", False):
            # Normalize weights relative to equal-weight: (w - 1/N) / (1/N) = N*w - 1
            # This centers weights around 0 and scales by N
            w_norm = self._w * self.n_assets - 1.0
        else:
            w_norm = self._w
        
        return np.concatenate([feats, w_norm]).astype(np.float32)

    def render(self, mode: str = "human") -> None:
        ts = self.prices.index[self.t]  # Use self.t consistently
        print(f"[{ts.date()}] t={self.t} value={self._value:.6f} w={np.round(self._w, 3)}")

    def _per_asset_cost_bps(self, t: int) -> np.ndarray:
        """
        Return a vector of per-asset 'bps per 1 unit turnover' for step t:
        fee_bps + slippage_bps + (spread in bps)/2
        (crossing the bid-ask once costs ~half the spread)
        If cost model is disabled, fallback to legacy flat transaction_cost.
        """
        cfg = self.config
        n = self.n_assets

        if not getattr(cfg, "cost_enabled", False):
            # legacy flat: convert fraction to bps for comparability
            flat_bps = float(cfg.transaction_cost) * 10_000.0
            return np.full(n, flat_bps, dtype=np.float64)

        fee = float(getattr(cfg, "fee_bps", 0.0))
        slp = float(getattr(cfg, "slippage_bps", 0.0))

        # SPREAD: either fixed or proportional to 20d daily vol
        spread_bps = np.full(n, float(getattr(cfg, "spread_fixed_bps", 0.0)), dtype=np.float64)
        if cfg.spread_type == "vol20":
            assert self._vol20 is not None  # computed in __init__ whenever the vol20 model is enabled
            spread_bps = float(cfg.spread_k_vol_to_bps) * self._vol20[t]  # daily vol -> bps
        elif cfg.spread_type != "fixed":
            raise ValueError(f"Unknown spread_type={cfg.spread_type!r}")

        # crossing half-spread once + fee + slippage
        per_asset_bps = fee + slp + 0.5 * spread_bps
        return per_asset_bps


def make_env(prices=None, features=None, env_cfg: EnvConfig | None = None, seed: int = 0):
    """
    Factory function that returns an environment initializer callable.
    
    This function creates a factory for creating PortfolioEnv instances wrapped
    in a Monitor. It's designed to work with Stable Baselines3's DummyVecEnv,
    which expects a callable that returns an environment instance.
    
    Args:
        prices: DataFrame of asset prices (indexed by date)
        features: DataFrame of features (indexed by date, must align with prices)
        env_cfg: EnvConfig instance for environment configuration
        seed: Random seed for reproducibility
    
    Returns:
        Callable that, when called, returns a Monitor-wrapped PortfolioEnv
    
    Example:
        env_factory = make_env(prices_df, features_df, env_config, seed=42)
        env = env_factory()  # Returns Monitor(PortfolioEnv(...))
        
        # Or use with DummyVecEnv:
        vec_env = DummyVecEnv([make_env(prices, features, cfg, seed=42)])
    """
    def _init():
        if prices is None or features is None or env_cfg is None:
            raise ValueError(
                "make_env requires (prices, features, env_cfg, seed); "
                "called with missing or None arguments."
            )
        env = PortfolioEnv(prices=prices, features=features, config=env_cfg)
        # Episode randomness is seeded through reset(seed=...) by the VecEnv; seed the spaces too
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        return Monitor(env)
    
    return _init


# self-test 
if __name__ == "__main__":
    prices_path, feats_path = PRICES_PATH, FEATURES_PATH
    if not (prices_path.exists() and feats_path.exists()):
        raise SystemExit(
            f"Missing processed files.\nExpected:\n  {prices_path}\n  {feats_path}\nRun: python -m src.data && python -m src.features"
        )
    prices = pd.read_parquet(prices_path)
    feats = pd.read_parquet(feats_path)
    env = PortfolioEnv(prices, feats, EnvConfig(transaction_cost=0.0005, seed=123))
    obs, _ = env.reset()
    for _ in range(10):
        a = np.random.randn(env.n_assets)
        obs, r, done, trunc, info = env.step(a)
        env.render()
        if done or trunc: break
    print("Final value:", env._value)
