"""
env.py — RL environment for portfolio optimization.

State  : concat(features_t, current_portfolio_weights)
Action : target weights (projected to the probability simplex: long-only, sum=1)
Reward : log(1 + portfolio_return_{t->t+1}) - transaction_cost * L1(turnover)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, cast
from pathlib import Path
import numpy as np
import pandas as pd

# Paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
PROC_DIR = PROJECT_ROOT / "data" / "processed"

# Gymnasium imports
try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM = True
except Exception:
    _GYM = False

    class _Box:
        def __init__(self, low, high, shape, dtype): self.low, self.high, self.shape, self.dtype = low, high, shape, dtype
    class _Env: pass
    spaces = type("spaces", (), {"Box": _Box})
    gym = type("gym", (), {"Env": _Env})


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

    start: Optional[int] = None   # starting row index within provided data
    end: Optional[int] = None     # ending row index (inclusive or exclusive per your env)
    max_steps: Optional[int] = None  # cap episode length we you want

    # Reward definition: "arith" (default) or "log"
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
    spread_vol_col_suffix: str = "_s20"

    # Observation normalization (optional, for RL training stability)
    normalize_obs_weights: bool = False  # If True, normalize weights relative to equal-weight

    # Advanced reward shaping for higher Sharpe
    reward_vol_scaling: bool = False  # Scale reward by Sharpe-like metric (preserves returns, reduces vol → higher Sharpe AND final value)
    reward_sharpe_bonus: float = 0.0  # Bonus for high rolling Sharpe (0.0 = disabled)
    reward_vol_window: int = 20  # Window for rolling volatility/Sharpe calculation

    @property
    def cash_daily_rate(self) -> float:
        # 252 trading days; adjust we want to use calendar days
        return float(self.cash_rate_annual) / 252.0


class PortfolioEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, prices: pd.DataFrame, features: pd.DataFrame, config: Optional[EnvConfig] = None) -> None:
        super().__init__()
        self.cfg = config or EnvConfig()
        self.config = self.cfg
        self.rng = np.random.default_rng(self.cfg.seed)
        
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

        # Precompute vol column indices if using vol20 spread (optimization: avoid repeated lookups)
        self._vol_col_indices = None
        if self.cfg.cost_enabled and getattr(self.cfg, "spread_type", "fixed") == "vol20":
            suffix = str(getattr(self.cfg, "spread_vol_col_suffix", "_s20"))
            vol_cols = [f"{c}{suffix}" for c in self.assets]
            if all(col in self.features.columns for col in vol_cols):
                try:
                    self._vol_col_indices = [self.features.columns.get_loc(c) for c in vol_cols]
                except (KeyError, IndexError):
                    # MultiIndex or other edge case - will use fallback in _per_asset_cost_bps
                    self._vol_col_indices = None

        # Episode bounds
        T = len(self.prices)
        start_val = getattr(self.cfg, "start", None)
        end_val = getattr(self.cfg, "end", None)
        self._start = 0 if start_val is None else int(cast(int, start_val))
        self._end = (T - 2) if end_val is None else int(cast(int, end_val))
        if not (0 <= self._start <= self._end <= T - 2):
            raise ValueError(f"Invalid (start,end)=({self._start},{self._end}) for T={T}")

        # Convention: first reward uses return from (t-1 -> t), so start at t=1
        self._t0 = max(self._start + 1, 1)
        self.t = self._t0  # Use self.t consistently (not self._t)

        # Start equal-weight (or your preferred initial weights)
        self._w = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self._value = 1.0

        # Track portfolio returns for volatility/Sharpe-based reward shaping
        self._portfolio_returns = []  # List of portfolio returns for rolling calculations

        # Done flag MUST exist before first step()
        self.done = False

        # Observation and action spaces
        obs_dim = self._n_features + self.n_assets
        self.observation_space = cast(Any, spaces).Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = cast(Any, spaces).Box(low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            try:
                import random
                np.random.seed(seed)
                random.seed(seed)
            except Exception:
                pass
        self.t = self._t0
        self._w = np.ones(self.n_assets, dtype=float) / self.n_assets
        self._value = 1.0
        self._portfolio_returns = []  # Reset return history
        self.done = False
        obs = self._obs()   # whatever you use to build an observation at time t
        info = {}
        return obs, info
    
    def step(self, action):
        """
        One-step transition with leak-free accounting:
        - Reward for period t uses returns r_t from (t-1 -> t) and the *previous* weights w_prev.
        - Trading to new weights w_new happens *now*; the cost is charged this step.
        - New weights become the holdings for the *next* period (t+1).
        """
        # ---- indices / guards ----
        # self.t should point to the current period end (so r_t = returns at self.t)
        # After reset(), set self.t = 1 (so first reward uses R[1] = return from day0->day1).
        if self.done:
            return self._obs(), 0.0, True, False, {}

        # Previous weights and this period's asset returns (arith)
        # Use views instead of copies for performance (only copy when needed for info dict)
        w_prev = self._w
        r_t = self._rets[self.t, :]

        # Gross portfolio return for this step
        mode = getattr(self.config, "reward_mode", "arith")
        if mode == "arith":
            ret_gross = float(np.dot(w_prev, r_t))
        elif mode == "log":
            ret_gross = float(np.log1p(np.dot(w_prev, r_t)))
        else:
            raise ValueError(f"Unknown reward_mode={mode!r}")

        # Project action to simplex of length n_assets
        a = np.asarray(action, dtype=float).ravel()
        w_target = _project_to_simplex(a)
        w_new = w_target

        # Turnover & cost (only when weights change)
        dw = np.abs(w_new - w_prev)
        turnover = float(dw.sum())
        tc = float(getattr(self.config, "transaction_cost", 0.0))
        cost = tc * turnover if turnover > 1e-12 else 0.0
        base = ret_gross - cost
        rw   = float(getattr(self.config, "return_weight", 1.0))
        pen_turn = float(getattr(self.config, "turnover_penalty", 0.0)) * turnover
        
        # Track portfolio return for volatility/Sharpe calculations
        self._portfolio_returns.append(base)
        
        # Advanced reward shaping for higher Sharpe
        reward = rw * base - pen_turn
        
        # Volatility scaling: scale reward by Sharpe-like metric (preserves returns, reduces vol)
        # This encourages the agent to reduce volatility while maintaining returns → increases Sharpe AND final value
        if getattr(self.config, "reward_vol_scaling", False):
            vol_window = int(getattr(self.config, "reward_vol_window", 20))
            if len(self._portfolio_returns) >= vol_window:
                recent_rets = np.array(self._portfolio_returns[-vol_window:], dtype=np.float64)
                rolling_mean = float(np.mean(recent_rets))
                rolling_vol = float(np.std(recent_rets))
                if rolling_vol > 1e-6:
                    # Scale by Sharpe-like metric: mean / (vol + epsilon)
                    # This preserves returns while penalizing high volatility
                    # Higher Sharpe → higher scaled reward (encourages both high returns AND low vol)
                    sharpe_scale = rolling_mean / (rolling_vol + 0.01)
                    # Normalize to reasonable scale (avoid extreme values)
                    # Clip to [0.1, 10.0] to prevent reward explosion or collapse
                    sharpe_scale = np.clip(sharpe_scale, 0.1, 10.0)
                    reward = reward * sharpe_scale
        
        # Sharpe bonus: add bonus for high rolling Sharpe
        sharpe_bonus_coef = float(getattr(self.config, "reward_sharpe_bonus", 0.0))
        if sharpe_bonus_coef > 0.0:
            vol_window = int(getattr(self.config, "reward_vol_window", 20))
            if len(self._portfolio_returns) >= vol_window:
                recent_rets = np.array(self._portfolio_returns[-vol_window:], dtype=np.float64)
                rolling_mean = float(np.mean(recent_rets))
                rolling_vol = float(np.std(recent_rets))
                if rolling_vol > 1e-6:
                    rolling_sharpe = rolling_mean / rolling_vol * np.sqrt(252.0)  # Annualized
                    # Bonus proportional to Sharpe (clipped to avoid extreme values)
                    sharpe_bonus = sharpe_bonus_coef * np.clip(rolling_sharpe, -5.0, 5.0)
                    reward = reward + sharpe_bonus

        # Advance state
        self._w = w_new
        self.t += 1
        self.done = self.t >= self._end

        obs = self._obs()
        # Update portfolio value for tracking
        self._value = self._value * (1.0 + base)
        
        # Only copy arrays that are returned in info (to avoid reference issues)
        info = {
            "weights": w_new.copy(),  # New weights, copy for safety
            "prev_weights": w_prev.copy(),  # Previous weights, copy for safety
            "w_prev": w_prev.copy(),   # alias expected by tests
            "r_t": r_t.copy(),  # Returns, copy for safety
            "ret_gross": float(ret_gross),
            "cost": float(cost),
            "ret_net": float(reward),
            "turnover": float(turnover),
            "value": float(self._value),  # Current portfolio value
        }
        return obs, reward, self.done, False, info

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
        if getattr(cfg, "spread_type", "fixed") == "vol20":
            # Use precomputed vol column indices if available (optimization)
            if self._vol_col_indices is not None:
                vol = self._features_array[t, self._vol_col_indices].astype(np.float64)
            else:
                # Fallback: compute indices on-the-fly or use DataFrame
                suffix = str(getattr(cfg, "spread_vol_col_suffix", "_s20"))
                vol_cols = [f"{c}{suffix}" for c in self.assets]
                if all(col in self.features.columns for col in vol_cols):
                    try:
                        col_indices = [self.features.columns.get_loc(c) for c in vol_cols]
                        vol = self._features_array[t, col_indices].astype(np.float64)
                    except (KeyError, IndexError):
                        # Fallback to DataFrame if column lookup fails (e.g., MultiIndex)
                        vol = self.features.iloc[t][vol_cols].to_numpy(dtype=np.float64)
                else:
                    vol = np.zeros(n, dtype=np.float64)
            k = float(getattr(cfg, "spread_k_vol_to_bps", 8000.0))
            spread_bps = k * vol  # convert vol to bps via a scalar

        # crossing half-spread once + fee + slippage
        per_asset_bps = fee + slp + 0.5 * spread_bps
        return per_asset_bps


# self-test 
if __name__ == "__main__":
    prices_path = PROC_DIR / "prices_adj.parquet"
    feats_path = PROC_DIR / "features.parquet"
    if not (prices_path.exists() and feats_path.exists()):
        raise SystemExit(
            f"Missing processed files.\nExpected:\n  {prices_path}\n  {feats_path}\nRun src/data.py then src/features.py."
        )
    prices = pd.read_parquet(prices_path)
    feats = pd.read_parquet(feats_path)
    env = PortfolioEnv(prices, feats, EnvConfig(transaction_cost=0.0005, seed=123))
    obs, _ = env.reset()
    for _ in range(10):
        a = np.random.randn(env.n_assets)
        obs, r, done, trunc, info = env.step(a)
        env.render()
        if done: break
    print("Final value:", env._value)
