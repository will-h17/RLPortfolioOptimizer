"""
env.py — RL environment for portfolio optimization.

State  : concat(features_t, current_portfolio_weights)
Action : target weights (projected to the probability simplex: long-only, sum=1)
Reward : log(1 + portfolio_return_{t->t+1}) - transaction_cost * L1(turnover)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- Paths ----------
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
    transaction_cost: float = 0.0  # legacy fallback (fraction, e.g., 0.0005 for 5 bps)
    include_cash: bool = True
    cash_rate_annual: float = 0.0
    seed: int = 0

    start: Optional[int] = None   # starting row index within provided data
    end: Optional[int] = None     # ending row index (inclusive or exclusive per your env)
    max_steps: Optional[int] = None  # cap episode length if you want

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

    @property
    def cash_daily_rate(self) -> float:
        # 252 trading days; adjust if you use calendar days
        return float(self.cash_rate_annual) / 252.0


class PortfolioEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, prices: pd.DataFrame, features: pd.DataFrame, config: Optional[EnvConfig] = None) -> None:
        super().__init__()
        self.cfg = config or EnvConfig()
        # keep `config` and `cfg` consistent so other methods can use either attribute
        self.config = self.cfg
        self.rng = np.random.default_rng(self.cfg.seed)
        
        # Expect DataFrames with same index and asset columns
        assert isinstance(prices, pd.DataFrame) and isinstance(features, pd.DataFrame)
        common = prices.index.intersection(features.index)
        if len(common) == 0:
            raise ValueError("prices and features have disjoint indices.")
        self.prices = prices.loc[common]
        self.features = features.loc[common]
        self.asset_cols = list(self.prices.columns)
        self.n_assets = len(self.asset_cols)

        # Precompute arithmetic daily returns (T x N)
        self._rets = self.prices.pct_change().fillna(0.0).to_numpy(dtype=float)
        self.t_index = self.prices.index.to_numpy()

        # Episode bounds
        self._start = 0 if getattr(self.cfg, "start", None) is None else int(self.cfg.start)
        self._end = len(self.t_index) - 1 if getattr(self.cfg, "end", None) is None else int(self.cfg.end)
        self._start = max(0, min(self._start, self._end - 1))
        self._end = max(self._start + 1, min(self._end, len(self.t_index) - 1))

        # Convention: first reward uses return from (t-1 -> t), so start at t=1
        self._t0 = max(self._start + 1, 1)
        self.t = self._t0

        # Start equal-weight (or your preferred initial weights)
        self._w = np.ones(self.n_assets, dtype=float) / self.n_assets

        # Done flag MUST exist before first step()
        self.done = False

        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("prices index must be DatetimeIndex.")

        # Align on dates
        idx = prices.index.intersection(features.index)
        if idx.empty:
            raise ValueError("prices and features have disjoint indices.")
        self.prices = prices.loc[idx].copy()
        self.features = features.loc[idx].copy()

        if self.prices.isna().any().any() or self.features.isna().any().any():
            raise ValueError("NaNs detected; ensure data.py and features.py produced clean outputs.")

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

        # Precompute simple returns P[t+1]/P[t]-1
        P = self.prices.to_numpy(dtype=np.float64)
        self._rets = P[1:] / P[:-1] - 1.0
        if self.cfg.include_cash:
            cash_idx = self.assets.index("__CASH__")
            self._rets[:, cash_idx] = float(self.cfg.cash_daily_rate)

        T = len(self.prices)
        self._start = 0 if self.cfg.start is None else int(self.cfg.start)
        self._end = (T - 2) if self.cfg.end is None else int(self.cfg.end)
        if not (0 <= self._start <= self._end <= T - 2):
            raise ValueError(f"Invalid (start,end)=({self._start},{self._end}) for T={T}")

        obs_dim = self.features.shape[1] + self.n_assets
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)

        self._t = 0
        self._w = np.ones(self.n_assets) / self.n_assets
        self._value = 1.0

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

        # Previous weights and this period’s asset returns (arith)
        w_prev = self._w.astype(float).copy()
        r_t = self._rets[self.t, :].astype(float).copy()

        # Gross portfolio return for this step
        mode = getattr(self.config, "reward_mode", "arith")
        if mode == "arith":
            ret_gross = float(np.dot(w_prev, r_t))
        elif mode == "log":
            ret_gross = float(np.log1p(np.dot(w_prev, r_t)))
        else:
            raise ValueError(f"Unknown reward_mode={mode!r}")

        # Project action to simplex of length n_assets (implement your own if different)
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
        reward = rw * base - pen_turn

        # Advance state
        self._w = w_new
        self.t += 1
        self.done = self.t >= self._end

        obs = self._obs()
        info = {
            "weights": w_new.copy(),
            "prev_weights": w_prev.copy(),
            "w_prev": w_prev.copy(),   # alias expected by tests
            "r_t": r_t.copy(),
            "ret_gross": float(ret_gross),
            "cost": float(cost),
            "ret_net": float(reward),
            "turnover": float(turnover),
        }
        return obs, reward, self.done, False, info

    def _obs(self) -> np.ndarray:
        feats = self.features.iloc[self._t].to_numpy(dtype=np.float64)
        return np.concatenate([feats, self._w]).astype(np.float32)

    def render(self, mode: str = "human") -> None:
        ts = self.prices.index[self._t]
        print(f"[{ts.date()}] t={self._t} value={self._value:.6f} w={np.round(self._w, 3)}")

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
            # Features must contain per-asset rolling std of daily returns at time t
            suffix = str(getattr(cfg, "spread_vol_col_suffix", "_s20"))
            vol_cols = [f"{c}{suffix}" for c in self.assets]  # e.g., ["AAPL_s20", "MSFT_s20", ...]
            # If any col missing, treat as 0 spread contribution
            # self.features is a DataFrame indexed by time with these columns
            if all(col in self.features.columns for col in vol_cols):
                vol = self.features.iloc[t][vol_cols].to_numpy(dtype=np.float64)  # daily vol (e.g., 0.012)
                k = float(getattr(cfg, "spread_k_vol_to_bps", 8000.0))
                spread_bps = k * vol  # convert vol to bps via a scalar
            # else: keep defaults (zeros or fixed)

        # crossing half-spread once + fee + slippage
        per_asset_bps = fee + slp + 0.5 * spread_bps
        return per_asset_bps


# --------------------------- self-test --------------------------- #
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
