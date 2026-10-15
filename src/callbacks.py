"""
callbacks.py — Training callbacks.

ValSharpeCallback replaces SB3's EvalCallback for model selection: EvalCallback ranks checkpoints
by episode reward, which includes reward shaping, whereas we care about the risk-adjusted return
of the portfolio itself.
"""

from __future__ import annotations

import csv
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from src.backtest import rollout_policy
from src.env import PortfolioEnv
from src.utils.metrics import sharpe_ratio


class ValSharpeCallback(BaseCallback):
    """
    Every `eval_freq` environment steps (summed over parallel envs), roll the current policy out
    deterministically on the validation env, and save it to `best_model_path` when the annualized
    Sharpe of its net returns is the best so far. Scores are appended to `log_path` (CSV).
    """

    def __init__(self, val_env: PortfolioEnv, eval_freq: int, best_model_path: Path, log_path: Path, verbose: int = 1):
        super().__init__(verbose)
        self.val_env = val_env
        self.eval_freq = int(eval_freq)
        self.best_model_path = Path(best_model_path)
        self.log_path = Path(log_path)
        self.best_sharpe = float("-inf")
        self._next_eval = self.eval_freq
        self._last_eval = -1

    def _on_training_start(self) -> None:
        with self.log_path.open("w", newline="") as f:
            csv.writer(f).writerow(["timesteps", "val_sharpe", "val_final_value", "is_best"])

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_eval:
            self._next_eval += self.eval_freq
            self._evaluate()
        return True

    def _on_training_end(self) -> None:
        if self._last_eval != self.num_timesteps:
            self._evaluate()  # make sure the final policy is scored too

    def _evaluate(self) -> None:
        self._last_eval = self.num_timesteps
        res = rollout_policy(self.model, self.val_env)
        sharpe = sharpe_ratio(res["returns"])
        is_best = sharpe > self.best_sharpe
        if is_best:
            self.best_sharpe = sharpe
            self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.best_model_path))
        with self.log_path.open("a", newline="") as f:
            csv.writer(f).writerow([self.num_timesteps, sharpe, float(res["values"][-1]), is_best])
        self.logger.record("eval/val_sharpe", sharpe)
        if self.verbose:
            flag = "  <- new best" if is_best else ""
            print(f"[val] timesteps={self.num_timesteps:,}  Sharpe={sharpe:.3f}{flag}")
