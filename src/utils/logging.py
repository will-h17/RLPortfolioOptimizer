"""
logging.py — Training logger: SB3's stdout table plus a per-update progress.csv in the run folder
(and TensorBoard event files when tensorboard is installed).
"""

from __future__ import annotations

import importlib.util

from stable_baselines3.common.logger import Logger, configure


def make_logger(log_dir: str) -> Logger:
    outputs = ["stdout", "csv"]
    if importlib.util.find_spec("tensorboard") is not None:
        outputs.append("tensorboard")
    return configure(log_dir, outputs)
