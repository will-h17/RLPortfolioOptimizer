"""
jobs.py — Long-running tasks started from the web app.

Each job takes a `report(percent, log)` function (percent is None when progress is unknown) and
returns a result for the page to render, or raises JobError with the log tail. Jobs run inside
Dash background callbacks: each in its own process, so pressing Stop kills the job and any
process it started (the CLI tools below run as subprocesses).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from src.paths import ARTIFACTS_DIR, CONFIGS_DIR, PRICES_PATH, PROJECT_ROOT, REPORTS_DIR

Report = Callable[[float | None, str], None]
ProgressParser = Callable[[str], float | None]


class JobError(RuntimeError):
    def __init__(self, message: str, log: str = ""):
        super().__init__(message)
        self.log = log


def stream_module(args: list[str], report: Report, progress: ProgressParser | None = None,
                  tail: int = 60) -> str:
    """
    Run `python -m <args>` with this interpreter from the repo root, reporting the last `tail`
    output lines (and a percentage from `progress`, if given) about twice a second.
    Returns the final log tail; raises JobError if the command fails.
    """
    cmd = [sys.executable, "-u", "-m", *args]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    lines: deque[str] = deque(maxlen=tail)
    percent: float | None = None
    last_report = 0.0
    with subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env, text=True, bufsize=1,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as proc:
        assert proc.stdout is not None
        for raw in proc.stdout:
            for line in raw.rstrip("\n").split("\r"):  # progress bars redraw with \r
                if line.strip():
                    lines.append(line)
                    if progress is not None:
                        percent = progress(line) or percent
            now = time.monotonic()
            if now - last_report >= 0.5:
                report(percent, "\n".join(lines))
                last_report = now
        returncode = proc.wait()
    log = "\n".join(lines)
    report(percent, log)
    if returncode != 0:
        raise JobError(f"`python -m {args[0]}` failed (exit code {returncode})", log)
    return log


def _find(pattern: str, log: str, what: str) -> str:
    matches = re.findall(pattern, log)
    if not matches:
        raise JobError(f"Could not find {what} in the job output", log)
    return matches[-1]


def _parse_symbols(text: str) -> list[str]:
    symbols = [s.strip().upper() for s in (text or "").split(",") if s.strip()]
    if not symbols:
        raise JobError("Enter at least one ticker symbol")
    return symbols


# ---------------- data ----------------

def download_data(report: Report, symbols_text: str, force: bool) -> list[Path]:
    from src.data import download_universe
    symbols = _parse_symbols(symbols_text)
    report(None, f"Downloading {', '.join(symbols)} from yfinance...")
    return download_universe(symbols, force=force)


def process_data(report: Report, symbols_text: str) -> pd.DataFrame:
    from src.data import load_and_merge
    symbols = _parse_symbols(symbols_text)
    report(None, f"Merging {', '.join(symbols)} into {PRICES_PATH.name}...")
    return load_and_merge(symbols)


def generate_features(report: Report, lookbacks_text: str) -> pd.DataFrame:
    from src.features import make_features
    try:
        lookbacks = [int(x) for x in (lookbacks_text or "").split(",") if x.strip()]
    except ValueError as e:
        raise JobError(f"Lookbacks must be whole numbers: {e}") from e
    if not lookbacks:
        raise JobError("Enter at least one lookback window")
    if not PRICES_PATH.exists():
        raise JobError("No processed prices yet. Download and process data first.")
    report(None, f"Computing features with lookbacks {lookbacks}...")
    return make_features(pd.read_parquet(PRICES_PATH), lookbacks=lookbacks)


# ---------------- training and evaluation (CLI tools as subprocesses) ----------------

def train(report: Report, config_name: str) -> Path:
    from src.config_loader import load_config
    config_path = CONFIGS_DIR / config_name
    total = load_config(config_path).train.total_timesteps

    def timesteps(line: str) -> float | None:
        m = re.search(r"total_timesteps\s*\|\s*(\d+)", line)
        return min(100.0, 100.0 * int(m.group(1)) / total) if m else None

    log = stream_module(["src.train", "--config", str(config_path)], report, progress=timesteps)
    return Path(_find(r"Run recorded in: (.+)", log, "the run folder").strip())


def backtest(report: Report, run_name: str) -> str:
    return stream_module(["src.backtest", "--run_dir", str(ARTIFACTS_DIR / run_name)], report)


def cost_sweep(report: Report, run_name: str) -> Path:
    run_dir = ARTIFACTS_DIR / run_name
    out_csv = run_dir / "eval" / "cost_sweep.csv"
    # The run's own config.json supplies the data paths and split, so the sweep matches training
    stream_module(["src.sweep_costs", "--config", str(run_dir / "config.json"), "--run_dir", str(run_dir),
                   "--out_csv", str(out_csv)], report)
    return out_csv


def grid_eval(report: Report, grid_config: str) -> Path:
    log = stream_module(["src.eval_grid", "--grid", str(CONFIGS_DIR / grid_config)], report)
    return Path(_find(r"Wrote \d+ rows to (.+)", log, "the results file").strip())


def hpo(report: Report, config_name: str, n_trials: int, study_name: str, resume: bool) -> Path:
    out_dir = ARTIFACTS_DIR / "hpo"
    args = ["src.hpo.run_optuna", "--config", str(CONFIGS_DIR / config_name), "--n-trials", str(n_trials),
            "--study-name", study_name, "--output-dir", str(out_dir)]

    def trials(line: str) -> float | None:
        m = re.search(r"Trial (\d+) finished", line)
        return min(100.0, 100.0 * (int(m.group(1)) + 1) / n_trials) if m else None

    stream_module(args + (["--resume"] if resume else []), report, progress=trials)
    return out_dir / f"{study_name}_summary.json"


def walk_forward(report: Report, config_name: str, seeds: list[int], first_test_year: int,
                 timesteps: int | None, name: str) -> Path:
    args = ["src.walk_forward", "--config", str(CONFIGS_DIR / config_name), "--seeds", *map(str, seeds),
            "--first-test-year", str(first_test_year), "--name", name]
    if timesteps:
        args += ["--timesteps", str(timesteps)]
    state = {"total": 0, "done": 0}

    def runs(line: str) -> float | None:
        if m := re.search(r"\[wf\] (\d+) folds x (\d+) seeds", line):
            state["total"] = int(m.group(1)) * int(m.group(2))
        elif re.search(r"\[wf\] fold \S+ seed \d+: test Sharpe", line):
            state["done"] += 1
        return 100.0 * state["done"] / state["total"] if state["total"] else None

    stream_module(args, report, progress=runs)
    return ARTIFACTS_DIR / "walk_forward" / name / "summary.csv"


def make_reports(report: Report, run_name: str | None) -> list[Path]:
    from src.report import generate_report
    report(None, "Running baselines and the RL agent over the full history...")
    generate_report(ARTIFACTS_DIR / run_name if run_name else None)
    return sorted(REPORTS_DIR.glob("*.png"))
