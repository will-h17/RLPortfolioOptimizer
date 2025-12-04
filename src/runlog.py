"""
runlog.py - Training run recording and artifact management.

This module provides the RunRecorder class, which manages the creation and organization
of training run directories. Each training run gets a timestamped directory that stores:
- Configuration files (frozen snapshot of training parameters)
- Model checkpoints and best models
- Evaluation metrics and results
- Git commit hash for reproducibility
- Other artifacts generated during training

The RunRecorder ensures that every training run is fully reproducible by saving all
configuration and version information needed to recreate the exact conditions.
"""

from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from datetime import datetime
import subprocess
import shutil
import json


def _git_commit_hash() -> str | None:
    """
    Get the current git commit hash for reproducibility tracking.
    
    Returns:
        Git commit hash as string, or None if not in a git repository or git is unavailable.
    """
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


class RunRecorder:
    def __init__(self, run_name: str, out_root: str, cfg_obj) -> None:
        """
        Initialize a new training run recorder.
        
        Args:
            run_name: Name identifier for this training run (e.g., "ppo_dirichlet_baseline")
            out_root: Root directory where run directories are created (typically "artifacts")
            cfg_obj: Configuration object (dataclass) to save as JSON
        """
        # Create timestamped directory name: YYYYMMDD-HHMMSS__run_name
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        # Ensure run name only contains alphanumeric, dash, and underscore
        safe_name = "".join(c for c in run_name if c.isalnum() or c in ("-", "_"))
        self.run_dir = Path(out_root) / f"{ts}__{safe_name}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for organizing artifacts
        (self.run_dir / "models").mkdir(exist_ok=True)              # Model checkpoints
        (self.run_dir / "models" / "best").mkdir(parents=True, exist_ok=True)  # Best model storage
        (self.run_dir / "eval").mkdir(exist_ok=True)                 # Evaluation results

        # Save frozen configuration snapshot for reproducibility
        # This ensures we can always see exactly what parameters were used
        cfg_json_path = self.run_dir / "config.json"
        with cfg_json_path.open("w") as f:
            json.dump(asdict(cfg_obj), f, indent=2)

        # Record git commit hash for code version tracking
        # Helps identify which version of the codebase produced this run
        commit = _git_commit_hash()
        (self.run_dir / "commit.txt").write_text(commit or "NO_GIT")

    def path(self, *parts: str) -> Path:
        """
        Get a path relative to the run directory.
        
        Convenience method for constructing paths within the run directory.
        
        Args:
            *parts: Path components to join (e.g., "models", "best", "model.zip")
            
        Returns:
            Path object pointing to the specified location within the run directory
            
        Example:
            rec.path("scaler.json")  # -> run_dir/scaler.json
            rec.path("models", "best", "model.zip")  # -> run_dir/models/best/model.zip
        """
        return self.run_dir.joinpath(*parts)

    def save_metrics(self, metrics: dict, name: str = "metrics.json") -> None:
        """
        Save evaluation metrics to a JSON file in the run directory.
        
        Args:
            metrics: Dictionary of metric names to values (e.g., {"sharpe": 1.5, "max_drawdown": -0.2})
            name: Filename to save metrics as (default: "metrics.json")
        """
        p = self.run_dir / name
        with p.open("w") as f:
            json.dump(metrics, f, indent=2)

    def copy_in(self, src: str | Path, dest_rel: str) -> None:
        """
        Copy a file into the run directory.
        
        Useful for copying model files, logs, or other artifacts into the run directory
        for permanent storage. Creates parent directories if needed.
        
        Args:
            src: Source file path (absolute or relative)
            dest_rel: Destination path relative to run directory (e.g., "models/checkpoint.zip")
        """
        src = Path(src)
        dest = self.run_dir / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
