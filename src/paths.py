"""
paths.py — Project directory layout, resolved from the repo root (not the current working directory).
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"

PRICES_PATH = PROC_DIR / "prices_adj.parquet"
FEATURES_PATH = PROC_DIR / "features.parquet"


def resolve(path: str | Path) -> Path:
    """Return path unchanged if absolute, otherwise relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
