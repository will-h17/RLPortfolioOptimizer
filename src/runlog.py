from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from datetime import datetime
import subprocess
import shutil
import json

def _git_commit_hash() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None

class RunRecorder:
    def __init__(self, run_name: str, out_root: str, cfg_obj) -> None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_name = "".join(c for c in run_name if c.isalnum() or c in ("-", "_"))
        self.run_dir = Path(out_root) / f"{ts}__{safe_name}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # subdirs for models/checkpoints/eval
        (self.run_dir / "models").mkdir(exist_ok=True)
        (self.run_dir / "models" / "best").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "eval").mkdir(exist_ok=True)

        # save config (json copy), and commit hash
        cfg_json_path = self.run_dir / "config.json"
        with cfg_json_path.open("w") as f:
            json.dump(asdict(cfg_obj), f, indent=2)

        commit = _git_commit_hash()
        (self.run_dir / "commit.txt").write_text(commit or "NO_GIT")

    def path(self, *parts: str) -> Path:
        return self.run_dir.joinpath(*parts)

    def save_metrics(self, metrics: dict, name: str = "metrics.json") -> None:
        p = self.run_dir / name
        with p.open("w") as f:
            json.dump(metrics, f, indent=2)

    def copy_in(self, src: str | Path, dest_rel: str) -> None:
        src = Path(src)
        dest = self.run_dir / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
