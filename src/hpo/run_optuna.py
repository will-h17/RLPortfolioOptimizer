from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import optuna
from optuna.pruners import MedianPruner
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from src.backtest import evaluate_sb3_model
from src.config_loader import load_config
from src.env import load_run_env_config
from src.paths import ARTIFACTS_DIR
from src.runlog import resolve_run_model
from src.scaler import FitOnTrainScaler
from src.train import train_once
from src.utils.data_utils import prepare_dataset


# Search space (uses current config)
def suggest_params(trial: optuna.trial.Trial, base_cfg):
    cfg = deepcopy(base_cfg)

    # PPO core knobs (SB3)
    cfg.ppo.learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    cfg.ppo.clip_range    = trial.suggest_float("clip_range",    0.05, 0.30)
    cfg.ppo.ent_coef      = trial.suggest_float("ent_coef",      1e-4, 1e-1, log=True)
    cfg.ppo.gae_lambda    = trial.suggest_float("gae_lambda",    0.80, 0.99)
    cfg.ppo.gamma         = trial.suggest_float("gamma",         0.90, 0.999)

    # Reward shaping: penalty per unit of L1 turnover. Daily returns are ~1e-3, so larger
    # values make the agent refuse to trade. (drawdown/volatility penalties are not
    # implemented in the env, so they are not searched.)
    cfg.reward.turnover_penalty = trial.suggest_float("turnover_penalty", 0.0, 1e-3)

    # Trial budget: shorten timesteps for exploration
    cfg.train.total_timesteps = int(cfg.hpo.train_timesteps)

    # Distinguish run folders by trial number
    base = getattr(cfg, "run_name", "run")
    cfg.run_name = f"{base}__t{trial.number:03d}"

    return cfg


# Validation scoring (Sharpe)
def eval_on_validation(run_dir: Path, cfg) -> float:
    """
    Recreate the validation slice exactly as in train.py, transform with the TRAIN scaler
    saved in run_dir, then evaluate best checkpoint with your existing function.
    """
    P_va, X_va = prepare_dataset(cfg).val

    # Transform VAL with the TRAIN-fitted scaler from this run
    scaler = FitOnTrainScaler.load(Path(run_dir) / "scaler.json", columns=X_va.columns)
    X_vaz = scaler.transform(X_va)

    res = evaluate_sb3_model(P_va, X_vaz, str(resolve_run_model(run_dir)), load_run_env_config(run_dir))
    return float(res["metrics"]["sharpe"])


# JSON storage helpers
def save_trial_to_json(trial: optuna.trial.FrozenTrial, study_name: str, output_dir: Path):
    """Save a single trial result to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_file = output_dir / f"{study_name}_trial_{trial.number:04d}.json"
    
    trial_data = {
        "trial_number": trial.number,
        "params": trial.params,
        "value": trial.value,
        "state": trial.state.name if hasattr(trial.state, 'name') else str(trial.state),
        "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
        "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
    }
    
    trial_file.write_text(json.dumps(trial_data, indent=2))
    return trial_file

def save_study_summary(study: optuna.Study, study_name: str, output_dir: Path):
    """Save study summary to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / f"{study_name}_summary.json"
    
    # Get all trials data
    trials_data = []
    for trial in study.trials:
        trials_data.append({
            "trial_number": trial.number,
            "params": trial.params,
            "value": trial.value,
            "state": trial.state.name if hasattr(trial.state, 'name') else str(trial.state),
        })
    
    summary = {
        "study_name": study_name,
        "n_trials": len(study.trials),
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number if study.best_trial else None,
        "trials": trials_data,
    }
    
    summary_file.write_text(json.dumps(summary, indent=2))
    return summary_file

# Optuna objective
def objective(trial: optuna.trial.Trial, base_cfg):
    cfg_trial = suggest_params(trial, base_cfg)

    # Run a normal training job
    run_dir = train_once(cfg_trial)
    trial.set_user_attr("run_dir", str(run_dir))

    # Score on the validation slice
    return eval_on_validation(run_dir, cfg_trial)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to your YAML/JSON config")
    ap.add_argument("--n-trials", type=int, default=30, help="Total completed trials wanted in the study")
    ap.add_argument("--study-name", default="ppo_hpo")
    ap.add_argument("--output-dir", default=str(ARTIFACTS_DIR / "hpo"), help="Directory to save JSON results")
    ap.add_argument("--resume", action="store_true", help="Continue an existing study with the same name")
    args = ap.parse_args()

    base_cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Persist the study in an append-only journal file so interrupted runs can resume
    journal_path = output_dir / f"{args.study_name}.journal"
    if journal_path.exists() and not args.resume:
        raise SystemExit(f"Study journal already exists at {journal_path}. "
                         f"Pass --resume to continue it, or use a different --study-name.")
    storage = JournalStorage(JournalFileBackend(str(journal_path)))
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=storage,
        load_if_exists=args.resume,
        pruner=MedianPruner(n_warmup_steps=1),
    )

    n_done = len(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)))
    n_remaining = max(0, args.n_trials - n_done)
    if n_done:
        print(f"[resume] {n_done} completed trials found; running {n_remaining} more")

    # Save each finished trial to JSON as soon as it completes
    def _save_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            save_trial_to_json(trial, args.study_name, output_dir)

    study.optimize(
        lambda t: objective(t, base_cfg),
        n_trials=n_remaining,
        callbacks=[_save_trial],
        gc_after_trial=True,
    )

    # Save final summary
    summary_file = save_study_summary(study, args.study_name, output_dir)

    print("\n=== HPO complete ===")
    print("Best Sharpe:", study.best_value)
    print("Best params:", study.best_params)
    print(f"\nResults saved to: {output_dir}")
    print(f"Summary: {summary_file}")
    print(f"Individual trials: {output_dir}/{args.study_name}_trial_*.json")
    print(f"Study journal (for --resume): {journal_path}")


if __name__ == "__main__":
    main()
