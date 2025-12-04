"""
report.py — Generate performance visualizations

- Runs backtests for Buy & Hold, Equal Weight, RL Agent
- Plots equity curves, metrics comparison, and allocation heatmap
"""

from __future__ import annotations
import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.backtest import buy_and_hold, equal_weight, evaluate_sb3_model
from src.scaler import FitOnTrainScaler
from src.utils.data_utils import align_after_load


def _find_latest_run(run_name_pattern: str = "ppo_dirichlet_baseline") -> Path | None:
    """Find the latest training run directory."""
    artifacts_dir = Path("artifacts")
    if not artifacts_dir.exists():
        return None
    
    # Find all runs matching the pattern
    runs = sorted(artifacts_dir.glob(f"*__{run_name_pattern}"), key=os.path.getmtime, reverse=True)
    return runs[0] if runs else None


def generate_report(run_dir: str | Path | None = None):
    """
    Generate performance visualizations.
    
    Args:
        run_dir: Path to training run directory. If None, auto-finds latest run.
    """
    # Load Data
    prices = pd.read_parquet("data/processed/prices_adj.parquet")
    features = pd.read_parquet("data/processed/features.parquet")
    
    # Align data
    prices, features = align_after_load(prices, features)

    # Baselines
    bh = buy_and_hold(prices, features)
    ew = equal_weight(prices, features)

    # RL Agent
    rl = None
    if run_dir is None:
        run_dir = _find_latest_run()
    
    if run_dir:
        run_path = Path(run_dir)
        scaler_path = run_path / "scaler.json"
        model_path = run_path / "models" / "best" / "best_model.zip"
        
        # Fallback to other model paths if best_model doesn't exist
        if not model_path.exists():
            model_path = run_path / "models" / "ppo_dirichlet_sb3.zip"
        if not model_path.exists():
            # Try to find any model zip
            model_zips = list(run_path.glob("models/**/*.zip"))
            if model_zips:
                model_path = model_zips[0]
        
        if scaler_path.exists() and model_path.exists():
            try:
                # Load scaler (this will tell us what columns it expects)
                scaler = FitOnTrainScaler.load(scaler_path, columns=features.columns)
                
                # Get the columns the scaler was trained on
                # Try multiple ways to get scaler columns (for compatibility)
                scaler_cols = None
                if hasattr(scaler, 'mu_') and scaler.mu_ is not None:
                    scaler_cols = list(scaler.mu_.index)
                elif hasattr(scaler, 'columns_'):
                    scaler_cols = list(scaler.columns_)
                
                if scaler_cols is None:
                    # Fallback: use all feature columns
                    scaler_cols = list(features.columns)
                    print("Warning: Could not determine scaler columns, using all features")
                
                # Only use features that the scaler knows about
                features_to_scale = features[[c for c in scaler_cols if c in features.columns]]
                
                if len(features_to_scale.columns) != len(scaler_cols):
                    missing = set(scaler_cols) - set(features_to_scale.columns)
                    print(f"Warning: Missing {len(missing)} feature columns that scaler expects")
                    print(f"  Missing: {list(missing)[:5]}...")
                
                # Apply scaler
                features_scaled = scaler.transform(features_to_scale)
                
                # Align scaled features to prices
                common = prices.index.intersection(features_scaled.index)
                prices_eval = prices.loc[common]
                features_eval = features_scaled.loc[common]
                
                print(f"Using model from: {model_path}")
                print(f"Using scaler from: {scaler_path}")
                print(f"Observation space: {features_eval.shape[1]} features + {prices_eval.shape[1]} assets = {features_eval.shape[1] + prices_eval.shape[1]} total")
                
                rl = evaluate_sb3_model(prices_eval, features_eval, str(model_path))
            except Exception as e:
                print(f"RL Agent (SB3) not available: {repr(e)}")
                import traceback
                traceback.print_exc()
        else:
            print(f"RL Agent not available: missing scaler or model")
            if not scaler_path.exists():
                print(f"  Missing: {scaler_path}")
            if not model_path.exists():
                print(f"  Missing: {model_path}")
    else:
        print("RL Agent not available: no training run found")
        print("  Expected: artifacts/*__ppo_dirichlet_baseline/")

    # Visualization
    os.makedirs("reports", exist_ok=True)

    # 1. Equity Curves
    plt.figure(figsize=(10, 5))
    plt.plot(bh["equity_curve"], label="Buy & Hold")
    plt.plot(ew["equity_curve"], label="Equal Weight")
    if rl:
        plt.plot(rl["equity_curve"], label="RL Agent")
    plt.title("Equity Curves")
    plt.xlabel("Time (days)")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/equity_curves.png")
    plt.close()

    # 2. Metrics Comparison - Separate subplots for better visibility
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Performance Metrics Comparison", fontsize=16, fontweight='bold')
    
    strategies = ["Buy & Hold", "Equal Weight"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    # Extract metrics
    bh_metrics = bh["metrics"]
    ew_metrics = ew["metrics"]
    rl_metrics = rl["metrics"] if rl else None
    
    # 1. Final Value (top-left)
    ax = axes[0, 0]
    strategies_plot = strategies.copy()
    values = [bh_metrics["final_value"], ew_metrics["final_value"]]
    if rl_metrics:
        strategies_plot.append("RL Agent")
        values.append(rl_metrics["final_value"])
    bars = ax.bar(strategies_plot, values, color=colors[:len(values)])
    ax.set_ylabel("Final Portfolio Value", fontweight='bold')
    ax.set_title("Final Value", fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Sharpe Ratio (top-right)
    ax = axes[0, 1]
    values = [bh_metrics["sharpe"], ew_metrics["sharpe"]]
    if rl_metrics:
        values.append(rl_metrics["sharpe"])
    bars = ax.bar(strategies_plot, values, color=colors[:len(values)])
    ax.set_ylabel("Annualized Sharpe Ratio", fontweight='bold')
    ax.set_title("Sharpe Ratio", fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Max Drawdown (bottom-left) - Note: max_drawdown is negative
    ax = axes[1, 0]
    values = [bh_metrics["max_drawdown"], ew_metrics["max_drawdown"]]
    if rl_metrics:
        values.append(rl_metrics["max_drawdown"])
    bars = ax.bar(strategies_plot, values, color=colors[:len(values)])
    ax.set_ylabel("Maximum Drawdown", fontweight='bold')
    ax.set_title("Max Drawdown (negative = loss)", fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='top' if height < 0 else 'bottom', fontweight='bold')
    
    # 4. Turnover (bottom-right)
    ax = axes[1, 1]
    values = [bh_metrics["turnover"], ew_metrics["turnover"]]
    if rl_metrics:
        values.append(rl_metrics["turnover"])
    bars = ax.bar(strategies_plot, values, color=colors[:len(values)])
    ax.set_ylabel("Average Turnover per Period", fontweight='bold')
    ax.set_title("Portfolio Turnover", fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("reports/metrics_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. RL Allocation Heatmap
    if rl:
        plt.figure(figsize=(12, 6))
        sns.heatmap(rl["weights"].T, cmap="viridis", cbar=True)
        plt.title("RL Agent Portfolio Weights")
        plt.xlabel("Time (days)")
        plt.ylabel("Assets")
        plt.tight_layout()
        plt.savefig("reports/rl_allocation_heatmap.png")
        plt.close()

    print("Report generated in reports/ folder:")
    print("- reports/equity_curves.png")
    print("- reports/metrics_comparison.png")
    if rl:
        print("- reports/rl_allocation_heatmap.png")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate performance reports")
    ap.add_argument("--run_dir", type=str, default=None, 
                    help="Path to training run directory (auto-finds latest if not provided)")
    args = ap.parse_args()
    generate_report(args.run_dir)
