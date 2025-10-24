"""
report.py — Generate performance visualizations for RL portfolio optimizer.

- Runs backtests for Buy & Hold, Equal Weight, RL Agent
- Plots equity curves, metrics comparison, and allocation heatmap
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.backtest import buy_and_hold, equal_weight, evaluate_sb3_model


def generate_report():
    # ---------------- Load Data ---------------- #
    prices = pd.read_parquet("data/processed/prices_adj.parquet")
    features = pd.read_parquet("data/processed/features.parquet")

    # ---------------- Baselines ---------------- #
    bh = buy_and_hold(prices, features)
    ew = equal_weight(prices, features)

    # ---------------- RL Agent ---------------- #
    rl = None
    try:
        rl = evaluate_sb3_model(prices, features, "models/ppo_dirichlet_sb3.zip")
    except Exception as e:
        print("RL Agent (SB3) not available:", repr(e))

    # ---------------- Visualization ---------------- #
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

    # 2. Metrics Comparison
    labels = ["Final Value", "Sharpe", "Max DD", "Turnover"]
    bh_vals = [bh["metrics"][k] for k in ["final_value", "sharpe", "max_drawdown", "turnover"]]
    ew_vals = [ew["metrics"][k] for k in ["final_value", "sharpe", "max_drawdown", "turnover"]]
    rl_vals = [rl["metrics"][k] for k in ["final_value", "sharpe", "max_drawdown", "turnover"]] if rl else [0, 0, 0, 0]

    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, bh_vals, width, label="Buy & Hold")
    ax.bar(x, ew_vals, width, label="Equal Weight")
    if rl:
        ax.bar(x + width, rl_vals, width, label="RL Agent")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Performance Metrics Comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig("reports/metrics_comparison.png")
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
    generate_report()
