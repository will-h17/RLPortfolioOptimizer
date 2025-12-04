# Reinforcement Learning Portfolio Optimizer

A complete system for training and evaluating reinforcement learning agents that optimize investment portfolio allocations. The system uses Proximal Policy Optimization (PPO) to learn how to rebalance portfolios across multiple assets, automatically adapting to market conditions while managing transaction costs and risk.

## Overview

This project provides an end-to-end pipeline for training reinforcement learning models on historical market data. The trained models learn to allocate capital across assets in a way that maximizes risk-adjusted returns. The system also includes comprehensive evaluation tools that compare the RL agent against traditional portfolio strategies like equal-weight allocation, volatility-targeted strategies, and momentum-based approaches.

The system is designed to be reproducible and config-driven. Every training run creates a complete record including the configuration used, model checkpoints, evaluation metrics, and version information. This makes it easy to compare different approaches and understand what works best.

## How the System Works

The project follows a standard machine learning workflow: data preparation, feature engineering, model training, and evaluation. Here's how the main components interact:

```
Data Sources (yfinance)
    |
    v
[data.py] Downloads historical prices
    |
    v
[features.py] Generates technical indicators
    |
    v
[config_loader.py] Loads configuration
    |
    v
[train.py] Orchestrates training:
    |-- [splits.py] Creates train/val/test splits
    |-- [scaler.py] Fits feature scaler on training data
    |-- [env.py] Creates trading environment
    |-- [agent.py] Builds PPO model with Dirichlet policy
    |-- [repro.py] Sets seeds and tracks versions
    |
    v
Trained Model (saved to artifacts/)
    |
    v
[backtest.py] Evaluates model performance
    |-- [baselines.py] Runs comparison strategies
    |
    v
[sweep_costs.py] Tests cost sensitivity
    |
    v
[eval_grid.py] Runs breadth evaluation
    |
    v
[report.py] Generates visualizations
```

## Project Structure

```
.
├── configs/              Configuration files for training and evaluation
├── data/
│   ├── raw/             Downloaded market data (CSV files)
│   └── processed/       Processed price and feature data (Parquet files)
├── src/                 Main source code
│   ├── data.py          Data download and processing
│   ├── features.py      Feature engineering
│   ├── config_loader.py Configuration file parsing
│   ├── train.py         Main training script
│   ├── env.py           Trading environment (Gymnasium)
│   ├── agent.py         PPO agent and policy definition
│   ├── scaler.py        Feature normalization
│   ├── splits.py         Date splitting utilities
│   ├── backtest.py      Model evaluation
│   ├── baselines.py     Baseline strategy implementations
│   ├── sweep_costs.py   Cost sensitivity analysis
│   ├── eval_grid.py     Breadth evaluation across scenarios
│   ├── report.py        Visualization and reporting
│   ├── repro.py         Reproducibility utilities
│   └── utils/           Shared utility functions
├── tests/               Unit tests
├── artifacts/           Training run outputs
└── app.py               Web interface (Dash application)
```

## Key Files

### Core Training Components

**train.py** - Main training orchestrator. Handles the complete training workflow: loads configuration, splits data into train/validation/test sets, fits the feature scaler on training data only, creates the trading environment, builds the PPO agent, runs training with checkpointing and evaluation callbacks, and saves all artifacts to a timestamped directory.

**env.py** - Implements the trading environment as a Gymnasium-compatible environment. The environment manages portfolio state, processes actions (target portfolio weights), calculates rewards based on returns and transaction costs, and tracks portfolio value over time. Actions are projected to a probability simplex to ensure valid portfolio allocations.

**agent.py** - Defines the reinforcement learning agent using Stable Baselines3's PPO algorithm. Implements a custom Dirichlet policy head that outputs portfolio weight allocations. The Dirichlet distribution naturally enforces the constraint that weights sum to one and are non-negative.

**config_loader.py** - Parses YAML configuration files and converts them into structured dataclass objects. Handles validation and provides default values for optional parameters. Ensures type safety and clear error messages.

### Data Processing

**data.py** - Downloads historical market data using yfinance and processes it into a unified price dataset. Handles data alignment, missing value handling, and saves processed data in Parquet format for efficient access.

**features.py** - Generates technical features from price data including returns, log returns, momentum indicators, rolling volatility measures, and exponential moving averages. Features are designed to capture market dynamics that the RL agent can learn from.

**scaler.py** - Implements feature normalization that fits only on training data to prevent look-ahead bias. Saves scaler parameters to JSON for consistent application during evaluation.

**splits.py** - Utilities for splitting time series data into training, validation, and test periods. Handles date alignment and ensures proper temporal ordering.

### Evaluation and Analysis

**backtest.py** - Runs backtests of the trained RL model and compares against baseline strategies. Computes performance metrics including Sharpe ratio, maximum drawdown, turnover, and final portfolio value.

**baselines.py** - Implements classical portfolio allocation strategies: buy-and-hold, equal-weight, volatility-targeted equal-weight, inverse-variance parity, and cross-sectional momentum. All baselines use the same cost-aware backtesting engine for fair comparison.

**sweep_costs.py** - Tests how sensitive the trained model is to different transaction cost assumptions. Sweeps costs from 0 to 10 basis points and reports performance degradation.

**eval_grid.py** - Runs comprehensive evaluation across multiple scenarios: different asset universes, different time periods, and different feature sets. Produces summary statistics for robustness analysis.

**report.py** - Generates visualization reports including equity curves, metrics comparisons, and allocation heatmaps. Automatically finds the latest training run and loads the appropriate model and scaler.

### Supporting Components

**repro.py** - Sets random seeds for reproducibility and collects library version information. Ensures that training runs can be reproduced exactly.

**utils/data_utils.py** - Shared utilities for data alignment and date handling used across multiple modules.

**utils/logging.py** - Logging setup for training progress and TensorBoard integration.

## Command Line Interface Guide

This section provides a step-by-step guide for using the system via the command line. The web interface (see APP_INFO.md) provides the same functionality through a graphical interface.

### Prerequisites

Install required dependencies:
```bash
pip install -r requirements.txt
```

### Step 1: Unit Tests

Before running the full pipeline, verify that core functionality works correctly:

```bash
pytest tests/ -v
```

This runs tests for environment mathematics, simplex projection, turnover calculations, and look-ahead bias checks. All tests should pass before proceeding.

### Step 2: Data Preparation

Download and process market data:

```bash
# Download historical data for your chosen assets
python -m src.data
```

This downloads data from yfinance and creates `data/processed/prices_adj.parquet`. The default configuration uses a set of ETFs, but you can modify the symbols in `src/data.py`.

Verify the data:
```python
import pandas as pd
prices = pd.read_parquet("data/processed/prices_adj.parquet")
print(f"Date range: {prices.index.min()} to {prices.index.max()}")
print(f"Assets: {list(prices.columns)}")
```

### Step 3: Feature Engineering

Generate technical features from price data:

```bash
python -m src.features
```

This creates `data/processed/features.parquet` with features like returns, momentum, volatility, and moving averages. Features are aligned with the price data and handle warm-up periods where some features may be NaN.

### Step 4: Training

Train the RL agent. Start with a quick test run to verify everything works:

```bash
python -m src.train --config configs/test_quick.yaml
```

This uses a configuration with fewer training steps (10,000) for quick verification. Once confirmed, run a full training:

```bash
python -m src.train --config configs/rlppo.yaml
```

Training creates a directory under `artifacts/` with timestamp and run name. The directory contains:
- `config.json` - Frozen configuration for this run
- `versions.json` - Library versions and seeds
- `scaler.json` - Fitted feature scaler
- `models/` - Model checkpoints and best model
- `eval/metrics.json` - Evaluation metrics

Training typically takes 15-30 minutes for 2,000,000 timesteps depending on hardware. Monitor the console output for progress updates.

### Step 5: Evaluation

Evaluate the trained model against baselines:

```bash
# Find the latest training run
LATEST_RUN=$(ls -td artifacts/*__ppo_dirichlet_baseline | head -1)

# Run backtest comparison
python -m src.backtest --run_dir $LATEST_RUN
```

This compares the RL agent against buy-and-hold, equal-weight, and other baseline strategies. Results are saved to `baseline_results.csv` in the run directory.

### Step 6: Cost Sensitivity Analysis

Test how transaction costs affect performance:

```bash
python -m src.sweep_costs \
    --config configs/rlppo.yaml \
    --run_dir $LATEST_RUN \
    --out_csv $LATEST_RUN/eval/cost_sweep.csv
```

This sweeps costs from 0 to 10 basis points and shows how performance degrades. Results help understand the robustness of the strategy to different cost assumptions.

### Step 7: Grid Evaluation

Run comprehensive evaluation across multiple scenarios:

```bash
# Quick grid test
python -m src.eval_grid --grid configs/eval_grid_mini.yaml

# Full grid evaluation
python -m src.eval_grid --grid configs/eval_grid.yaml
```

Grid evaluation tests the model across different asset universes, time periods, and feature configurations. Results are saved in `eval/breadth/` with detailed CSV files.

### Step 8: Generate Reports

Create visualization reports:

```bash
python -m src.report
```

This generates PNG files in the `reports/` directory:
- `equity_curves.png` - Portfolio value over time
- `metrics_comparison.png` - Bar charts comparing strategies
- `rl_allocation_heatmap.png` - Asset allocation over time

The report script automatically finds the latest training run and loads the appropriate model.

### Step 9: Hyperparameter Optimization (Optional)

Use Optuna to find optimal hyperparameters:

```bash
python -m src.hpo.run_optuna \
    --config configs/rlppo.yaml \
    --n-trials 30 \
    --study-name ppo_hpo \
    --output-dir artifacts/hpo
```

This runs multiple training trials with different hyperparameters and finds the best combination. Each trial trains a full model, so this can take 10-15 hours for 30 trials. Results are saved as JSON files in the output directory.

You can resume an interrupted study:
```bash
python -m src.hpo.run_optuna \
    --config configs/rlppo.yaml \
    --n-trials 30 \
    --study-name ppo_hpo \
    --output-dir artifacts/hpo \
    --resume
```

## Configuration Files

Configuration files are YAML files that specify all parameters for training and evaluation. Key sections include:

- `run_name` - Identifier for this training run
- `seed` - Random seed for reproducibility
- `paths` - File paths for data and output
- `dates` - Train/validation/test period boundaries
- `env` - Environment parameters (costs, cash rate, etc.)
- `reward` - Reward function parameters
- `ppo` - PPO hyperparameters (learning rate, network size, etc.)
- `train` - Training parameters (timesteps, checkpoint frequency, etc.)

Example configurations are provided in the `configs/` directory. Start with `test_quick.yaml` for quick testing, then use `rlppo.yaml` or `rlppo_sharpe_optimized.yaml` for production training.

## Understanding Results

After training, check the evaluation metrics in `artifacts/<run>/eval/metrics.json`. Key metrics include:

- `sharpe` - Sharpe ratio (risk-adjusted return)
- `final_value` - Final portfolio value (starting from 1.0)
- `max_drawdown` - Maximum peak-to-trough decline
- `turnover` - Average portfolio turnover per period
- `num_steps` - Number of trading periods

Compare these metrics against baseline strategies to assess whether the RL agent is learning useful allocation patterns. A good model should achieve higher Sharpe ratio and final value than simple baselines while maintaining reasonable turnover.

## Troubleshooting

**Training stops early**: Check that your date ranges are valid and that you have enough data. The environment needs at least 100 rows of training data.

**Poor performance**: Try different hyperparameters or increase training time. The model may need more timesteps to learn effective strategies.

**Memory errors**: Reduce batch size or number of parallel environments. The default configuration should work on most systems, but very large feature sets may require adjustment.

**Import errors**: Ensure all dependencies are installed. Use `pip install -r requirements.txt` if a requirements file is provided, or install packages individually.

## Next Steps

Once you have a working model:

1. Experiment with different configurations to improve performance
2. Try different asset universes or time periods
3. Analyze allocation patterns to understand what the model learned
4. Use hyperparameter optimization to find better settings
5. Compare results across multiple training runs

For a graphical interface that makes these operations easier, see APP_INFO.md for the web application guide.
