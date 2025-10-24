# Reinforcement Portfolio Optimizer

An end-to-end pipeline for training, evaluating, and comparing a reinforcement-learning portfolio allocator against robust baselines (vol-targeted equal-weight, inverse-variance parity, and cross-sectional momentum). The project is config-driven, reproducible, and leaves a complete run-of-record (config, seeds/versions, scaler, model, metrics) under `artifacts/`.

---

## Project structure

.
├─ configs/
│ ├─ rlppo.yaml # main training config (assets, dates, costs, PPO hparams, seed)
│ └─ eval_grid.yaml # breadth eval (universes, periods, ablations)
├─ data/
│ └─ processed/
│ ├─ prices_adj.parquet # (T×N) adjusted closes (columns = tickers)
│ └─ features.parquet # (T×F) raw feature matrix (NaN warm-up allowed)
├─ src/
│ ├─ train.py # splits, scaler fit, PPO train, TEST eval, run record
│ ├─ env.py # PortfolioEnv + EnvConfig (reward, costs, projection, info)
│ ├─ agent.py # SB3 PPO factory (policy sizes/hparams)
│ ├─ backtest.py # baselines + RL eval; saves baseline_results
│ ├─ sweep_costs.py # cost sensitivity sweep (0–10 bps) on TEST
│ ├─ eval_grid.py # multi-universe/period ablation evaluation
│ ├─ scaler.py # FitOnTrainScaler (fit on train; JSON save/load)
│ ├─ baselines.py # VT-EW, IVP, Momentum + cost-aware backtest engine
│ ├─ splits.py # date_slices / walk_forward helpers
│ ├─ data.py, features.py # data/feature prep
│ ├─ report.py # optional plotting/report helpers
│ ├─ repro.py # seeding + versions collection
│ └─ init.py # makes src/ importable
├─ tests/
│ ├─ conftest.py # path bootstrap + tiny synthetic market fixture
│ ├─ test_env_math.py # PnL timing, simplex projection, turnover
│ ├─ test_turnover_and_metrics.py # Sharpe / MaxDD utilities
│ └─ test_no_lookahead.py # off-by-one / no look-ahead
└─ artifacts/
└─ <timestamp>__<run_name>/
├─ config.json # frozen config for this run
├─ versions.json # seed + library versions
├─ scaler.json # train-fitted scaler
├─ models/
│ ├─ ppo_dirichlet_sb3.zip
│ └─ best/best_model.zip
└─ eval/
├─ metrics.json
├─ cost_sweep.csv
├─ baseline_results.csv
└─ breadth/
├─ grid_results.csv
├─ grid_sharpe_median.csv
└─ errors.csv

**Key files (expanded):**
- `train.py` — Orchestrates the whole experiment: parses config, constructs date splits (train/val/test), fits the feature scaler on TRAIN only, builds the Gym env and SB3 PPO agent, runs training with callbacks (checkpoints / eval), evaluates the saved policy on TEST, and writes a run-of-record (config.json, versions.json, scaler.json, model files, and eval metrics) into an artifacts/<timestamp>__<run_name>/ folder.

- `env.py` — Trading environment (Gym/Gymnasium compatible). Implements state construction (features + current weights), action projection to the long-only simplex, reward calculation (arithmetic or log returns minus transaction costs and optional penalties), cost models (fixed / vol-based spread, fees, slippage), and exposes rich `info` for diagnostics (per-step returns, costs, turnover).

- `agent.py` — Factory for the RL agent and policy. Encodes architecture choices (Dirichlet / Gaussian policy variants, hidden layer sizes), SB3 PPO hyper-parameters wiring, and helper utilities to load/save policies consistently for training and evaluation.

- `backtest.py` — Lightweight backtester and baseline runner used for final evaluation and sanity checks. Aligns prices & features, runs a policy function across the test period (used for RL policy or simple baselines), and computes performance metrics (cumulative returns/equity, Sharpe, max drawdown, average turnover). Saves baseline and RL evaluation CSVs under artifacts.

- `sweep_costs.py` — Post-training utility that replays a trained policy over TEST while sweeping transaction cost parameters (e.g., 0–10 bps). Produces a cost-sensitivity table (cost_sweep.csv) showing how net performance degrades with higher implicit trading costs.

- `eval_grid.py` — Higher-level evaluation harness for breadth experiments: iterates across universes, date windows, and ablations (feature sets, lookbacks, cost assumptions) to produce comparative tables and summary stats used for robustness analysis.

- `scaler.py` — Implements `FitOnTrainScaler`: fit on TRAIN features only, serialize/deserialize scaler parameters (JSON), and provide transform/inverse_transform helpers used at training and evaluation time to ensure no look-ahead leaks.

- `baselines.py` — Implementations of classical allocation baselines (vol-targeted equal-weight, inverse-variance parity, cross-sectional momentum) and a cost-aware backtest engine used to compare RL vs deterministic strategies on the same evaluation pipeline.

- `splits.py` — Date slicing and walk-forward utilities that convert config date strings into index-aligned train/val/test splits, plus helpers to clamp dates to available data.

- `data.py` / `features.py` — Data ingestion and feature engineering: load raw prices, compute returns/log-returns, momentum, rolling vol, EMAs, save processed Parquet artifacts used downstream.

- `report.py` — Optional plotting and report helpers for generating time-series plots, turnover diagnostics, and run summaries from artifacts.

- `repro.py` — Reproducibility helpers: seed setting for Python / NumPy / Torch, and collection of library versions to include in the run record.

- `tests/` — Unit tests and small integration checks (PnL timing/no look-ahead, simplex projection, turnover and metric calculations). Helps to run these before large experiments.
