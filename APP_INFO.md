# Web Application User Guide

The web app runs the whole workflow from a browser: getting data, training agents, and evaluating them against simple baselines. Everything it does can also be done from the command line (see README.md).

## Getting Started

### Installation

Install the project with the app extra in its own environment (Python 3.12):

```bash
conda create -n rlpo python=3.12 -y
conda activate rlpo
pip install -e ".[app]" -c constraints.txt
```

In VS Code, select the `rlpo` interpreter (Command Palette → "Python: Select Interpreter").

### Starting the Application

```bash
python app.py
```

Open `http://127.0.0.1:8050` in your browser.

Options:
- `--port 8051` uses another port.
- `--debug` turns on Dash dev tools and auto-reload while you edit the app.
- `--host 0.0.0.0` lets other machines on your network connect. Never combine it with `--debug`: the debugger lets anyone who can reach the page run code on your machine.

Stopping the server (Ctrl-C) also stops any job it is running.

## How Jobs Work

Pages that do long work (downloading, training, backtests, sweeps, walk-forward, search) have the same controls:

- **Start**: the job runs in its own background process, so you can switch pages while it runs.
- **Progress bar**: shows a percentage when the job can measure it (training timesteps, walk-forward runs, search trials), otherwise an animated bar.
- **Log**: the job's latest output, updated about twice a second.
- **Stop**: ends the job right away, including any training it started. A stopped training run keeps whatever it had already saved.
- **Result**: a summary, table or chart once the job finishes. Errors show the message and the end of the log.

Jobs belong to the browser tab that started them. Reloading the page detaches you from a running job, but the files it writes still appear when it finishes.

## Pages

### 🏠 Dashboard
Shows whether prices and features exist, how many completed runs and walk-forward studies there are, the latest run, and links through the workflow.

### 📥 Data
Enter ticker symbols (for example `SPY,QQQ,TLT`).
1. **Download data** fetches daily adjusted prices from yfinance into `data/raw/`. Cached files are reused unless you tick "Re-download".
2. **Process & merge** combines them into `data/processed/prices_adj.parquet`.

### 🔧 Features
Enter lookback windows in days (default `5,20`) and generate `data/processed/features.parquet`: returns, momentum, volatility and trend features for every asset.

### ⚙️ Training
Choose a config. The preview shows its run name, timesteps, parallel environments, episode setup, reward, costs and date split. If the config is invalid, the preview lists each invalid field instead.

- `test_quick.yaml` takes a few seconds and only checks that the pipeline works.
- `rlppo_v2.yaml` is the recommended setup (about 2.5 minutes for 1,000,000 timesteps on a laptop CPU).

The result shows the run name and its test-split metrics.

### 📊 Evaluation
Backtests a run and the baselines over the full price history. That history includes the run's training period, so the agent's numbers here are in-sample. Use Walk-forward for a fair comparison.

### 💰 Cost Sensitivity
Re-runs a trained agent on its test split with bid-ask spreads from 0 to 10 bps, using the run's own config, and charts Sharpe against cost.

### 🧭 Walk-forward
The result to trust. For each test year, it trains a fresh agent per seed on all earlier data, chooses the checkpoint on the preceding six months, and tests on that year. Equal weight, buy & hold and inverse variance are run on the same windows with identical costs.

- **Results** shows any finished study: the full-period out-of-sample summary and a Sharpe-by-year chart.
- **Run a study** starts one. A full study (10 years × 3 seeds with `rlppo_v2.yaml`) takes about 75 minutes. Set "Timesteps per run" to 50000 for a quick check.
- Re-running with the same study name resumes it, skipping finished years and seeds.

### 🌐 Grid Evaluation
Evaluates the latest run (or the one named in the grid config) across asset universes, periods and feature subsets.

### 🔍 Hyperparameters
Searches PPO and turnover-penalty settings with Optuna, scoring each trial by validation Sharpe. Each trial trains a model for `hpo.train_timesteps` steps, so a 30-trial search takes hours. Tick "Resume" to continue an existing study with the same name.

### 📈 Reports
Renders equity curves, metric comparisons and an allocation heatmap for a run (or the latest one) to `reports/`, and shows them on the page.

### 📉 Performance
Explores one run in detail:
- its test-split Sharpe;
- its equity curve against buy & hold and equal weight over the full history (in-sample, as on the Evaluation page);
- how its validation Sharpe changed during training;
- how its allocation shifted over time.

## Troubleshooting

- **Download fails**: check your internet connection and the symbols. yfinance returns nothing for unknown tickers, and the job says so.
- **Training fails immediately**: make sure prices and features exist (Dashboard), then read the error and log under the job.
- **A config is rejected**: the Training preview names the invalid field. Fix it in the config or in the base file it `extends`.
- **The app won't start**: activate the `rlpo` environment and reinstall with `pip install -e ".[app]" -c constraints.txt`.
- **Checking the app without a browser**: `pytest tests/test_app_smoke.py -v` loads every page and the main callbacks.
