# Web Application User Guide

A walk-through guide for using the web-interface for this portfolio optimizer system. For those not wanting to use the CLI, the web application provides an end-to-end workflow for model use and evaluation.

## Getting Started

### Installation

First, install the web application dependencies (if not already installed):

```bash
pip install -r requirements_app.txt
```

### Starting the Application

Start the web server:

```bash
python app.py
```

The application will start on `http://localhost:8050`. Open this URL in your web browser.

## Application Overview

The application is organized into several pages accessible from the sidebar:

- Dashboard - Overview and quick status
- Data Acquisition - Download and process market data
- Features - Generate technical indicators
- Train Model - Train RL agents
- Evaluation - Compare models against baselines
- Cost Analysis - Test cost sensitivity
- Grid Evaluation - Run comprehensive tests
- Hyperparameter Optimization - Find optimal settings
- Reports - Generate visualizations
- Performance - View detailed results

## Step-by-Step Workflow

### Step 1: Prepare Data

Navigate to the "Data Acquisition" page.

Enter the stock symbols you want to use, separated by commas. For example: `SPY,QQQ,TLT,GLD,IWM`

Click "Download Data" to fetch historical prices. The system will download data from yfinance and show progress in the status area.

Once download completes, click "Process & Merge" to combine the data into a single file. This creates the processed price data that the rest of the system uses.

### Step 2: Generate Features

Navigate to the "Features" page.

Enter lookback periods for feature generation, separated by commas. For example: `5,20` creates features with 5-day and 20-day windows.

Click "Generate Features" to create technical indicators from the price data. This includes returns, momentum, volatility measures, and moving averages.

The status area will show progress. When complete, features are saved and ready for training.

### Step 3: Train a Model

Navigate to the "Train Model" page.

Select a configuration file from the dropdown. The preview area shows key parameters from the selected configuration. For quick testing, choose `test_quick.yaml`. For production training, use `rlppo.yaml` or `rlppo_sharpe_optimized.yaml`.

Review the configuration preview to understand the training parameters. Warnings will appear if the configuration has very short training times or other issues.

Click "Start Training" to begin. Training runs in the background, so you can navigate to other pages while it runs. The status area shows progress and will update when training completes.

Training can take 30-60 minutes for a full run depending on the number of timesteps specified in the configuration. Monitor the console output or status area for progress updates.

### Step 4: Evaluate Performance

Navigate to the "Evaluation" page.

Select your training run from the dropdown. The dropdown shows all available runs with timestamps.

Click "Run Backtest" to compare your RL agent against baseline strategies. Results appear in a table showing metrics for buy-and-hold, equal-weight, and your RL agent.

Review the metrics to see how your model performs. Key metrics include Sharpe ratio, final portfolio value, maximum drawdown, and turnover.

### Step 5: Analyze Costs

Navigate to the "Cost Analysis" page.

Select your training run and the configuration file used for training.

Click "Run Cost Sweep" to test how transaction costs affect performance. The system tests costs from 0 to 10 basis points and displays results in a chart.

This analysis helps understand how robust your strategy is to different cost assumptions. Strategies that degrade significantly with small cost increases may not be practical for real trading.

### Step 6: View Detailed Results

Navigate to the "Performance" page.

Select your training run from the dropdown.

The page displays several visualizations:

- Equity curves showing portfolio value over time for different strategies
- Metrics comparison charts showing how strategies compare
- Allocation heatmap showing how the RL agent allocated weights across assets over time

Use these visualizations to understand what the model learned and how it behaves in different market conditions.

### Step 7: Generate Reports

Navigate to the "Reports" page.

Optionally select a specific training run, or leave empty to use the latest run.

Click "Generate Reports" to create visualization files. Reports are saved to the `reports/` directory as PNG files.

These reports can be used for documentation, presentations, or further analysis outside the web interface.

## Advanced Features

### Grid Evaluation

The "Grid Evaluation" page allows comprehensive testing across multiple scenarios. Select a grid configuration file and click "Run Grid Evaluation" to test the model across different asset universes, time periods, and feature sets.

This is useful for understanding model robustness and identifying scenarios where the strategy works well or poorly.

### Hyperparameter Optimization

The "Hyperparameter Optimization" page uses Optuna to automatically search for optimal hyperparameters.

Select a base configuration file, specify the number of trials, and provide a study name. Click "Start HPO" to begin. Each trial trains a full model, so this process can take 10-15 hours for 30 trials depending on hardware.

You can resume interrupted studies by checking the "Resume from existing study" option. Results are saved as JSON files in the output directory.

## Understanding the Interface

### Status Messages

All pages show status messages that indicate:
- When operations are in progress
- When operations complete successfully
- Any errors that occur

Status messages include details about what happened and where results were saved.

### Progress Indicators

Long-running operations show progress bars or status updates. Training shows progress through timesteps, and other operations show completion status.

### Error Handling

If an error occurs, the status area will display an error message with details. Common issues include:
- Missing data files (run data acquisition first)
- Invalid configurations (check the config file)
- Insufficient data (ensure date ranges are valid)

## Tips for Effective Use

Start with quick tests using `test_quick.yaml` to verify everything works before running long training sessions.

Monitor training progress through the status area or console output. Training output is captured and displayed.

Compare multiple training runs by selecting different runs in the Performance page. This helps identify which configurations work best.

Use the Dashboard page to get a quick overview of system status, including available data, features, and training runs.

## Troubleshooting

If data download fails, check your internet connection and verify that the stock symbols are valid. Some symbols may not be available in yfinance.

If training fails immediately, check that data and features have been generated. The system needs processed price and feature data before training can begin.

If the application won't start, verify that all dependencies are installed. Run `pip install dash dash-bootstrap-components plotly` to ensure required packages are available.

If pages don't load correctly, check the browser console for JavaScript errors. The application requires a modern web browser.

## Next Steps

After you have a trained model:

1. Experiment with different configurations to improve performance
2. Use hyperparameter optimization to find better settings
3. Run grid evaluations to test robustness
4. Generate reports for documentation
5. Compare results across multiple training runs

The web interface provides access to all functionality, but you can also use the command line tools described in README.md for more advanced workflows or automation.

