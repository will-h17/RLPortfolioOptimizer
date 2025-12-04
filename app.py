"""
Comprehensive Dash App for RL Portfolio Optimizer
All-in-one interface for the entire workflow
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from pathlib import Path
import pandas as pd
import numpy as np
import json
import os
import glob
import threading
import time
import sys
import io
import argparse
from datetime import datetime
from typing import Dict, List, Optional

# Import project modules
from src.config_loader import load_config
from src.train import train_once
from src.backtest import buy_and_hold, equal_weight, evaluate_sb3_model
from src.baselines import (
    CostCfg, vt_equal_weight, ivp_inverse_variance, momentum_xs, _run_backtest
)
from src.scaler import FitOnTrainScaler
from src.utils.data_utils import align_after_load
from src.data import download_universe, load_and_merge
from src.features import make_features
from src.report import generate_report

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "RL Portfolio Optimizer - All-in-One"

# Paths
CONFIGS_DIR = Path("configs")
ARTIFACTS_DIR = Path("artifacts")
REPORTS_DIR = Path("reports")
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
REPORTS_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Global state for all background tasks
task_status = {
    "data": {"running": False, "message": "", "progress": 0},
    "features": {"running": False, "message": "", "progress": 0},
    "training": {"running": False, "message": "", "progress": 0, "run_dir": None},
    "backtest": {"running": False, "message": "", "progress": 0},
    "cost_sweep": {"running": False, "message": "", "progress": 0},
    "grid_eval": {"running": False, "message": "", "progress": 0},
    "hpo": {"running": False, "message": "", "progress": 0},
    "report": {"running": False, "message": "", "progress": 0},
}
training_status = task_status["training"]  # Backward compatibility

# Layout Components

def create_navbar():
    """Create the top navigation bar."""
    return dbc.Navbar(
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("📈 RL Portfolio Optimizer", className="mb-0"),
                        html.Small("Intelligent Portfolio Rebalancing", className="text-muted")
                    ])
                ], width="auto"),
            ], align="center", className="g-0"),
        ], fluid=True),
        color="primary",
        dark=True,
        className="mb-4"
    )

def create_sidebar():
    """Create the sidebar navigation with all pages."""
    return html.Div([
        dbc.Nav([
            dbc.NavLink("🏠 Dashboard", href="/", active="exact", id="nav-dashboard", className="mb-2"),
            dbc.NavLink("📥 Data Acquisition", href="/data", active="exact", id="nav-data", className="mb-2"),
            dbc.NavLink("🔧 Features", href="/features", active="exact", id="nav-features", className="mb-2"),
            dbc.NavLink("⚙️ Training", href="/train", active="exact", id="nav-train", className="mb-2"),
            dbc.NavLink("📊 Evaluation", href="/evaluate", active="exact", id="nav-evaluate", className="mb-2"),
            dbc.NavLink("💰 Cost Analysis", href="/costs", active="exact", id="nav-costs", className="mb-2"),
            dbc.NavLink("🌐 Grid Evaluation", href="/grid", active="exact", id="nav-grid", className="mb-2"),
            dbc.NavLink("🔍 Hyperparameter Tuning", href="/hpo", active="exact", id="nav-hpo", className="mb-2"),
            dbc.NavLink("📈 Reports", href="/reports", active="exact", id="nav-reports", className="mb-2"),
            dbc.NavLink("📉 Performance", href="/performance", active="exact", id="nav-performance", className="mb-2"),
        ], vertical=True, pills=True, className="bg-light p-3 rounded")
    ], className="sidebar")

# Page Components

def create_dashboard_page():
    """Create the main dashboard page."""
    prices_exist = (PROC_DIR / "prices_adj.parquet").exists()
    features_exist = (PROC_DIR / "features.parquet").exists()
    latest_run = None
    if ARTIFACTS_DIR.exists():
        runs = sorted(ARTIFACTS_DIR.glob("*__*"), key=os.path.getmtime, reverse=True)
        latest_run = runs[0] if runs else None
    
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("Dashboard", className="mb-4"),
                html.P("Welcome to the RL Portfolio Optimizer. Use the sidebar to navigate through the workflow.",
                       className="text-muted mb-4"),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Data Status", className="card-title"),
                                html.H3("✅" if prices_exist else "❌", className="text-center"),
                                html.P("Prices data" if prices_exist else "No prices data", className="text-center text-muted")
                            ])
                        ])
                    ], width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Features Status", className="card-title"),
                                html.H3("✅" if features_exist else "❌", className="text-center"),
                                html.P("Features data" if features_exist else "No features data", className="text-center text-muted")
                            ])
                        ])
                    ], width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Training Runs", className="card-title"),
                                html.H3(str(len(list(ARTIFACTS_DIR.glob("*__*")))) if ARTIFACTS_DIR.exists() else "0", className="text-center"),
                                html.P("Total runs", className="text-center text-muted")
                            ])
                        ])
                    ], width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Latest Run", className="card-title"),
                                html.P(latest_run.name if latest_run else "None", className="text-center", style={"fontSize": "0.9em"}),
                                html.P("Most recent", className="text-center text-muted")
                            ])
                        ])
                    ], width=3),
                ], className="mb-4"),
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Quick Actions", className="card-title mb-3"),
                        dbc.ListGroup([
                            dbc.ListGroupItem([html.Strong("1. Data Acquisition"), " - Download market data"], action=True, href="/data"),
                            dbc.ListGroupItem([html.Strong("2. Feature Engineering"), " - Generate features"], action=True, href="/features"),
                            dbc.ListGroupItem([html.Strong("3. Training"), " - Train your RL model"], action=True, href="/train"),
                            dbc.ListGroupItem([html.Strong("4. Evaluation"), " - Evaluate model performance"], action=True, href="/evaluate"),
                        ])
                    ])
                ]),
            ], width=12)
        ])
    ], fluid=True)

def create_data_page():
    """Create the data acquisition page."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("Data Acquisition", className="mb-4"),
                html.P("Download and process market data for your portfolio.", className="text-muted mb-4"),
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Symbols Configuration", className="card-title mb-3"),
                        dbc.Label("Stock Symbols (comma-separated)", html_for="symbols-input"),
                        dbc.Input(id="symbols-input", type="text", value="SPY,QQQ,TLT", placeholder="SPY,QQQ,TLT"),
                        html.Small("Enter stock ticker symbols separated by commas", className="text-muted mt-2 d-block"),
                        dbc.Checkbox(id="force-download", label="Force re-download (ignore cache)", className="mt-3"),
                    ])
                ], className="mb-4"),
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Actions", className="card-title mb-3"),
                        dbc.Button("Download Data", id="download-button", color="primary", size="lg", className="me-2"),
                        dbc.Button("Process & Merge", id="process-button", color="success", size="lg"),
                        html.Div(id="data-status", className="mt-4"),
                        dcc.Interval(id="data-interval", interval=1000, n_intervals=0, disabled=True),
                    ])
                ], className="mb-4"),
                html.Div(id="data-info"),
            ], width=12)
        ])
    ], fluid=True)

def create_features_page():
    """Create the feature engineering page."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("Feature Engineering", className="mb-4"),
                html.P("Generate technical features from price data.", className="text-muted mb-4"),
                dbc.Alert([html.Strong("Prerequisites: "), "You need processed price data (prices_adj.parquet) before generating features."],
                         color="info", className="mb-4"),
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Feature Configuration", className="card-title mb-3"),
                        dbc.Label("Lookback Windows (comma-separated)", html_for="lookbacks-input"),
                        dbc.Input(id="lookbacks-input", type="text", value="5,20", placeholder="5,20"),
                        html.Small("Enter lookback periods in days, separated by commas", className="text-muted mt-2 d-block"),
                    ])
                ], className="mb-4"),
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Actions", className="card-title mb-3"),
                        dbc.Button("Generate Features", id="generate-features-button", color="primary", size="lg"),
                        html.Div(id="features-status", className="mt-4"),
                        dcc.Interval(id="features-interval", interval=1000, n_intervals=0, disabled=True),
                    ])
                ], className="mb-4"),
                html.Div(id="features-info"),
            ], width=12)
        ])
    ], fluid=True)

def create_evaluation_page():
    """Create the evaluation/backtest page."""
    runs = []
    if ARTIFACTS_DIR.exists():
        runs = sorted([r.name for r in ARTIFACTS_DIR.glob("*__*")], reverse=True)
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("Model Evaluation", className="mb-4"),
                html.P("Run backtests and compare your model against baseline strategies.", className="text-muted mb-4"),
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Select Training Run", className="card-title mb-3"),
                        dcc.Dropdown(id="eval-run-select", options=[{"label": r, "value": r} for r in runs],
                                    placeholder="Select a training run...", clearable=True),
                        html.Div(id="eval-run-info", className="mt-3"),
                    ])
                ], className="mb-4"),
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Actions", className="card-title mb-3"),
                        dbc.Button("Run Backtest", id="run-backtest-button", color="primary", size="lg", disabled=len(runs) == 0),
                        html.Div(id="backtest-status", className="mt-4"),
                        dcc.Interval(id="backtest-interval", interval=1000, n_intervals=0, disabled=True),
                    ])
                ], className="mb-4"),
                html.Div(id="backtest-results"),
            ], width=12)
        ])
    ], fluid=True)

def create_cost_analysis_page():
    """Create the cost sensitivity analysis page."""
    runs = []
    if ARTIFACTS_DIR.exists():
        runs = sorted([r.name for r in ARTIFACTS_DIR.glob("*__*")], reverse=True)
    config_files = sorted([f.name for f in CONFIGS_DIR.glob("*.yaml") if "eval_grid" not in f.name.lower()])
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("Cost Sensitivity Analysis", className="mb-4"),
                html.P("Test how transaction costs affect model performance.", className="text-muted mb-4"),
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Configuration", className="card-title mb-3"),
                        dbc.Label("Training Run", html_for="cost-run-select"),
                        dcc.Dropdown(id="cost-run-select", options=[{"label": r, "value": r} for r in runs],
                                    placeholder="Select a training run...", clearable=True),
                        dbc.Label("Config File", html_for="cost-config-select", className="mt-3"),
                        dcc.Dropdown(id="cost-config-select", options=[{"label": f, "value": f} for f in config_files],
                                    placeholder="Select config file..."),
                    ])
                ], className="mb-4"),
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Actions", className="card-title mb-3"),
                        dbc.Button("Run Cost Sweep", id="run-cost-sweep-button", color="primary", size="lg"),
                        html.Div(id="cost-sweep-status", className="mt-4"),
                        dcc.Interval(id="cost-sweep-interval", interval=1000, n_intervals=0, disabled=True),
                    ])
                ], className="mb-4"),
                html.Div(id="cost-sweep-results"),
            ], width=12)
        ])
    ], fluid=True)

def create_grid_eval_page():
    """Create the grid evaluation page."""
    grid_configs = sorted([f.name for f in CONFIGS_DIR.glob("*eval_grid*.yaml")])
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("Grid Evaluation", className="mb-4"),
                html.P("Evaluate model across multiple universes, periods, and feature sets.", className="text-muted mb-4"),
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Grid Configuration", className="card-title mb-3"),
                        dbc.Label("Grid Config File", html_for="grid-config-select"),
                        dcc.Dropdown(id="grid-config-select", options=[{"label": f, "value": f} for f in grid_configs],
                                    placeholder="Select grid config..."),
                    ])
                ], className="mb-4"),
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Actions", className="card-title mb-3"),
                        dbc.Button("Run Grid Evaluation", id="run-grid-button", color="primary", size="lg"),
                        html.Div(id="grid-status", className="mt-4"),
                        dcc.Interval(id="grid-interval", interval=2000, n_intervals=0, disabled=True),
                    ])
                ], className="mb-4"),
                html.Div(id="grid-results"),
            ], width=12)
        ])
    ], fluid=True)

def create_hpo_page():
    """Create the hyperparameter optimization page."""
    config_files = sorted([f.name for f in CONFIGS_DIR.glob("*.yaml") if "eval_grid" not in f.name.lower()])
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("Hyperparameter Optimization", className="mb-4"),
                html.P("Use Optuna to find optimal hyperparameters for your model.", className="text-muted mb-4"),
                dbc.Alert([html.Strong("⚠️ Time Warning: "), "HPO can take 10-15 hours for 30 trials. Each trial trains a full model."],
                         color="warning", className="mb-4"),
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Configuration", className="card-title mb-3"),
                        dbc.Label("Base Config File", html_for="hpo-config-select"),
                        dcc.Dropdown(id="hpo-config-select", options=[{"label": f, "value": f} for f in config_files],
                                    placeholder="Select base config..."),
                        dbc.Label("Number of Trials", html_for="hpo-n-trials", className="mt-3"),
                        dbc.Input(id="hpo-n-trials", type="number", value=30, min=1, max=100),
                        dbc.Label("Study Name", html_for="hpo-study-name", className="mt-3"),
                        dbc.Input(id="hpo-study-name", type="text", value="ppo_hpo", placeholder="ppo_hpo"),
                        dbc.Checkbox(id="hpo-resume", label="Resume from existing study", className="mt-3"),
                    ])
                ], className="mb-4"),
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Actions", className="card-title mb-3"),
                        dbc.Button("Start HPO", id="start-hpo-button", color="primary", size="lg"),
                        html.Div(id="hpo-status", className="mt-4"),
                        dcc.Interval(id="hpo-interval", interval=5000, n_intervals=0, disabled=True),
                    ])
                ], className="mb-4"),
                html.Div(id="hpo-results"),
            ], width=12)
        ])
    ], fluid=True)

def create_reports_page():
    """Create the reports generation page."""
    runs = []
    if ARTIFACTS_DIR.exists():
        runs = sorted([r.name for r in ARTIFACTS_DIR.glob("*__*")], reverse=True)
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("Generate Reports", className="mb-4"),
                html.P("Create visualizations and performance reports.", className="text-muted mb-4"),
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Configuration", className="card-title mb-3"),
                        dbc.Label("Training Run (optional)", html_for="report-run-select"),
                        dcc.Dropdown(id="report-run-select", options=[{"label": r, "value": r} for r in runs],
                                    placeholder="Select run (or leave empty for latest)...", clearable=True),
                    ])
                ], className="mb-4"),
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Actions", className="card-title mb-3"),
                        dbc.Button("Generate Reports", id="generate-reports-button", color="primary", size="lg"),
                        html.Div(id="report-status", className="mt-4"),
                        dcc.Interval(id="report-interval", interval=1000, n_intervals=0, disabled=True),
                    ])
                ], className="mb-4"),
                html.Div(id="report-results"),
            ], width=12)
        ])
    ], fluid=True)

# Training Page

def create_training_page():
    """Create the training interface page."""
    # Get available config files, but filter out evaluation grid configs
    all_configs = sorted([f.name for f in CONFIGS_DIR.glob("*.yaml") if f.is_file()])
    
    # Filter out evaluation grid configs (they don't have run_name, seed, etc.)
    config_files = [f for f in all_configs if "eval_grid" not in f.lower()]
    
    if not config_files:
        # Fallback: show all configs but warn user
        config_files = all_configs
    
    # Prefer production configs over test configs
    production_configs = [f for f in config_files if "test" not in f.lower() and "quick" not in f.lower() and "eval" not in f.lower()]
    default_config = production_configs[0] if production_configs else (config_files[0] if config_files else None)
    
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("Train RL Model", className="mb-4"),
                html.P("Select a configuration file and start training your portfolio optimization model.",
                       className="text-muted mb-4"),
                
                # Warning about quick configs
                dbc.Alert([
                    html.Strong("⚠️ Training Time Warning: "),
                    "Quick test configs (e.g., test_quick.yaml) use only 10,000 timesteps and will produce poor results. ",
                    "For production-quality models, use configs like 'rlppo_sharpe_optimized.yaml' (500k+ timesteps). ",
                    "Training can take 30-60 minutes for full configs."
                ], color="warning", className="mb-4"),
                
                # Config selection
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Configuration", className="card-title mb-3"),
                        dbc.Label("Config File", html_for="config-select"),
                        dcc.Dropdown(
                            id="config-select",
                            options=[{"label": f, "value": f} for f in config_files],
                            value=default_config,
                            placeholder="Select a config file..."
                        ),
                        html.Div(id="config-preview", className="mt-3"),
                    ])
                ], className="mb-4"),
                
                # Training controls
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Training Controls", className="card-title mb-3"),
                        dbc.Button(
                            "Start Training",
                            id="train-button",
                            color="primary",
                            size="lg",
                            className="me-2",
                            disabled=False
                        ),
                        dbc.Button(
                            "Stop Training",
                            id="stop-button",
                            color="danger",
                            size="lg",
                            disabled=True
                        ),
                        html.Div(id="training-status", className="mt-4"),
                        dcc.Interval(
                            id="training-interval",
                            interval=2000,  # Update every 2 seconds
                            n_intervals=0,
                            disabled=True
                        ),
                    ])
                ], className="mb-4"),
                
                # Training progress
                html.Div(id="training-progress"),
                
            ], width=12)
        ])
    ], fluid=True)

# Performance Page

def create_performance_page():
    """Create the performance visualization page."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("Performance Analysis", className="mb-4"),
                html.P("View and compare model performance against baseline strategies.",
                       className="text-muted mb-4"),
                
                # Run selection
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Select Training Run", className="card-title mb-3"),
                        dcc.Dropdown(
                            id="run-select",
                            placeholder="Select a training run...",
                            clearable=True
                        ),
                        html.Div(id="run-info", className="mt-3"),
                    ])
                ], className="mb-4"),
                
                # Metrics cards
                html.Div(id="metrics-cards", className="mb-4"),
                
                # Charts
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Equity Curves", className="card-title"),
                                dcc.Graph(id="equity-curves-chart")
                            ])
                        ])
                    ], width=12, className="mb-4"),
                ]),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Metrics Comparison", className="card-title"),
                                dcc.Graph(id="metrics-chart")
                            ])
                        ])
                    ], width=6, className="mb-4"),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Allocation Heatmap", className="card-title"),
                                dcc.Graph(id="allocation-heatmap")
                            ])
                        ])
                    ], width=6, className="mb-4"),
                ]),
                
            ], width=12)
        ])
    ], fluid=True)

# Main App Layout

app.layout = dbc.Container([
    dcc.Location(id="url", refresh=False),
    create_navbar(),
    dbc.Row([
        dbc.Col([
            create_sidebar()
        ], width=2),
        dbc.Col([
            html.Div(id="page-content")
        ], width=10)
    ]),
    # Store for task statuses
    dcc.Store(id="task-store", data=task_status),
    dcc.Store(id="training-store", data=training_status),  # Backward compatibility
], fluid=True)

# Callbacks

@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    """Route to different pages."""
    if pathname == "/data":
        return create_data_page()
    elif pathname == "/features":
        return create_features_page()
    elif pathname == "/train":
        return create_training_page()
    elif pathname == "/evaluate":
        return create_evaluation_page()
    elif pathname == "/costs":
        return create_cost_analysis_page()
    elif pathname == "/grid":
        return create_grid_eval_page()
    elif pathname == "/hpo":
        return create_hpo_page()
    elif pathname == "/reports":
        return create_reports_page()
    elif pathname == "/performance":
        return create_performance_page()
    else:
        return create_dashboard_page()

@app.callback(
    Output("config-preview", "children"),
    Input("config-select", "value")
)
def update_config_preview(config_file):
    """Show preview of selected config."""
    if not config_file:
        return html.Div("No config selected")
    
    try:
        # Check if this is a training config (has run_name)
        import yaml
        config_path = CONFIGS_DIR / config_file
        with open(config_path) as f:
            raw_config = yaml.safe_load(f)
        
        if "run_name" not in raw_config:
            return dbc.Alert([
                html.Strong("⚠️ Invalid Training Config: "),
                f"This config file ({config_file}) is not a training configuration. ",
                "It appears to be an evaluation grid config. Please select a training config like ",
                html.Code("rlppo_sharpe_optimized.yaml"), " or ", html.Code("rlppo_improved.yaml"), "."
            ], color="danger", className="mb-2")
        
        cfg = load_config(config_path)
        
        # Estimate training time (rough: ~1000 timesteps per second on average hardware)
        timesteps = cfg.train.total_timesteps
        est_minutes = max(1, int(timesteps / 1000 / 60))
        
        # Warning for quick configs
        warning = None
        if timesteps < 50000:
            warning = dbc.Alert([
                html.Strong("⚠️ Low Training Steps: "),
                f"This config only uses {timesteps:,} timesteps. ",
                "For good performance, use at least 200,000+ timesteps (recommended: 500,000+)."
            ], color="danger", className="mb-2")
        elif timesteps < 200000:
            warning = dbc.Alert([
                html.Strong("ℹ️ Moderate Training Steps: "),
                f"This config uses {timesteps:,} timesteps. ",
                "For best performance, consider using 500,000+ timesteps."
            ], color="warning", className="mb-2")
        
        return html.Div([
            html.H6("Configuration Preview", className="mb-2"),
            warning,
            html.Pre(json.dumps({
                "run_name": cfg.run_name,
                "seed": cfg.seed,
                "total_timesteps": f"{timesteps:,}",
                "estimated_time": f"~{est_minutes} minutes",
                "learning_rate": cfg.ppo.learning_rate,
                "gamma": cfg.ppo.gamma,
                "batch_size": cfg.ppo.batch_size,
            }, indent=2), className="bg-light p-3 rounded")
        ])
    except Exception as e:
        return dbc.Alert([
            html.Strong("Error loading config: "),
            str(e)
        ], color="danger", className="mb-2")

@app.callback(
    [Output("run-select", "options"),
     Output("eval-run-select", "options"),
     Output("cost-run-select", "options"),
     Output("report-run-select", "options")],
    Input("url", "pathname")
)
def update_run_options(pathname):
    """Update available training runs for all dropdowns."""
    runs = sorted(ARTIFACTS_DIR.glob("*__*"), key=os.path.getmtime, reverse=True)
    options = [{"label": f"{r.name} ({datetime.fromtimestamp(r.stat().st_mtime).strftime('%Y-%m-%d %H:%M')})",
                "value": r.name} for r in runs[:50]]  # Show last 50 runs
    return options, options, options, options

def train_model_async(config_file):
    """Train model in background thread."""
    global training_status
    import sys
    import io
    from datetime import datetime
    
    # Use a Tee-like approach: capture AND print to console
    class TeeOutput:
        def __init__(self, original, capture):
            self.original = original
            self.capture = capture
        
        def write(self, text):
            self.original.write(text)  # Print to console
            self.capture.write(text)   # Capture for app display
            self.original.flush()
        
        def flush(self):
            self.original.flush()
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    try:
        training_status["running"] = True
        training_status["message"] = "Loading configuration..."
        training_status["progress"] = 0
        
        cfg = load_config(CONFIGS_DIR / config_file)
        total_steps = cfg.train.total_timesteps
        
        # Print to console AND capture
        print(f"\n[APP {datetime.now().strftime('%H:%M:%S')}] Starting training: {cfg.run_name}")
        print(f"[APP] Total timesteps: {total_steps:,}")
        print(f"[APP] Estimated time: ~{int(total_steps/1000/60)} minutes")
        print(f"[APP] Training output will appear below...\n")
        
        training_status["message"] = f"Starting training: {cfg.run_name} ({total_steps:,} timesteps)..."
        training_status["progress"] = 5
        
        # Tee output: both console and capture
        sys.stdout = TeeOutput(old_stdout, stdout_capture)
        sys.stderr = TeeOutput(old_stderr, stderr_capture)
        
        import time
        start_time = time.time()
        
        # Actually run training
        run_dir = train_once(cfg)
        
        # Restore output
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        # Get captured output
        stdout_text = stdout_capture.getvalue()
        stderr_text = stderr_capture.getvalue()
        
        elapsed = time.time() - start_time
        
        print(f"\n[APP {datetime.now().strftime('%H:%M:%S')}] Training completed in {int(elapsed/60)}m {int(elapsed%60)}s")
        print(f"[APP] Run directory: {run_dir}\n")
        
        # Check if training actually completed
        if elapsed < 60 and total_steps > 100000:
            # Suspiciously fast - likely an error
            error_msg = f"Training completed too quickly ({int(elapsed)}s) for {total_steps:,} timesteps!\n"
            error_msg += f"This suggests an error occurred.\n\n"
            error_msg += f"Stdout:\n{stdout_text[-1000:]}\n\n"
            error_msg += f"Stderr:\n{stderr_text[-1000:]}"
            training_status["message"] = error_msg
            training_status["progress"] = 0
        else:
            # Verify metrics file exists
            metrics_path = Path(run_dir) / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)
                timesteps_actual = metrics.get("timesteps", "N/A")
                training_status["run_dir"] = str(run_dir)
                training_status["message"] = f"Training complete! Run saved to: {run_dir.name}\n"
                training_status["message"] += f"Timesteps: {timesteps_actual:,}\n"
                training_status["message"] += f"Time taken: {int(elapsed/60)}m {int(elapsed%60)}s\n\n"
                training_status["message"] += f"Last output:\n{stdout_text[-500:]}"
            else:
                training_status["run_dir"] = str(run_dir)
                training_status["message"] = f"Training complete! Run saved to: {run_dir.name} (took {int(elapsed/60)}m {int(elapsed%60)}s)\n"
                training_status["message"] += f"Warning: metrics.json not found at {metrics_path}\n\n"
                training_status["message"] += f"Last output:\n{stdout_text[-500:]}"
            training_status["progress"] = 100
        
    except Exception as e:
        # Restore output
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        import traceback
        error_details = traceback.format_exc()
        stdout_text = stdout_capture.getvalue()
        stderr_text = stderr_capture.getvalue()
        
        print(f"\n[APP ERROR {datetime.now().strftime('%H:%M:%S')}] Training failed: {str(e)}")
        print(f"See traceback above for details.\n")
        
        error_msg = f"Error: {str(e)}\n\n"
        error_msg += f"Traceback:\n{error_details}\n\n"
        error_msg += f"Stdout:\n{stdout_text[-1000:]}\n\n"
        error_msg += f"Stderr:\n{stderr_text[-1000:]}"
        
        training_status["message"] = error_msg
        training_status["progress"] = 0
    finally:
        training_status["running"] = False
        # Ensure output is restored
        sys.stdout = old_stdout
        sys.stderr = old_stderr

@app.callback(
    [Output("train-button", "disabled"),
     Output("stop-button", "disabled"),
     Output("training-interval", "disabled"),
     Output("training-status", "children"),
     Output("training-progress", "children"),
     Output("training-store", "data")],
    [Input("train-button", "n_clicks"),
     Input("stop-button", "n_clicks"),
     Input("training-interval", "n_intervals")],
    [State("config-select", "value"),
     State("training-store", "data")],
    prevent_initial_call=True  # Prevent callback from firing on page load
)
def handle_training(train_clicks, stop_clicks, n_intervals, config_file, store_data):
    """Handle training start/stop and progress updates."""
    ctx = callback_context
    if not ctx.triggered:
        return False, True, True, "", "", training_status
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    # Only start training if train-button was clicked (n_clicks > 0) AND training is not already running
    if trigger_id == "train-button" and train_clicks and train_clicks > 0 and config_file and not training_status["running"]:
        # Validate config file before starting
        try:
            import yaml
            config_path = CONFIGS_DIR / config_file
            with open(config_path) as f:
                raw_config = yaml.safe_load(f)
            
            if "run_name" not in raw_config:
                error_msg = f"Cannot train with {config_file}: This is not a training config file. " \
                           f"Please select a training config (e.g., rlppo_sharpe_optimized.yaml)"
                return False, True, True, dbc.Alert(error_msg, color="danger"), "", training_status
            
            # Start training in background thread
            thread = threading.Thread(target=train_model_async, args=(config_file,))
            thread.daemon = True
            thread.start()
            return True, False, False, "Training started...", "", training_status
        except Exception as e:
            error_msg = f"Error validating config: {str(e)}"
            return False, True, True, dbc.Alert(error_msg, color="danger"), "", training_status
    
    # If train-button was clicked but training is already running, do nothing
    elif trigger_id == "train-button":
        # Training already running, don't start another
        return True, False, False, training_status.get("message", "Training already in progress..."), "", training_status
    
    elif trigger_id == "stop-button":
        training_status["running"] = False
        training_status["message"] = "Training stopped by user"
        return False, True, True, "Training stopped", "", training_status
    
    elif trigger_id == "training-interval":
        # Update progress - this should ONLY update the display, not start training
        status_msg = training_status.get("message", "")
        progress = training_status.get("progress", 0)
        
        if training_status["running"]:
            # Show estimated progress with time info
            progress_bar = dbc.Progress(
                value=progress,
                label=f"{progress}%",
                striped=True,
                animated=True,
                className="mb-2"
            )
            
            # Show status message with potential error details
            status_display = html.Div([
                html.P(status_msg, className="mb-2", style={"white-space": "pre-wrap"}),
                html.P("Training in progress... This may take 30-90 minutes for 5M timesteps.", 
                      className="text-muted small mb-2"),
                progress_bar,
                html.P("Note: If training completes in under 5 minutes, there may be an error. Check the message above.",
                      className="text-warning small mt-2")
            ])
            
            # Return current state - don't change button states or start new training
            return (True, False, False, status_display, "", training_status)
        else:
            if training_status.get("run_dir"):
                return (False, True, True,
                       html.Div([
                           dbc.Alert("Training complete!", color="success", className="mb-2"),
                           html.P(f"Run directory: {training_status['run_dir']}", className="text-muted")
                       ]), "", training_status)
            return False, True, True, status_msg, "", training_status
    
    return False, True, True, "", "", training_status

# Data Acquisition Callbacks

def download_data_async(symbols_str, force):
    """Download data in background thread."""
    global task_status
    task_status["data"]["running"] = True
    task_status["data"]["message"] = "Starting download..."
    task_status["data"]["progress"] = 0
    try:
        symbols = [s.strip().upper() for s in symbols_str.split(",") if s.strip()]
        task_status["data"]["message"] = f"Downloading {len(symbols)} symbols..."
        task_status["data"]["progress"] = 10
        paths = download_universe(symbols, force=force)
        task_status["data"]["message"] = f"Downloaded {len(paths)} files successfully!"
        task_status["data"]["progress"] = 50
        task_status["data"]["message"] += f"\nFiles: {', '.join([p.name for p in paths])}"
        task_status["data"]["progress"] = 100
    except Exception as e:
        task_status["data"]["message"] = f"Error: {str(e)}"
        task_status["data"]["progress"] = 0
    finally:
        task_status["data"]["running"] = False

def process_data_async(symbols_str):
    """Process and merge data in background thread."""
    global task_status
    task_status["data"]["running"] = True
    task_status["data"]["message"] = "Processing data..."
    task_status["data"]["progress"] = 0
    try:
        symbols = [s.strip().upper() for s in symbols_str.split(",") if s.strip()]
        task_status["data"]["progress"] = 30
        df = load_and_merge(symbols)
        task_status["data"]["message"] = f"Processed {len(df)} rows, {len(df.columns)} assets"
        task_status["data"]["progress"] = 100
    except Exception as e:
        task_status["data"]["message"] = f"Error: {str(e)}"
        task_status["data"]["progress"] = 0
    finally:
        task_status["data"]["running"] = False

@app.callback(
    [Output("data-status", "children"),
     Output("data-interval", "disabled"),
     Output("data-info", "children")],
    [Input("download-button", "n_clicks"),
     Input("process-button", "n_clicks"),
     Input("data-interval", "n_intervals")],
    [State("symbols-input", "value"),
     State("force-download", "value")]
)
def handle_data_ops(download_clicks, process_clicks, n_intervals, symbols, force):
    """Handle data download and processing."""
    ctx = callback_context
    if not ctx.triggered:
        prices_exist = (PROC_DIR / "prices_adj.parquet").exists()
        info = dbc.Alert(f"Prices file exists: {prices_exist}", color="success" if prices_exist else "warning") if prices_exist else ""
        return "", True, info
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if trigger_id == "download-button" and download_clicks and not task_status["data"]["running"]:
        thread = threading.Thread(target=download_data_async, args=(symbols or "SPY,QQQ,TLT", bool(force)))
        thread.daemon = True
        thread.start()
        return "Downloading...", False, ""
    elif trigger_id == "process-button" and process_clicks and not task_status["data"]["running"]:
        thread = threading.Thread(target=process_data_async, args=(symbols or "SPY,QQQ,TLT",))
        thread.daemon = True
        thread.start()
        return "Processing...", False, ""
    elif trigger_id == "data-interval":
        status = task_status["data"]
        if status["running"]:
            return dbc.Alert(status["message"], color="info"), False, ""
        elif status["progress"] == 100:
            prices_exist = (PROC_DIR / "prices_adj.parquet").exists()
            info = dbc.Alert(f"✅ {status['message']}\nPrices file exists: {prices_exist}", color="success") if prices_exist else dbc.Alert(status["message"], color="info")
            return dbc.Alert(status["message"], color="success"), True, info
        else:
            return status["message"] or "", not status["running"], ""
    
    return "", True, ""

# Features Callbacks

def generate_features_async(lookbacks_str):
    """Generate features in background thread."""
    global task_status
    task_status["features"]["running"] = True
    task_status["features"]["message"] = "Generating features..."
    task_status["features"]["progress"] = 0
    try:
        lookbacks = [int(x.strip()) for x in lookbacks_str.split(",") if x.strip()]
        task_status["features"]["progress"] = 20
        prices_path = PROC_DIR / "prices_adj.parquet"
        if not prices_path.exists():
            raise FileNotFoundError("Prices file not found. Please download data first.")
        prices = pd.read_parquet(prices_path)
        task_status["features"]["progress"] = 40
        features = make_features(prices, lookbacks=lookbacks)
        task_status["features"]["message"] = f"Generated {len(features)} rows, {len(features.columns)} features"
        task_status["features"]["progress"] = 100
    except Exception as e:
        task_status["features"]["message"] = f"Error: {str(e)}"
        task_status["features"]["progress"] = 0
    finally:
        task_status["features"]["running"] = False

@app.callback(
    [Output("features-status", "children"),
     Output("features-interval", "disabled"),
     Output("features-info", "children")],
    [Input("generate-features-button", "n_clicks"),
     Input("features-interval", "n_intervals")],
    [State("lookbacks-input", "value")]
)
def handle_features(n_clicks, n_intervals, lookbacks):
    """Handle feature generation."""
    ctx = callback_context
    if not ctx.triggered:
        features_exist = (PROC_DIR / "features.parquet").exists()
        info = dbc.Alert(f"Features file exists: {features_exist}", color="success" if features_exist else "warning") if features_exist else ""
        return "", True, info
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if trigger_id == "generate-features-button" and n_clicks and not task_status["features"]["running"]:
        thread = threading.Thread(target=generate_features_async, args=(lookbacks or "5,20",))
        thread.daemon = True
        thread.start()
        return "Generating features...", False, ""
    elif trigger_id == "features-interval":
        status = task_status["features"]
        if status["running"]:
            return dbc.Alert(status["message"], color="info"), False, ""
        elif status["progress"] == 100:
            features_exist = (PROC_DIR / "features.parquet").exists()
            info = dbc.Alert(f"✅ {status['message']}\nFeatures file exists: {features_exist}", color="success") if features_exist else dbc.Alert(status["message"], color="info")
            return dbc.Alert(status["message"], color="success"), True, info
        else:
            return status["message"] or "", not status["running"], ""
    
    return "", True, ""

# Evaluation/Backtest Callbacks

def run_backtest_async(run_name):
    """Run backtest in background thread."""
    global task_status
    task_status["backtest"]["running"] = True
    task_status["backtest"]["message"] = "Running backtest..."
    task_status["backtest"]["progress"] = 0
    try:
        run_dir = ARTIFACTS_DIR / run_name
        task_status["backtest"]["progress"] = 20
        
        # Import and run backtest
        import subprocess
        result = subprocess.run(
            ["python", "-m", "src.backtest", "--run_dir", str(run_dir)],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            task_status["backtest"]["message"] = f"✅ Backtest complete!\n{result.stdout[-500:]}"
            task_status["backtest"]["progress"] = 100
        else:
            task_status["backtest"]["message"] = f"Error: {result.stderr[-500:]}"
            task_status["backtest"]["progress"] = 0
    except Exception as e:
        task_status["backtest"]["message"] = f"Error: {str(e)}"
        task_status["backtest"]["progress"] = 0
    finally:
        task_status["backtest"]["running"] = False

@app.callback(
    [Output("backtest-status", "children"),
     Output("backtest-interval", "disabled"),
     Output("backtest-results", "children")],
    [Input("run-backtest-button", "n_clicks"),
     Input("backtest-interval", "n_intervals")],
    [State("eval-run-select", "value")]
)
def handle_backtest(n_clicks, n_intervals, run_name):
    """Handle backtest execution."""
    ctx = callback_context
    if not ctx.triggered:
        return "", True, ""
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if trigger_id == "run-backtest-button" and n_clicks and run_name and not task_status["backtest"]["running"]:
        thread = threading.Thread(target=run_backtest_async, args=(run_name,))
        thread.daemon = True
        thread.start()
        return "Running backtest...", False, ""
    elif trigger_id == "backtest-interval":
        status = task_status["backtest"]
        if status["running"]:
            return dbc.Alert(status["message"], color="info"), False, ""
        elif status["progress"] == 100:
            # Try to load results
            results_path = ARTIFACTS_DIR / "baseline_results.csv"
            if results_path.exists():
                df = pd.read_csv(results_path)
                table = dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True, className="mt-3")
                return dbc.Alert(status["message"], color="success"), True, table
            return dbc.Alert(status["message"], color="success"), True, ""
        else:
            return status["message"] or "", not status["running"], ""
    
    return "", True, ""

# Cost Analysis Callbacks

def run_cost_sweep_async(run_name, config_file):
    """Run cost sweep in background thread."""
    global task_status
    task_status["cost_sweep"]["running"] = True
    task_status["cost_sweep"]["message"] = "Running cost sweep..."
    task_status["cost_sweep"]["progress"] = 0
    try:
        run_dir = ARTIFACTS_DIR / run_name
        config_path = CONFIGS_DIR / config_file
        out_csv = run_dir / "eval" / "cost_sweep.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        
        task_status["cost_sweep"]["progress"] = 20
        
        import subprocess
        result = subprocess.run(
            ["python", "-m", "src.sweep_costs", "--config", str(config_path), "--run_dir", str(run_dir), "--out_csv", str(out_csv)],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode == 0:
            task_status["cost_sweep"]["message"] = f"✅ Cost sweep complete!"
            task_status["cost_sweep"]["progress"] = 100
        else:
            task_status["cost_sweep"]["message"] = f"Error: {result.stderr[-500:]}"
            task_status["cost_sweep"]["progress"] = 0
    except Exception as e:
        task_status["cost_sweep"]["message"] = f"Error: {str(e)}"
        task_status["cost_sweep"]["progress"] = 0
    finally:
        task_status["cost_sweep"]["running"] = False

@app.callback(
    [Output("cost-sweep-status", "children"),
     Output("cost-sweep-interval", "disabled"),
     Output("cost-sweep-results", "children")],
    [Input("run-cost-sweep-button", "n_clicks"),
     Input("cost-sweep-interval", "n_intervals")],
    [State("cost-run-select", "value"),
     State("cost-config-select", "value")]
)
def handle_cost_sweep(n_clicks, n_intervals, run_name, config_file):
    """Handle cost sweep execution."""
    ctx = callback_context
    if not ctx.triggered:
        return "", True, ""
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if trigger_id == "run-cost-sweep-button" and n_clicks and run_name and config_file and not task_status["cost_sweep"]["running"]:
        thread = threading.Thread(target=run_cost_sweep_async, args=(run_name, config_file))
        thread.daemon = True
        thread.start()
        return "Running cost sweep...", False, ""
    elif trigger_id == "cost-sweep-interval":
        status = task_status["cost_sweep"]
        if status["running"]:
            return dbc.Alert(status["message"], color="info"), False, ""
        elif status["progress"] == 100:
            # Try to load results
            if run_name:
                results_path = ARTIFACTS_DIR / run_name / "eval" / "cost_sweep.csv"
                if results_path.exists():
                    df = pd.read_csv(results_path)
                    fig = px.line(df, x="base_bps", y="sharpe", title="Sharpe Ratio vs Transaction Cost")
                    return dbc.Alert(status["message"], color="success"), True, dcc.Graph(figure=fig)
            return dbc.Alert(status["message"], color="success"), True, ""
        else:
            return status["message"] or "", not status["running"], ""
    
    return "", True, ""

# Grid Evaluation Callbacks

def run_grid_eval_async(grid_config):
    """Run grid evaluation in background thread."""
    global task_status
    task_status["grid_eval"]["running"] = True
    task_status["grid_eval"]["message"] = "Running grid evaluation..."
    task_status["grid_eval"]["progress"] = 0
    try:
        grid_path = CONFIGS_DIR / grid_config
        task_status["grid_eval"]["progress"] = 20
        
        import subprocess
        result = subprocess.run(
            ["python", "-m", "src.eval_grid", "--grid", str(grid_path)],
            capture_output=True,
            text=True,
            timeout=3600
        )
        
        if result.returncode == 0:
            task_status["grid_eval"]["message"] = f"✅ Grid evaluation complete!"
            task_status["grid_eval"]["progress"] = 100
        else:
            task_status["grid_eval"]["message"] = f"Error: {result.stderr[-500:]}"
            task_status["grid_eval"]["progress"] = 0
    except Exception as e:
        task_status["grid_eval"]["message"] = f"Error: {str(e)}"
        task_status["grid_eval"]["progress"] = 0
    finally:
        task_status["grid_eval"]["running"] = False

@app.callback(
    [Output("grid-status", "children"),
     Output("grid-interval", "disabled"),
     Output("grid-results", "children")],
    [Input("run-grid-button", "n_clicks"),
     Input("grid-interval", "n_intervals")],
    [State("grid-config-select", "value")]
)
def handle_grid_eval(n_clicks, n_intervals, grid_config):
    """Handle grid evaluation execution."""
    ctx = callback_context
    if not ctx.triggered:
        return "", True, ""
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if trigger_id == "run-grid-button" and n_clicks and grid_config and not task_status["grid_eval"]["running"]:
        thread = threading.Thread(target=run_grid_eval_async, args=(grid_config,))
        thread.daemon = True
        thread.start()
        return "Running grid evaluation...", False, ""
    elif trigger_id == "grid-interval":
        status = task_status["grid_eval"]
        if status["running"]:
            return dbc.Alert(status["message"], color="info"), False, ""
        elif status["progress"] == 100:
            return dbc.Alert(status["message"], color="success"), True, ""
        else:
            return status["message"] or "", not status["running"], ""
    
    return "", True, ""

# HPO Callbacks

def run_hpo_async(config_file, n_trials, study_name, resume):
    """Run HPO in background thread."""
    global task_status
    task_status["hpo"]["running"] = True
    task_status["hpo"]["message"] = f"Starting HPO with {n_trials} trials..."
    task_status["hpo"]["progress"] = 0
    try:
        config_path = CONFIGS_DIR / config_file
        output_dir = ARTIFACTS_DIR / "hpo"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        task_status["hpo"]["progress"] = 5
        
        import subprocess
        cmd = ["python", "-m", "src.hpo.run_optuna", "--config", str(config_path),
               "--n-trials", str(n_trials), "--study-name", study_name, "--output-dir", str(output_dir)]
        if resume:
            cmd.append("--resume")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=86400)  # 24 hour timeout
        
        if result.returncode == 0:
            task_status["hpo"]["message"] = f"✅ HPO complete! Best params: {result.stdout[-500:]}"
            task_status["hpo"]["progress"] = 100
        else:
            task_status["hpo"]["message"] = f"Error: {result.stderr[-500:]}"
            task_status["hpo"]["progress"] = 0
    except Exception as e:
        task_status["hpo"]["message"] = f"Error: {str(e)}"
        task_status["hpo"]["progress"] = 0
    finally:
        task_status["hpo"]["running"] = False

@app.callback(
    [Output("hpo-status", "children"),
     Output("hpo-interval", "disabled"),
     Output("hpo-results", "children")],
    [Input("start-hpo-button", "n_clicks"),
     Input("hpo-interval", "n_intervals")],
    [State("hpo-config-select", "value"),
     State("hpo-n-trials", "value"),
     State("hpo-study-name", "value"),
     State("hpo-resume", "value")]
)
def handle_hpo(n_clicks, n_intervals, config_file, n_trials, study_name, resume):
    """Handle HPO execution."""
    ctx = callback_context
    if not ctx.triggered:
        return "", True, ""
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if trigger_id == "start-hpo-button" and n_clicks and config_file and not task_status["hpo"]["running"]:
        thread = threading.Thread(target=run_hpo_async, args=(config_file, n_trials or 30, study_name or "ppo_hpo", bool(resume)))
        thread.daemon = True
        thread.start()
        return "Starting HPO...", False, ""
    elif trigger_id == "hpo-interval":
        status = task_status["hpo"]
        if status["running"]:
            return dbc.Alert(status["message"], color="info"), False, ""
        elif status["progress"] == 100:
            # Try to load summary
            summary_path = ARTIFACTS_DIR / "hpo" / f"{study_name or 'ppo_hpo'}_summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    summary = json.load(f)
                return dbc.Alert(status["message"], color="success"), True, html.Pre(json.dumps(summary, indent=2))
            return dbc.Alert(status["message"], color="success"), True, ""
        else:
            return status["message"] or "", not status["running"], ""
    
    return "", True, ""

# Reports Callbacks

def generate_reports_async(run_name):
    """Generate reports in background thread."""
    global task_status
    task_status["report"]["running"] = True
    task_status["report"]["message"] = "Generating reports..."
    task_status["report"]["progress"] = 0
    try:
        task_status["report"]["progress"] = 20
        run_dir = ARTIFACTS_DIR / run_name if run_name else None
        generate_report(run_dir)
        task_status["report"]["message"] = "✅ Reports generated successfully!"
        task_status["report"]["progress"] = 100
    except Exception as e:
        task_status["report"]["message"] = f"Error: {str(e)}"
        task_status["report"]["progress"] = 0
    finally:
        task_status["report"]["running"] = False

@app.callback(
    [Output("report-status", "children"),
     Output("report-interval", "disabled"),
     Output("report-results", "children")],
    [Input("generate-reports-button", "n_clicks"),
     Input("report-interval", "n_intervals")],
    [State("report-run-select", "value")]
)
def handle_reports(n_clicks, n_intervals, run_name):
    """Handle report generation."""
    ctx = callback_context
    if not ctx.triggered:
        return "", True, ""
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if trigger_id == "generate-reports-button" and n_clicks and not task_status["report"]["running"]:
        thread = threading.Thread(target=generate_reports_async, args=(run_name,))
        thread.daemon = True
        thread.start()
        return "Generating reports...", False, ""
    elif trigger_id == "report-interval":
        status = task_status["report"]
        if status["running"]:
            return dbc.Alert(status["message"], color="info"), False, ""
        elif status["progress"] == 100:
            # Show report files
            report_files = list(REPORTS_DIR.glob("*.png"))
            if report_files:
                images = [html.Img(src=f"/assets/{f.name}", style={"width": "100%", "margin": "10px"}) for f in report_files]
                return dbc.Alert(status["message"], color="success"), True, html.Div(images)
            return dbc.Alert(status["message"], color="success"), True, ""
        else:
            return status["message"] or "", not status["running"], ""
    
    return "", True, ""

# Performance Callbacks (existing)

@app.callback(
    [Output("run-info", "children"),
     Output("metrics-cards", "children"),
     Output("equity-curves-chart", "figure"),
     Output("metrics-chart", "figure"),
     Output("allocation-heatmap", "figure")],
    Input("run-select", "value")
)
def update_performance(run_dir):
    """Update performance visualizations based on selected run."""
    if not run_dir or not Path(run_dir).exists():
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="Select a training run to view performance",
                                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return "", "", empty_fig, empty_fig, empty_fig
    
    run_path = Path(run_dir)
    
    # Load metrics
    metrics = {}
    metrics_path = run_path / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    
    # Run info
    timesteps = metrics.get('timesteps', 'N/A')
    if timesteps != 'N/A' and isinstance(timesteps, (int, float)):
        timesteps = f"{int(timesteps):,}"
    
    run_info = html.Div([
        html.H6("Run Information", className="mb-2"),
        html.P(f"Run: {run_path.name}", className="mb-1"),
        html.P(f"Timesteps: {timesteps}", className="mb-1 text-muted"),
        html.P(f"Test Sharpe: {metrics.get('results', {}).get('sharpe', 'N/A'):.3f}" if metrics.get('results', {}).get('sharpe') is not None else "Test Sharpe: N/A", className="mb-1 text-muted"),
    ])
    
    # Load data
    try:
        prices_path = Path("data/processed/prices_adj.parquet")
        features_path = Path("data/processed/features.parquet")
        
        if not prices_path.exists() or not features_path.exists():
            raise FileNotFoundError("Data files not found. Please run data preparation first.")
        
        prices = pd.read_parquet(prices_path)
        features = pd.read_parquet(features_path)
        prices, features = align_after_load(prices, features)
        
        # Baselines
        bh = buy_and_hold(prices, features)
        ew = equal_weight(prices, features)
        
        # RL Agent
        rl = None
        scaler_path = run_path / "scaler.json"
        model_path = run_path / "models" / "best" / "best_model.zip"
        
        if not model_path.exists():
            model_zips = list(run_path.glob("models/**/*.zip"))
            if model_zips:
                model_path = model_zips[0]
        
        if scaler_path.exists() and model_path.exists():
            scaler = FitOnTrainScaler.load(scaler_path, columns=features.columns)
            scaler_cols = list(scaler.mu_.index) if hasattr(scaler, 'mu_') and scaler.mu_ is not None else list(features.columns)
            features_to_scale = features[[c for c in scaler_cols if c in features.columns]]
            features_scaled = scaler.transform(features_to_scale)
            common = prices.index.intersection(features_scaled.index)
            prices_eval = prices.loc[common]
            features_eval = features_scaled.loc[common]
            rl = evaluate_sb3_model(prices_eval, features_eval, str(model_path))
        
        # Metrics cards
        metrics_cards = dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Sharpe Ratio", className="text-muted"),
                        html.H3(f"{rl['metrics']['sharpe']:.2f}" if rl else "N/A", className="mb-0")
                    ])
                ], color="primary", outline=True)
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Final Value", className="text-muted"),
                        html.H3(f"{rl['metrics']['final_value']:.2f}x" if rl else "N/A", className="mb-0")
                    ])
                ], color="success", outline=True)
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Max Drawdown", className="text-muted"),
                        html.H3(f"{rl['metrics']['max_drawdown']:.1%}" if rl else "N/A", className="mb-0")
                    ])
                ], color="warning", outline=True)
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Turnover", className="text-muted"),
                        html.H3(f"{rl['metrics']['turnover']:.3f}" if rl else "N/A", className="mb-0")
                    ])
                ], color="info", outline=True)
            ], width=3),
        ])
        
        # Equity curves
        equity_fig = go.Figure()
        equity_fig.add_trace(go.Scatter(
            y=bh["equity_curve"],
            mode="lines",
            name="Buy & Hold",
            line=dict(color="blue")
        ))
        equity_fig.add_trace(go.Scatter(
            y=ew["equity_curve"],
            mode="lines",
            name="Equal Weight",
            line=dict(color="green")
        ))
        if rl:
            equity_fig.add_trace(go.Scatter(
                y=rl["equity_curve"],
                mode="lines",
                name="RL Agent",
                line=dict(color="red", width=2)
            ))
        equity_fig.update_layout(
            title="Equity Curves Over Time",
            xaxis_title="Time Step",
            yaxis_title="Portfolio Value",
            hovermode="x unified",
            template="plotly_white"
        )
        
        # Metrics comparison
        strategies = ["Buy & Hold", "Equal Weight"]
        sharpe_vals = [bh["metrics"]["sharpe"], ew["metrics"]["sharpe"]]
        final_vals = [bh["metrics"]["final_value"], ew["metrics"]["final_value"]]
        
        if rl:
            strategies.append("RL Agent")
            sharpe_vals.append(rl["metrics"]["sharpe"])
            final_vals.append(rl["metrics"]["final_value"])
        
        metrics_fig = go.Figure()
        metrics_fig.add_trace(go.Bar(
            x=strategies,
            y=sharpe_vals,
            name="Sharpe Ratio",
            marker_color="steelblue"
        ))
        metrics_fig.update_layout(
            title="Sharpe Ratio Comparison",
            yaxis_title="Sharpe Ratio",
            template="plotly_white",
            showlegend=False
        )
        
        # Allocation heatmap
        if rl and "weights" in rl:
            weights_array = np.array(rl["weights"])
            n_weights = weights_array.shape[1] if len(weights_array.shape) > 1 else 0
            
            # Handle case where weights might include cash or have different shape
            if n_weights > 0:
                # Get asset names - weights might include cash, so we need to handle this
                # If weights has more columns than prices, it likely includes cash
                if n_weights == len(prices.columns):
                    asset_names = list(prices.columns)
                elif n_weights == len(prices.columns) + 1:
                    # Includes cash - use prices columns + cash
                    asset_names = list(prices.columns) + ["CASH"]
                else:
                    # Fallback: use generic names
                    asset_names = [f"Asset_{i}" for i in range(n_weights)]
                
                # Ensure we have the right number of names
                if len(asset_names) != n_weights:
                    asset_names = [f"Asset_{i}" for i in range(n_weights)]
                
                weights_df = pd.DataFrame(weights_array, columns=asset_names)
                # Sample every 20th row for performance
                weights_sample = weights_df.iloc[::20]
                
                heatmap_fig = px.imshow(
                    weights_sample.T,
                    labels=dict(x="Time Step", y="Asset", color="Weight"),
                    title="Portfolio Allocation Over Time",
                    aspect="auto",
                    color_continuous_scale="Viridis"
                )
            else:
                heatmap_fig = go.Figure()
                heatmap_fig.add_annotation(
                    text="No allocation data available",
                    xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
                )
        else:
            heatmap_fig = go.Figure()
            heatmap_fig.add_annotation(
                text="No allocation data available",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
        
        return run_info, metrics_cards, equity_fig, metrics_fig, heatmap_fig
        
    except Exception as e:
        error_msg = f"Error loading performance data: {str(e)}"
        empty_fig = go.Figure()
        empty_fig.add_annotation(text=error_msg, xref="paper", yref="paper",
                                x=0.5, y=0.5, showarrow=False)
        return html.Div(error_msg, className="text-danger"), "", empty_fig, empty_fig, empty_fig

# Run App

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)

