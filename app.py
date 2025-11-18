"""
Dash App for RL Portfolio Optimizer
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
from datetime import datetime

# Import project modules
from src.config_loader import load_config
from src.train import train_once
from src.backtest import buy_and_hold, equal_weight, evaluate_sb3_model
from src.baselines import (
    CostCfg, vt_equal_weight, ivp_inverse_variance, momentum_xs, _run_backtest
)
from src.scaler import FitOnTrainScaler
from src.utils.data_utils import align_after_load

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "RL Portfolio Optimizer"

# Paths
CONFIGS_DIR = Path("configs")
ARTIFACTS_DIR = Path("artifacts")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# Global state for training
training_status = {"running": False, "progress": 0, "message": "", "run_dir": None}

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
    """Create the sidebar navigation."""
    return html.Div([
        dbc.Nav([
            dbc.NavLink([
                "📊 Performance"
            ], href="/", active="exact", id="nav-performance", className="mb-2"),
            dbc.NavLink([
                "⚙️ Train Model"
            ], href="/train", active="exact", id="nav-train", className="mb-2"),
        ], vertical=True, pills=True, className="bg-light p-3 rounded")
    ], className="sidebar")

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
    # Store for training status
    dcc.Store(id="training-store", data=training_status),
], fluid=True)

# Callbacks

@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    """Route to different pages."""
    if pathname == "/train":
        return create_training_page()
    else:
        return create_performance_page()

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
    Output("run-select", "options"),
    Input("url", "pathname")
)
def update_run_options(pathname):
    """Update available training runs."""
    runs = sorted(ARTIFACTS_DIR.glob("*__*"), key=os.path.getmtime, reverse=True)
    return [{"label": f"{r.name} ({datetime.fromtimestamp(r.stat().st_mtime).strftime('%Y-%m-%d %H:%M')})",
              "value": str(r)} for r in runs[:50]]  # Show last 50 runs

def train_model_async(config_file):
    """Train model in background thread."""
    global training_status
    import sys
    import io
    
    # Capture stdout/stderr to see training progress
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
        training_status["message"] = f"Starting training: {cfg.run_name} ({total_steps:,} timesteps)..."
        training_status["progress"] = 5
        
        # Redirect output to capture training logs
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        
        import time
        start_time = time.time()
        
        # Log that we're about to start
        print(f"[APP] Starting training with {total_steps:,} timesteps")
        print(f"[APP] This should take approximately {int(total_steps/1000/60)} minutes")
        
        # Actually run training
        run_dir = train_once(cfg)
        
        # Restore output
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        # Get captured output
        stdout_text = stdout_capture.getvalue()
        stderr_text = stderr_capture.getvalue()
        
        elapsed = time.time() - start_time
        
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
            training_status["run_dir"] = str(run_dir)
            training_status["message"] = f"Training complete! Run saved to: {run_dir.name} (took {int(elapsed/60)}m {int(elapsed%60)}s)\n\nLast output:\n{stdout_text[-500:]}"
            training_status["progress"] = 100
        
    except Exception as e:
        # Restore output
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        import traceback
        error_details = traceback.format_exc()
        stdout_text = stdout_capture.getvalue()
        stderr_text = stderr_capture.getvalue()
        
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
     State("training-store", "data")]
)
def handle_training(train_clicks, stop_clicks, n_intervals, config_file, store_data):
    """Handle training start/stop and progress updates."""
    ctx = callback_context
    if not ctx.triggered:
        return False, True, True, "", "", training_status
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if trigger_id == "train-button" and config_file:
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
    
    elif trigger_id == "stop-button":
        training_status["running"] = False
        training_status["message"] = "Training stopped by user"
        return False, True, True, "Training stopped", "", training_status
    
    elif trigger_id == "training-interval":
        # Update progress
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
    run_info = html.Div([
        html.H6("Run Information", className="mb-2"),
        html.P(f"Run: {run_path.name}", className="mb-1"),
        html.P(f"Timesteps: {metrics.get('timesteps', 'N/A')}", className="mb-1 text-muted"),
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

