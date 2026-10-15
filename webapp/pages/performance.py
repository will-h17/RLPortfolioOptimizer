import json

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

from src.backtest import buy_and_hold, equal_weight, evaluate_sb3_model
from src.env import load_run_env_config
from src.paths import ARTIFACTS_DIR, FEATURES_PATH, PRICES_PATH
from src.runlog import resolve_run_model
from src.scaler import FitOnTrainScaler
from src.utils.data_utils import align_after_load
from webapp.components import card, dropdown, page, run_names, run_options

dash.register_page(__name__, path="/performance", name="📉 Performance", title="Performance", order=10)


def layout():
    names = run_names()
    return page(
        "Run Performance",
        "One run over the full price history against buy & hold and equal weight. The history includes the "
        "run's training period, so this view is in-sample; see Walk-forward for out-of-sample results.",
        card("Training run", dropdown("perf-run", run_options(), value=names[0] if names else None,
                                      placeholder="Select a run..."), html.Div(id="perf-info", className="mt-3")),
        dcc.Loading([
            html.Div(id="perf-cards", className="mb-4"),
            card("Equity curves", dcc.Graph(id="perf-equity")),
            dbc.Row([
                dbc.Col(card("Sharpe ratio", dcc.Graph(id="perf-sharpe")), lg=6),
                dbc.Col(card("Validation Sharpe during training", dcc.Graph(id="perf-val")), lg=6),
            ]),
            card("Allocation over time", dcc.Graph(id="perf-weights")),
        ]),
    )


def _empty(text: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=text, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    fig.update_layout(template="plotly_white", xaxis_visible=False, yaxis_visible=False)
    return fig


def _metric(title: str, value: str, color: str) -> dbc.Col:
    return dbc.Col(dbc.Card(dbc.CardBody([html.H6(title, className="text-muted"), html.H3(value, className="mb-0")]),
                            color=color, outline=True), md=3)


@dash.callback(
    Output("perf-info", "children"), Output("perf-cards", "children"), Output("perf-equity", "figure"),
    Output("perf-sharpe", "figure"), Output("perf-val", "figure"), Output("perf-weights", "figure"),
    Input("perf-run", "value"),
)
def update(run_name):
    if not run_name:
        empty = _empty("Select a training run")
        return "", "", empty, empty, empty, empty
    run_path = ARTIFACTS_DIR / run_name
    if not (PRICES_PATH.exists() and FEATURES_PATH.exists()):
        empty = _empty("No processed data: run the Data and Features pages first")
        return "", "", empty, empty, empty, empty

    metrics_path = run_path / "metrics.json"
    test = json.loads(metrics_path.read_text()).get("results", {}) if metrics_path.exists() else {}
    info = html.Small(f"Test split: Sharpe {test['sharpe']:.3f} over {test.get('num_steps', '?')} days"
                      if "sharpe" in test else "No test metrics recorded for this run", className="text-muted")

    prices, features = align_after_load(pd.read_parquet(PRICES_PATH), pd.read_parquet(FEATURES_PATH))
    bh, ew = buy_and_hold(prices, features), equal_weight(prices, features)

    scaler = FitOnTrainScaler.load(run_path / "scaler.json", columns=features.columns)
    assert scaler.mu_ is not None
    feats = scaler.transform(features.loc[:, [c for c in scaler.mu_.index if c in features.columns]])
    rl = evaluate_sb3_model(prices, feats, str(resolve_run_model(run_path)), load_run_env_config(run_path))
    m = rl["metrics"]

    cards = dbc.Row([
        _metric("Sharpe ratio", f"{m['sharpe']:.2f}", "primary"),
        _metric("Final value", f"{m['final_value']:.2f}x", "success"),
        _metric("Max drawdown", f"{m['max_drawdown']:.1%}", "warning"),
        _metric("Turnover", f"{m['turnover']:.3f}", "info"),
    ])

    equity = go.Figure()
    for name, curve, width in (("Buy & hold", bh["equity_curve"], 1), ("Equal weight", ew["equity_curve"], 1),
                               ("RL agent", rl["equity_curve"], 2)):
        equity.add_trace(go.Scatter(x=prices.index[-len(curve):], y=curve, mode="lines", name=name,
                                    line={"width": width}))
    equity.update_layout(template="plotly_white", hovermode="x unified", yaxis_title="Portfolio value",
                         yaxis_type="log")

    sharpe = px.bar(x=["Buy & hold", "Equal weight", "RL agent"],
                    y=[bh["metrics"]["sharpe"], ew["metrics"]["sharpe"], m["sharpe"]],
                    labels={"x": "", "y": "Sharpe"}, template="plotly_white")

    val_csv = run_path / "eval" / "val_sharpe.csv"
    if val_csv.exists():
        val = pd.read_csv(val_csv)
        val_fig = px.line(val, x="timesteps", y="val_sharpe", markers=True, template="plotly_white",
                          labels={"val_sharpe": "Validation Sharpe"})
    else:
        val_fig = _empty("Not recorded (runs before validation-Sharpe selection)")

    weights = np.asarray(rl["weights"])
    assets = list(prices.columns) + (["CASH"] if weights.shape[1] == prices.shape[1] + 1 else [])
    sample = pd.DataFrame(weights, columns=pd.Index(assets[: weights.shape[1]]),
                          index=prices.index[-len(weights):]).iloc[::20]
    weights_fig = px.area(sample, template="plotly_white", labels={"value": "Weight", "index": "", "variable": "Asset"})

    return info, cards, equity, sharpe, val_fig, weights_fig
