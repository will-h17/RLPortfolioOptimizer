import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import State, html

from src.paths import ARTIFACTS_DIR
from webapp import jobs
from webapp.components import card, dropdown, job_panel, page, register_job, run_options, table

dash.register_page(__name__, path="/evaluate", name="Evaluation", title="Evaluation", order=4)


def layout():
    return page(
        "Backtest a Run", "Run a trained agent and the baselines over the full price history (this includes the training period, so it is in-sample for the agent; use Walk-forward for an honest comparison).",
        card("Training run", dropdown("evaluate-run", run_options(), placeholder="Select a run...")),
        card("Backtest", job_panel("backtest", "Run backtest")),
    )


def _render(log: str):
    summary = [line for line in log.splitlines() if line.startswith(("Buy & Hold", "Equal Weight", "RL Agent"))]
    baselines = ARTIFACTS_DIR / "baseline_results.csv"
    return html.Div([
        dbc.Alert("Backtest complete.", color="success"),
        html.Pre("\n".join(summary), className="bg-light p-2 small", style={"whiteSpace": "pre-wrap"}),
        html.H6("Cost-aware baselines", className="mt-3"),
        table(pd.read_csv(baselines)) if baselines.exists() else None,
    ])


register_job("backtest", [State("evaluate-run", "value")], jobs.backtest, _render)
