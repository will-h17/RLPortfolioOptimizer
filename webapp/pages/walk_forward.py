from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import Input, Output, State, dcc, html

from src.paths import ARTIFACTS_DIR
from webapp import jobs
from webapp.components import card, dropdown, job_panel, page, register_job, table, training_configs

dash.register_page(__name__, path="/walk-forward", name="🧭 Walk-forward", title="Walk-forward", order=6)

STUDIES_DIR = ARTIFACTS_DIR / "walk_forward"
SUMMARY_COLUMNS = ["strategy", "folds", "seeds", "oos_sharpe", "oos_sharpe_seed_std", "oos_cagr", "oos_max_dd",
                   "fold_sharpe_mean", "beats_ew_sharpe", "turnover_mean"]


def _studies() -> list[str]:
    return sorted(p.parent.name for p in STUDIES_DIR.glob("*/summary.csv")) if STUDIES_DIR.exists() else []


def layout():
    configs, studies = training_configs(), _studies()
    return page(
        "Walk-forward Evaluation",
        "Train one agent per test year and seed, and compare it with equal weight, buy & hold, and inverse variance "
        "under identical costs. This is the result to trust. Re-running a study with the same name resumes it.",
        card("Results", dropdown("wf-study", studies, value=studies[-1] if studies else None,
                                 placeholder="No studies yet"), html.Div(id="wf-study-view", className="mt-3")),
        card("Run a study",
             dbc.Row([
                 dbc.Col([dbc.Label("Config"), dropdown("wf-config", configs,
                                                        value="rlppo_v2.yaml" if "rlppo_v2.yaml" in configs else None)], md=6),
                 dbc.Col([dbc.Label("Study name"), dbc.Input(id="wf-name", type="text", value="ppo_v2_wf")], md=6),
             ], className="mb-3"),
             dbc.Row([
                 dbc.Col([dbc.Label("Seeds (comma-separated)"), dbc.Input(id="wf-seeds", type="text", value="0,1,2")], md=4),
                 dbc.Col([dbc.Label("First test year"), dbc.Input(id="wf-year", type="number", value=2016)], md=4),
                 dbc.Col([dbc.Label("Timesteps per run (blank = config)"),
                          dbc.Input(id="wf-timesteps", type="number", placeholder="from config")], md=4),
             ], className="mb-3"),
             dbc.Alert("A full study trains folds x seeds models: about 75 minutes for 10 years x 3 seeds with "
                       "rlppo_v2.yaml. Use 50000 timesteps for a quick check.", color="info"),
             job_panel("wf", "Run walk-forward")),
    )


@dash.callback(Output("wf-study-view", "children"), Input("wf-study", "value"))
def show_study(name):
    if not name:
        return ""
    return _study_view(STUDIES_DIR / name / "summary.csv")


def _study_view(summary_csv: Path):
    summary = pd.read_csv(summary_csv)
    runs = pd.read_csv(summary_csv.parent / "runs.csv", dtype={"fold": str})
    per_fold = runs.groupby(["fold", "strategy"], as_index=False)["sharpe"].mean()
    fig = px.line(per_fold, x="fold", y="sharpe", color="strategy", markers=True, template="plotly_white",
                  labels={"fold": "Test year", "sharpe": "Sharpe (RL: mean over seeds)"})
    return html.Div([
        html.H6("Full-period out-of-sample results (folds stitched together)"),
        table(summary.loc[:, [c for c in SUMMARY_COLUMNS if c in summary.columns]]),
        dcc.Graph(figure=fig),
    ])


def _run(report, config_name, seeds_text, first_year, timesteps, name):
    try:
        seeds = [int(s) for s in (seeds_text or "").split(",") if s.strip()]
    except ValueError as e:
        raise jobs.JobError(f"Seeds must be whole numbers: {e}") from e
    if not (config_name and seeds and first_year and name):
        raise jobs.JobError("Choose a config, seeds, a first test year, and a study name")
    return jobs.walk_forward(report, config_name, seeds, int(first_year), int(timesteps) if timesteps else None, name)


register_job(
    "wf",
    [State("wf-config", "value"), State("wf-seeds", "value"), State("wf-year", "value"),
     State("wf-timesteps", "value"), State("wf-name", "value")],
    _run,
    lambda summary_csv: html.Div([dbc.Alert(f"Study saved to {summary_csv.parent}", color="success"),
                                  _study_view(summary_csv)]),
    require_all=False,
)
