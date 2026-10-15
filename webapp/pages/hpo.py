import json

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import State, html

from webapp import jobs
from webapp.components import card, dropdown, job_panel, page, register_job, table, training_configs

dash.register_page(__name__, path="/hpo", name="🔍 Hyperparameters", title="Hyperparameter Search", order=8)


def layout():
    return page(
        "Hyperparameter Search", "Search PPO and reward settings with Optuna, scoring each trial by validation Sharpe.",
        dbc.Alert("Each trial trains a full model (hpo.train_timesteps in the config), so 30 trials take hours.",
                  color="warning"),
        card("Settings",
             dbc.Label("Base config"), dropdown("hpo-config", training_configs(), placeholder="Select a config..."),
             dbc.Label("Number of trials", className="mt-3"),
             dbc.Input(id="hpo-trials", type="number", value=30, min=1, max=500),
             dbc.Label("Study name", className="mt-3"), dbc.Input(id="hpo-study", type="text", value="ppo_hpo"),
             dbc.Checkbox(id="hpo-resume", label="Resume an existing study with this name", value=False,
                          className="mt-3")),
        card("Search", job_panel("hpo", "Start search")),
    )


def _render(summary_path):
    summary = json.loads(summary_path.read_text())
    best = pd.DataFrame([{"parameter": k, "value": v} for k, v in summary["best_params"].items()])
    return html.Div([
        dbc.Alert(f"Best validation Sharpe {summary['best_value']:.3f} (trial {summary['best_trial_number']}) "
                  f"of {summary['n_trials']}.", color="success"),
        table(best, float_format="{:.6g}"),
    ])


register_job(
    "hpo", [State("hpo-config", "value"), State("hpo-trials", "value"), State("hpo-study", "value"),
            State("hpo-resume", "value")],
    lambda report, config, trials, study, resume: jobs.hpo(report, config, int(trials), study, bool(resume)),
    _render,
    require_all=False,
)
