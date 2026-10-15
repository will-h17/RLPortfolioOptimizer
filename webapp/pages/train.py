import json

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, dcc, html

from src.config_loader import load_config
from src.paths import CONFIGS_DIR
from webapp import jobs
from webapp.components import card, dropdown, error, job_panel, page, register_job, table, training_configs

dash.register_page(__name__, path="/train", name="⚙️ Training", title="Training", order=3)

DEFAULT_CONFIG = "rlppo_v2.yaml"


def layout():
    configs = training_configs()
    return page(
        "Train a Model", "Train a PPO agent from a config. Stop kills the training process; nothing is kept from a stopped run except what it already saved.",
        card("Configuration",
             dropdown("train-config", configs, value=DEFAULT_CONFIG if DEFAULT_CONFIG in configs else None,
                      placeholder="Select a config..."),
             html.Div(id="train-preview", className="mt-3")),
        card("Training", job_panel("train", "Start training")),
    )


@dash.callback(Output("train-preview", "children"), Input("train-config", "value"))
def preview(config_name):
    if not config_name:
        return ""
    try:
        cfg = load_config(CONFIGS_DIR / config_name)
    except ValueError as e:
        return error("This config is invalid:", str(e))
    episodes = (f"random {cfg.env.episode_length}-day windows" if cfg.env.random_start
                else "one pass over the whole training split")
    cost = (f"cost model: {cfg.cost_model.spread.type} spread" if cfg.cost_model.enabled
            else f"flat {cfg.env.transaction_cost_bps} bps")
    summary = {
        "run_name": cfg.run_name,
        "timesteps": f"{cfg.train.total_timesteps:,}",
        "parallel envs": cfg.train.n_envs,
        "episodes": episodes,
        "reward": cfg.env.reward_mode,
        "costs": cost,
        "split": f"train ≤ {cfg.dates.train_end}, val ≤ {cfg.dates.val_end}, test ≤ {cfg.dates.test_end}",
    }
    warning = (dbc.Alert(f"Only {cfg.train.total_timesteps:,} timesteps: fine for checking the pipeline, "
                         "too few to learn a useful policy.", color="warning", className="mb-2")
               if cfg.train.total_timesteps < 200_000 else None)
    return html.Div([warning, html.Pre(json.dumps(summary, indent=2), className="bg-light p-3 rounded mb-0")])


def _render(run_dir):
    metrics = json.loads((run_dir / "metrics.json").read_text())
    results = {k: v for k, v in metrics["results"].items() if isinstance(v, (int, float))}
    return html.Div([
        dbc.Alert(["Training complete: ", html.Strong(run_dir.name), ". ",
                   dcc.Link("Open it on the Performance page", href="/performance"), "."], color="success"),
        html.H6("Test split (best validation checkpoint)"),
        table(pd.DataFrame([results])),
    ])


register_job("train", [State("train-config", "value")], jobs.train, _render)
