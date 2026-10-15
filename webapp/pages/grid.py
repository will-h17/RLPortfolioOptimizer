import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import State, html

from webapp import jobs
from webapp.components import card, dropdown, grid_configs, job_panel, page, register_job, table

dash.register_page(__name__, path="/grid", name="🌐 Grid Evaluation", title="Grid Evaluation", order=7)


def layout():
    configs = grid_configs()
    return page(
        "Grid Evaluation", "Evaluate the latest (or configured) run across asset universes, periods, and feature subsets.",
        card("Grid config", dropdown("grid-config", configs, value=configs[0] if configs else None)),
        card("Evaluate", job_panel("grid", "Run grid evaluation")),
    )


def _render(results_csv):
    df = pd.read_csv(results_csv)
    return html.Div([dbc.Alert(f"Saved {results_csv}", color="success"), table(df)])


register_job("grid", [State("grid-config", "value")], jobs.grid_eval, _render)
