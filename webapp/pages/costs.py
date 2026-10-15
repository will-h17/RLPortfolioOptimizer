import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import State, dcc, html

from webapp import jobs
from webapp.components import card, dropdown, job_panel, page, register_job, run_options, table

dash.register_page(__name__, path="/costs", name="Cost Sensitivity", title="Cost Sensitivity", order=5)


def layout():
    return page(
        "Cost Sensitivity", "Re-run a trained agent on its test split with spreads from 0 to 10 bps, using the run's own config.",
        card("Training run", dropdown("costs-run", run_options(), placeholder="Select a run...")),
        card("Sweep", job_panel("sweep", "Run cost sweep")),
    )


def _render(csv_path):
    df = pd.read_csv(csv_path)
    fig = px.line(df, x="base_bps", y="sharpe", markers=True, template="plotly_white",
                  labels={"base_bps": "Spread (bps)", "sharpe": "Test Sharpe"})
    return html.Div([dbc.Alert(f"Saved {csv_path}", color="success"), dcc.Graph(figure=fig), table(df)])


register_job("sweep", [State("costs-run", "value")], jobs.cost_sweep, _render)
