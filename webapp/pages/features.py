import dash
import dash_bootstrap_components as dbc
from dash import State

from src.paths import FEATURES_PATH, PRICES_PATH
from webapp import jobs
from webapp.components import card, job_panel, page, register_job

dash.register_page(__name__, path="/features", name="Features", title="Features", order=2)


def layout():
    return page(
        "Feature Engineering", "Compute returns, momentum, volatility, and trend features for every asset.",
        None if PRICES_PATH.exists() else dbc.Alert("No processed prices yet: run the Data page first.", color="warning"),
        card("Lookback windows",
             dbc.Label("Window lengths in days (comma-separated)", html_for="features-lookbacks"),
             dbc.Input(id="features-lookbacks", type="text", value="5,20")),
        card("Generate", job_panel("features", f"Write {FEATURES_PATH.name}", cancellable=False)),
    )


register_job(
    "features", [State("features-lookbacks", "value")],
    jobs.generate_features,
    lambda df: dbc.Alert(f"Wrote {FEATURES_PATH.name}: {len(df):,} rows x {df.shape[1]} features", color="success"),
)
