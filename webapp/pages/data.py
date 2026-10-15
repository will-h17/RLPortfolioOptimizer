import dash
import dash_bootstrap_components as dbc
from dash import State

from src.paths import PRICES_PATH
from webapp import jobs
from webapp.components import card, job_panel, page, register_job

dash.register_page(__name__, path="/data", name="Data", title="Data", order=1)


def layout():
    return page(
        "Data Acquisition", "Download daily adjusted prices from yfinance and merge them into one price table.",
        card("Symbols",
             dbc.Label("Ticker symbols (comma-separated)", html_for="data-symbols"),
             dbc.Input(id="data-symbols", type="text", value="SPY,QQQ,TLT"),
             dbc.Checkbox(id="data-force", label="Re-download even if cached", value=False, className="mt-3")),
        card("1. Download", job_panel("download", "Download data")),
        card("2. Process & merge", job_panel("process", f"Write {PRICES_PATH.name}")),
    )


register_job(
    "download", [State("data-symbols", "value"), State("data-force", "value")],
    lambda report, symbols, force: jobs.download_data(report, symbols, bool(force)),
    lambda paths: dbc.Alert(f"Downloaded/cached {len(paths)} file(s): {', '.join(p.name for p in paths)}",
                            color="success"),
    require_all=False,
)

register_job(
    "process", [State("data-symbols", "value")],
    jobs.process_data,
    lambda df: dbc.Alert(f"Wrote {PRICES_PATH.name}: {len(df):,} rows x {df.shape[1]} assets "
                         f"({df.index.min().date()} to {df.index.max().date()})", color="success"),
)
