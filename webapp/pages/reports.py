import dash
import dash_bootstrap_components as dbc
from dash import State, html

from src.paths import REPORTS_DIR
from webapp import jobs
from webapp.components import card, dropdown, image, job_panel, page, register_job, run_options

dash.register_page(__name__, path="/reports", name="Reports", title="Reports", order=9)


def layout():
    return page(
        "Reports", f"Render equity curves, metric comparisons, and the allocation heatmap to {REPORTS_DIR.name}/.",
        card("Training run",
             dropdown("reports-run", run_options(), placeholder="Latest completed run", clearable=True)),
        card("Generate", job_panel("reports", "Generate reports", cancellable=False)),
    )


register_job(
    "reports", [State("reports-run", "value")],
    jobs.make_reports,
    lambda pngs: html.Div([dbc.Alert(f"Saved {len(pngs)} image(s) to {REPORTS_DIR}", color="success"),
                           *[image(p) for p in pngs]]),
    require_all=False,  # empty run = latest
)
