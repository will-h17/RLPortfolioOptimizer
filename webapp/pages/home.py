import dash
import dash_bootstrap_components as dbc
from dash import html

from src.paths import ARTIFACTS_DIR, FEATURES_PATH, PRICES_PATH
from webapp.components import card, page, run_names

dash.register_page(__name__, path="/", name="🏠 Dashboard", title="Dashboard", order=0)


def _stat(title: str, value: str, note: str) -> dbc.Col:
    return dbc.Col(card(title, html.H3(value, className="text-center"),
                        html.P(note, className="text-center text-muted mb-0")), md=6, lg=3)


def layout():
    runs = run_names()
    studies = sorted((ARTIFACTS_DIR / "walk_forward").glob("*/summary.csv")) if ARTIFACTS_DIR.exists() else []
    steps = [
        ("/data", "1. Data", "download and merge prices"),
        ("/features", "2. Features", "compute the model's inputs"),
        ("/train", "3. Training", "train an agent from a config"),
        ("/walk-forward", "4. Walk-forward", "the reliable out-of-sample test"),
        ("/performance", "5. Performance", "inspect a single run"),
    ]
    return page(
        "Dashboard", "Use the sidebar to move through the workflow.",
        dbc.Row([
            _stat("Prices", "✅" if PRICES_PATH.exists() else "❌", PRICES_PATH.name),
            _stat("Features", "✅" if FEATURES_PATH.exists() else "❌", FEATURES_PATH.name),
            _stat("Completed runs", str(len(runs)), "with a saved model"),
            _stat("Walk-forward studies", str(len(studies)), "with a summary"),
        ]),
        card("Latest run", html.P(runs[0] if runs else "None yet", className="mb-0")),
        card("Workflow", dbc.ListGroup([
            dbc.ListGroupItem([html.Strong(label), f" — {desc}"], href=href, action=True)
            for href, label, desc in steps
        ])),
    )
