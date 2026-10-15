"""
components.py — Shared layout pieces and the background-job wiring used by every page.

A job is a button that runs a function from webapp/jobs.py in a Dash background callback: the
page shows a progress bar and the job's live log, the Stop button kills the job's process, and the
result is rendered when it finishes. No state is shared between browser sessions or jobs.
"""

from __future__ import annotations

import base64
import traceback
import uuid
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, dcc, html

from src.paths import ARTIFACTS_DIR, CONFIGS_DIR
from src.runlog import has_model
from webapp.jobs import JobError

# ---------------- shell ----------------

def navbar() -> dbc.Navbar:
    return dbc.Navbar(
        dbc.Container([
            html.Div([
                html.H4("📈 RL Portfolio Optimizer", className="mb-0 text-white"),
                html.Small("Reinforcement-learning portfolio allocation", className="text-white-50"),
            ]),
        ], fluid=True),
        color="primary", dark=True, className="mb-4",
    )


def sidebar() -> html.Div:
    links = [dbc.NavLink(page["name"], href=page["relative_path"], active="exact", className="mb-1")
             for page in dash.page_registry.values()]
    return html.Div(dbc.Nav(links, vertical=True, pills=True, className="bg-light p-3 rounded"))


def page(title: str, subtitle: str, *children: Any) -> dbc.Container:
    return dbc.Container([
        html.H2(title, className="mb-2"),
        html.P(subtitle, className="text-muted mb-4"),
        *children,
    ], fluid=True)


def card(title: str | None, *children: Any, class_name: str = "mb-4") -> dbc.Card:
    body = ([html.H5(title, className="card-title mb-3")] if title else []) + list(children)
    return dbc.Card(dbc.CardBody(body), className=class_name)


def table(df: pd.DataFrame, float_format: str = "{:.3f}") -> dbc.Table:
    shown = df.copy()
    for col in shown.select_dtypes("float").columns:
        shown[col] = shown[col].map(float_format.format)
    # from_dataframe is attached to Table at runtime, so type checkers cannot see it
    return dbc.Table.from_dataframe(shown, striped=True, bordered=True, hover=True, size="sm", className="mt-2")  # pyright: ignore[reportAttributeAccessIssue]


def image(path: Path) -> html.Img:
    """Inline a PNG (files outside assets/ are not served by Dash)."""
    data = base64.b64encode(path.read_bytes()).decode()
    return html.Img(src=f"data:image/png;base64,{data}", style={"width": "100%"}, className="mb-3")


def error(message: str, log: str = "") -> html.Div:
    return html.Div([
        dbc.Alert(message, color="danger", className="mb-2"),
        html.Pre(log, className="bg-light p-2 small", style={"whiteSpace": "pre-wrap"}) if log else None,
    ])


# ---------------- choices ----------------

def run_names() -> list[str]:
    """Completed training runs, newest first."""
    runs = [p for p in ARTIFACTS_DIR.glob("*__*") if p.is_dir() and has_model(p)] if ARTIFACTS_DIR.exists() else []
    return [p.name for p in sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)]


def run_options() -> list[dict[str, str]]:
    def label(name: str) -> str:
        stamp = datetime.fromtimestamp((ARTIFACTS_DIR / name).stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        return f"{name} ({stamp})"
    return [{"label": label(n), "value": n} for n in run_names()]


def training_configs() -> list[str]:
    """Training configs at the top of configs/ (base/ holds shared fragments, eval_grid* is another format)."""
    return sorted(p.name for p in CONFIGS_DIR.glob("*.yaml") if not p.name.startswith("eval_grid"))


def grid_configs() -> list[str]:
    return sorted(p.name for p in CONFIGS_DIR.glob("eval_grid*.yaml"))


def dropdown(id_: str, options: Sequence[str] | list[dict[str, str]], value: str | None = None,
             placeholder: str = "Select...", clearable: bool = False) -> dcc.Dropdown:
    opts = [o if isinstance(o, dict) else {"label": o, "value": o} for o in options]
    return dcc.Dropdown(id=id_, options=opts, value=value, placeholder=placeholder, clearable=clearable)


# ---------------- background jobs ----------------

def job_panel(prefix: str, start_label: str, cancellable: bool = True) -> html.Div:
    """Start/Stop buttons, progress bar, live log, and result area for a job registered with register_job."""
    return html.Div([
        # Fresh on every page load: makes each click's background-job cache key unique, so a click can
        # never be answered with the result or progress of an earlier job that had the same inputs
        dcc.Store(id=f"{prefix}-token", data=uuid.uuid4().hex),
        dbc.Button(start_label, id=f"{prefix}-start", color="primary", size="lg", className="me-2"),
        dbc.Button("Stop", id=f"{prefix}-cancel", color="danger", size="lg", disabled=True,
                   style=None if cancellable else {"display": "none"}),
        dbc.Progress(id=f"{prefix}-progress", value=0, className="mt-3", style={"height": "1.25rem"}),
        html.Pre(id=f"{prefix}-log", className="bg-light p-2 mt-3 small",
                 style={"maxHeight": "22rem", "overflowY": "auto", "whiteSpace": "pre-wrap"}),
        html.Div(id=f"{prefix}-result", className="mt-3"),
    ])


def register_job(prefix: str, states: list[State], run: Callable[..., Any],
                 render: Callable[[Any], Any], require_all: bool = True) -> None:
    """
    Wire a job_panel to `run(report, *state_values)` as a cancellable background callback, then
    show `render(result)`. With require_all, the job refuses to start until every input is filled.
    """
    @dash.callback(
        Output(f"{prefix}-result", "children"),
        Input(f"{prefix}-start", "n_clicks"),
        State(f"{prefix}-token", "data"),
        *states,
        background=True,
        running=[
            (Output(f"{prefix}-start", "disabled"), True, False),
            (Output(f"{prefix}-cancel", "disabled"), False, True),
            (Output(f"{prefix}-progress", "animated"), True, False),
            (Output(f"{prefix}-progress", "striped"), True, False),
        ],
        cancel=[Input(f"{prefix}-cancel", "n_clicks")],
        progress=[Output(f"{prefix}-progress", "value"), Output(f"{prefix}-progress", "label"),
                  Output(f"{prefix}-log", "children")],
        prevent_initial_call=True,
    )
    def _job(set_progress, _n_clicks, _token, *values):
        if require_all and any(v is None or v == "" for v in values):
            return dbc.Alert("Fill in every field before starting.", color="warning")

        def report(percent: float | None, log: str) -> None:
            # Unknown progress shows a full animated bar without a percentage
            set_progress((100 if percent is None else percent, "" if percent is None else f"{percent:.0f}%", log))

        set_progress((0, "", "Starting..."))
        try:
            result = run(report, *values)
        except JobError as e:
            return error(str(e), e.log)
        except Exception as e:  # show unexpected failures in the page instead of a silent dead job
            return error(f"{type(e).__name__}: {e}", traceback.format_exc())
        set_progress((100, "Done", dash.no_update))
        return render(result)
