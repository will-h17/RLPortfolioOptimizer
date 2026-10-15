"""
RL Portfolio Optimizer web app.

Pages live in webapp/pages (one module per page, registered with Dash Pages). Long-running work
runs in Dash background callbacks managed by a disk cache: each job gets its own process, so jobs
survive page navigation and the Stop buttons can kill them. See APP_INFO.md.

    python app.py [--port 8050] [--debug]
"""

from __future__ import annotations

import argparse
import signal
import sys

import dash
import dash_bootstrap_components as dbc
import diskcache
import psutil
from dash import DiskcacheManager

from src.paths import PROJECT_ROOT
from webapp.components import navbar, sidebar

JOB_CACHE_DIR = PROJECT_ROOT / ".cache" / "webapp-jobs"
job_cache = diskcache.Cache(str(JOB_CACHE_DIR))

app = dash.Dash(
    __name__,
    use_pages=True,
    pages_folder="webapp/pages",
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    background_callback_manager=DiskcacheManager(job_cache),
    title="RL Portfolio Optimizer",
)
server = app.server  # for WSGI servers, e.g. `gunicorn app:server`


def layout():
    return dbc.Container([
        navbar(),
        dbc.Row([
            dbc.Col(sidebar(), md=3, lg=2, className="mb-4"),
            dbc.Col(dash.page_container, md=9, lg=10),
        ]),
    ], fluid=True)


app.layout = layout


def stop_jobs() -> None:
    """Kill every process this server started (background jobs and the tools they run)."""
    for child in psutil.Process().children(recursive=True):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description="RL Portfolio Optimizer web app")
    ap.add_argument("--host", default="127.0.0.1",
                    help="Interface to bind; use 0.0.0.0 to allow other machines on your network")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--debug", action="store_true",
                    help="Enable Dash dev tools and auto-reload. Never combine with a public --host: "
                         "the debugger allows running arbitrary code")
    args = ap.parse_args()

    job_cache.clear()  # results/progress left by a previous server session must never be served
    # Background jobs run in child processes; stop them with the server instead of leaving them
    # running (and writing runs) after the app is closed
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        app.run(debug=args.debug, host=args.host, port=args.port)
    finally:
        stop_jobs()


if __name__ == "__main__":
    main()
