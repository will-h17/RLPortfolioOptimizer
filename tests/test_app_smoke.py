"""
Smoke tests for the Dash app: the shell loads, every page renders through the Dash Pages router,
the synchronous callbacks work against real data, and the job runner streams/raises correctly.
Uses Dash's Flask test client, so no server or browser is needed. Skipped when the app extras
(pip install -e ".[app]") are not installed.
"""
import json

import pytest

pytest.importorskip("dash")
pytest.importorskip("dash_bootstrap_components")
pytest.importorskip("diskcache")

import app as app_module  # noqa: E402
from src.paths import ARTIFACTS_DIR, FEATURES_PATH, PRICES_PATH  # noqa: E402
from src.runlog import latest_run  # noqa: E402
from webapp import jobs  # noqa: E402

PAGES = ["/", "/data", "/features", "/train", "/evaluate", "/costs", "/walk-forward", "/grid", "/hpo",
         "/reports", "/performance"]


@pytest.fixture(scope="module")
def client():
    c = app_module.app.server.test_client()
    assert c.get("/").status_code == 200  # first request also validates every page's layout and callback ids
    return c


def _call(client, outputs, inputs):
    """POST a callback request the way the Dash renderer does; returns the decoded response."""
    def _out(o):
        cid, prop = o.split(".")
        return {"id": cid, "property": prop}

    multi = len(outputs) > 1
    payload = {
        "output": (".." + "...".join(outputs) + "..") if multi else outputs[0],
        "outputs": [_out(o) for o in outputs] if multi else _out(outputs[0]),
        "inputs": [{"id": cid, "property": prop, "value": value} for cid, prop, value in inputs],
        "changedPropIds": [f"{inputs[0][0]}.{inputs[0][1]}"],
        "state": [],
    }
    resp = client.post("/_dash-update-component", json=payload)
    assert resp.status_code == 200, resp.get_data(as_text=True)[:2000]
    return json.loads(resp.get_data(as_text=True))["response"]


def test_shell_and_dependencies_load(client):
    for route in ("/_dash-layout", "/_dash-dependencies"):
        assert client.get(route).status_code == 200, route


def _hrefs(node):
    if isinstance(node, dict):
        if "href" in node:
            yield node["href"]
        for value in node.values():
            yield from _hrefs(value)
    elif isinstance(node, list):
        for value in node:
            yield from _hrefs(value)


def test_sidebar_lists_every_page(client):
    layout = json.loads(client.get("/_dash-layout").get_data(as_text=True))
    assert set(PAGES) <= set(_hrefs(layout))


@pytest.mark.parametrize("pathname", PAGES)
def test_every_page_renders(client, pathname):
    response = _call(client, ["_pages_content.children", "_pages_store.data"],
                     [("_pages_location", "pathname", pathname), ("_pages_location", "search", "")])
    assert response["_pages_content"]["children"]
    assert "404" not in json.dumps(response["_pages_content"]["children"])[:200]


def test_training_config_preview(client):
    response = _call(client, ["train-preview.children"], [("train-config", "value", "rlppo_v2.yaml")])
    text = json.dumps(response)
    assert "ppo_v2" in text and "random 252-day windows" in text


def test_walk_forward_results_view(client):
    studies = sorted((ARTIFACTS_DIR / "walk_forward").glob("*/summary.csv"))
    if not studies:
        pytest.skip("no walk-forward study yet")
    response = _call(client, ["wf-study-view.children"], [("wf-study", "value", studies[-1].parent.name)])
    assert "Equal weight" in json.dumps(response)


def test_performance_page_evaluates_latest_run(client):
    if not (PRICES_PATH.exists() and FEATURES_PATH.exists()):
        pytest.skip("processed data not available")
    try:
        run = latest_run(ARTIFACTS_DIR)
    except FileNotFoundError:
        pytest.skip("no completed training run")

    outputs = ["perf-info.children", "perf-cards.children", "perf-equity.figure", "perf-sharpe.figure",
               "perf-val.figure", "perf-weights.figure"]
    response = _call(client, outputs, [("perf-run", "value", run.name)])
    names = {t.get("name") for t in response["perf-equity"]["figure"]["data"]}
    assert names == {"Buy & hold", "Equal weight", "RL agent"}


def test_stream_module_reports_output_and_raises_on_failure():
    reports = []
    log = jobs.stream_module(["platform"], lambda pct, text: reports.append(text))
    assert log.strip() and reports[-1] == log

    with pytest.raises(jobs.JobError) as exc:
        jobs.stream_module(["no_such_module_xyz"], lambda pct, text: None)
    assert "No module named" in exc.value.log
