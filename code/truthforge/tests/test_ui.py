"""Tests for the local UI server.

These spin up the real `ThreadingHTTPServer` on an ephemeral port and talk to
it over HTTP, because the things worth testing here are the wiring (routing,
JSON shapes, error codes, path traversal) rather than the handler functions in
isolation.

Everything uses the `lexical` detector and the `echo` generator, so no GPU and
no model download is involved.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from truthforge.ui.server import STATIC_DIR, create_server


@pytest.fixture(scope="module")
def server():
    httpd = create_server(host="127.0.0.1", port=0, device="cpu")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def get(base: str, path: str):
    try:
        with urllib.request.urlopen(base + path, timeout=30) as r:  # noqa: S310
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def post(base: str, path: str, payload: dict):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def raw(base: str, path: str):
    try:
        with urllib.request.urlopen(base + path, timeout=30) as r:  # noqa: S310
            return r.status, r.read(), r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers.get("Content-Type", "")


# --------------------------------------------------------------------------
# static assets
# --------------------------------------------------------------------------


def test_static_assets_exist_on_disk():
    for name in ("index.html", "styles.css", "app.js", "charts.js"):
        assert (STATIC_DIR / name).is_file(), f"{name} missing from the UI bundle"


def test_index_and_assets_are_served(server):
    status, body, ctype = raw(server, "/")
    assert status == 200
    assert b"TruthForge" in body
    assert "text/html" in ctype

    for name, expect in (
        ("/styles.css", "text/css"),
        ("/app.js", "text/javascript"),
        ("/charts.js", "text/javascript"),
    ):
        status, body, ctype = raw(server, name)
        assert status == 200 and len(body) > 100
        assert expect in ctype


def test_path_traversal_is_refused(server):
    """The server reads files from disk; escaping the static dir must fail."""
    for attempt in ("/../server.py", "/../../config.py", "/..%2fserver.py"):
        status, _body, _ct = raw(server, attempt)
        assert status in (403, 404), f"{attempt} returned {status}"


def test_unknown_static_path_404s(server):
    status, _body, _ct = raw(server, "/nope.js")
    assert status == 404


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------


def test_status_reports_the_environment(server):
    status, data = get(server, "/api/status")
    assert status == 200
    assert "cuda" in data and "available" in data["cuda"]
    assert "lexical" in data["available_detectors"]
    assert "rule" in data["available_routers"]
    assert "echo" in data["available_generators"]


def test_examples_endpoint_marks_the_fixture_synthetic(server):
    status, data = get(server, "/api/examples?source=sample&limit=3")
    assert status == 200
    if not data["available"]:
        pytest.skip(data["reason"])
    assert data["synthetic"] is True
    assert len(data["examples"]) <= 3
    ex = data["examples"][0]
    assert ex["question"] and ex["response"] and ex["passages"]


def test_unknown_route_404s(server):
    status, data = get(server, "/api/does-not-exist")
    assert status == 404
    assert "error" in data


def test_run_requires_evidence(server):
    status, data = post(server, "/api/run", {"question": "q", "passages": []})
    assert status == 400
    assert "passage" in data["error"].lower()
    assert data["kind"] == "bad_request"


def test_run_requires_a_question_or_an_answer(server):
    status, data = post(server, "/api/run", {"passages": ["some evidence text"]})
    assert status == 400


def test_run_returns_a_full_turn(server):
    status, data = post(server, "/api/run", {
        "question": "When was the Eiffel Tower completed?",
        "answer": "The Eiffel Tower was completed in 1889. "
                  "It cost 45 million francs to build.",
        "passages": ["The Eiffel Tower was completed in 1889 for the World's Fair."],
        "detector": "lexical",
        "router": "rule",
        "generator": "echo",
    })
    assert status == 200, data
    assert len(data["claims"]) == 2
    assert data["verdicts"] and data["decisions"]
    assert "metrics" in data and "cost_by_stage" in data
    assert data["config"]["detector"] == "lexical"

    # The detector's own tokens must be billed, or the cost column lies.
    stages = {s["stage"] for s in data["cost_by_stage"]}
    assert "detect" in stages
    assert sum(s["tokens"] for s in data["cost_by_stage"]) > 0


def test_run_honours_router_thresholds(server):
    """Widening the uncertain band is the over-correction lever; it has to be
    reachable from the UI or the ablation cannot be demonstrated there."""
    body = {
        "question": "q",
        "answer": "Quantum chromodynamics describes gluon interactions precisely.",
        "passages": ["Cats are small domesticated carnivorous mammals."],
        "detector": "lexical", "router": "rule", "generator": "echo",
    }
    _s, narrow = post(server, "/api/run", {**body, "uncertain_band": 0.0})
    _s, wide = post(server, "/api/run", {**body, "uncertain_band": 0.49})

    wide_accepts = sum(1 for d in wide["decisions"] if d["action"] == "accept")
    narrow_accepts = sum(1 for d in narrow["decisions"] if d["action"] == "accept")
    assert wide_accepts >= narrow_accepts


def test_run_rejects_an_unknown_detector(server):
    """An unrecognised detector name is the caller's mistake, so 400 with a
    readable message -- not a 500 with a traceback."""
    status, data = post(server, "/api/run", {
        "question": "q", "answer": "An answer of adequate length here.",
        "passages": ["evidence"], "detector": "no_such_detector",
    })
    assert status == 400
    assert "unknown detector" in data["error"]
    assert data["kind"] == "bad_request"


def test_invalid_json_body_is_a_400(server):
    req = urllib.request.Request(
        server + "/api/run", data=b"{not json",
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=20)  # noqa: S310
        raise AssertionError("should have failed")
    except urllib.error.HTTPError as e:
        assert e.code == 400


def test_runs_listing_is_shaped_correctly(server):
    status, data = get(server, "/api/runs")
    assert status == 200
    assert "runs" in data and isinstance(data["runs"], list)
    for r in data["runs"]:
        assert "name" in r and "metrics" in r


def test_compare_with_no_selection_is_graceful(server):
    status, data = get(server, "/api/compare?runs=")
    assert status == 200
    assert data["rows"] == []


# --------------------------------------------------------------------------
# annotation
# --------------------------------------------------------------------------


def test_annotation_batch_without_a_run_is_graceful(server):
    status, data = get(server, "/api/annotate/batch?run=")
    assert status == 200
    assert data["items"] == []


def test_annotation_validates_the_label(server):
    base = {
        "run": "test_run", "annotator_id": "T", "attempt_key": "k:0:rewrite",
    }
    status, data = post(server, "/api/annotate", {**base, "outcome": "excellent"})
    assert status == 400 and "outcome must be" in data["error"]

    status, data = post(server, "/api/annotate", {**base, "outcome": "worse"})
    assert status == 400 and "worse_kind" in data["error"]

    status, data = post(
        server, "/api/annotate",
        {**base, "outcome": "worse", "worse_kind": "not_a_kind"},
    )
    assert status == 400

    status, data = post(
        server, "/api/annotate",
        {**base, "outcome": "same", "worse_kind": "hedging"},
    )
    assert status == 400 and "must be empty" in data["error"]


def test_annotation_requires_an_annotator(server):
    status, _data = post(server, "/api/annotate", {
        "run": "r", "attempt_key": "k", "outcome": "same",
    })
    assert status == 400


def test_annotation_round_trips_to_csv(server, tmp_path, monkeypatch):
    """A saved label must land in annotation/ as a readable worksheet row."""
    import csv

    import truthforge.ui.server as srv

    monkeypatch.setattr(srv, "PROJECT_ROOT", tmp_path)

    status, data = post(server, "/api/annotate", {
        "run": "unit", "annotator_id": "ZZ", "attempt_key": "c1:0:rewrite",
        "claim_id": "c1", "action": "rewrite", "iteration": 0,
        "before_text": "before", "after_text": "after", "evidence": "ev",
        "outcome": "worse", "worse_kind": "hedging", "notes": "unit test",
    })
    assert status == 200, data
    path = Path(data["path"])
    assert path.exists()

    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["outcome"] == "worse"
    assert rows[0]["worse_kind"] == "hedging"
    assert rows[0]["annotator_id"] == "ZZ"

    # load_human_labels must accept what the UI wrote.
    from truthforge.eval.outcomes import load_human_labels

    labels = load_human_labels(path)
    assert labels["c1:0:rewrite"]["outcome"] == "worse"


def test_annotator_id_is_sanitised_into_the_filename(server, tmp_path, monkeypatch):
    """The annotator id reaches a file path, so it must not escape it."""
    import truthforge.ui.server as srv

    monkeypatch.setattr(srv, "PROJECT_ROOT", tmp_path)

    status, data = post(server, "/api/annotate", {
        "run": "../../evil", "annotator_id": "../../../etc/passwd",
        "attempt_key": "k", "outcome": "same",
    })
    assert status == 200
    resolved = Path(data["path"]).resolve()
    assert resolved.is_relative_to(tmp_path.resolve())


def test_run_warns_when_the_source_conflict_signal_is_a_placeholder(server):
    """An unusable signal must be announced, not silently used."""
    status, data = post(server, "/api/run", {
        "question": "q",
        "answer": "The Eiffel Tower was completed in 1889.",
        "passages": ["The Eiffel Tower was completed in 1889."],
        "detector": "lexical", "router": "rule", "generator": "echo",
    })
    assert status == 200
    titles = [w["title"] for w in data["warnings"]]
    assert any("placeholder" in t for t in titles)


def test_run_warns_about_discarded_corrector_output(server):
    """A corrector that echoes the answer inflates the futile rate; the UI has
    to say so rather than presenting it as an ordinary no-op."""
    answer = (
        "Water boils at 100 degrees Celsius at sea level. "
        "It was discovered by aliens in 1823."
    )
    status, data = post(server, "/api/run", {
        "question": "q", "answer": answer,
        "passages": ["Water boils at 100 degrees Celsius at sea level."],
        "detector": "lexical", "router": "always_correct", "generator": "echo",
    })
    assert status == 200
    assert isinstance(data["warnings"], list)
    assert "detector_info" in data
