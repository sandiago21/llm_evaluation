import sys
import types
from unittest.mock import MagicMock

import pytest


def _install_router_stubs():
    """Stub the router modules before importing the FastAPI app.

    `src.router.model` loads a HuggingFace tokenizer at import time, which
    we don't want hitting the network during unit tests. We replace both
    `src.router.model` and `src.router.inference` with lightweight fakes.
    """
    model_stub = types.ModuleType("src.router.model")
    model_stub.load_router_model = MagicMock(return_value=None)
    model_stub.tokenizer = MagicMock()
    sys.modules["src.router.model"] = model_stub

    inference_stub = types.ModuleType("src.router.inference")
    inference_stub.predict_best_model = MagicMock(
        return_value={"recommended_model": "mistral", "confidence": 0.83}
    )
    sys.modules["src.router.inference"] = inference_stub

    return model_stub, inference_stub


@pytest.fixture
def app_modules(monkeypatch):
    # CI may set SKIP_ROUTER_MODEL=true globally to bypass the real router at
    # startup. These tests rely on the stubbed router actually being invoked,
    # so force the flag off before re-importing the API module.
    monkeypatch.delenv("SKIP_ROUTER_MODEL", raising=False)

    model_stub, inference_stub = _install_router_stubs()

    # Drop any cached import of the api/task_manager so they pick up our stubs.
    for name in (
        "src.api.main",
        "src.tasks.task_manager",
    ):
        sys.modules.pop(name, None)

    import src.api  # noqa: F401
    if hasattr(src.api, "main"):
        delattr(src.api, "main")

    from fastapi.testclient import TestClient

    from src.api import main as api_main

    client = TestClient(api_main.app)
    yield client, api_main, inference_stub


def test_evaluate_returns_task_id(app_modules, monkeypatch):
    client, api_main, _ = app_modules

    monkeypatch.setattr(api_main, "launch_evaluation_task", lambda dataset, model_names: "task-1")

    response = client.post(
        "/evaluate",
        json={
            "samples": [{"query": "q", "expected_answer": "a"}],
            "models": ["mistral"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body == {"task_id": "task-1", "status": "pending"}


def test_evaluate_forwards_samples_and_models(app_modules, monkeypatch):
    client, api_main, _ = app_modules

    received = {}

    def fake_launch(dataset, model_names):
        received["models"] = model_names
        received["samples"] = [(s.query, s.expected_answer) for s in dataset.samples]
        return "task-x"

    monkeypatch.setattr(api_main, "launch_evaluation_task", fake_launch)

    client.post(
        "/evaluate",
        json={
            "samples": [
                {"query": "q1", "expected_answer": "a1"},
                {"query": "q2", "expected_answer": "a2"},
            ],
            "models": ["mistral", "llama3"],
        },
    )

    assert received["models"] == ["mistral", "llama3"]
    assert received["samples"] == [("q1", "a1"), ("q2", "a2")]


def test_evaluate_validates_request_body(app_modules):
    client, _, _ = app_modules
    response = client.post("/evaluate", json={"samples": [{"query": "q"}], "models": ["x"]})
    assert response.status_code == 422


def test_task_status_returns_runner_state(app_modules, monkeypatch):
    client, api_main, _ = app_modules

    monkeypatch.setattr(
        api_main,
        "get_task_status",
        lambda task_id: {"status": "completed", "result": {"mistral": []}, "error": None},
    )

    response = client.get("/tasks/abc")
    assert response.status_code == 200
    assert response.json()["status"] == "completed"


def test_route_returns_router_prediction(app_modules):
    client, _, _ = app_modules

    response = client.post("/route", json={"query": "explain swaps"})
    assert response.status_code == 200
    body = response.json()
    assert body == {"recommended_model": "mistral", "confidence": 0.83}


def test_route_requires_query_field(app_modules):
    client, _, _ = app_modules
    response = client.post("/route", json={})
    assert response.status_code == 422


def test_response_has_request_id_header(app_modules, monkeypatch):
    client, api_main, _ = app_modules
    monkeypatch.setattr(api_main, "get_task_status", lambda task_id: {"status": "pending"})

    response = client.get("/tasks/whatever")
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) > 0


def _reload_api_main():
    """Force a clean re-import of src.api.main so module-level constants
    (e.g. SKIP_ROUTER_MODEL) get re-evaluated against the current env."""
    import importlib

    for name in ("src.api.main", "src.router.model", "src.router.inference"):
        sys.modules.pop(name, None)

    import src.api  # noqa: F401
    # `from src.api import main` reads src.api.__dict__["main"], which still
    # holds a stale reference after popping sys.modules. Clear it so the
    # next import really reruns the module body.
    if hasattr(src.api, "main"):
        delattr(src.api, "main")

    return importlib.import_module("src.api.main")


def test_route_returns_503_when_router_skipped(monkeypatch):
    """When SKIP_ROUTER_MODEL is set, /route must not import the router
    and must respond 503 instead of attempting inference."""
    monkeypatch.setenv("SKIP_ROUTER_MODEL", "true")

    api_main = _reload_api_main()

    from fastapi.testclient import TestClient

    client = TestClient(api_main.app)
    response = client.post("/route", json={"query": "anything"})

    assert response.status_code == 503
    assert "SKIP_ROUTER_MODEL" in response.json()["detail"]


def test_docs_endpoint_reachable_when_router_skipped(monkeypatch):
    """Smoke-test parity: /docs must serve even without the router model."""
    monkeypatch.setenv("SKIP_ROUTER_MODEL", "true")

    api_main = _reload_api_main()

    from fastapi.testclient import TestClient

    client = TestClient(api_main.app)
    response = client.get("/docs")
    assert response.status_code == 200
