"""End-to-end smoke test using the deterministic mock provider."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eval_pipeline.api import create_app
from eval_pipeline.cache import ResultsCache
from eval_pipeline.config import Settings
from eval_pipeline.dataset import EvalDataset
from eval_pipeline.runner import run_evaluation


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings()
    s.provider = "mock"
    s.judge.provider = "mock"
    s.cache.directory = str(tmp_path / "cache")
    s.runner.max_workers_per_model = 2
    return s


def _toy_dataset() -> EvalDataset:
    return EvalDataset.from_records(
        [
            {"query": "What is 2+2?", "expected_answer": "4"},
            {"query": "Capital of France?", "expected_answer": "Paris"},
        ]
    )


def test_versioning_is_deterministic(settings: Settings) -> None:
    ds1 = _toy_dataset()
    ds2 = _toy_dataset()
    v1 = ds1.version(["alpha", "beta"], settings.judge)
    v2 = ds2.version(["beta", "alpha"], settings.judge)  # order-independent
    v3 = ds2.version(["alpha", "gamma"], settings.judge)  # different models -> different version
    assert v1 == v2
    assert v1 != v3


def test_run_evaluation_writes_cache_and_summary(settings: Settings) -> None:
    dataset = _toy_dataset()
    models = ["alpha", "beta"]
    outcome = run_evaluation(dataset, models, settings)
    assert outcome.failed_models == []
    assert sorted(outcome.completed_models) == models
    assert set(dataset.inferences.keys()) == set(models)
    assert all(len(records) == len(dataset.samples) for records in dataset.inferences.values())

    # Re-running with the same inputs hits cache for both models.
    again = run_evaluation(_toy_dataset(), models, settings)
    assert sorted(again.cached_models) == models
    assert again.completed_models == []


def test_partial_cache_hit_only_runs_missing_model(settings: Settings) -> None:
    dataset = _toy_dataset()
    run_evaluation(dataset, ["alpha"], settings)

    fresh = _toy_dataset()
    outcome = run_evaluation(fresh, ["alpha", "beta"], settings)
    assert outcome.cached_models == ["alpha"]
    assert outcome.completed_models == ["beta"]


def test_api_health_evaluate_results_flow(settings: Settings) -> None:
    app = create_app(settings)
    client = TestClient(app)

    assert client.get("/health").json() == {"status": "ok", "provider": "mock"}

    payload = {
        "models": ["alpha", "beta"],
        "samples": [
            {"query": "What is 2+2?", "expected_answer": "4"},
            {"query": "Capital of France?", "expected_answer": "Paris"},
        ],
        "sync": True,
    }
    resp = client.post("/evaluate", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "completed"
    assert sorted(body["models_done"]) == ["alpha", "beta"]
    version = body["version"]

    results = client.get(f"/results/{version}").json()
    assert sorted(results["manifest"]["models"]) == ["alpha", "beta"]
    assert sorted(results["results"].keys()) == ["alpha", "beta"]
    for rows in results["results"].values():
        assert len(rows) == len(payload["samples"])
        for row in rows:
            assert row["generated_answer"]
            assert row["latency_seconds"] >= 0
            assert row["token_count"] >= 0

    # /route is reserved for the ML-engineer extension and must signal that.
    assert client.post("/route", json={"query": "anything"}).status_code == 501


def test_cache_round_trip(settings: Settings) -> None:
    cache = ResultsCache(settings.cache.directory)
    dataset = _toy_dataset()
    run_evaluation(dataset, ["alpha"], settings, cache=cache)
    model_version = dataset.model_version("alpha", settings.judge)
    assert cache.has_model(model_version)
    records = cache.read_model(model_version)
    assert len(records) == len(dataset.samples)
    assert records[0].model == "alpha"

    # Manifest is keyed by aggregate version and lists per-model addressing.
    version = dataset.version(["alpha"], settings.judge)
    manifest = cache.read_manifest(version)
    assert manifest is not None
    assert manifest["model_versions"] == {"alpha": model_version}
