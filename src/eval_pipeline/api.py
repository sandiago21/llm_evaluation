"""FastAPI service exposing the evaluation pipeline.

Endpoints
---------
GET  /health                  liveness probe
POST /evaluate                kick off (or return cached) evaluation
GET  /tasks/{task_id}         task status
GET  /results/{version}       per-model summary + records for a version
POST /route                   ML-engineer extension (not implemented yet)
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .cache import ResultsCache
from .config import Settings, load_settings
from .dataset import EvalDataset
from .logging_config import configure_logging, new_request_id, set_request_id
from .runner import run_evaluation
from .tasks import TaskRegistry, TaskStatus

log = logging.getLogger(__name__)


# ---------------------------- request models ------------------------------


class SampleIn(BaseModel):
    query: str
    expected_answer: str


class EvaluateRequest(BaseModel):
    models: list[str] = Field(..., min_length=1)
    samples: list[SampleIn] | None = None
    dataset_path: str | None = None
    sync: bool = False  # if true, wait for completion and return summary

    def to_dataset(self) -> EvalDataset:
        if self.samples:
            return EvalDataset.from_records([s.model_dump() for s in self.samples])
        if self.dataset_path:
            return EvalDataset.from_path(self.dataset_path)
        raise HTTPException(400, "Provide either 'samples' or 'dataset_path'")


class RouteRequest(BaseModel):
    query: str


# ------------------------------ app factory -------------------------------


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    configure_logging(level=settings.logging.level, json=settings.logging.as_json)
    log.info("starting eval-pipeline api", extra={"provider": settings.provider})

    cache = ResultsCache(settings.cache.directory)
    tasks = TaskRegistry()

    app = FastAPI(title="LLM Evaluation Pipeline", version="0.1.0")
    app.state.settings = settings
    app.state.cache = cache
    app.state.tasks = tasks

    @app.middleware("http")
    async def _request_id_middleware(request: Request, call_next):
        rid = request.headers.get("x-request-id") or new_request_id()
        set_request_id(rid)
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        return response

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "provider": settings.provider}

    @app.post("/evaluate")
    def evaluate(req: EvaluateRequest) -> JSONResponse:
        dataset = req.to_dataset()
        version = dataset.version(req.models, settings.judge)
        task = tasks.create(models=req.models, version=version)

        def _run() -> None:
            tasks.mark_running(task.task_id)
            try:
                outcome = run_evaluation(
                    dataset,
                    req.models,
                    settings,
                    cache=cache,
                    on_model_done=lambda m, cached: tasks.mark_model_done(task.task_id, m, from_cache=cached),
                    on_model_failed=lambda m: tasks.mark_model_failed(task.task_id, m),
                )
                tasks.mark_completed(task.task_id, dataset.summary())
                log.info(
                    "evaluation finished",
                    extra={
                        "task_id": task.task_id,
                        "version": outcome.version,
                        "cached": outcome.cached_models,
                        "completed": outcome.completed_models,
                        "failed": outcome.failed_models,
                    },
                )
            except Exception as exc:  # noqa: BLE001 — boundary
                log.exception("evaluation crashed", extra={"task_id": task.task_id})
                tasks.mark_failed(task.task_id, str(exc))

        if req.sync:
            _run()
            state = tasks.get(task.task_id)
            assert state is not None
            return JSONResponse(state.to_dict())

        # Detached thread keeps the API responsive without an external broker.
        threading.Thread(target=_run, name=f"eval-{task.task_id[:8]}", daemon=True).start()
        return JSONResponse({"task_id": task.task_id, "version": version, "status": TaskStatus.PENDING.value}, status_code=202)

    @app.get("/tasks/{task_id}")
    def task_status(task_id: str) -> dict[str, Any]:
        state = tasks.get(task_id)
        if state is None:
            raise HTTPException(404, f"unknown task {task_id}")
        return state.to_dict()

    @app.get("/results/{version}")
    def results(version: str) -> dict[str, Any]:
        manifest = cache.read_manifest(version)
        if manifest is None:
            raise HTTPException(404, f"no results for version {version}")
        model_versions: dict[str, str] = manifest.get("model_versions", {})
        per_model: dict[str, Any] = {}
        for m, mv in model_versions.items():
            records = cache.read_model(mv)
            per_model[m] = [r.to_dict() for r in records]
        return {"manifest": manifest, "results": per_model}

    @app.post("/route")
    def route(_: RouteRequest) -> dict[str, Any]:
        # Reserved for the ML-engineer dynamic-router extension.
        raise HTTPException(status_code=501, detail="dynamic model router not implemented yet")

    return app


# Module-level app for `uvicorn eval_pipeline.api:app` and the `eval-api` script.
app = create_app()


def run() -> None:  # pragma: no cover - thin entrypoint
    import uvicorn

    settings = app.state.settings
    uvicorn.run(
        "eval_pipeline.api:app",
        host=settings.api.host,
        port=settings.api.port,
        log_config=None,
    )
