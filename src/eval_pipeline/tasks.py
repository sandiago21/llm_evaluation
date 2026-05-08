"""In-memory background task registry.

Tasks track the progress of an evaluation run launched by the API. The
registry is process-local and safe for concurrent access. A production
deployment would typically swap this for Redis/Celery/RQ; the interface is
deliberately thin to make that substitution easy.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskState:
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    version: str | None = None
    models: list[str] = field(default_factory=list)
    models_done: list[str] = field(default_factory=list)
    models_failed: list[str] = field(default_factory=list)
    cached_models: list[str] = field(default_factory=list)
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    result_summary: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "version": self.version,
            "models": self.models,
            "models_done": self.models_done,
            "models_failed": self.models_failed,
            "cached_models": self.cached_models,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result_summary": self.result_summary,
        }


class TaskRegistry:
    def __init__(self) -> None:
        self._tasks: dict[str, TaskState] = {}
        self._lock = threading.Lock()

    def create(self, *, models: list[str], version: str | None = None) -> TaskState:
        tid = uuid.uuid4().hex
        state = TaskState(task_id=tid, models=list(models), version=version)
        with self._lock:
            self._tasks[tid] = state
        return state

    def update(self, task_id: str, **fields: Any) -> TaskState:
        with self._lock:
            state = self._tasks[task_id]
            for k, v in fields.items():
                setattr(state, k, v)
            return state

    def mark_running(self, task_id: str) -> None:
        self.update(task_id, status=TaskStatus.RUNNING, started_at=time.time())

    def mark_model_done(self, task_id: str, model: str, *, from_cache: bool = False) -> None:
        with self._lock:
            state = self._tasks[task_id]
            state.models_done.append(model)
            if from_cache:
                state.cached_models.append(model)

    def mark_model_failed(self, task_id: str, model: str) -> None:
        with self._lock:
            self._tasks[task_id].models_failed.append(model)

    def mark_completed(self, task_id: str, summary: dict[str, Any]) -> None:
        self.update(
            task_id,
            status=TaskStatus.COMPLETED,
            finished_at=time.time(),
            result_summary=summary,
        )

    def mark_failed(self, task_id: str, error: str) -> None:
        self.update(
            task_id,
            status=TaskStatus.FAILED,
            finished_at=time.time(),
            error=error,
        )

    def get(self, task_id: str) -> TaskState | None:
        with self._lock:
            return self._tasks.get(task_id)

    def list(self) -> list[TaskState]:
        with self._lock:
            return list(self._tasks.values())
