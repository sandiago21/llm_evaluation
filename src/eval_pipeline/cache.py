"""Filesystem cache for inference results.

Layout::

    <cache_dir>/by_model/<model_version>.parquet     # per-model results
    <cache_dir>/versions/<version>/manifest.json     # aggregate-run lineage

Per-model results are addressed by ``model_version`` — a hash of
(samples, judge, model) — so that adding a new model to a run does not
invalidate previously-computed model results. The manifest under
``versions/<version>`` captures the aggregate run (samples + models + judge)
for lineage and points at the per-model entries it covers.

A ``threading.Lock`` per filesystem path serialises concurrent writes from
different model threads.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .dataset import InferenceRecord

log = logging.getLogger(__name__)


class ResultsCache:
    def __init__(self, directory: str | Path) -> None:
        self._root = Path(directory)
        (self._root / "by_model").mkdir(parents=True, exist_ok=True)
        (self._root / "versions").mkdir(parents=True, exist_ok=True)
        self._global_lock = threading.Lock()
        self._path_locks: dict[Path, threading.Lock] = defaultdict(threading.Lock)

    # ---- internal -----------------------------------------------------
    def _model_path(self, model_version: str) -> Path:
        return self._root / "by_model" / f"{model_version}.parquet"

    def _version_dir(self, version: str) -> Path:
        d = self._root / "versions" / version
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _lock_for(self, path: Path) -> threading.Lock:
        with self._global_lock:
            return self._path_locks[path]

    # ---- public API ---------------------------------------------------
    def has_model(self, model_version: str) -> bool:
        return self._model_path(model_version).exists()

    def write_manifest(self, version: str, manifest: dict[str, Any]) -> None:
        path = self._version_dir(version) / "manifest.json"
        with self._lock_for(path):
            with path.open("w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, sort_keys=True, ensure_ascii=False)

    def read_manifest(self, version: str) -> dict[str, Any] | None:
        path = self._version_dir(version) / "manifest.json"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def write_model(self, model_version: str, records: list[InferenceRecord]) -> Path:
        path = self._model_path(model_version)
        df = pd.DataFrame([r.to_dict() for r in records])
        with self._lock_for(path):
            df.to_parquet(path, index=False)
        log.info("cache write", extra={"model_version": model_version, "rows": len(df)})
        return path

    def read_model(self, model_version: str) -> list[InferenceRecord]:
        path = self._model_path(model_version)
        if not path.exists():
            return []
        df = pd.read_parquet(path)
        return [
            InferenceRecord(
                sample_index=int(row["sample_index"]),
                query=str(row["query"]),
                expected_answer=str(row["expected_answer"]),
                model=str(row["model"]),
                generated_answer=str(row["generated_answer"]),
                latency_seconds=float(row["latency_seconds"]),
                token_count=int(row["token_count"]),
                correctness=_unbox(row.get("correctness")),
                error=_str_or_none(row.get("error")),
            )
            for _, row in df.iterrows()
        ]


def _unbox(value: Any) -> bool | float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (bool,)):
        return value
    return value


def _str_or_none(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value)
