"""Multi-model parallel evaluation runner.

One worker thread is spawned per model; within a model the samples can be
processed by an inner thread pool (``runner.max_workers_per_model``) but
results are written back in sample order. Results for each model are flushed
to the cache as soon as that model's run finishes; the global cache lock
ensures concurrent writes from different model threads remain consistent.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

from .cache import ResultsCache
from .config import Settings
from .dataset import EvalDataset, InferenceRecord
from .inference import evaluate_one
from .judge import Judge
from .providers import get_provider
from .providers.base import InferenceProvider

log = logging.getLogger(__name__)


@dataclass
class RunOutcome:
    version: str
    dataset: EvalDataset
    cached_models: list[str]
    completed_models: list[str]
    failed_models: list[str]


def run_evaluation(
    dataset: EvalDataset,
    models: list[str],
    settings: Settings,
    *,
    cache: ResultsCache | None = None,
    on_model_done: Callable[[str, bool], None] | None = None,
    on_model_failed: Callable[[str], None] | None = None,
) -> RunOutcome:
    """Run evaluation across ``models`` in parallel and persist per-model results.

    Cache hits are loaded directly; misses run inference and write a new
    Parquet file under ``cache/<version>/<model>.parquet``.
    """
    cache = cache or ResultsCache(settings.cache.directory)
    judge_provider = get_provider(settings, settings.judge.provider) if settings.judge.enabled else None
    judge = Judge(judge_provider, settings.judge) if judge_provider else None

    version = dataset.version(models, settings.judge)
    model_versions = {m: dataset.model_version(m, settings.judge) for m in models}
    cache.write_manifest(
        version,
        {
            "version": version,
            "samples_hash": dataset.samples_hash(),
            "num_samples": len(dataset.samples),
            "models": sorted(set(models)),
            "model_versions": model_versions,
            "judge": {
                "enabled": settings.judge.enabled,
                "provider": settings.judge.provider,
                "model": settings.judge.model,
                "temperature": settings.judge.temperature,
            },
        },
    )

    cached_models = sorted(m for m in models if cache.has_model(model_versions[m]))
    todo = [m for m in models if m not in cached_models]
    log.info(
        "evaluation plan",
        extra={"version": version, "cached": cached_models, "todo": todo, "samples": len(dataset.samples)},
    )

    # Load cache hits first.
    for m in cached_models:
        dataset.inferences[m] = cache.read_model(model_versions[m])
        if on_model_done:
            on_model_done(m, True)

    completed: list[str] = []
    failed: list[str] = []
    completed_lock = threading.Lock()

    if not todo:
        return RunOutcome(
            version=version,
            dataset=dataset,
            cached_models=cached_models,
            completed_models=completed,
            failed_models=failed,
        )

    def _run_model(model: str) -> tuple[str, list[InferenceRecord] | None, str | None]:
        try:
            provider = get_provider(settings)
            records = _run_for_model(
                model=model,
                dataset=dataset,
                provider=provider,
                judge=judge,
                max_workers=settings.runner.max_workers_per_model,
            )
            cache.write_model(model_versions[model], records)
            return model, records, None
        except Exception as exc:  # noqa: BLE001 — boundary
            log.exception("model run failed", extra={"model": model})
            return model, None, str(exc)

    # One thread per model.
    with ThreadPoolExecutor(max_workers=max(len(todo), 1), thread_name_prefix="model") as pool:
        futures = [pool.submit(_run_model, m) for m in todo]
        for fut in as_completed(futures):
            model, records, err = fut.result()
            if err is not None or records is None:
                with completed_lock:
                    failed.append(model)
                if on_model_failed:
                    on_model_failed(model)
                continue
            dataset.inferences[model] = records
            with completed_lock:
                completed.append(model)
            if on_model_done:
                on_model_done(model, False)

    return RunOutcome(
        version=version,
        dataset=dataset,
        cached_models=cached_models,
        completed_models=sorted(completed),
        failed_models=sorted(failed),
    )


def _run_for_model(
    *,
    model: str,
    dataset: EvalDataset,
    provider: InferenceProvider,
    judge: Judge | None,
    max_workers: int,
) -> list[InferenceRecord]:
    n = len(dataset.samples)
    results: list[InferenceRecord | None] = [None] * n

    def _one(idx: int) -> tuple[int, InferenceRecord]:
        sample = dataset.samples[idx]
        rec = evaluate_one(
            sample_index=idx,
            query=sample.query,
            expected_answer=sample.expected_answer,
            model=model,
            provider=provider,
            judge=judge,
        )
        return idx, rec

    with ThreadPoolExecutor(max_workers=max(max_workers, 1), thread_name_prefix=f"sample-{model}") as pool:
        for fut in as_completed([pool.submit(_one, i) for i in range(n)]):
            idx, rec = fut.result()
            results[idx] = rec

    # Type narrowing: every slot has been filled.
    return [r for r in results if r is not None]
