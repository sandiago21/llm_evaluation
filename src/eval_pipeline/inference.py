"""Atomic inference function: run one (query, expected_answer) through one model."""

from __future__ import annotations

import logging
import time

from .dataset import InferenceRecord
from .judge import Judge
from .providers.base import InferenceError, InferenceProvider

log = logging.getLogger(__name__)


def evaluate_one(
    *,
    sample_index: int,
    query: str,
    expected_answer: str,
    model: str,
    provider: InferenceProvider,
    judge: Judge | None,
) -> InferenceRecord:
    """Run a single (query, model) inference and grade it.

    Always returns an ``InferenceRecord``; on failure ``error`` is populated
    and the answer-related fields are empty/zero so downstream aggregation
    still works.
    """
    start = time.perf_counter()
    try:
        result = provider.generate(query, model)
    except InferenceError as exc:
        log.warning("inference failed", extra={"model": model, "sample_index": sample_index, "error": str(exc)})
        return InferenceRecord(
            sample_index=sample_index,
            query=query,
            expected_answer=expected_answer,
            model=model,
            generated_answer="",
            latency_seconds=time.perf_counter() - start,
            token_count=0,
            correctness=None,
            error=str(exc),
        )
    latency = time.perf_counter() - start
    correctness: bool | None = None
    if judge is not None:
        correctness = judge.score(query, expected_answer, result.text)
    return InferenceRecord(
        sample_index=sample_index,
        query=query,
        expected_answer=expected_answer,
        model=model,
        generated_answer=result.text,
        latency_seconds=latency,
        token_count=result.token_count,
        correctness=correctness,
    )
