import hashlib
import json
from typing import List

from src.models.dataset import EvaluationDataset


def compute_model_version(
    dataset: EvaluationDataset,
    model_name: str,
    judge_model: str,
    judge_prompt_version: str = "v1",
) -> str:
    """
    Per-model cache key: depends only on (samples, judge config, model_name).

    Adding a new model to a run does NOT invalidate already-cached models
    — each model's cache entry is reused as long as the samples and judge
    config are unchanged.
    """

    sample_payload = [
        {
            "query": sample.query,
            "expected_answer": sample.expected_answer,
        }
        for sample in dataset.samples
    ]

    payload = {
        "samples": sample_payload,
        "judge_model": judge_model,
        "judge_prompt_version": judge_prompt_version,
        "model": model_name,
    }

    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
    )

    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_dataset_version(
    dataset: EvaluationDataset,
    model_names: List[str],
    judge_model: str,
    judge_prompt_version: str = "v1",
) -> str:
    """
    Deterministically computes a dataset version hash.

    Version depends on:
    - dataset samples
    - evaluated models
    - judge model configuration
    - judge prompt version

    Same inputs -> same hash.
    """

    # -------------------------
    # Normalize sample payload
    # -------------------------
    sample_payload = [
        {
            "query": sample.query,
            "expected_answer": sample.expected_answer,
        }
        for sample in dataset.samples
    ]

    # -------------------------
    # Construct version payload
    # -------------------------
    version_payload = {
        "samples": sample_payload,
        "models": sorted(model_names),
        "judge_model": judge_model,
        "judge_prompt_version": judge_prompt_version,
    }

    # -------------------------
    # Deterministic serialization
    # -------------------------
    serialized = json.dumps(
        version_payload,
        sort_keys=True,
        ensure_ascii=False,
    )

    # -------------------------
    # SHA256 version hash
    # -------------------------
    version_hash = hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()

    return version_hash
