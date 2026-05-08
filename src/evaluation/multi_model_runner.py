from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import List

from src.models.dataset import EvaluationDataset
from src.evaluation.atomic_function import evaluate_query_pair

from src.evaluation.cache import (
    resolve_cached_models,
    save_model_cache,
)

from src.core.config import JUDGE_MODEL


write_lock = threading.Lock()


def _run_model_on_dataset(
    model_name: str,
    dataset: EvaluationDataset,
):

    results = []

    for sample in dataset.samples:

        result = evaluate_query_pair(
            query=sample.query,
            expected_answer=sample.expected_answer,
            model_name=model_name,
        )

        results.append(result)

    return results


def run_multi_model_evaluation(
    dataset: EvaluationDataset,
    model_names: List[str],
):
    """
    Parallel evaluation with partial cache loading.
    """

    # -------------------------
    # Resolve cache first
    # -------------------------
    version, missing_models = resolve_cached_models(
        dataset=dataset,
        model_names=model_names,
        judge_model=JUDGE_MODEL,
    )

    # -------------------------
    # Early return if fully cached
    # -------------------------
    if len(missing_models) == 0:

        print("[✓] Fully loaded from cache")

        return dataset

    # -------------------------
    # Worker function
    # -------------------------
    def worker(model_name: str):

        results = _run_model_on_dataset(
            model_name=model_name,
            dataset=dataset,
        )

        with write_lock:

            dataset.inferences[model_name] = results

            save_model_cache(
                version=version,
                model_name=model_name,
                results=results,
            )

        print(f"[✓] Cached {model_name}")

    # -------------------------
    # Parallel execution
    # -------------------------
    with ThreadPoolExecutor(
        max_workers=len(missing_models)
    ) as executor:

        futures = [
            executor.submit(worker, model_name)
            for model_name in missing_models
        ]

        for future in as_completed(futures):
            future.result()

    return dataset
