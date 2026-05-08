from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import List

from src.models.dataset import EvaluationDataset, InferenceResult
from src.evaluation.atomic_function import evaluate_query_answer_pair


# -------------------------
# Global lock for safe writes
# -------------------------
write_lock = threading.Lock()


# -------------------------
# Worker per model
# -------------------------

def _run_model_on_dataset(
    model_name: str,
    dataset: EvaluationDataset,
) -> List[InferenceResult]:
    """
    Runs inference for a single model across all samples.
    """

    results: List[InferenceResult] = []

    for sample in dataset.samples:
        result = evaluate_query_answer_pair(
            query_answer=sample,
            model_name=model_name,
        )
        results.append(result)

    return results


# -------------------------
# Main orchestrator
# -------------------------

def run_multi_model_evaluation(
    dataset: EvaluationDataset,
    model_names: List[str],
) -> EvaluationDataset:
    """
    Runs multiple models in parallel (one thread per model)
    and aggregates results into dataset.inferences.
    """

    def worker(model_name: str):
        results = _run_model_on_dataset(model_name, dataset)

        # Thread-safe write into shared dataset
        with write_lock:
            dataset.inferences[model_name] = results

        return model_name, results


    # -------------------------
    # Thread pool execution
    # -------------------------
    with ThreadPoolExecutor(max_workers=len(model_names)) as executor:

        futures = [
            executor.submit(worker, model)
            for model in model_names
        ]

        for future in as_completed(futures):
            model_name, _ = future.result()
            print(f"[✓] Completed model: {model_name}")

    return dataset
