import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from src.models.dataset import EvaluationDataset
from src.evaluation.multi_model_runner import (
    run_multi_model_evaluation,
)


# -------------------------
# In-memory task registry
# -------------------------

TASKS: Dict[str, Dict] = {}

TASK_LOCK = threading.Lock()


# -------------------------
# Thread pool for jobs
# -------------------------

executor = ThreadPoolExecutor(max_workers=10)


# -------------------------
# Background task launcher
# -------------------------

def launch_evaluation_task(
    dataset: EvaluationDataset,
    model_names: List[str],
) -> str:
    """
    Launches evaluation as a background task.
    Returns task_id immediately.
    """

    task_id = str(uuid.uuid4())

    # -------------------------
    # Initialize task state
    # -------------------------
    with TASK_LOCK:

        TASKS[task_id] = {
            "status": "pending",
            "models": model_names,
            "error": None,
            "result": None,
        }

    # -------------------------
    # Background worker
    # -------------------------
    def task_runner():

        try:

            with TASK_LOCK:
                TASKS[task_id]["status"] = "running"

            result_dataset = run_multi_model_evaluation(
                dataset=dataset,
                model_names=model_names,
            )

            print(f"Check here: {result_dataset}")

            with TASK_LOCK:

                TASKS[task_id]["status"] = "completed"

                TASKS[task_id]["result"] = result_dataset.inferences

        except Exception as e:

            with TASK_LOCK:

                TASKS[task_id]["status"] = "failed"

                TASKS[task_id]["error"] = str(e)

    # -------------------------
    # Submit async task
    # -------------------------
    executor.submit(task_runner)

    return task_id


# -------------------------
# Status checker
# -------------------------

def get_task_status(task_id: str):
    """
    Returns current task metadata/status.
    """

    with TASK_LOCK:

        if task_id not in TASKS:
            return {
                "status": "not_found",
            }

        return TASKS[task_id]
