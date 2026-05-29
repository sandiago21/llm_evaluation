import time
import uuid
from unittest.mock import patch

import pytest

from src.tasks import task_manager
from src.tasks.task_manager import (
    TASKS,
    get_task_status,
    launch_evaluation_task,
)


@pytest.fixture(autouse=True)
def clear_tasks():
    TASKS.clear()
    yield
    # Drain any still-running workers so they can't run the real
    # run_multi_model_evaluation against a non-existent Ollama after
    # the test's patch context has been torn down.
    deadline = time.time() + 2.0
    while time.time() < deadline:
        in_flight = [
            t for t in TASKS.values() if t["status"] in {"pending", "running"}
        ]
        if not in_flight:
            break
        time.sleep(0.01)
    TASKS.clear()


def _wait_until(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_launch_returns_uuid_task_id(eval_dataset):
    with patch.object(task_manager, "run_multi_model_evaluation", return_value=eval_dataset):
        task_id = launch_evaluation_task(eval_dataset, ["mistral"])
        # Drain the worker before the patch is reverted, otherwise it
        # would call the real run_multi_model_evaluation against Ollama.
        assert _wait_until(
            lambda: get_task_status(task_id)["status"] in {"completed", "failed"}
        )

    uuid.UUID(task_id)


def test_launch_registers_task_with_initial_state(eval_dataset):
    """Initial state can be either 'pending' or 'running' depending on
    scheduling — both are valid pre-completion states."""
    with patch.object(task_manager, "run_multi_model_evaluation", return_value=eval_dataset):
        task_id = launch_evaluation_task(eval_dataset, ["mistral"])

        state = get_task_status(task_id)
        assert state["status"] in {"pending", "running", "completed"}
        assert state["models"] == ["mistral"]
        assert state["error"] is None

        # Drain before patch teardown so the worker doesn't escape.
        assert _wait_until(
            lambda: get_task_status(task_id)["status"] in {"completed", "failed"}
        )


def test_completed_task_carries_inferences(eval_dataset, inference_result):
    eval_dataset.inferences["mistral"] = [inference_result]

    with patch.object(task_manager, "run_multi_model_evaluation", return_value=eval_dataset):
        task_id = launch_evaluation_task(eval_dataset, ["mistral"])
        # Wait inside the patch so the worker thread observes the mock.
        assert _wait_until(lambda: get_task_status(task_id)["status"] == "completed")

    state = get_task_status(task_id)
    assert state["result"] == {"mistral": [inference_result]}
    assert state["error"] is None


def test_failed_task_records_error(eval_dataset):
    with patch.object(
        task_manager,
        "run_multi_model_evaluation",
        side_effect=RuntimeError("boom"),
    ):
        task_id = launch_evaluation_task(eval_dataset, ["mistral"])
        assert _wait_until(lambda: get_task_status(task_id)["status"] == "failed")

    state = get_task_status(task_id)
    assert state["error"] == "boom"
    assert state["result"] is None


def test_get_task_status_unknown_id_returns_not_found():
    state = get_task_status("does-not-exist")
    assert state == {"status": "not_found"}


def test_launch_invokes_runner_with_correct_args(eval_dataset):
    with patch.object(task_manager, "run_multi_model_evaluation", return_value=eval_dataset) as run:
        task_id = launch_evaluation_task(eval_dataset, ["mistral", "llama3"])
        assert _wait_until(lambda: get_task_status(task_id)["status"] in {"completed", "failed"})
        run.assert_called_once_with(dataset=eval_dataset, model_names=["mistral", "llama3"])
