from unittest.mock import patch

from src.evaluation import multi_model_runner
from src.evaluation.multi_model_runner import run_multi_model_evaluation
from src.models.dataset import InferenceResult


def _result(value):
    return InferenceResult(
        generated_answer=value,
        latency_seconds=0.1,
        token_count=1,
        correctness=1.0,
    )


def test_run_returns_dataset_unchanged_when_all_models_cached(tmp_cache_dir, eval_dataset, inference_result):
    """If resolve_cached_models reports nothing missing, no worker should run."""
    def fake_resolve(dataset, model_names, judge_model, judge_prompt_version="v1"):
        for m in model_names:
            dataset.inferences[m] = [inference_result]
        return ({m: "v" for m in model_names}, [])

    with patch.object(multi_model_runner, "resolve_cached_models", side_effect=fake_resolve), \
         patch.object(multi_model_runner, "evaluate_query_answer_pair") as worker, \
         patch.object(multi_model_runner, "save_model_cache") as saver:
        result = run_multi_model_evaluation(eval_dataset, ["mistral", "llama3"])

    assert result is eval_dataset
    assert set(result.inferences.keys()) == {"mistral", "llama3"}
    worker.assert_not_called()
    saver.assert_not_called()


def test_run_evaluates_only_missing_models(tmp_cache_dir, eval_dataset, inference_result):
    """Cached model should be skipped; missing one should be evaluated and saved."""
    def fake_resolve(dataset, model_names, judge_model, judge_prompt_version="v1"):
        dataset.inferences["mistral"] = [inference_result, inference_result]
        return ({"mistral": "vM", "llama3": "vL"}, ["llama3"])

    with patch.object(multi_model_runner, "resolve_cached_models", side_effect=fake_resolve), \
         patch.object(
             multi_model_runner,
             "evaluate_query_answer_pair",
             return_value=_result("from-llama3"),
         ) as worker, \
         patch.object(multi_model_runner, "save_model_cache") as saver:
        result = run_multi_model_evaluation(eval_dataset, ["mistral", "llama3"])

    # llama3 called once per sample (2 samples in fixture).
    assert worker.call_count == 2
    assert "mistral" in result.inferences
    assert "llama3" in result.inferences
    assert len(result.inferences["llama3"]) == 2

    saver.assert_called_once()
    saved_kwargs = saver.call_args.kwargs
    assert saved_kwargs["model_name"] == "llama3"
    assert saved_kwargs["version"] == "vL"


def test_run_executes_multiple_missing_models(tmp_cache_dir, eval_dataset):
    """Each missing model should produce its own cache save."""
    def fake_resolve(dataset, model_names, judge_model, judge_prompt_version="v1"):
        return ({m: f"v-{m}" for m in model_names}, list(model_names))

    with patch.object(multi_model_runner, "resolve_cached_models", side_effect=fake_resolve), \
         patch.object(
             multi_model_runner,
             "evaluate_query_answer_pair",
             return_value=_result("ans"),
         ), \
         patch.object(multi_model_runner, "save_model_cache") as saver:
        run_multi_model_evaluation(eval_dataset, ["mistral", "llama3", "gemma2"])

    saved_models = {call.kwargs["model_name"] for call in saver.call_args_list}
    assert saved_models == {"mistral", "llama3", "gemma2"}
