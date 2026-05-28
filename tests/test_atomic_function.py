from unittest.mock import patch

from src.evaluation import atomic_function
from src.evaluation.atomic_function import evaluate_query_answer_pair
from src.models.dataset import InferenceResult, QueryAnswerSample


def test_evaluate_returns_inference_result():
    sample = QueryAnswerSample(query="2+2?", expected_answer="4")

    with patch.object(
        atomic_function,
        "generate",
        return_value={"response": "4", "token_count": 5},
    ), patch.object(
        atomic_function,
        "judge_correctness",
        return_value=1.0,
    ):
        result = evaluate_query_answer_pair(sample, "mistral")

    assert isinstance(result, InferenceResult)
    assert result.generated_answer == "4"
    assert result.token_count == 5
    assert result.correctness == 1.0
    assert result.latency_seconds >= 0.0


def test_evaluate_passes_query_to_generate_and_expected_to_judge():
    sample = QueryAnswerSample(query="QUERY", expected_answer="EXPECTED")

    with patch.object(
        atomic_function,
        "generate",
        return_value={"response": "GENERATED", "token_count": 3},
    ) as gen, patch.object(
        atomic_function,
        "judge_correctness",
        return_value=0.5,
    ) as judge:
        evaluate_query_answer_pair(sample, "mistral")

    gen.assert_called_once_with(model_name="mistral", prompt="QUERY")
    judge.assert_called_once_with(
        expected_answer="EXPECTED",
        generated_answer="GENERATED",
    )


def test_evaluate_handles_missing_token_count_in_generate_output():
    sample = QueryAnswerSample(query="q", expected_answer="a")

    with patch.object(
        atomic_function,
        "generate",
        return_value={"response": "ans"},
    ), patch.object(
        atomic_function,
        "judge_correctness",
        return_value=0.0,
    ):
        result = evaluate_query_answer_pair(sample, "mistral")

    assert result.token_count == 0


def test_evaluate_measures_latency():
    sample = QueryAnswerSample(query="q", expected_answer="a")
    fake_times = iter([100.0, 102.5])

    with patch.object(
        atomic_function,
        "generate",
        return_value={"response": "ok", "token_count": 1},
    ), patch.object(
        atomic_function,
        "judge_correctness",
        return_value=1.0,
    ), patch.object(atomic_function.time, "time", side_effect=lambda: next(fake_times)):
        result = evaluate_query_answer_pair(sample, "mistral")

    assert result.latency_seconds == 2.5
