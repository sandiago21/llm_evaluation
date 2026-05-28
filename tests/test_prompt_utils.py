from unittest.mock import patch

from src.models.dataset import EvaluationDataset, InferenceResult
from src.prompts import utils as prompt_utils
from src.prompts.utils import evaluate_prompt, rewrite_prompt


def test_rewrite_prompt_returns_stripped_response():
    with patch.object(
        prompt_utils,
        "generate",
        return_value={"response": "  better prompt  \n", "token_count": 4},
    ):
        out = rewrite_prompt("base", 0.5)

    assert out == "better prompt"


def test_rewrite_prompt_includes_failure_examples_when_provided():
    with patch.object(
        prompt_utils,
        "generate",
        return_value={"response": "x", "token_count": 1},
    ) as gen:
        rewrite_prompt("base", 0.5, failure_examples="Q: foo  A: bar")

    sent = gen.call_args.kwargs["prompt"]
    assert "Q: foo" in sent
    assert "Failure cases" in sent


def test_rewrite_prompt_omits_failure_section_when_none():
    with patch.object(
        prompt_utils,
        "generate",
        return_value={"response": "x", "token_count": 1},
    ) as gen:
        rewrite_prompt("base", 0.5)

    assert "Failure cases" not in gen.call_args.kwargs["prompt"]


def test_rewrite_prompt_uses_optimizer_model():
    with patch.object(
        prompt_utils,
        "generate",
        return_value={"response": "x", "token_count": 1},
    ) as gen:
        rewrite_prompt("base", 0.5)

    assert gen.call_args.kwargs["model_name"] == prompt_utils.MODEL


def test_evaluate_prompt_applies_template_to_queries_without_mutating_input():
    dataset = EvaluationDataset.from_samples(
        [
            {"query": "raw1", "expected_answer": "e1"},
            {"query": "raw2", "expected_answer": "e2"},
        ]
    )

    captured_templated_dataset = {}

    def fake_run(templated_dataset, model_names):
        captured_templated_dataset["ds"] = templated_dataset
        templated_dataset.inferences[model_names[0]] = [
            InferenceResult("ok1", 0.1, 1, 1.0),
            InferenceResult("ok2", 0.1, 1, 0.0),
        ]
        return templated_dataset

    with patch.object(prompt_utils, "run_multi_model_evaluation", side_effect=fake_run):
        score, failures = evaluate_prompt(
            "mistral",
            "Answer: {query}",
            dataset,
        )

    # Score is mean correctness across the two samples.
    assert score == 0.5

    # Templated dataset must carry the formatted prompt, not the raw query.
    templated_queries = [s.query for s in captured_templated_dataset["ds"].samples]
    assert templated_queries == ["Answer: raw1", "Answer: raw2"]

    # The original dataset should be untouched.
    assert [s.query for s in dataset.samples] == ["raw1", "raw2"]


def test_evaluate_prompt_failure_examples_target_worst_samples():
    dataset = EvaluationDataset.from_samples(
        [
            {"query": "q1", "expected_answer": "e1"},
            {"query": "q2", "expected_answer": "e2"},
            {"query": "q3", "expected_answer": "e3"},
        ]
    )

    def fake_run(templated_dataset, model_names):
        templated_dataset.inferences[model_names[0]] = [
            InferenceResult("good", 0.1, 1, 1.0),
            InferenceResult("bad", 0.1, 1, 0.0),
            InferenceResult("mid", 0.1, 1, 0.5),
        ]
        return templated_dataset

    with patch.object(prompt_utils, "run_multi_model_evaluation", side_effect=fake_run):
        _, failures = evaluate_prompt(
            "mistral",
            "{query}",
            dataset,
            n_failure_examples=2,
        )

    # The worst (correctness 0.0) sample must appear; the perfect one must not.
    assert "Generated: bad" in failures
    assert "Generated: good" not in failures
