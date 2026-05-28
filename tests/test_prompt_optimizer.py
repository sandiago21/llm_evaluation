from unittest.mock import patch

from src.prompts import prompt_optimizer
from src.prompts.prompt_optimizer import optimize_prompt


def test_optimize_returns_best_prompt_across_iterations(eval_dataset):
    """Optimizer must pick the highest-scoring prompt seen so far."""
    scores = iter([0.4, 0.9, 0.6])
    rewrites = iter(["rewrite-1", "rewrite-2", "rewrite-3"])

    def fake_evaluate(model_name, prompt_template, dataset):
        return next(scores), "failures"

    def fake_rewrite(current_prompt, score, failure_examples=None):
        return next(rewrites)

    with patch.object(prompt_optimizer, "evaluate_prompt", side_effect=fake_evaluate), \
         patch.object(prompt_optimizer, "rewrite_prompt", side_effect=fake_rewrite):
        result = optimize_prompt(
            weak_model="mistral",
            validation_dataset=eval_dataset,
            initial_prompt="initial",
            iterations=3,
        )

    # Iteration 0 evaluates "initial" (0.4)
    # Iteration 1 evaluates "rewrite-1" (0.9, new best)
    # Iteration 2 evaluates "rewrite-2" (0.6, not better)
    assert result["best_score"] == 0.9
    assert result["best_prompt"] == "rewrite-1"
    assert len(result["history"]) == 3


def test_optimize_history_records_each_iteration(eval_dataset):
    with patch.object(prompt_optimizer, "evaluate_prompt", return_value=(0.5, "")), \
         patch.object(prompt_optimizer, "rewrite_prompt", return_value="next-prompt"):
        result = optimize_prompt(
            weak_model="mistral",
            validation_dataset=eval_dataset,
            initial_prompt="start",
            iterations=4,
        )

    assert [h["iteration"] for h in result["history"]] == [0, 1, 2, 3]
    assert all(h["score"] == 0.5 for h in result["history"])


def test_optimize_rewrites_from_best_prompt_not_latest(eval_dataset):
    """After a regression, the next rewrite should be seeded from the
    best-so-far prompt, not from the most recently scored candidate."""
    scores = iter([0.4, 0.9, 0.1])
    rewrites = iter(["after-initial", "after-best", "after-best-again"])
    rewrite_inputs = []

    def fake_evaluate(model_name, prompt_template, dataset):
        return next(scores), "failures"

    def fake_rewrite(current_prompt, score, failure_examples=None):
        rewrite_inputs.append(current_prompt)
        return next(rewrites)

    with patch.object(prompt_optimizer, "evaluate_prompt", side_effect=fake_evaluate), \
         patch.object(prompt_optimizer, "rewrite_prompt", side_effect=fake_rewrite):
        optimize_prompt(
            weak_model="mistral",
            validation_dataset=eval_dataset,
            initial_prompt="initial",
            iterations=3,
        )

    # 1st rewrite seeded from "initial" (best after iter 0).
    # 2nd rewrite seeded from "after-initial" (best after iter 1, score 0.9).
    # 3rd rewrite ALSO seeded from "after-initial" — iter 2 score 0.1 didn't beat best.
    assert rewrite_inputs[0] == "initial"
    assert rewrite_inputs[1] == "after-initial"
    assert rewrite_inputs[2] == "after-initial"


def test_optimize_zero_iterations_returns_neutral_state(eval_dataset):
    with patch.object(prompt_optimizer, "evaluate_prompt") as ev, \
         patch.object(prompt_optimizer, "rewrite_prompt") as rw:
        result = optimize_prompt(
            weak_model="mistral",
            validation_dataset=eval_dataset,
            initial_prompt="initial",
            iterations=0,
        )

    ev.assert_not_called()
    rw.assert_not_called()
    assert result["best_prompt"] == "initial"
    assert result["best_score"] == float("-inf")
    assert result["history"] == []
