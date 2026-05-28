from unittest.mock import patch

import pytest

from src.evaluation import judge as judge_module
from src.evaluation.judge import judge_correctness


def _mock_generate(response):
    return patch.object(
        judge_module,
        "generate",
        return_value={"response": response, "token_count": 1},
    )


def test_judge_correctness_parses_float():
    with _mock_generate("0.5"):
        assert judge_correctness("expected", "generated") == 0.5


def test_judge_correctness_handles_one_point_zero():
    with _mock_generate("1.0"):
        assert judge_correctness("a", "a") == 1.0


def test_judge_correctness_handles_zero():
    with _mock_generate("0"):
        assert judge_correctness("a", "b") == 0.0


def test_judge_correctness_strips_whitespace():
    with _mock_generate("  0.75\n"):
        assert judge_correctness("a", "a") == 0.75


def test_judge_correctness_clamps_above_one():
    with _mock_generate("2.5"):
        assert judge_correctness("a", "a") == 1.0


def test_judge_correctness_clamps_below_zero():
    with _mock_generate("-0.5"):
        assert judge_correctness("a", "a") == 0.0


def test_judge_correctness_falls_back_to_zero_on_malformed_output():
    with _mock_generate("totally not a number"):
        assert judge_correctness("a", "b") == 0.0


def test_judge_correctness_sends_prompt_with_expected_and_generated():
    with patch.object(
        judge_module,
        "generate",
        return_value={"response": "1.0", "token_count": 1},
    ) as gen:
        judge_correctness("EXP", "GEN")

    prompt = gen.call_args.kwargs["prompt"]
    assert "EXP" in prompt
    assert "GEN" in prompt


def test_judge_correctness_uses_configured_judge_model():
    with patch.object(
        judge_module,
        "generate",
        return_value={"response": "1.0", "token_count": 1},
    ) as gen:
        judge_correctness("a", "b")

    assert gen.call_args.kwargs["model_name"] == judge_module.JUDGE_MODEL
