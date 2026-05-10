from typing import Union
from src.evaluation.ollama_client import generate
from src.core.config import (
    JUDGE_MODEL,
)


JUDGE_PROMPT = """
You are an expert evaluator.

Compare the expected answer and generated answer.

Expected Answer:
{expected}

Generated Answer:
{generated}

Return ONLY a score between 0 and 1:
- 1.0 = fully correct
- 0.5 = partially correct
- 0.0 = incorrect

Do not explain. Output only the number.
"""


def judge_correctness(
    expected_answer: str,
    generated_answer: str,
) -> float:
    """
    Uses an LLM to evaluate correctness of a model output.
    """

    prompt = JUDGE_PROMPT.format(
        expected=expected_answer,
        generated=generated_answer,
    )

    result = generate(
        model_name=JUDGE_MODEL,
        prompt=prompt,
    )

    raw_output = result.get("response", "").strip()

    try:
        score = float(raw_output)
    except Exception:
        score = 0.0  # fallback for malformed judge output

    # clamp score into valid range
    score = max(0.0, min(1.0, score))

    return score
