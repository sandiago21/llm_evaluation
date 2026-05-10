from pydantic import BaseModel, Field

import time
from typing import Dict, Any

from src.models.dataset import QueryAnswerSample, InferenceResult
from src.evaluation.ollama_client import generate
from src.evaluation.judge import judge_correctness


from dataclasses import dataclass




def evaluate_query_answer_pair(
    query_answer: QueryAnswerSample,
    model_name: str,
) -> InferenceResult:
    """
    Atomic evaluation function for a single (QueryAnswerSample, model) pair.

    Returns:
        atomic_result: InferenceResult

    """

    # -------------------------
    # 1. Measure generation time
    # -------------------------
    start_time = time.time()

    generation_result = generate(
        model_name=model_name,
        prompt=query_answer.query,
    )

    latency_seconds = time.time() - start_time

    generated_answer = generation_result["response"]
    token_count = generation_result.get("token_count", 0)

    # -------------------------
    # 2. LLM-as-a-Judge scoring
    # -------------------------
    correctness = judge_correctness(
        expected_answer=query_answer.expected_answer,
        generated_answer=generated_answer,
    )

    # -------------------------
    # 3. Return structured output
    # -------------------------

    atomic_result = InferenceResult(
        generated_answer=generated_answer,
        latency_seconds=latency_seconds,
        token_count=token_count,
        correctness=correctness,
    )

    return atomic_result
