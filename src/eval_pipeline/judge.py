"""LLM-as-a-Judge correctness scoring."""

from __future__ import annotations

import logging

from .config import JudgeConfig
from .providers.base import InferenceError, InferenceProvider

log = logging.getLogger(__name__)


class Judge:
    """Wraps a provider+model for correctness grading.

    The judge formats a prompt comparing the expected and generated answers,
    asks the model for a single-token verdict, and returns a boolean.
    """

    def __init__(self, provider: InferenceProvider, cfg: JudgeConfig) -> None:
        self._provider = provider
        self._cfg = cfg

    def score(self, query: str, expected_answer: str, generated_answer: str) -> bool | None:
        if not self._cfg.enabled:
            return None
        prompt = self._cfg.prompt_template.format(
            query=query,
            expected_answer=expected_answer,
            generated_answer=generated_answer,
        )
        try:
            result = self._provider.generate(
                prompt,
                self._cfg.model,
                temperature=self._cfg.temperature,
            )
        except InferenceError as exc:
            log.warning("judge call failed", extra={"error": str(exc)})
            return None
        verdict = result.text.strip().upper()
        if verdict.startswith("CORRECT"):
            return True
        if verdict.startswith("INCORRECT"):
            return False
        # Some models echo the prompt or add filler — look for the keyword.
        if "INCORRECT" in verdict:
            return False
        if "CORRECT" in verdict:
            return True
        log.warning("unparseable judge verdict", extra={"verdict": verdict[:200]})
        return None
