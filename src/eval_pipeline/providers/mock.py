"""Deterministic offline provider used for tests and CI.

The mock derives its response from a SHA-256 of (model, prompt) so different
models produce different but reproducible outputs. Token count is the simple
whitespace count of the generated text. The provider also recognises a small
"judge" protocol: if the prompt asks for CORRECT/INCORRECT we synthesise a
response based on a string-overlap heuristic between the expected and
generated answers found inside the prompt template.
"""

from __future__ import annotations

import hashlib
import re

from .base import GenerationResult, InferenceProvider


class MockProvider(InferenceProvider):
    name = "mock"

    _JUDGE_RE = re.compile(
        r"Expected[^:]*:\s*(?P<expected>.*?)\n.*?Assistant[^:]*:\s*(?P<actual>.*?)\n",
        re.DOTALL | re.IGNORECASE,
    )

    def generate(self, prompt: str, model: str, *, temperature: float | None = None) -> GenerationResult:
        if "CORRECT or INCORRECT" in prompt or "Reply CORRECT" in prompt:
            return self._judge(prompt)
        digest = hashlib.sha256(f"{model}\x00{prompt}".encode()).hexdigest()
        # Synthesize a short, model-flavoured answer.
        words = [
            "answer",
            "response",
            "summary",
            "explanation",
            "context",
            "detail",
        ]
        idx = int(digest[:8], 16) % len(words)
        text = f"[{model}] mock {words[idx]} for: {prompt[:64].strip()} ({digest[:6]})"
        return GenerationResult(text=text, token_count=len(text.split()))

    def _judge(self, prompt: str) -> GenerationResult:
        match = self._JUDGE_RE.search(prompt)
        if not match:
            return GenerationResult(text="INCORRECT", token_count=1)
        expected = _tokens(match.group("expected"))
        actual = _tokens(match.group("actual"))
        if not expected:
            return GenerationResult(text="INCORRECT", token_count=1)
        overlap = len(expected & actual) / len(expected)
        verdict = "CORRECT" if overlap >= 0.5 else "INCORRECT"
        return GenerationResult(text=verdict, token_count=1)


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {m.lower() for m in _TOKEN_RE.findall(text)}
