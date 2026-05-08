"""Provider protocol and shared types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class InferenceError(RuntimeError):
    """Raised when a provider fails to generate after exhausting retries."""


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_count: int


@runtime_checkable
class InferenceProvider(Protocol):
    name: str

    def generate(self, prompt: str, model: str, *, temperature: float | None = None) -> GenerationResult:
        ...
