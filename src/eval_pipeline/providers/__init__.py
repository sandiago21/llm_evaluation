"""Inference provider abstraction.

Providers expose a single ``generate(prompt, model)`` method returning a
``GenerationResult``. The factory ``get_provider`` selects an implementation
based on the application settings.
"""

from __future__ import annotations

from ..config import Settings
from .base import GenerationResult, InferenceError, InferenceProvider
from .mock import MockProvider
from .ollama import OllamaProvider


def get_provider(settings: Settings, name: str | None = None) -> InferenceProvider:
    """Build a provider instance by name; defaults to ``settings.provider``."""
    selected = (name or settings.provider).lower()
    if selected == "mock":
        return MockProvider()
    if selected == "ollama":
        return OllamaProvider(settings.ollama, settings.inference)
    raise ValueError(f"Unknown provider: {selected!r}")


__all__ = [
    "GenerationResult",
    "InferenceError",
    "InferenceProvider",
    "MockProvider",
    "OllamaProvider",
    "get_provider",
]
