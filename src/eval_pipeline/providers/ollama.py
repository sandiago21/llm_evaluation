"""Ollama HTTP client with retry/backoff."""

from __future__ import annotations

import logging

import httpx
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import InferenceConfig, OllamaConfig
from .base import GenerationResult, InferenceError, InferenceProvider

log = logging.getLogger(__name__)


class OllamaProvider(InferenceProvider):
    name = "ollama"

    def __init__(self, ollama_cfg: OllamaConfig, inference_cfg: InferenceConfig) -> None:
        self._cfg = ollama_cfg
        self._inf = inference_cfg
        self._client = httpx.Client(
            base_url=ollama_cfg.base_url,
            timeout=ollama_cfg.request_timeout_s,
        )

    def generate(self, prompt: str, model: str, *, temperature: float | None = None) -> GenerationResult:
        @retry(
            reraise=True,
            stop=stop_after_attempt(self._inf.max_retries),
            wait=wait_exponential(
                multiplier=self._inf.retry_initial_backoff_s,
                max=self._inf.retry_max_backoff_s,
            ),
            retry=retry_if_exception_type((httpx.HTTPError, InferenceError)),
        )
        def _call() -> GenerationResult:
            payload: dict = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self._inf.temperature if temperature is None else temperature,
                    "num_predict": self._inf.max_tokens,
                },
            }
            try:
                resp = self._client.post(self._cfg.generate_path, json=payload)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                log.warning("ollama request failed", extra={"model": model, "error": str(exc)})
                raise
            data = resp.json()
            text = data.get("response", "")
            # Ollama returns eval_count for generated tokens; fall back to whitespace count.
            tokens = int(data.get("eval_count") or len(text.split()))
            if not text:
                raise InferenceError(f"empty response from ollama for model {model!r}")
            return GenerationResult(text=text, token_count=tokens)

        try:
            return _call()
        except RetryError as exc:  # pragma: no cover - tenacity reraises by default
            raise InferenceError(str(exc)) from exc

    def close(self) -> None:
        self._client.close()
