"""Configuration loading: YAML defaults + environment overrides (env wins)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

ENV_PREFIX = "EVAL_"
ENV_NESTED_DELIMITER = "__"
DEFAULT_CONFIG_PATH = Path("config/default.yaml")


class OllamaConfig(BaseModel):
    base_url: str = "http://ollama:11434"
    request_timeout_s: float = 120.0
    generate_path: str = "/api/generate"


class JudgeConfig(BaseModel):
    enabled: bool = True
    provider: Literal["ollama", "mock"] = "ollama"
    model: str = "llama3.2"
    temperature: float = 0.0
    prompt_template: str = (
        "Question: {query}\nExpected: {expected_answer}\n"
        "Assistant: {generated_answer}\nReply CORRECT or INCORRECT."
    )


class InferenceConfig(BaseModel):
    temperature: float = 0.0
    max_tokens: int = 512
    request_timeout_s: float = 120.0
    max_retries: int = 3
    retry_initial_backoff_s: float = 1.0
    retry_max_backoff_s: float = 16.0


class CacheConfig(BaseModel):
    directory: str = "./cache"


class RunnerConfig(BaseModel):
    max_workers_per_model: int = 4


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class LoggingConfig(BaseModel):
    """Logging settings.

    The YAML/env key remains ``json`` (e.g. ``EVAL_LOGGING__JSON=false``); the
    Python attribute is ``as_json`` to avoid shadowing ``BaseModel.json``.
    """

    level: str = "INFO"
    as_json: bool = Field(default=True, alias="json", description="Emit logs as JSON.")

    model_config = {"populate_by_name": True}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a mapping at the top level")
    return data


class _YamlSource(PydanticBaseSettingsSource):
    """Pydantic-settings source that returns values from a YAML file."""

    def __init__(self, settings_cls: type[BaseSettings], path: Path) -> None:
        super().__init__(settings_cls)
        self._data = _load_yaml(path)

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        value = self._data.get(field_name)
        return value, field_name, False

    def __call__(self) -> dict[str, Any]:
        return dict(self._data)


def _yaml_path() -> Path:
    return Path(os.environ.get("EVAL_CONFIG_FILE", DEFAULT_CONFIG_PATH))


class Settings(BaseSettings):
    """Application settings.

    Resolution order (highest priority first): init kwargs → environment
    variables (``EVAL_*`` with ``__`` as nesting delimiter) → YAML file
    (``config/default.yaml`` or ``EVAL_CONFIG_FILE``).
    """

    provider: Literal["ollama", "mock"] = "ollama"
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    runner: RunnerConfig = Field(default_factory=RunnerConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter=ENV_NESTED_DELIMITER,
        case_sensitive=False,
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _YamlSource(settings_cls, _yaml_path()),
            file_secret_settings,
        )


def load_settings(config_path: Path | str | None = None) -> Settings:
    """Construct Settings using YAML + env (env overrides YAML)."""
    if config_path is not None:
        os.environ["EVAL_CONFIG_FILE"] = str(config_path)
    return Settings()


__all__ = [
    "APIConfig",
    "CacheConfig",
    "InferenceConfig",
    "JudgeConfig",
    "LoggingConfig",
    "OllamaConfig",
    "RunnerConfig",
    "Settings",
    "load_settings",
]
