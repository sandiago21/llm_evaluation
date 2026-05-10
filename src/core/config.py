import os
import yaml


# -------------------------
# Load YAML config
# -------------------------

with open("configs/config.yaml", "r") as f:
    yaml_config = yaml.safe_load(f)


# -------------------------
# Ollama Configuration
# -------------------------

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    yaml_config["ollama"]["host"],
)

OLLAMA_TIMEOUT_SECONDS = int(
    os.getenv(
        "OLLAMA_TIMEOUT_SECONDS",
        yaml_config["ollama"]["timeout_seconds"],
    )
)


# -------------------------
# Judge Configuration
# -------------------------

JUDGE_MODEL = os.getenv(
    "JUDGE_MODEL",
    yaml_config["judge"]["model"],
)

JUDGE_PROMPT_VERSION = os.getenv(
    "JUDGE_PROMPT_VERSION",
    yaml_config["judge"]["prompt_version"],
)

JUDGE_TEMPERATURE = float(
    os.getenv(
        "JUDGE_TEMPERATURE",
        yaml_config["judge"]["temperature"],
    )
)

JUDGE_MAX_TOKENS = int(
    os.getenv(
        "JUDGE_MAX_TOKENS",
        yaml_config["judge"]["max_tokens"],
    )
)


# -------------------------
# Cache Configuration
# -------------------------

CACHE_DIRECTORY = os.getenv(
    "CACHE_DIRECTORY",
    yaml_config["cache"]["directory"],
)


# -------------------------
# API Configuration
# -------------------------

API_HOST = os.getenv(
    "API_HOST",
    yaml_config["api"]["host"],
)

API_PORT = int(
    os.getenv(
        "API_PORT",
        yaml_config["api"]["port"],
    )
)
