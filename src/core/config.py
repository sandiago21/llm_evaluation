import os
import yaml


with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)


JUDGE_MODEL = os.getenv(
    "JUDGE_MODEL",
    config["judge"]["model"],
)

JUDGE_TEMPERATURE = float(
    os.getenv(
        "JUDGE_TEMPERATURE",
        config["judge"]["temperature"],
    )
)

JUDGE_MAX_TOKENS = int(
    os.getenv(
        "JUDGE_MAX_TOKENS",
        config["judge"]["max_tokens"],
    )
)