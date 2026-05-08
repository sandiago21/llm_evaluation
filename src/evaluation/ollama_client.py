import requests

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from requests.exceptions import (
    Timeout,
    ConnectionError,
    HTTPError,
)

from src.core.config import (
    OLLAMA_HOST,
    OLLAMA_TIMEOUT_SECONDS,
)

from src.core.logging import logger
from src.core.request_context import (
    get_request_id,
)



# -------------------------
# Retry policy
# -------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(
        multiplier=1,
        min=2,
        max=10,
    ),
    retry=retry_if_exception_type(
        (
            Timeout,
            ConnectionError,
            HTTPError,
        )
    ),
)
def generate(
    model_name: str,
    prompt: str,
):

    request_id = get_request_id()

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
    }

    logger.info(
        f"Starting inference with model={model_name}",
        extra={
            "request_id": request_id,
        },
    )

    try:

        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json=payload,
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )

        response.raise_for_status()

        data = response.json()

        logger.info(
            f"Inference completed model={model_name}",
            extra={
                "request_id": request_id,
            },
        )

        return {
            "response": data.get("response", ""),
            "token_count": data.get("eval_count", 0),
        }

    except Timeout:

        logger.exception(
            f"Timeout during inference model={model_name}",
            extra={
                "request_id": request_id,
            },
        )

        raise

    except HTTPError:

        logger.exception(
            f"HTTP error during inference model={model_name}",
            extra={
                "request_id": request_id,
            },
        )

        raise

    except ConnectionError:

        logger.exception(
            f"Connection error during inference model={model_name}",
            extra={
                "request_id": request_id,
            },
        )

        raise

    except Exception:

        logger.exception(
            f"Unexpected inference failure model={model_name}",
            extra={
                "request_id": request_id,
            },
        )

        raise
