import requests
from typing import Dict
from tenacity import retry, stop_after_attempt, wait_exponential


OLLAMA_HOST = "http://localhost:11434"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def generate(model_name: str, prompt: str) -> Dict:
    """
    Calls local Ollama model and returns:
    - response text
    - token count (approx from API if available)
    """

    response = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )

    response.raise_for_status()
    data = response.json()

    return {
        "response": data.get("response", ""),
        "token_count": data.get("eval_count", 0),
    }
