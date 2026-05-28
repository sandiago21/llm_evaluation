from unittest.mock import MagicMock, patch

import pytest
from requests.exceptions import ConnectionError, HTTPError, Timeout
from tenacity import RetryError, wait_none

from src.evaluation import ollama_client


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    """Skip tenacity's exponential backoff so retry tests don't sleep."""
    monkeypatch.setattr(ollama_client.generate.retry, "wait", wait_none())
    ollama_client.generate.retry.statistics.clear()
    yield


def _mock_response(payload=None, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload or {"response": "hi", "eval_count": 7}
    resp.raise_for_status.return_value = None
    return resp


def test_generate_returns_response_and_token_count():
    with patch.object(ollama_client.requests, "post", return_value=_mock_response()) as post:
        result = ollama_client.generate(model_name="mistral", prompt="hello")

    assert result == {"response": "hi", "token_count": 7}
    post.assert_called_once()


def test_generate_sends_correct_payload():
    with patch.object(ollama_client.requests, "post", return_value=_mock_response()) as post:
        ollama_client.generate(model_name="mistral", prompt="hello")

    _, kwargs = post.call_args
    assert kwargs["json"] == {
        "model": "mistral",
        "prompt": "hello",
        "stream": False,
    }
    assert "timeout" in kwargs


def test_generate_handles_missing_token_count():
    with patch.object(
        ollama_client.requests,
        "post",
        return_value=_mock_response({"response": "ok"}),
    ):
        result = ollama_client.generate(model_name="mistral", prompt="hi")

    assert result["token_count"] == 0


def test_generate_retries_on_timeout_then_succeeds():
    call_count = {"n": 0}

    def side_effect(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise Timeout("slow")
        return _mock_response()

    with patch.object(ollama_client.requests, "post", side_effect=side_effect):
        result = ollama_client.generate(model_name="mistral", prompt="hi")

    assert result["response"] == "hi"
    assert call_count["n"] == 2


def test_generate_retries_on_connection_error_then_gives_up():
    with patch.object(ollama_client.requests, "post", side_effect=ConnectionError("nope")) as post:
        with pytest.raises(RetryError) as excinfo:
            ollama_client.generate(model_name="mistral", prompt="hi")

    assert post.call_count == 3
    # Underlying cause should still be the original ConnectionError.
    assert isinstance(excinfo.value.last_attempt.exception(), ConnectionError)


def test_generate_does_not_retry_on_unexpected_exception():
    with patch.object(ollama_client.requests, "post", side_effect=ValueError("bug")) as post:
        with pytest.raises(ValueError):
            ollama_client.generate(model_name="mistral", prompt="hi")

    assert post.call_count == 1


def test_generate_raises_on_http_error_after_retries():
    resp = MagicMock()
    resp.raise_for_status.side_effect = HTTPError("500")
    with patch.object(ollama_client.requests, "post", return_value=resp) as post:
        with pytest.raises(RetryError) as excinfo:
            ollama_client.generate(model_name="mistral", prompt="hi")

    assert post.call_count == 3
    assert isinstance(excinfo.value.last_attempt.exception(), HTTPError)
