import json
import logging

from src.core.logging import JsonFormatter


def _record(**kwargs):
    defaults = dict(
        name="scoring-api",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    defaults.update(kwargs)
    return logging.LogRecord(**defaults)


def test_formatter_emits_required_fields():
    record = _record()
    out = json.loads(JsonFormatter().format(record))

    assert out["level"] == "INFO"
    assert out["message"] == "hello"
    assert out["module"]
    assert "timestamp" in out


def test_formatter_renders_format_args():
    record = _record(msg="value=%s", args=("42",))
    out = json.loads(JsonFormatter().format(record))
    assert out["message"] == "value=42"


def test_formatter_includes_request_id_when_present():
    record = _record()
    record.request_id = "req-7"
    out = json.loads(JsonFormatter().format(record))
    assert out["request_id"] == "req-7"


def test_formatter_omits_request_id_when_absent():
    record = _record()
    out = json.loads(JsonFormatter().format(record))
    assert "request_id" not in out


def test_formatter_includes_exception():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys
        record = _record(exc_info=sys.exc_info())
        out = json.loads(JsonFormatter().format(record))
        assert "ValueError" in out["exception"]
        assert "boom" in out["exception"]
