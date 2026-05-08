"""Structured JSON logging with request-ID propagation via contextvars."""

from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar

try:
    from pythonjsonlogger.json import JsonFormatter
except ImportError:  # older python-json-logger
    from pythonjsonlogger.jsonlogger import JsonFormatter  # type: ignore[no-redef]

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def configure_logging(level: str = "INFO", json: bool = True) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    handler = logging.StreamHandler(stream=sys.stdout)
    if json:
        formatter: logging.Formatter = JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s %(request_id)s",
            rename_fields={"asctime": "ts", "levelname": "level"},
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] [req=%(request_id)s] %(message)s"
        )
    handler.setFormatter(formatter)
    handler.addFilter(RequestIdFilter())
    root.addHandler(handler)


def new_request_id() -> str:
    rid = uuid.uuid4().hex
    request_id_var.set(rid)
    return rid


def set_request_id(rid: str | None) -> None:
    request_id_var.set(rid)


def get_request_id() -> str | None:
    return request_id_var.get()
