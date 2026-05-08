import threading
import uuid


_request_context = threading.local()


def set_request_id(request_id: str):
    _request_context.request_id = request_id


def get_request_id() -> str:

    return getattr(
        _request_context,
        "request_id",
        "unknown",
    )


def generate_request_id() -> str:
    return str(uuid.uuid4())
