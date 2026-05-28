import threading
import uuid

from src.core.request_context import (
    generate_request_id,
    get_request_id,
    set_request_id,
)


def test_generate_request_id_is_uuid4():
    rid = generate_request_id()
    parsed = uuid.UUID(rid)
    assert parsed.version == 4


def test_generate_request_id_is_unique():
    ids = {generate_request_id() for _ in range(100)}
    assert len(ids) == 100


def test_set_and_get_request_id():
    set_request_id("abc-123")
    assert get_request_id() == "abc-123"


def test_get_request_id_unknown_in_fresh_thread():
    captured = {}

    def worker():
        captured["value"] = get_request_id()

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert captured["value"] == "unknown"


def test_request_id_isolated_per_thread():
    set_request_id("main-thread")
    captured = {}

    def worker():
        set_request_id("worker-thread")
        captured["worker"] = get_request_id()

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert captured["worker"] == "worker-thread"
    assert get_request_id() == "main-thread"
