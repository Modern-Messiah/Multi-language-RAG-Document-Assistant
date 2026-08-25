"""Request correlation, the access log, and readiness.

The two questions this exists to answer: "which log lines belong to the request
that failed", and "is this backend able to serve, or only able to answer a
socket". Both were unanswerable before.
"""
import logging

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.observability import (
    REQUEST_ID_HEADER,
    RequestIdFilter,
    configure_logging,
    current_request_id,
    new_request_id,
    readiness,
)
from tests.conftest import make_settings

# =========================
# The id itself
# =========================

def test_generated_ids_differ():
    assert new_request_id() != new_request_id()


def test_a_generated_id_is_short_enough_to_read_out():
    """It exists to be quoted by a person in a support message."""
    generated = new_request_id()

    assert len(generated) == 16
    assert generated.isalnum()


def test_there_is_no_id_outside_a_request():
    assert current_request_id() == ""


# =========================
# On the response
# =========================

def test_every_response_carries_an_id(api):
    for path, params in (
        ("/health", None),
        ("/ready", None),
        ("/documents", {"user_id": "u1"}),
    ):
        response = api.get(path, params=params)
        assert response.headers.get(REQUEST_ID_HEADER), f"{path} has no id"


def test_a_rejected_request_still_carries_an_id(api):
    """The failures are the ones worth correlating."""
    response = api.get("/documents", params={"user_id": "not a valid id"})

    assert response.status_code == 422
    assert response.headers.get(REQUEST_ID_HEADER)


def test_an_unauthenticated_request_carries_an_id(api):
    response = api.get("/documents", params={"user_id": "u1"},
                       headers={"X-API-Key": "wrong"})

    assert response.status_code == 401
    assert response.headers.get(REQUEST_ID_HEADER)


def test_two_requests_get_different_ids(api):
    first = api.get("/documents", params={"user_id": "u1"})
    second = api.get("/documents", params={"user_id": "u1"})

    assert first.headers[REQUEST_ID_HEADER] != second.headers[REQUEST_ID_HEADER]


def test_a_streaming_response_carries_an_id(api):
    """The SSE endpoint is why the middleware is raw ASGI rather than
    BaseHTTPMiddleware, so it needs its own check."""
    with api.stream(
        "POST", "/query/stream",
        json={"question": "hi", "language": "Auto", "user_id": "u1"},
    ) as response:
        assert response.headers.get(REQUEST_ID_HEADER)
        assert response.status_code == 200


# =========================
# An id supplied by the caller
# =========================

def test_a_callers_id_is_kept(api):
    """A proxy or a client-side trace should not be renamed halfway through."""
    response = api.get("/documents", params={"user_id": "u1"},
                       headers={REQUEST_ID_HEADER: "trace-abc.123_XYZ"})

    assert response.headers[REQUEST_ID_HEADER] == "trace-abc.123_XYZ"


@pytest.mark.parametrize("hostile", [
    "has spaces",
    "semi;colon",
    "a" * 65,
    "",
])
def test_an_unusable_id_is_replaced_not_echoed(api, hostile):
    """It reaches the log and the response, so it is untrusted input."""
    response = api.get("/documents", params={"user_id": "u1"},
                       headers={REQUEST_ID_HEADER: hostile})

    assert response.headers[REQUEST_ID_HEADER] != hostile
    assert len(response.headers[REQUEST_ID_HEADER]) == 16


@pytest.mark.parametrize("raw", [
    b"line\nbreak",
    b"\xd1\x89",          # utf-8 bytes, which is what a non-ASCII id arrives as
    b"\xff",              # not valid utf-8 at all
    b"tab\there",
])
def test_hostile_bytes_never_reach_the_log(raw):
    """Tested at the byte level because an HTTP client will not send these: a
    newline in a header value would let a caller forge log lines, and httpx
    refuses the request before it exists. A raw socket does not.
    """
    from app.observability import RequestContextMiddleware

    scope = {"type": "http", "headers": [(b"x-request-id", raw)]}

    assert RequestContextMiddleware._incoming_id(scope) == ""


# =========================
# In the log
# =========================

@pytest.fixture()
def stamped(caplog):
    """caplog captures through its own handler, so the filter has to be applied
    to it the way configure_logging applies it to the app's handlers."""
    added = RequestIdFilter()
    caplog.handler.addFilter(added)
    try:
        yield caplog
    finally:
        caplog.handler.removeFilter(added)


def test_the_id_reaches_log_lines_from_other_modules(api, stamped):
    """The point of the whole exercise: the lines from app.rag.* are the ones
    worth correlating, and they are written by code that knows nothing about
    HTTP."""
    with stamped.at_level(logging.INFO):
        response = api.post(
            "/upload",
            params={"user_id": "u1"},
            files={"file": ("notes.txt", b"Annual leave is 28 days.", "text/plain")},
        )

    assert response.status_code == 200
    request_id = response.headers[REQUEST_ID_HEADER]
    from_the_rag_layer = [r for r in stamped.records if r.name.startswith("app.rag.")]

    assert from_the_rag_layer, "nothing under app.rag logged, so this proves nothing"
    assert all(r.request_id == request_id for r in from_the_rag_layer), (
        [(r.name, r.request_id) for r in from_the_rag_layer]
    )


def test_lines_outside_a_request_say_so_rather_than_borrowing_an_id(stamped):
    with stamped.at_level(logging.INFO):
        logging.getLogger("app.rag.chain").info("startup line")

    assert stamped.records[-1].request_id == "-"


def test_one_access_line_per_request(api, caplog):
    with caplog.at_level(logging.INFO, logger="app.observability"):
        api.get("/documents", params={"user_id": "u1"})

    lines = [r.getMessage() for r in caplog.records]
    assert len(lines) == 1, lines
    assert "GET /documents -> 200" in lines[0]
    assert "user=u1" in lines[0]
    assert "ms" in lines[0]


def test_the_access_line_names_the_status_of_a_failure(api, caplog):
    with caplog.at_level(logging.INFO, logger="app.observability"):
        api.get("/documents", params={"user_id": "u1"}, headers={"X-API-Key": "no"})

    assert "-> 401" in caplog.records[0].getMessage()


def test_probe_traffic_is_not_logged_while_it_passes(api, caplog):
    """The Compose healthcheck polls every few seconds; logging each poll
    buries real traffic."""
    with caplog.at_level(logging.INFO, logger="app.observability"):
        api.get("/health")
        api.get("/ready")

    assert caplog.records == []


def test_a_failing_probe_is_logged(api, caplog):
    """Which is the one moment someone needs to see it."""
    api.app_state.embeddings.collection = None

    with caplog.at_level(logging.INFO, logger="app.observability"):
        response = api.get("/ready")

    assert response.status_code == 503
    assert "/ready -> 503" in caplog.records[-1].getMessage()


def test_an_unvalidated_user_id_cannot_forge_a_log_line(api, caplog):
    """The access line is written for rejected requests too, so at that point
    user_id has not been through the pattern check."""
    with caplog.at_level(logging.INFO, logger="app.observability"):
        api.get("/documents", params={"user_id": "u1\nINFO forged line"})

    message = caplog.records[0].getMessage()
    assert "forged" not in message, message
    assert "user=invalid" in message


def test_configure_logging_does_not_stack_filters():
    """It runs once per app instance and the suite builds many, so a duplicate
    filter per call would end up stamping thousands of records.

    Asserted on a handler this test owns: the root logger's handlers are shared
    with pytest, and other tests here add filters of their own.
    """
    handler = logging.StreamHandler()
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        configure_logging()
        configure_logging()
        configure_logging()

        stamps = [f for f in handler.filters if isinstance(f, RequestIdFilter)]
        assert len(stamps) == 1
    finally:
        root.removeHandler(handler)


def test_a_handler_installed_by_someone_else_gets_the_format():
    """basicConfig only formats handlers it created itself, so a handler that
    was already there would print records without the id column."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        configure_logging()

        assert "request_id" in handler.formatter._fmt
    finally:
        root.removeHandler(handler)


def test_the_log_format_has_a_place_for_the_id():
    from app.observability import LOG_FORMAT

    assert "%(request_id)s" in LOG_FORMAT


# =========================
# /ready
# =========================

def test_ready_is_ready(api):
    response = api.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {"startup": "ok", "vector_store": "ok"},
    }


def test_ready_needs_no_api_key(api):
    """An orchestrator's probe has no credentials."""
    assert api.get("/ready", headers={"X-API-Key": ""}).status_code == 200


def test_ready_is_503_before_startup_finishes(tmp_path, fake_openai_embeddings):
    """The process answers sockets well before the vector store is open, which
    is exactly the window in which /health lies."""
    app = create_app(make_settings(tmp_path))
    client = TestClient(app, raise_server_exceptions=False)  # no lifespan

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"] == {"startup": "failed"}


def test_health_still_answers_before_startup(tmp_path, fake_openai_embeddings):
    """Liveness and readiness are different questions; /health must keep
    answering or an orchestrator will restart a starting container."""
    app = create_app(make_settings(tmp_path))
    client = TestClient(app, raise_server_exceptions=False)

    assert client.get("/health").status_code == 200


def test_ready_notices_an_unusable_store(api):
    """count() answers 0 for a collection that was never opened, so the first
    version of this check passed while the store was unusable."""
    api.app_state.embeddings.collection = None

    response = api.get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["vector_store"] == "failed"


def test_ready_notices_a_store_that_raises(api):
    def boom():
        raise RuntimeError("disk gone")

    api.app_state.embeddings.ping = boom

    response = api.get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["vector_store"] == "failed"


def test_ready_does_not_leak_why(api):
    """It is unauthenticated, and an exception message carries paths."""
    def boom():
        raise RuntimeError("/srv/secret/chroma is unreadable")

    api.app_state.embeddings.ping = boom

    body = api.get("/ready").text

    assert "secret" not in body
    assert "RuntimeError" not in body


def test_readiness_does_not_call_openai(api):
    """Readiness must not depend on a third party's quota: a rate limit would
    otherwise pull every replica out of the load balancer."""
    calls = []
    api.app_state.rag_chain.client.chat.completions.create = (
        lambda **kwargs: calls.append(kwargs)
    )
    api.app_state.embeddings.embeddings.embed_query = (
        lambda text: calls.append(text)
    )

    assert api.get("/ready").status_code == 200
    assert calls == []


def test_readiness_reports_both_checks_by_name(api):
    ok, checks = readiness(api.app_state)

    assert ok
    assert set(checks) == {"startup", "vector_store"}


def test_ready_is_dispatched_off_the_event_loop(tmp_path):
    """It reads from ChromaDB, which blocks. As a coroutine it would stall
    every other request in the process while answering a probe - the same
    mistake the upload handler used to make."""
    import inspect

    app = create_app(make_settings(tmp_path))
    endpoint = next(
        route.endpoint for route in app.routes
        if getattr(route, "path", None) == "/ready"
    )

    assert not inspect.iscoroutinefunction(endpoint), (
        "/ready must be a plain def so FastAPI runs it in a threadpool"
    )


# =========================
# A crash
# =========================

def test_a_crash_answers_json_naming_the_request(api):
    """Without the handler, an unhandled exception produced a plain-text 500
    from the server, outside this app - so the response carried no id, and the
    one failure a user most needs to report was the one they could not point
    at."""
    from fastapi import APIRouter

    router = APIRouter()

    @router.get("/_boom")
    def boom():
        raise RuntimeError("unhandled")

    api.app.include_router(router)

    response = api.get("/_boom")

    assert response.status_code == 500
    body = response.json()
    assert body["detail"] == "Internal server error"
    assert body["request_id"] == response.headers[REQUEST_ID_HEADER]


def test_a_crash_does_not_leak_the_exception(api):
    from fastapi import APIRouter

    router = APIRouter()

    @router.get("/_boom2")
    def boom():
        raise RuntimeError("/srv/secret/path exploded")

    api.app.include_router(router)

    body = api.get("/_boom2").text

    assert "secret" not in body
    assert "RuntimeError" not in body


# =========================
# What the clients show
# =========================

def test_a_user_is_given_the_id_for_a_failure_they_cannot_fix():
    from clients.backend import describe_error

    message = describe_error(500, "Internal server error", request_id="abc123")

    assert "abc123" in message


@pytest.mark.parametrize("status", [401, 403, 500, 502, 503, 504])
def test_the_id_is_shown_for_operator_and_server_failures(status):
    from clients.backend import describe_error

    assert "abc123" in describe_error(status, "whatever", request_id="abc123")


@pytest.mark.parametrize("status", [400, 404, 409, 413, 422, 429])
def test_the_id_is_not_shown_where_the_user_can_act(status):
    """"Unsupported file format (request 4f2a...)" is noise: the message
    already says what to change."""
    from clients.backend import describe_error

    assert "abc123" not in describe_error(status, "Fix your file", request_id="abc123")


def test_no_id_no_parenthetical():
    from clients.backend import describe_error

    assert describe_error(500, "Boom") == "Boom"


def test_the_id_is_read_off_the_response_header():
    from clients.backend import error_from_response

    class Response:
        status_code = 503
        headers = {REQUEST_ID_HEADER: "from-header"}

        def json(self):
            return {"detail": "Backend unavailable"}

    assert "from-header" in error_from_response(Response())


def test_the_body_carries_the_id_when_a_proxy_strips_the_header():
    from clients.backend import error_from_response

    class Response:
        status_code = 500
        headers = {}

        def json(self):
            return {"detail": "Internal server error", "request_id": "from-body"}

    assert "from-body" in error_from_response(Response())


def test_a_response_without_headers_still_describes_the_error():
    """A 502 HTML page from a proxy has neither our header nor a JSON body."""
    from clients.backend import error_from_response

    class Response:
        status_code = 502

        def json(self):
            raise ValueError("not json")

    assert error_from_response(Response()) == "Backend error (HTTP 502)"
