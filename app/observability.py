"""Request correlation and readiness.

Two questions were unanswerable before this module existed.

"A user says the assistant failed at 14:32 - which log lines are theirs?" The
log held interleaved lines from every concurrent request, with nothing tying
them together. Handlers run in a threadpool, so even the order was not a hint.

"Is the backend ready to serve, or merely running?" `/health` returns 200 as
soon as the process answers a socket, which it does before the vector store is
open. An orchestrator that routes on `/health` sends traffic into 503s.

So: every request carries an id, every log line records it, and the id comes
back in a response header so a user can quote it. And `/ready` reports whether
the components a request actually needs are usable.
"""
import logging
import re
import secrets
import time
from contextvars import ContextVar
from urllib.parse import parse_qs

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

# Where the middleware leaves the id on the ASGI scope. The context variable is
# the normal way to read it, but an error handler running *outside* this
# middleware (Starlette's ServerErrorMiddleware is the outermost layer) sees the
# same scope dict and nothing else.
SCOPE_KEY = "rag.request_id"

_request_id: ContextVar[str] = ContextVar("request_id", default="")

# Ids reach the log and are echoed to the client, so a client-supplied one is
# untrusted input: a newline in it would let a caller forge log lines, and 4 KB
# of it would bury the message. Anything outside this shape is replaced with a
# generated id rather than rejected - the request is not the caller's fault to
# fix, and refusing it would break a proxy that stamps a format we did not
# anticipate.
_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

# 16 hex characters: enough that ids do not collide in any log a person will
# read, short enough to say out loud, and the same length as the file hashes
# already in this API.
_ID_BYTES = 8

# user_id is echoed into the access line, and at that point it has not been
# validated yet - the line is written for rejected requests too.
_USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Probe endpoints, polled every few seconds by the Compose healthcheck. Logging
# each poll buries real traffic, so they are logged only when they fail: a
# readiness probe that starts failing is exactly what someone needs to see.
_QUIET_PATHS = frozenset({"/health", "/ready"})


def new_request_id() -> str:
    return secrets.token_hex(_ID_BYTES)


def current_request_id() -> str:
    """The id of the request being served, or "" outside a request."""
    return _request_id.get()


def request_id_of(request) -> str:
    """The id of a request, for a handler that has the Request object.

    Prefers the scope over the context variable so it also works from an error
    handler outside the middleware, where the variable has been reset.
    """
    return request.scope.get(SCOPE_KEY) or current_request_id()


class RequestIdFilter(logging.Filter):
    """Add `request_id` to every record so the format string can use it.

    Attached to handlers, not loggers: a filter on a logger only sees records
    logged through that logger, while a filter on a handler sees everything the
    handler emits - including records from `app.rag.*`, which is where the
    interesting lines are.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            # "-" rather than "" so the column stays visible for startup lines
            # and background work, which belong to no request.
            record.request_id = current_request_id() or "-"
        return True


LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Install the format and the filter. Safe to call more than once."""
    logging.basicConfig(level=level, format=LOG_FORMAT)

    root = logging.getLogger()
    for handler in root.handlers:
        if not any(isinstance(f, RequestIdFilter) for f in handler.filters):
            handler.addFilter(RequestIdFilter())
        # basicConfig only installs a formatter the first time it creates a
        # handler; a handler someone else installed keeps its own, which would
        # drop the id.
        if handler.formatter is None or "request_id" not in (
            handler.formatter._fmt or ""
        ):
            handler.setFormatter(logging.Formatter(LOG_FORMAT))


def _user_id_of(scope) -> str:
    raw = parse_qs(scope.get("query_string", b"").decode("latin-1")).get("user_id", [""])
    candidate = raw[0]
    if not candidate:
        return "-"
    # Unvalidated at this point, and it goes straight into a log line.
    return candidate if _USER_ID_PATTERN.match(candidate) else "invalid"


class RequestContextMiddleware:
    """Assign an id, echo it back, and log one line per request.

    Written as raw ASGI rather than with `@app.middleware("http")`: that
    decorator wraps the response in an extra task, which is a poor fit for the
    SSE endpoint that streams for as long as an answer takes.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = self._incoming_id(scope) or new_request_id()
        scope[SCOPE_KEY] = request_id
        token = _request_id.set(request_id)

        status = {"code": 500}
        header = (REQUEST_ID_HEADER.lower().encode("latin-1"), request_id.encode("ascii"))

        async def send_with_id(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                headers = message.setdefault("headers", [])
                if not any(name == header[0] for name, _ in headers):
                    headers.append(header)
            await send(message)

        started = time.perf_counter()
        try:
            await self.app(scope, receive, send_with_id)
        finally:
            self._log(scope, status["code"], time.perf_counter() - started)
            _request_id.reset(token)

    @staticmethod
    def _incoming_id(scope) -> str:
        wanted = REQUEST_ID_HEADER.lower().encode("latin-1")
        for name, value in scope.get("headers", []):
            if name == wanted:
                candidate = value.decode("latin-1")
                return candidate if _ID_PATTERN.match(candidate) else ""
        return ""

    @staticmethod
    def _log(scope, status: int, seconds: float) -> None:
        path = scope.get("path", "")
        if path in _QUIET_PATHS and status < 400:
            return
        logger.info(
            "%s %s -> %s in %d ms (user=%s)",
            scope.get("method", "?"),
            path,
            status,
            seconds * 1000,
            _user_id_of(scope),
        )


# =========================
# Readiness
# =========================
# Deliberately not checked: OpenAI. Readiness would then depend on a third
# party's availability and on quota, so a rate limit at the wrong moment would
# take every replica out of the load balancer while the backend was perfectly
# able to serve cached and non-LLM endpoints. A failing OpenAI call surfaces as
# a 429 or 504 on the request that needed it, which is where it belongs.

def readiness(app_state) -> tuple:
    """Report whether the components a request needs are usable.

    Returns (ready, checks). `checks` names each component and says "ok" or
    "failed" - never why, because /ready is unauthenticated and an exception
    message can carry a filesystem path.
    """
    checks = {}

    if getattr(app_state, "vectorstore", None) is None:
        # Startup has not finished; the process is answering sockets already.
        checks["startup"] = "failed"
        return False, checks
    checks["startup"] = "ok"

    try:
        # ping(), not count(): count() answers 0 for a collection that was
        # never opened, so the check passed while the store was unusable. Found
        # by breaking the store and watching /ready still say "ready".
        app_state.embeddings.ping()
        checks["vector_store"] = "ok"
    except Exception:
        logger.exception("Readiness check failed: vector store is not usable")
        checks["vector_store"] = "failed"

    return all(value == "ok" for value in checks.values()), checks
