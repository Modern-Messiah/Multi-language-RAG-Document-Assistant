"""What the Streamlit UI and the Telegram bot both need to talk to the backend.

Both clients carried their own copy of the error-description logic, and the
copies had already drifted apart: the bot learned to hide a 401 from end users
who cannot act on it, the frontend learned to flatten pydantic validation
lists, and neither picked up the other's fix. Same story for the language list,
which existed in three places with nothing keeping them in step.

Only configuration and pure helpers live here. The transport stays in each
client, because the two use different libraries: `requests` in Streamlit,
`httpx` in the bot.
"""
import os

from app.rag.languages import AUTO_LANGUAGE, SUPPORTED_LANGUAGES

__all__ = [
    "AUTO_LANGUAGE",
    "SUPPORTED_LANGUAGES",
    "DEFAULT_BACKEND_URL",
    "DEFAULT_MAX_FILE_SIZE",
    "OPERATOR_ERROR",
    "api_headers",
    "backend_url",
    "describe_error",
    "error_from_response",
    "max_file_bytes",
    "max_file_mb",
]

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"
DEFAULT_MAX_FILE_SIZE = 30 * 1024 * 1024

# Shown instead of the backend's own words when the failure is a deployment
# problem: an end user can do nothing about a rejected API key, and the detail
# ("Invalid or missing API key") tells them more about the setup than they need.
OPERATOR_ERROR = "The assistant is not configured correctly. Please contact the operator."

# Statuses that mean "the client is not allowed in", i.e. an operator problem.
_OPERATOR_STATUSES = (401, 403)


def backend_url() -> str:
    """Base URL of the backend. docker-compose overrides this per service."""
    return os.getenv("BACKEND_URL", DEFAULT_BACKEND_URL) or DEFAULT_BACKEND_URL


def api_headers() -> dict:
    """The shared-secret header.

    An empty value is legitimate: the backend runs with authentication disabled
    in development and ignores the header.
    """
    return {"X-API-Key": os.getenv("BACKEND_API_KEY", "")}


def max_file_bytes() -> int:
    """Mirror the backend's MAX_FILE_SIZE so the UI cannot advertise a wrong limit."""
    raw = os.getenv("MAX_FILE_SIZE", "")
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_FILE_SIZE
    return value if value > 0 else DEFAULT_MAX_FILE_SIZE


def max_file_mb() -> int:
    return max_file_bytes() // (1024 * 1024)


def describe_error(status_code: int, detail=None) -> str:
    """Turn a failed backend response into one line fit to show a user.

    `detail` is whatever the JSON body's "detail" key held, or None when the
    body was absent or not JSON at all.
    """
    if status_code in _OPERATOR_STATUSES:
        return OPERATOR_ERROR

    if isinstance(detail, list):
        # pydantic validation errors: [{"loc": [...], "msg": "...", ...}, ...]
        messages = [
            str(item.get("msg", item)) if isinstance(item, dict) else str(item)
            for item in detail
        ]
        joined = "; ".join(m for m in messages if m)
        if joined:
            return joined
    elif isinstance(detail, str) and detail.strip():
        return detail
    elif isinstance(detail, dict) and detail:
        return str(detail)

    return f"Backend error (HTTP {status_code})"


def error_from_response(response) -> str:
    """describe_error for a requests or httpx response.

    Both expose .status_code and .json(); .json() raises on a non-JSON body,
    which is exactly what a proxy's HTML 502 page or an empty 500 produces.
    """
    detail = None
    try:
        payload = response.json()
    except Exception:
        payload = None
    if isinstance(payload, dict):
        detail = payload.get("detail")
    return describe_error(response.status_code, detail)
