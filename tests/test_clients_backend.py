"""The shared client helpers - previously untested code in two copies.

Neither client had a single test. Both carried their own backend_error, and the
copies had drifted: the bot hid a 401 from end users, the frontend flattened
pydantic validation lists, and neither had the other's fix.
"""
from types import SimpleNamespace

import pytest

from clients import backend


class FakeResponse:
    """Stands in for a requests or httpx response - both expose these two."""

    def __init__(self, status_code, payload=None, raises=False):
        self.status_code = status_code
        self._payload = payload
        self._raises = raises

    def json(self):
        if self._raises:
            raise ValueError("not JSON")
        return self._payload


# =========================
# Configuration from the environment
# =========================

def test_backend_url_defaults_to_loopback(monkeypatch):
    monkeypatch.delenv("BACKEND_URL", raising=False)

    assert backend.backend_url() == backend.DEFAULT_BACKEND_URL


def test_backend_url_uses_the_environment(monkeypatch):
    monkeypatch.setenv("BACKEND_URL", "http://backend:8000")

    assert backend.backend_url() == "http://backend:8000"


def test_empty_backend_url_falls_back_to_the_default(monkeypatch):
    """docker-compose sets it; an empty value must not produce "/upload"."""
    monkeypatch.setenv("BACKEND_URL", "")

    assert backend.backend_url() == backend.DEFAULT_BACKEND_URL


def test_api_headers_carry_the_shared_secret(monkeypatch):
    monkeypatch.setenv("BACKEND_API_KEY", "s3cret")

    assert backend.api_headers() == {"X-API-Key": "s3cret"}


def test_api_headers_are_empty_when_auth_is_disabled(monkeypatch):
    """An empty key is legitimate: the backend then skips the check."""
    monkeypatch.delenv("BACKEND_API_KEY", raising=False)

    assert backend.api_headers() == {"X-API-Key": ""}


@pytest.mark.parametrize(
    "raw,expected_mb",
    [
        ("31457280", 30),
        ("5242880", 5),
        ("1048576", 1),
    ],
)
def test_max_file_size_follows_the_environment(monkeypatch, raw, expected_mb):
    monkeypatch.setenv("MAX_FILE_SIZE", raw)

    assert backend.max_file_bytes() == int(raw)
    assert backend.max_file_mb() == expected_mb


@pytest.mark.parametrize("raw", ["", "not-a-number", "0", "-1"])
def test_unusable_max_file_size_falls_back_to_the_default(monkeypatch, raw):
    monkeypatch.setenv("MAX_FILE_SIZE", raw)

    assert backend.max_file_bytes() == backend.DEFAULT_MAX_FILE_SIZE


def test_max_file_size_default_matches_the_backend(monkeypatch):
    """The UI must not advertise a limit the API will refuse."""
    from app.config import Settings

    monkeypatch.delenv("MAX_FILE_SIZE", raising=False)
    settings = Settings(_env_file=None, openai_api_key="k")

    assert backend.max_file_bytes() == settings.max_file_size


# =========================
# describe_error
# =========================

@pytest.mark.parametrize("status", [401, 403])
def test_auth_failures_are_never_relayed_to_the_user(status):
    """"Invalid or missing API key" is for the operator, not the end user."""
    message = backend.describe_error(status, "Invalid or missing API key")

    assert message == backend.OPERATOR_ERROR
    assert "API key" not in message


def test_a_plain_detail_is_passed_through():
    assert backend.describe_error(400, "File too large. Maximum allowed size is 30 MB.") == (
        "File too large. Maximum allowed size is 30 MB."
    )


def test_pydantic_validation_lists_are_flattened():
    detail = [
        {"loc": ["query", "user_id"], "msg": "string does not match pattern", "type": "x"},
        {"loc": ["body", "question"], "msg": "field required", "type": "y"},
    ]

    message = backend.describe_error(422, detail)

    assert message == "string does not match pattern; field required"
    assert "loc" not in message, "the raw pydantic blob leaked into the UI"


def test_a_list_of_bare_strings_is_also_handled():
    assert backend.describe_error(422, ["first", "second"]) == "first; second"


def test_a_dict_detail_is_stringified_rather_than_dropped():
    message = backend.describe_error(500, {"code": "boom"})

    assert "boom" in message


@pytest.mark.parametrize("detail", [None, "", "   ", [], {}])
def test_a_missing_detail_falls_back_to_the_status(detail):
    assert backend.describe_error(503, detail) == "Backend error (HTTP 503)"


def test_the_fallback_names_the_actual_status():
    assert "418" in backend.describe_error(418, None)


# =========================
# error_from_response
# =========================

def test_error_from_response_reads_the_detail():
    response = FakeResponse(400, {"detail": "Only TXT and PDF files are supported"})

    assert backend.error_from_response(response) == "Only TXT and PDF files are supported"


def test_error_from_response_survives_a_non_json_body():
    """A proxy's HTML 502 page used to raise inside the error path itself."""
    response = FakeResponse(502, raises=True)

    assert backend.error_from_response(response) == "Backend error (HTTP 502)"


def test_error_from_response_survives_a_json_body_that_is_not_an_object():
    response = FakeResponse(500, ["unexpected"])

    assert backend.error_from_response(response) == "Backend error (HTTP 500)"


def test_error_from_response_hides_auth_failures():
    response = FakeResponse(401, {"detail": "Invalid or missing API key"})

    assert backend.error_from_response(response) == backend.OPERATOR_ERROR


def test_error_from_response_accepts_a_minimal_duck_type():
    """Both requests and httpx satisfy this; nothing else is required."""
    response = SimpleNamespace(status_code=400, json=lambda: {"detail": "nope"})

    assert backend.error_from_response(response) == "nope"
