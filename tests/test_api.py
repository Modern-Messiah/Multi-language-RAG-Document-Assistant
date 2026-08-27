"""API behavior: auth, user_id validation, traversal safety, dedup, health, honest errors."""
import pytest

from tests.conftest import TEST_API_KEY, make_settings

TXT = b"RAG systems combine retrieval with generation to answer questions from documents."


def _upload(api, filename="doc.txt", content=TXT, user_id="u1"):
    params = {"user_id": user_id} if user_id is not None else {}
    return api.post(
        "/upload",
        params=params,
        files={"file": (filename, content, "text/plain")},
    )


# =========================
# Authentication
# =========================

def test_request_without_api_key_is_rejected(api):
    response = api.post(
        "/query",
        headers={"X-API-Key": ""},
        json={"question": "hi", "user_id": "u1"},
    )
    assert response.status_code == 401


def test_wrong_api_key_is_rejected(api):
    response = api.post(
        "/upload",
        headers={"X-API-Key": "wrong-key"},
        params={"user_id": "u1"},
        files={"file": ("doc.txt", TXT, "text/plain")},
    )
    assert response.status_code == 401


def test_health_needs_no_api_key(api):
    response = api.get("/health", headers={"X-API-Key": ""})
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_auth_disabled_when_no_backend_key(tmp_path, fake_openai_embeddings):
    from fastapi.testclient import TestClient

    from app.main import create_app

    app = create_app(make_settings(tmp_path, backend_api_key=""))
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/query", json={"question": "hi", "user_id": "u1"})
        assert response.status_code == 200


# =========================
# user_id validation
# =========================

def test_upload_without_user_id_is_rejected(api):
    response = _upload(api, user_id=None)
    assert response.status_code == 422


def test_query_without_user_id_is_rejected(api):
    response = api.post("/query", json={"question": "hi"})
    assert response.status_code == 422


def test_clear_without_user_id_is_rejected(api):
    response = api.post("/clear")
    assert response.status_code == 422


def test_user_id_with_unsafe_characters_is_rejected(api):
    for bad in ("../x", "team a", "u\u0000id", "x" * 65):
        response = _upload(api, user_id=bad)
        assert response.status_code == 422, f"user_id {bad!r} was accepted"


# =========================
# Upload safety & dedup
# =========================

def test_upload_traversal_filename_stays_inside_upload_dir(api):
    """Assert on the resolved location, not on the absence of a guessed name.

    The previous version passed even with sanitization removed, because it only
    checked one hardcoded escape path.
    """
    for name in ("../evil.txt", "a/../../evil.txt", r"..\..\evil.txt", "/etc/evil.txt"):
        response = _upload(api, filename=name, user_id="u1")
        assert response.status_code == 200, f"{name!r}: {response.text}"

        # The stored name must be a plain basename with no traversal component.
        assert response.json()["filename"] == "evil.txt"

    from app.activity import ACTIVITY_DIRNAME

    owner_dir = (api.upload_dir / "u1").resolve()
    # Everything under the upload dir except the activity markers, which are
    # not uploads: the point is that a stored *upload* cannot land anywhere but
    # in its owner's directory.
    stored = [
        p for p in api.upload_dir.rglob("*")
        if p.is_file() and ACTIVITY_DIRNAME not in p.parts
    ]
    assert stored, "sanitized file was not stored under the upload dir"
    for path in stored:
        assert path.resolve().parent == owner_dir, f"{path} escaped {owner_dir}"


@pytest.mark.parametrize(
    "filename",
    [".", "..", "...", ".txt", "....txt", "_", "   ", "@@@", "/", "\\"],
)
def test_unusable_filenames_are_rejected(api, filename):
    response = _upload(api, filename=filename)

    assert response.status_code == 400, f"{filename!r} was accepted"
    assert [p for p in api.upload_dir.rglob("*") if p.is_file()] == []


def test_missing_filename_is_a_validation_error(api):
    """An empty filename means no file part at all, so FastAPI rejects it."""
    response = _upload(api, filename="")

    assert response.status_code == 422
    assert [p for p in api.upload_dir.rglob("*") if p.is_file()] == []


def test_dotfiles_are_rejected_by_design(api):
    """Sanitization strips leading dots rather than storing hidden files."""
    response = _upload(api, filename=".hidden.txt")

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid filename"


def test_upload_long_filename_is_accepted(api):
    long_name = "о" * 300 + ".txt"

    response = _upload(api, filename=long_name)

    assert response.status_code == 200
    stored = list(api.upload_dir.rglob("*.txt"))
    assert stored
    assert all(len(p.name.encode("utf-8")) <= 255 for p in stored)


def test_upload_duplicate_content_is_skipped(api):
    first = _upload(api, user_id="u1").json()
    second = _upload(api, user_id="u1").json()

    assert first.get("duplicate") is False
    assert first["chunks"] >= 1
    assert second.get("duplicate") is True
    assert second["chunks"] == 0


def test_upload_same_content_different_user_is_indexed(api):
    _upload(api, user_id="u1")
    response = _upload(api, user_id="u2").json()

    assert response.get("duplicate") is False
    assert response["chunks"] >= 1


def test_similar_user_ids_keep_separate_chunks(api):
    _upload(api, user_id="team_a")
    _upload(api, user_id="team-a")

    collection = api.app_state.embeddings.collection
    first = collection.get(where={"user_id": {"$eq": "team_a"}})
    second = collection.get(where={"user_id": {"$eq": "team-a"}})
    assert first["ids"], "first user's chunks missing or overwritten"
    assert second["ids"], "second user's chunks missing"


# =========================
# Settings wiring
# =========================

def test_chunk_size_setting_is_wired(tmp_path, fake_openai_embeddings):
    from fastapi.testclient import TestClient

    from app.main import create_app

    app = create_app(make_settings(tmp_path, chunk_size=120, chunk_overlap=0))
    body = ("word " * 200).encode()  # ~1000 chars -> several 120-char chunks

    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers["X-API-Key"] = TEST_API_KEY
        response = client.post(
            "/upload",
            params={"user_id": "u1"},
            files={"file": ("doc.txt", body, "text/plain")},
        )

    assert response.status_code == 200
    assert response.json()["chunks"] > 3


# =========================
# Health & honest errors
# =========================

def test_health_endpoint(api):
    from app.main import API_VERSION

    response = api.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": API_VERSION}


def test_clear_failure_returns_generic_500(api, monkeypatch):
    def boom(filter):
        raise RuntimeError("secret internal path C:\\private")

    monkeypatch.setattr(api.app_state.embeddings, "delete_documents", boom)

    response = api.post("/clear", params={"user_id": "u1"})

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail == "Failed to clear documents"
    assert "secret" not in detail


def test_upload_store_failure_returns_503(api, monkeypatch):
    def boom(documents, ids=None):
        raise RuntimeError("chroma exploded")

    monkeypatch.setattr(api.app_state.embeddings, "add_documents", boom)

    response = _upload(api)

    assert response.status_code == 503


def test_query_failure_returns_503(api, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("chroma exploded")

    monkeypatch.setattr(api.app_state.rag_chain, "ask", boom)

    response = api.post("/query", json={"question": "hi", "user_id": "u1"})

    assert response.status_code == 503


# =========================
# Clear removes vectors AND raw files
# =========================

def test_clear_removes_the_vectors(api):
    _upload(api, user_id="u1")
    assert api.app_state.embeddings.count() >= 1

    assert api.post("/clear", params={"user_id": "u1"}).status_code == 200
    assert api.app_state.embeddings.count() == 0


def test_clear_removes_the_uploaded_files(api):
    _upload(api, user_id="u1")
    assert [p for p in api.upload_dir.rglob("*") if p.is_file()]

    api.post("/clear", params={"user_id": "u1"})

    leftover = [p for p in api.upload_dir.rglob("*") if p.is_file()]
    assert leftover == [], "clear left the raw documents on disk"


def test_clear_only_touches_the_requesting_user(api):
    _upload(api, user_id="alice")
    _upload(api, filename="other.txt", content=b"Bob's own notes about vectors.", user_id="bob")

    api.post("/clear", params={"user_id": "alice"})

    remaining = api.app_state.embeddings.collection.get()
    owners = {m["user_id"] for m in remaining["metadatas"]}
    assert owners == {"bob"}
    assert (api.upload_dir / "bob").is_dir()
    assert not (api.upload_dir / "alice").exists()


def test_clear_is_idempotent_when_nothing_was_uploaded(api):
    assert api.post("/clear", params={"user_id": "nobody"}).status_code == 200


# =========================
# Query isolation
# =========================

def test_query_cannot_reach_another_users_documents(api):
    _upload(api, filename="secret.txt",
            content=b"The launch code for project Aurora is 4815162342.", user_id="alice")

    hits = api.app_state.vectorstore.similarity_search(
        "launch code", k=5, filter={"user_id": "bob"}
    )

    assert hits == [], "another user's chunks were retrievable"


# =========================
# Auth
# =========================

def test_non_ascii_api_key_from_client_returns_401_not_500(api):
    """secrets.compare_digest raises TypeError on non-ASCII str operands.

    Header values arrive as raw bytes, so a client can always send them; the
    comparison must reject them cleanly instead of blowing up into a 500.
    """
    response = api.post(
        "/clear",
        params={"user_id": "u1"},
        headers={"X-API-Key": "ключ-не-ascii".encode("utf-8")},
    )

    assert response.status_code == 401


def test_non_ascii_configured_key_is_refused_at_startup(tmp_path):
    """A non-ASCII BACKEND_API_KEY could never match, so fail loudly instead."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc:
        make_settings(tmp_path, backend_api_key="ключ-очень-секретный")

    assert "ASCII" in str(exc.value)
