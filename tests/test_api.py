"""Stage-1 API behavior: traversal safety, dedup, health, honest errors."""

TXT = b"RAG systems combine retrieval with generation to answer questions from documents."


def _upload(api, filename=b"doc.txt".decode(), content=TXT, user_id=None):
    params = {"user_id": user_id} if user_id else {}
    return api.post(
        "/upload",
        params=params,
        files={"file": (filename, content, "text/plain")},
    )


def test_upload_traversal_filename_stays_inside_upload_dir(api):
    response = _upload(api, filename="../evil.txt")

    assert response.status_code == 200
    escaped = api.upload_dir.parent / "evil.txt"
    assert not escaped.exists(), "file escaped the upload directory"
    stored = list(api.upload_dir.rglob("*evil.txt"))
    assert stored, "sanitized file was not stored under the upload dir"


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


def test_upload_long_filename_is_accepted(api):
    long_name = "о" * 300 + ".txt"

    response = _upload(api, filename=long_name)

    assert response.status_code == 200
    stored = list(api.upload_dir.rglob("*.txt"))
    assert stored
    assert all(len(p.name.encode("utf-8")) <= 255 for p in stored)


def test_anon_dedup_scoped_to_anon_not_all_users(api):
    assert _upload(api, user_id="u1").json()["duplicate"] is False

    anon_first = _upload(api).json()
    assert anon_first["duplicate"] is False, "anon upload must not be dropped because a named user has the same file"
    assert anon_first["chunks"] >= 1

    anon_second = _upload(api).json()
    assert anon_second["duplicate"] is True


def test_sanitized_user_collision_does_not_overwrite_chunks(api):
    import app.main as main

    _upload(api, user_id="team.a")
    _upload(api, user_id="team a")

    collection = main.embeddings.vectorstore._collection
    first = collection.get(where={"user_id": {"$eq": "team.a"}})
    second = collection.get(where={"user_id": {"$eq": "team a"}})
    assert first["ids"], "first user's chunks were overwritten"
    assert second["ids"], "second user's chunks missing"


def test_health_endpoint(api):
    response = api.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_clear_failure_returns_generic_500(api, monkeypatch):
    import app.main as main

    def boom(filter):
        raise RuntimeError("secret internal path C:\\private")

    monkeypatch.setattr(main.embeddings, "delete_documents", boom)
    monkeypatch.setattr(main, "vectorstore", object())

    response = api.post("/clear", params={"user_id": "u1"})

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail == "Failed to clear documents"
    assert "secret" not in detail


def test_query_store_failure_returns_503(api, monkeypatch):
    import app.main as main

    def boom(collection_name):
        raise RuntimeError("chroma exploded")

    monkeypatch.setattr(main.embeddings, "load_vectorstore", boom)

    response = api.post("/query", json={"question": "What is RAG?"})

    assert response.status_code == 503
