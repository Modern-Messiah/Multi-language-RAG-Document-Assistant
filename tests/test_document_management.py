"""Seeing and removing individual documents.

Before this there was no way to find out what was indexed - the Streamlit
sidebar only remembered the current browser session's uploads and the bot knew
nothing at all - and the only deletion was /clear, all or nothing. Fixing one
stale file meant wiping the corpus and re-uploading it.
"""
import pytest

TXT = b"RAG systems combine retrieval with generation to answer questions."
OTHER = b"Compost ratios, tomato varieties, and when to prune the raspberries."


def _upload(api, filename="doc.txt", content=TXT, user_id="u1"):
    return api.post(
        "/upload",
        params={"user_id": user_id},
        files={"file": (filename, content, "text/plain")},
    )


def _documents(api, user_id="u1"):
    response = api.get("/documents", params={"user_id": user_id})
    assert response.status_code == 200, response.text
    return response.json()


def _stored_files(api, user_id="u1"):
    owner_dir = api.upload_dir / user_id
    return sorted(p.name for p in owner_dir.iterdir()) if owner_dir.is_dir() else []


# =========================
# Listing
# =========================

def test_listing_an_empty_namespace(api):
    """A tenant with nothing indexed is an empty list, not an error."""
    body = _documents(api, "nobody")

    assert body == {"documents": [], "total_chunks": 0}


def test_uploaded_document_appears_in_the_listing(api):
    uploaded = _upload(api).json()

    body = _documents(api)

    assert len(body["documents"]) == 1
    entry = body["documents"][0]
    assert entry["source"] == "doc.txt"
    assert entry["file_hash"] == uploaded["file_hash"]
    assert entry["chunks"] >= 1
    assert entry["type"] == "txt"


def test_chunks_are_folded_back_into_documents(api, tmp_path):
    """Chunks are the storage unit; a person thinks in files."""
    from fastapi.testclient import TestClient

    from app.main import create_app
    from tests.conftest import TEST_API_KEY, make_settings

    app = create_app(make_settings(tmp_path, chunk_size=60, chunk_overlap=0))
    body = ("sentence number one. " * 30).encode()

    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers["X-API-Key"] = TEST_API_KEY
        client.post(
            "/upload", params={"user_id": "u1"},
            files={"file": ("long.txt", body, "text/plain")},
        )
        listing = client.get("/documents", params={"user_id": "u1"}).json()

    assert len(listing["documents"]) == 1, "one file must stay one entry"
    assert listing["documents"][0]["chunks"] > 3
    assert listing["total_chunks"] == listing["documents"][0]["chunks"]


def test_documents_are_sorted_by_name(api):
    for name in ("zebra.txt", "apple.txt", "Mango.txt"):
        _upload(api, filename=name, content=f"content of {name}".encode())

    sources = [d["source"] for d in _documents(api)["documents"]]

    assert sources == sorted(sources, key=str.lower)


def test_total_chunks_sums_the_listing(api):
    _upload(api, filename="a.txt", content=TXT)
    _upload(api, filename="b.txt", content=OTHER)

    body = _documents(api)

    assert body["total_chunks"] == sum(d["chunks"] for d in body["documents"])


def test_pdf_listing_reports_pages(api):
    from tests.test_upload_errors import _pdf

    api.post(
        "/upload", params={"user_id": "u1"},
        files={"file": ("paper.pdf", _pdf(), "application/pdf")},
    )

    entry = _documents(api)["documents"][0]

    assert entry["type"] == "pdf"
    assert entry["pages"] == 1


# =========================
# Tenant scoping
# =========================

def test_listing_shows_only_your_own_documents(api):
    _upload(api, filename="alice.txt", content=TXT, user_id="alice")
    _upload(api, filename="bob.txt", content=OTHER, user_id="bob")

    assert [d["source"] for d in _documents(api, "alice")["documents"]] == ["alice.txt"]
    assert [d["source"] for d in _documents(api, "bob")["documents"]] == ["bob.txt"]


def test_identical_content_is_listed_for_each_owner(api):
    """The content hash is the same across tenants by construction."""
    first = _upload(api, user_id="alice").json()
    second = _upload(api, user_id="bob").json()

    assert first["file_hash"] == second["file_hash"]
    assert len(_documents(api, "alice")["documents"]) == 1
    assert len(_documents(api, "bob")["documents"]) == 1


def test_listing_requires_a_valid_user_id(api):
    assert api.get("/documents").status_code == 422
    assert api.get("/documents", params={"user_id": "../etc"}).status_code == 422


def test_listing_requires_the_api_key(api):
    response = api.get(
        "/documents", params={"user_id": "u1"}, headers={"X-API-Key": "wrong"}
    )

    assert response.status_code == 401


# =========================
# Deleting one document
# =========================

def test_delete_removes_the_document_from_the_listing(api):
    _upload(api, filename="keep.txt", content=TXT)
    doomed = _upload(api, filename="drop.txt", content=OTHER).json()

    response = api.delete(
        f"/documents/{doomed['file_hash']}", params={"user_id": "u1"}
    )

    assert response.status_code == 200
    assert response.json()["chunks_removed"] >= 1
    assert [d["source"] for d in _documents(api)["documents"]] == ["keep.txt"]


def test_delete_removes_the_raw_file_too(api):
    uploaded = _upload(api).json()
    assert _stored_files(api), "nothing was stored to begin with"

    api.delete(f"/documents/{uploaded['file_hash']}", params={"user_id": "u1"})

    assert _stored_files(api) == [], "the raw upload outlived its vectors"


def test_delete_leaves_other_documents_alone(api):
    keep = _upload(api, filename="keep.txt", content=TXT).json()
    drop = _upload(api, filename="drop.txt", content=OTHER).json()

    api.delete(f"/documents/{drop['file_hash']}", params={"user_id": "u1"})

    remaining = _documents(api)["documents"]
    assert [d["file_hash"] for d in remaining] == [keep["file_hash"]]
    assert _stored_files(api) == [f"{keep['file_hash']}_keep.txt"]


def test_deleting_an_unknown_document_is_404(api):
    _upload(api)

    response = api.delete("/documents/" + "0" * 16, params={"user_id": "u1"})

    assert response.status_code == 404


def test_cannot_delete_another_tenants_document(api):
    alice = _upload(api, user_id="alice").json()

    response = api.delete(
        f"/documents/{alice['file_hash']}", params={"user_id": "bob"}
    )

    assert response.status_code == 404, "one tenant reached another's document"
    assert len(_documents(api, "alice")["documents"]) == 1, "alice's document vanished"


def test_deleting_shared_content_only_affects_the_caller(api):
    """Both own the same bytes, so both rows carry the same hash."""
    shared = _upload(api, user_id="alice").json()["file_hash"]
    _upload(api, user_id="bob")

    api.delete(f"/documents/{shared}", params={"user_id": "alice"})

    assert _documents(api, "alice")["documents"] == []
    assert len(_documents(api, "bob")["documents"]) == 1, "bob's copy was collateral"


@pytest.mark.parametrize("bad", ["zzz", "0" * 15, "0" * 17, "ZZZZZZZZZZZZZZZZ", "../../etc"])
def test_malformed_hashes_are_rejected(api, bad):
    response = api.delete(f"/documents/{bad}", params={"user_id": "u1"})

    assert response.status_code in (404, 422), f"{bad!r} reached the store"


def test_delete_requires_the_api_key(api):
    uploaded = _upload(api).json()

    response = api.delete(
        f"/documents/{uploaded['file_hash']}",
        params={"user_id": "u1"},
        headers={"X-API-Key": "wrong"},
    )

    assert response.status_code == 401
    assert len(_documents(api)["documents"]) == 1


# =========================
# Re-uploading a revision
# =========================

def test_same_name_new_content_replaces_the_old_revision(api):
    first = _upload(api, filename="report.txt", content=b"Revenue grew 12 percent.").json()
    second = _upload(api, filename="report.txt", content=b"Revenue grew 17 percent.").json()

    assert first["file_hash"] != second["file_hash"]
    assert second["replaced"] is True

    documents = _documents(api)["documents"]
    assert len(documents) == 1, "both revisions are still answering questions"
    assert documents[0]["file_hash"] == second["file_hash"]


def test_the_replaced_revisions_file_is_removed(api):
    _upload(api, filename="report.txt", content=b"Revenue grew 12 percent.")
    second = _upload(api, filename="report.txt", content=b"Revenue grew 17 percent.").json()

    assert _stored_files(api) == [f"{second['file_hash']}_report.txt"]


def test_the_old_revision_stops_answering_questions(api):
    _upload(api, filename="report.txt", content=b"Revenue grew by twelve percent.")
    _upload(api, filename="report.txt", content=b"Revenue grew by seventeen percent.")

    hits = api.app_state.vectorstore.similarity_search(
        "revenue", k=10, filter={"user_id": "u1"}
    )

    text = " ".join(hit.page_content for hit in hits)
    assert "seventeen" in text
    assert "twelve" not in text, "the superseded revision is still retrievable"


def test_a_first_upload_is_not_a_replacement(api):
    assert _upload(api).json()["replaced"] is False


def test_identical_content_is_a_duplicate_not_a_replacement(api):
    _upload(api, filename="report.txt")
    again = _upload(api, filename="report.txt").json()

    assert again["duplicate"] is True
    assert again["replaced"] is False
    assert again["chunks"] == 0


def test_replacement_is_scoped_to_the_owner(api):
    """Alice uploading her own report.txt must not retire bob's."""
    bob = _upload(api, filename="report.txt", content=b"Bob's own report.", user_id="bob").json()
    _upload(api, filename="report.txt", content=b"Alice's report.", user_id="alice")

    assert [d["file_hash"] for d in _documents(api, "bob")["documents"]] == [bob["file_hash"]]


def test_different_names_coexist(api):
    _upload(api, filename="a.txt", content=TXT)
    _upload(api, filename="b.txt", content=OTHER)

    assert len(_documents(api)["documents"]) == 2


# =========================
# The upload contract
# =========================

def test_upload_returns_the_hash_that_addresses_the_document(api):
    uploaded = _upload(api).json()

    response = api.delete(
        f"/documents/{uploaded['file_hash']}", params={"user_id": "u1"}
    )

    assert response.status_code == 200, "the hash upload returned did not address it"


def test_duplicate_upload_still_returns_the_hash(api):
    first = _upload(api).json()
    again = _upload(api).json()

    assert again["file_hash"] == first["file_hash"]
