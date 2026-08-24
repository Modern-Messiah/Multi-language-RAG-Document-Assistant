"""Upload failure modes: honest status codes, no orphaned files, size limit."""
import pytest

from tests.conftest import make_settings

TXT = b"RAG systems combine retrieval with generation to answer questions."

# Smallest PDF that pypdf can actually extract text from.
def _pdf(text: str = "RAG grounds answers in retrieved documents.") -> bytes:
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        None,  # filled in below
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    stream = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode()
    objects[3] = (
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode()
    return bytes(out)


def _upload(api, filename, content, user_id="u1"):
    return api.post(
        "/upload",
        params={"user_id": user_id},
        files={"file": (filename, content, "application/octet-stream")},
    )


def _stored(api):
    return [p for p in api.upload_dir.rglob("*") if p.is_file()]


# =========================
# Happy path: PDF
# =========================

def test_pdf_upload_is_indexed_with_page_metadata(api):
    response = _upload(api, "paper.pdf", _pdf())

    assert response.status_code == 200
    assert response.json()["chunks"] >= 1

    metadatas = api.app_state.embeddings.collection.get()["metadatas"]
    assert metadatas[0]["type"] == "pdf"
    assert metadatas[0]["page"] == 1
    assert metadatas[0]["source"] == "paper.pdf"


# =========================
# Corrupt input is a 400, not a 500
# =========================

def test_corrupt_pdf_is_rejected_with_400(api):
    response = _upload(api, "broken.pdf", b"this is definitely not a pdf")

    assert response.status_code == 400, response.text
    assert "corrupted" in response.json()["detail"].lower()


def test_corrupt_pdf_leaves_no_file_behind(api):
    _upload(api, "broken.pdf", b"this is definitely not a pdf")

    assert _stored(api) == [], "unparseable upload was kept on disk"


def test_corrupt_pdf_does_not_index_anything(api):
    _upload(api, "broken.pdf", b"this is definitely not a pdf")

    assert api.app_state.embeddings.count() == 0


def test_empty_file_is_rejected(api):
    response = _upload(api, "empty.txt", b"")

    assert response.status_code == 400
    assert _stored(api) == []


def test_whitespace_only_txt_is_rejected_without_orphan(api):
    response = _upload(api, "blank.txt", b"   \n\t  \n  ")

    assert response.status_code == 400
    assert _stored(api) == []


def test_indexing_failure_removes_the_saved_file(api, monkeypatch):
    def boom(documents, ids=None):
        raise RuntimeError("chroma exploded")

    monkeypatch.setattr(api.app_state.embeddings, "add_documents", boom)

    response = _upload(api, "doc.txt", TXT)

    assert response.status_code == 503
    assert _stored(api) == [], "file kept after a failed index"


# =========================
# Size limit
# =========================

def test_oversize_upload_is_rejected(tmp_path, fake_openai_embeddings):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from tests.conftest import TEST_API_KEY

    app = create_app(make_settings(tmp_path, max_file_size=1024))

    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers["X-API-Key"] = TEST_API_KEY
        response = client.post(
            "/upload",
            params={"user_id": "u1"},
            files={"file": ("big.txt", b"A" * 4096, "text/plain")},
        )

    assert response.status_code == 400
    # The message must report the configured limit, not a hardcoded 30 MB.
    assert "30 MB" not in response.json()["detail"]


def test_size_limit_message_reports_the_configured_limit(tmp_path, fake_openai_embeddings):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from tests.conftest import TEST_API_KEY

    app = create_app(make_settings(tmp_path, max_file_size=5 * 1024 * 1024))

    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers["X-API-Key"] = TEST_API_KEY
        response = client.post(
            "/upload",
            params={"user_id": "u1"},
            files={"file": ("big.txt", b"A" * (6 * 1024 * 1024), "text/plain")},
        )

    assert response.status_code == 400
    assert "5 MB" in response.json()["detail"]


# =========================
# Extension gate
# =========================

def test_unsupported_extensions_are_rejected(api):
    for filename in ("evil.exe", "no_extension", "notes.docx", "archive.txt.zip"):
        response = _upload(api, filename, TXT)
        assert response.status_code == 400, f"{filename} was accepted"
        assert _stored(api) == [], f"{filename} was written to disk"


def test_uppercase_extension_is_accepted(api):
    assert _upload(api, "doc.TXT", TXT).status_code == 200


# =========================
# Size rendering
# =========================

@pytest.mark.parametrize(
    "num_bytes,expected",
    [
        (30 * 1024 * 1024, "30 MB"),
        (5 * 1024 * 1024, "5 MB"),
        (int(1.5 * 1024 * 1024), "1.5 MB"),
        (1024 * 1024, "1 MB"),
        (100 * 1024, "100 KB"),
        (1024, "1 KB"),
        (1536, "1.5 KB"),
        (512, "512 bytes"),
        (0, "0 bytes"),
    ],
)
def test_human_size_never_rounds_a_real_limit_to_zero(num_bytes, expected):
    """A 1 KB limit used to be reported as "0 MB"."""
    from app.main import human_size

    assert human_size(num_bytes) == expected


def test_small_limit_is_reported_in_kb(tmp_path, fake_openai_embeddings):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from tests.conftest import TEST_API_KEY

    app = create_app(make_settings(tmp_path, max_file_size=1024))

    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers["X-API-Key"] = TEST_API_KEY
        response = client.post(
            "/upload",
            params={"user_id": "u1"},
            files={"file": ("big.txt", b"A" * 4096, "text/plain")},
        )

    assert response.status_code == 400
    assert "1 KB" in response.json()["detail"], response.json()
