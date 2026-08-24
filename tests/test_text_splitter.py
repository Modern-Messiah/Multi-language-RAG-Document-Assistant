"""TextChunker: chunk boundaries, overlap, metadata, and statistics."""
from langchain.schema import Document

from app.rag.text_splitter import TextChunker


def _doc(text, **metadata):
    return Document(page_content=text, metadata=metadata)


# =========================
# Chunk boundaries
# =========================

def test_short_document_stays_one_chunk():
    chunks = TextChunker(chunk_size=1000, chunk_overlap=100).split_documents(
        [_doc("A short paragraph.")]
    )

    assert len(chunks) == 1
    assert chunks[0].page_content == "A short paragraph."


def test_long_document_is_split_and_respects_chunk_size():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = TextChunker(chunk_size=200, chunk_overlap=0).split_documents([_doc(text)])

    assert len(chunks) > 1
    assert all(len(c.page_content) <= 200 for c in chunks)


def test_overlap_repeats_text_between_neighbours():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = TextChunker(chunk_size=200, chunk_overlap=80).split_documents([_doc(text)])

    assert len(chunks) > 2
    # Some token from the tail of a chunk must reappear at the head of the next.
    first_tail = chunks[0].page_content.split()[-1]
    assert first_tail in chunks[1].page_content


def test_zero_overlap_does_not_repeat():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = TextChunker(chunk_size=150, chunk_overlap=0).split_documents([_doc(text)])

    joined = " ".join(c.page_content for c in chunks)
    # With no overlap every distinct token appears exactly once.
    assert joined.count("word150") == 1


# =========================
# Metadata
# =========================

def test_source_metadata_is_preserved_on_every_chunk():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = TextChunker(chunk_size=150, chunk_overlap=0).split_documents(
        [_doc(text, source="report.pdf", type="pdf", page=3)]
    )

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["source"] == "report.pdf"
        assert chunk.metadata["type"] == "pdf"
        assert chunk.metadata["page"] == 3


def test_chunk_ids_are_sequential_and_totals_agree():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = TextChunker(chunk_size=150, chunk_overlap=0).split_documents([_doc(text)])

    assert [c.metadata["chunk_id"] for c in chunks] == list(range(len(chunks)))
    assert all(c.metadata["total_chunks"] == len(chunks) for c in chunks)


def test_chunk_size_metadata_matches_the_content():
    chunks = TextChunker(chunk_size=120, chunk_overlap=0).split_documents(
        [_doc(" ".join(f"word{i}" for i in range(200)))]
    )

    for chunk in chunks:
        assert chunk.metadata["chunk_size"] == len(chunk.page_content)


def test_multiple_documents_share_one_upload_wide_numbering():
    """chunk_id/total_chunks count the whole upload, not each source document.

    A multi-page PDF arrives as one Document per page, and the ids must stay
    unique across the batch — that is what makes the per-upload chunk ids in
    app/main.py collision-free.
    """
    pages = [_doc(" ".join(f"p{n}word{i}" for i in range(200)), page=n) for n in range(3)]

    chunks = TextChunker(chunk_size=150, chunk_overlap=0).split_documents(pages)

    ids = [c.metadata["chunk_id"] for c in chunks]
    assert ids == list(range(len(chunks))), "chunk ids collide across pages"
    assert len(set(ids)) == len(ids)


# =========================
# Degenerate input
# =========================

def test_no_documents_returns_empty_list():
    assert TextChunker().split_documents([]) == []


def test_split_text_on_blank_input_returns_empty_list():
    chunker = TextChunker()

    assert chunker.split_text("") == []
    assert chunker.split_text("   \n\t ") == []


def test_split_text_returns_plain_strings():
    chunks = TextChunker(chunk_size=100, chunk_overlap=0).split_text(
        " ".join(f"word{i}" for i in range(100))
    )

    assert len(chunks) > 1
    assert all(isinstance(c, str) for c in chunks)


# =========================
# Statistics
# =========================

def test_statistics_describe_the_chunks():
    chunks = TextChunker(chunk_size=150, chunk_overlap=0).split_documents(
        [_doc(" ".join(f"word{i}" for i in range(300)))]
    )

    stats = TextChunker.get_chunk_statistics(chunks)

    sizes = [len(c.page_content) for c in chunks]
    assert stats["total_chunks"] == len(chunks)
    assert stats["min_chunk_size"] == min(sizes)
    assert stats["max_chunk_size"] == max(sizes)
    assert stats["total_characters"] == sum(sizes)
    assert stats["min_chunk_size"] <= stats["avg_chunk_size"] <= stats["max_chunk_size"]


def test_statistics_on_empty_input_reports_an_error():
    assert "error" in TextChunker.get_chunk_statistics([])
