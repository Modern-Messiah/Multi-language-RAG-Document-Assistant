"""Unit tests for TableAwareSplitter and table-preserving chunking."""

from langchain_core.documents import Document

from app.rag.table_splitter import (
    TableAwareSplitter,
    extract_segments,
    is_markdown_separator_row,
    is_markdown_table_row,
)
from app.rag.text_splitter import TextChunker


def _doc(text: str, **metadata) -> Document:
    return Document(page_content=text, metadata=metadata)


# =========================
# Detection helpers
# =========================

def test_markdown_row_and_separator_detection():
    assert is_markdown_table_row("| Name | Price | Quantity |")
    assert is_markdown_table_row("| A | B |")
    assert not is_markdown_table_row("Just plain text with no pipes")
    assert not is_markdown_table_row("")

    assert is_markdown_separator_row("|---|---|---|")
    assert is_markdown_separator_row("|:---|:---:|---:|")
    assert is_markdown_separator_row("| - | - |")
    assert not is_markdown_separator_row("| Name | Price |")
    assert not is_markdown_separator_row("---")


def test_extract_segments_splits_prose_and_markdown_tables():
    text = (
        "Introduction paragraph.\n\n"
        "| Metric | FY2020 | FY2021 |\n"
        "|---|---|---|\n"
        "| Revenue | $500M | $600M |\n"
        "| Net Income | $50M | $65M |\n\n"
        "Conclusion paragraph after table."
    )
    segments = extract_segments(text)
    assert len(segments) == 3
    assert not segments[0].is_table
    assert segments[0].content.strip() == "Introduction paragraph."

    assert segments[1].is_table
    assert segments[1].table_block.table_type == "markdown"
    assert len(segments[1].table_block.rows) == 2

    assert not segments[2].is_table
    assert segments[2].content.strip() == "Conclusion paragraph after table."


def test_extract_segments_handles_html_tables():
    text = (
        "Overview prose.\n\n"
        "<table><thead><tr><th>Company</th><th>Ticker</th></tr></thead>"
        "<tbody><tr><td>Apple</td><td>AAPL</td></tr></tbody></table>\n\n"
        "Footer prose."
    )
    segments = extract_segments(text)
    assert len(segments) == 3
    assert not segments[0].is_table
    assert segments[1].is_table
    assert segments[1].table_block.table_type == "html"
    assert not segments[2].is_table


# =========================
# Table Preservation
# =========================

def test_markdown_table_preserved_intact():
    table_text = (
        "| Year | Revenue | Operating Income | Net Margin |\n"
        "|---|---|---|---|\n"
        "| 2018 | $514B | $21.9B | 4.2% |\n"
        "| 2019 | $524B | $20.5B | 3.9% |\n"
        "| 2020 | $559B | $22.5B | 4.0% |"
    )
    splitter = TableAwareSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents([_doc(table_text, source="annual_report.txt")])

    assert len(chunks) == 1
    assert chunks[0].page_content == table_text
    assert chunks[0].metadata["contains_table"] is True
    assert chunks[0].metadata["is_table"] is True
    assert chunks[0].metadata["source"] == "annual_report.txt"


def test_large_markdown_table_repeats_headers_on_each_chunk():
    header = "| Item | Amount | Currency | Note |"
    sep = "|---|---|---|---|"
    # Generate 30 rows to exceed chunk_size=300
    rows = [f"| Item #{i:02d} | ${i*100:,} | USD | Audited row {i} |" for i in range(1, 31)]
    full_table = f"{header}\n{sep}\n" + "\n".join(rows)

    splitter = TableAwareSplitter(chunk_size=350, chunk_overlap=0)
    chunks = splitter.split_documents([_doc(full_table)])

    assert len(chunks) > 1
    for chunk in chunks:
        lines = chunk.page_content.split("\n")
        assert lines[0] == header, "Every chunk must start with table header"
        assert lines[1] == sep, "Every chunk must have table separator"
        assert chunk.metadata["contains_table"] is True
        assert chunk.metadata["is_table"] is True


def test_html_table_preserved_intact():
    html_table = (
        "<table>\n"
        "  <tr><th>Segment</th><th>Sales</th></tr>\n"
        "  <tr><td>Retail</td><td>$400M</td></tr>\n"
        "  <tr><td>Wholesale</td><td>$150M</td></tr>\n"
        "</table>"
    )
    splitter = TableAwareSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents([_doc(html_table)])

    assert len(chunks) == 1
    assert "<table>" in chunks[0].page_content
    assert chunks[0].metadata["contains_table"] is True


# =========================
# TextChunker Integration
# =========================

def test_text_chunker_table_aware_preserves_table_structure():
    doc_text = (
        "Financial Highlights:\n\n"
        "| Quarter | EPS | Consensus | Beat |\n"
        "|---|---|---|---|\n"
        "| Q1 | $1.20 | $1.15 | Yes |\n"
        "| Q2 | $1.35 | $1.30 | Yes |\n\n"
        "The company reported record quarterly results."
    )
    chunker = TextChunker(chunk_size=1000, chunk_overlap=100, table_aware=True)
    chunks = chunker.split_documents([_doc(doc_text, source="earnings.md")])

    assert len(chunks) >= 1
    assert any(c.metadata.get("contains_table") for c in chunks)
    table_chunk = next(c for c in chunks if c.metadata.get("contains_table"))
    assert "| Quarter | EPS | Consensus | Beat |" in table_chunk.page_content
    assert "| Q1 | $1.20 | $1.15 | Yes |" in table_chunk.page_content


def test_text_chunker_table_aware_disabled_falls_back():
    doc_text = (
        "| A | B |\n"
        "|---|---|\n"
        "| 1 | 2 |"
    )
    chunker = TextChunker(chunk_size=1000, chunk_overlap=100, table_aware=False)
    chunks = chunker.split_documents([_doc(doc_text)])

    assert len(chunks) == 1
    # When table_aware is disabled, contains_table is False
    assert chunks[0].metadata.get("contains_table") is False


def test_text_chunker_upload_wide_numbering_with_mixed_documents():
    doc1 = _doc("Just simple prose without any tables.", doc_id=1)
    doc2 = _doc(
        "| Metric | 2020 |\n|---|---|\n| Profit | $10M |",
        doc_id=2,
    )
    doc3 = _doc("Another plain prose paragraph for testing.", doc_id=3)

    chunker = TextChunker(chunk_size=500, chunk_overlap=50, table_aware=True)
    chunks = chunker.split_documents([doc1, doc2, doc3])

    assert len(chunks) == 3
    for i, c in enumerate(chunks):
        assert c.metadata["chunk_id"] == i
        assert c.metadata["total_chunks"] == len(chunks)

    assert chunks[0].metadata["contains_table"] is False
    assert chunks[1].metadata["contains_table"] is True
    assert chunks[2].metadata["contains_table"] is False
