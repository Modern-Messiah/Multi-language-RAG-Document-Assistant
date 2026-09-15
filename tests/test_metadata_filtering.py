"""Unit and integration tests for Metadata Extraction and Filtering."""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from app.rag.chain import RAGChain
from app.rag.metadata_extractor import (
    build_chroma_filter,
    extract_document_metadata,
    extract_query_metadata,
)

# =========================
# Unit: Metadata Extraction
# =========================

def test_extract_document_metadata_from_filenames():
    cases = [
        (
            "WALMART_2020_10K_p50.txt",
            {"company": "WALMART", "year": 2020, "doc_type": "10-K"},
        ),
        (
            "Pfizer_2023Q2_10Q_p40.txt",
            {"company": "PFIZER", "year": 2023, "quarter": "Q2", "doc_type": "10-Q"},
        ),
        (
            "VERIZON_2022_10K_p76.txt",
            {"company": "VERIZON", "year": 2022, "doc_type": "10-K"},
        ),
        (
            "ULTABEAUTY_2023Q4_EARNINGS_p1.txt",
            {"company": "ULTABEAUTY", "year": 2023, "quarter": "Q4", "doc_type": "EARNINGS"},
        ),
        (
            "3M_2018_10K_p59.txt",
            {"company": "3M", "year": 2018, "doc_type": "10-K"},
        ),
    ]

    for fname, expected in cases:
        meta = extract_document_metadata(fname)
        for k, v in expected.items():
            assert meta.get(k) == v, f"Failed for {fname} on {k}: got {meta.get(k)}, expected {v}"


def test_extract_document_metadata_with_text_fallback():
    # Filename with no year or company
    fname = "annual_report.txt"
    text = "For the fiscal year ended January 31, 2022. WALMART INC. Form 10-K"
    meta = extract_document_metadata(fname, text=text)

    assert meta.get("year") == 2022
    assert meta.get("doc_type") == "10-K"


def test_extract_query_metadata_from_questions():
    # Query with company and FY year
    q1 = "What was Walmart's net revenue in FY2020?"
    m1 = extract_query_metadata(q1)
    assert m1.get("company") == "WALMART"
    assert m1.get("year") == 2020

    # Query with company and calendar year
    q2 = "Did Verizon increase its debt in 2022?"
    m2 = extract_query_metadata(q2)
    assert m2.get("company") == "VERIZON"
    assert m2.get("year") == 2022

    # Query with company and quarter
    q3 = "As of Q2 2023, is Pfizer spinning off any large business?"
    m3 = extract_query_metadata(q3)
    assert m3.get("company") == "PFIZER"
    assert m3.get("year") == 2023
    assert m3.get("quarter") == "Q2"

    # Multi-year comparative question: filters by company and restricts year scope via $in
    q5 = "What is the FY2018 - FY2020 3 year average unadjusted EBITDA % margin for Walmart?"
    m5 = extract_query_metadata(q5)
    assert m5.get("company") == "WALMART"
    assert m5.get("year") == {"$in": [2018, 2019, 2020]}

    # Multi-word alias company names
    assert extract_query_metadata("What was Boeing's revenue in 2022?").get("company") == "BOEING"
    assert extract_query_metadata("Did American Express increase dividends in 2021?").get("company") == "AMERICANEXPRESS"
    assert extract_query_metadata("What is the restructuring cost for AES Corporation?").get("company") == "AES"


# =========================
# Unit: Chroma Filter Builder
# =========================

def test_build_chroma_filter():
    # 1. Base tenant filter
    assert build_chroma_filter("tenant-123") == {"user_id": "tenant-123"}

    # 2. Single metadata field
    f1 = build_chroma_filter("tenant-123", {"year": 2020})
    assert f1 == {"$and": [{"user_id": "tenant-123"}, {"year": 2020}]}

    # 3. Multiple metadata fields
    f2 = build_chroma_filter("tenant-123", {"company": "WALMART", "year": 2020})
    assert f2 == {
        "$and": [
            {"user_id": "tenant-123"},
            {"company": "WALMART"},
            {"year": 2020},
        ]
    }

    # 4. Missing user_id raises ValueError
    with pytest.raises(ValueError):
        build_chroma_filter("")


# =========================
# Integration: RAGChain Retrieval
# =========================

class FakeVectorStore:
    def __init__(self, docs=None, side_effect=None):
        self.docs = docs or []
        self.side_effect = list(side_effect) if side_effect else None
        self.calls = []

    def similarity_search_with_score(self, query, k=5, filter=None):
        self.calls.append({"query": query, "k": k, "filter": filter})
        if self.side_effect is not None:
            batch = self.side_effect.pop(0) if self.side_effect else []
            return [(d, 0.2) for d in batch]
        return [(d, 0.2) for d in self.docs[:k]]


def test_rag_chain_applies_explicit_metadata_filter():
    doc = Document(
        page_content="Walmart 2020 sales: $524B",
        metadata={"source": "WALMART_2020_10K.txt", "company": "WALMART", "year": 2020},
    )
    fake_store = FakeVectorStore(docs=[doc])
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="Walmart 2020 sales were $524B"))
    ]

    chain = RAGChain(
        vectorstore=fake_store,
        client=mock_client,
        metadata_filtering_enabled=True,
    )

    res = chain.ask(
        question="What was revenue?",
        user_id="tenant-42",
        metadata_filter={"year": 2020, "company": "WALMART"},
    )

    assert res["answer"] == "Walmart 2020 sales were $524B"
    assert len(res["sources"]) == 1
    assert res["sources"][0]["metadata"]["company"] == "WALMART"
    assert res["sources"][0]["metadata"]["year"] == 2020

    # Verify vectorstore was queried with the scoped filter
    assert len(fake_store.calls) == 1
    passed_filter = fake_store.calls[0]["filter"]
    assert passed_filter == {
        "$and": [
            {"user_id": "tenant-42"},
            {"year": 2020},
            {"company": "WALMART"},
        ]
    }


def test_rag_chain_auto_detects_filter_from_question():
    doc = Document(
        page_content="Verizon 2022 debt info",
        metadata={"source": "VERIZON_2022.txt", "company": "VERIZON", "year": 2022},
    )
    fake_store = FakeVectorStore(docs=[doc])
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="Verizon increased debt in 2022"))
    ]

    chain = RAGChain(
        vectorstore=fake_store,
        client=mock_client,
        metadata_filtering_enabled=True,
    )

    chain.ask(
        question="What was Verizon debt in 2022?",
        user_id="tenant-42",
    )

    assert len(fake_store.calls) == 1
    passed_filter = fake_store.calls[0]["filter"]
    assert passed_filter == {
        "$and": [
            {"user_id": "tenant-42"},
            {"company": "VERIZON"},
            {"year": 2022},
        ]
    }


def test_rag_chain_fallbacks_to_tenant_wide_search_on_zero_results():
    doc = Document(page_content="General info", metadata={"source": "doc.txt"})
    # First call with strict filter returns empty; second call (fallback) returns document
    fake_store = FakeVectorStore(side_effect=[[], [doc]])

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="Answer from fallback"))
    ]

    chain = RAGChain(
        vectorstore=fake_store,
        client=mock_client,
        metadata_filtering_enabled=True,
    )

    res = chain.ask(
        question="What happened at Walmart in 2015?",
        user_id="tenant-42",
    )

    assert res["answer"] == "Answer from fallback"
    assert len(fake_store.calls) == 2
    # First call was scoped filter
    assert "$and" in fake_store.calls[0]["filter"]
    # Second call was tenant-wide fallback
    assert fake_store.calls[1]["filter"] == {"user_id": "tenant-42"}
