"""Unit tests for Okapi BM25 sparse index."""

import pytest
from langchain_core.documents import Document

from app.rag.bm25 import BM25Index, tokenize_multilingual


def test_tokenize_multilingual():
    text = "Привет, мир! Hello World 123. Финансовые отчеты: Walmart, FY2019."
    tokens = tokenize_multilingual(text)
    assert "привет" in tokens
    assert "мир" in tokens
    assert "hello" in tokens
    assert "world" in tokens
    assert "123" in tokens
    assert "финансовые" in tokens
    assert "walmart" in tokens
    assert "fy2019" in tokens


def test_bm25_empty_query_or_corpus():
    index = BM25Index()
    assert index.search("тест", owner="u1") == []

    index.add_documents([Document(page_content="текст документа", metadata={"user_id": "u1"})])
    assert index.search("", owner="u1") == []
    assert index.search("совершенно другое", owner="u1") == []


def test_bm25_ranking_russian():
    index = BM25Index()
    docs = [
        Document(
            page_content="Принято считать, что испанскую гитару первым перевел в электрическую Джордж Бошам.",
            metadata={"source": "doc_guitar.txt", "user_id": "user1"},
        ),
        Document(
            page_content="Электроэнергетика — наиболее важная отрасль современной мировой энергетики.",
            metadata={"source": "doc_energy.txt", "user_id": "user1"},
        ),
        Document(
            page_content="Джордж Вашингтон был первым всенародно избранным президентом США.",
            metadata={"source": "doc_history.txt", "user_id": "user1"},
        ),
    ]
    index.add_documents(docs, owner="user1")

    # Search for electric guitar
    results = index.search("Кто первым испанскую гитару сделал электрической?", owner="user1", k=3)
    assert len(results) > 0
    top_doc, top_score = results[0]
    assert top_doc.metadata["source"] == "doc_guitar.txt"
    assert top_score > 0


def test_bm25_tenant_isolation():
    index = BM25Index()
    docs_user1 = [
        Document(page_content="Секретный отчет пользователя один", metadata={"user_id": "user1"}),
    ]
    docs_user2 = [
        Document(page_content="Секретный отчет пользователя два", metadata={"user_id": "user2"}),
    ]
    index.add_documents(docs_user1, owner="user1")
    index.add_documents(docs_user2, owner="user2")

    res1 = index.search("секретный отчет", owner="user1")
    res2 = index.search("секретный отчет", owner="user2")

    assert len(res1) == 1
    assert "один" in res1[0][0].page_content
    assert len(res2) == 1
    assert "два" in res2[0][0].page_content


def test_bm25_delete_and_clear():
    index = BM25Index()
    docs = [
        Document(page_content="Документ файл один", metadata={"file_hash": "hash1", "user_id": "u1"}),
        Document(page_content="Документ файл два", metadata={"file_hash": "hash2", "user_id": "u1"}),
    ]
    index.add_documents(docs, owner="u1")
    assert index.count_owner("u1") == 2

    # Delete hash1
    removed = index.delete_by_file_hash("hash1", owner="u1")
    assert removed == 1
    assert index.count_owner("u1") == 1

    res = index.search("документ", owner="u1")
    assert len(res) == 1
    assert "два" in res[0][0].page_content

    # Clear owner
    index.clear_owner("u1")
    assert index.count_owner("u1") == 0
    assert index.search("документ", owner="u1") == []


def test_bm25_metadata_filter():
    index = BM25Index()
    docs = [
        Document(page_content="Отчет Walmart 2018", metadata={"company": "WALMART", "year": 2018, "user_id": "u1"}),
        Document(page_content="Отчет Walmart 2019", metadata={"company": "WALMART", "year": 2019, "user_id": "u1"}),
        Document(page_content="Отчет Apple 2018", metadata={"company": "APPLE", "year": 2018, "user_id": "u1"}),
    ]
    index.add_documents(docs, owner="u1")

    # Filter by company
    results = index.search("отчет", owner="u1", filter_dict={"company": "WALMART"})
    assert len(results) == 2
    for d, _ in results:
        assert d.metadata["company"] == "WALMART"

    # Filter with $in
    results_in = index.search("отчет", owner="u1", filter_dict={"year": {"$in": [2018]}})
    assert len(results_in) == 2
