"""RAGChain: prompt assembly, language rules, citation stripping, source list."""
from types import SimpleNamespace

from langchain.schema import Document

from app.rag.chain import LANG_RULES, RAGChain


class FakeVectorStore:
    def __init__(self, docs):
        self.docs = docs
        self.calls = []

    def similarity_search(self, query, k=3, filter=None):
        self.calls.append({"query": query, "k": k, "filter": filter})
        return self.docs


class FakeOpenAI:
    """Records the request and replays a canned answer."""

    def __init__(self, answer="The answer."):
        self.answer = answer
        self.calls = []
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=outer.answer))]
                )

        self.chat = SimpleNamespace(completions=_Completions())


def _doc(text, source):
    return Document(page_content=text, metadata={"source": source})


def _chain(docs, answer="The answer.", **kwargs):
    store = FakeVectorStore(docs)
    client = FakeOpenAI(answer)
    chain = RAGChain(store, client=client, **kwargs)
    return chain, store, client


# =========================
# Empty retrieval
# =========================

def test_no_documents_returns_early_without_calling_the_llm():
    chain, _, client = _chain([])

    result = chain.ask("anything", user_id="u1")

    assert result["sources"] == []
    assert "No relevant information" in result["answer"]
    assert client.calls == [], "LLM was called with no context"


# =========================
# Retrieval parameters
# =========================

def test_user_id_is_passed_as_a_metadata_filter():
    chain, store, _ = _chain([_doc("body", "a.txt")], top_k=7)

    chain.ask("q", user_id="alice")

    assert store.calls[0]["filter"] == {"user_id": "alice"}
    assert store.calls[0]["k"] == 7


# =========================
# Language rules
# =========================

def test_explicit_language_injects_its_rule():
    chain, _, client = _chain([_doc("body", "a.txt")])

    chain.ask("q", language="Қазақша", user_id="u1")

    system = client.calls[0]["messages"][0]["content"]
    assert LANG_RULES["Қазақша"] in system


def test_auto_language_asks_to_mirror_the_question():
    chain, _, client = _chain([_doc("body", "a.txt")])

    chain.ask("q", language="Auto", user_id="u1")

    system = client.calls[0]["messages"][0]["content"]
    assert "same language as the user's question" in system


def test_unknown_language_falls_back_instead_of_failing():
    chain, _, client = _chain([_doc("body", "a.txt")])

    result = chain.ask("q", language="Klingon", user_id="u1")

    system = client.calls[0]["messages"][0]["content"]
    assert "same language as the user's question" in system
    assert result["answer"] == "The answer."


# =========================
# Answer post-processing
# =========================

def test_bracket_citations_are_stripped():
    chain, _, _ = _chain([_doc("body", "a.txt")], answer="Sky is blue [1] and wet [23].")

    assert chain.ask("q", user_id="u1")["answer"] == "Sky is blue  and wet ."


def test_context_includes_every_retrieved_chunk_with_its_source():
    docs = [_doc("first chunk", "a.txt"), _doc("second chunk", "b.txt")]
    chain, _, client = _chain(docs)

    chain.ask("q", user_id="u1")

    user_message = client.calls[0]["messages"][1]["content"]
    assert "first chunk" in user_message and "second chunk" in user_message
    assert "Source: a.txt" in user_message and "Source: b.txt" in user_message


# =========================
# Sources
# =========================

def test_sources_are_deduplicated_and_numbered_from_one():
    docs = [
        _doc("chunk one", "a.txt"),
        _doc("chunk two", "a.txt"),
        _doc("chunk three", "b.txt"),
    ]
    chain, _, _ = _chain(docs)

    sources = chain.ask("q", user_id="u1")["sources"]

    assert [s["source"] for s in sources] == ["a.txt", "b.txt"]
    assert [s["id"] for s in sources] == [1, 2]


def test_source_preview_is_capped_at_200_chars():
    chain, _, _ = _chain([_doc("x" * 500, "a.txt")])

    assert len(chain.ask("q", user_id="u1")["sources"][0]["preview"]) == 200


def test_missing_source_metadata_becomes_unknown():
    chain, _, _ = _chain([Document(page_content="body", metadata={})])

    assert chain.ask("q", user_id="u1")["sources"][0]["source"] == "unknown"


# =========================
# Model parameters
# =========================

def test_model_and_temperature_reach_the_llm_call():
    chain, _, client = _chain([_doc("body", "a.txt")], model="gpt-4o", temperature=0.3)

    chain.ask("q", user_id="u1")

    assert client.calls[0]["model"] == "gpt-4o"
    assert client.calls[0]["temperature"] == 0.3
