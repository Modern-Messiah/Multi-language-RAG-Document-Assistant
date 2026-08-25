"""Streamed answers.

Five to fifteen seconds of a motionless spinner was the most visible latency in
the product. /query is unchanged; /query/stream is an addition that sends the
sources first - they are known before generation starts - then the answer as it
is produced.
"""
import json
from types import SimpleNamespace

import pytest
from langchain.schema import Document
from openai import APITimeoutError, RateLimitError

from app.rag.chain import NO_CONTEXT_ANSWER, CitationStripper, RAGChain


class Store:
    def __init__(self, docs=None):
        self.docs = docs if docs is not None else [
            Document(page_content="body of a.txt", metadata={"source": "a.txt"})
        ]

    def similarity_search(self, query, k=4, filter=None):
        return self.docs


class StreamingChat:
    """Emits an answer piece by piece, like the OpenAI stream does."""

    def __init__(self, pieces=("Hello ", "world."), finish_reason="stop"):
        self.pieces = list(pieces)
        self.calls = []
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                if not kwargs.get("stream"):
                    return SimpleNamespace(
                        choices=[SimpleNamespace(
                            message=SimpleNamespace(content="".join(outer.pieces)),
                            finish_reason=finish_reason,
                        )],
                        usage=None,
                    )

                def generate():
                    last = len(outer.pieces) - 1
                    for index, piece in enumerate(outer.pieces):
                        yield SimpleNamespace(choices=[SimpleNamespace(
                            delta=SimpleNamespace(content=piece),
                            finish_reason=finish_reason if index == last else None,
                        )])

                return generate()

        self.chat = SimpleNamespace(completions=_Completions())


def _events(chain, **kwargs):
    return list(chain.ask_stream(user_id="u1", **kwargs))


def _text_of(events):
    return "".join(e["text"] for e in events if e["type"] == "token")


# =========================
# CitationStripper
# =========================

def _strip(chunks):
    stripper = CitationStripper()
    return "".join(stripper.feed(c) for c in chunks) + stripper.flush()


def test_a_marker_in_one_chunk_is_removed():
    assert _strip(["Sky is blue [1] and wet."]) == "Sky is blue  and wet."


def test_a_marker_split_across_chunks_is_removed():
    """The whole reason this class exists: a per-chunk regex sees "[", "1", "]"
    as three harmless pieces and passes all of them through."""
    assert _strip(["Sky is blue ", "[", "1", "]", " and wet."]) == "Sky is blue  and wet."


def test_a_multi_digit_marker_split_across_chunks_is_removed():
    assert _strip(["a[", "12", "]b"]) == "ab"


def test_consecutive_markers_are_removed():
    assert _strip(["[1][2][3]done"]) == "done"


def test_an_unterminated_bracket_is_real_text():
    """"[12" at the end of a stream was never a citation, so the user sees it."""
    assert _strip(["text [12"]) == "text [12"


def test_a_lone_bracket_is_released_when_it_cannot_be_a_marker():
    assert _strip(["cost [", "USD]"]) == "cost [USD]"


def test_nothing_is_held_back_forever():
    stripper = CitationStripper()
    stripper.feed("trailing [")

    assert stripper.flush() == "["
    assert stripper.flush() == "", "flush must not repeat itself"


@pytest.mark.parametrize("chunks", [[""], [], ["", ""]])
def test_empty_input_is_empty_output(chunks):
    assert _strip(chunks) == ""


def test_ordinary_text_passes_through_unchanged():
    text = "A sentence with brackets [like this] and numbers 42."

    assert _strip([text]) == text


# =========================
# The event stream
# =========================

def test_sources_arrive_before_any_token():
    """They are known before generation, and showing them first is the point."""
    chain = RAGChain(Store(), client=StreamingChat())

    events = _events(chain, question="q")

    assert events[0]["type"] == "sources"
    assert [s["source"] for s in events[0]["sources"]] == ["a.txt"]


def test_the_stream_ends_with_done():
    chain = RAGChain(Store(), client=StreamingChat())

    assert _events(chain, question="q")[-1] == {"type": "done"}


def test_tokens_reassemble_into_the_answer():
    chain = RAGChain(Store(), client=StreamingChat(["Hel", "lo ", "wor", "ld."]))

    assert _text_of(_events(chain, question="q")) == "Hello world."


def test_citations_are_stripped_from_the_stream():
    chain = RAGChain(Store(), client=StreamingChat(["Blue ", "[", "1", "]", " sky."]))

    assert _text_of(_events(chain, question="q")) == "Blue  sky."


def test_streaming_is_requested_of_the_model():
    chat = StreamingChat()
    chain = RAGChain(Store(), client=chat)

    _events(chain, question="q")

    assert chat.calls[0]["stream"] is True


def test_empty_chunks_are_skipped():
    """The first chunk of an OpenAI stream carries a role and no content."""
    chain = RAGChain(Store(), client=StreamingChat(["", "text", ""]))

    assert _text_of(_events(chain, question="q")) == "text"


# =========================
# Nothing retrieved
# =========================

def test_no_documents_streams_the_message_without_calling_the_model():
    chat = StreamingChat()
    chain = RAGChain(Store(docs=[]), client=chat)

    events = _events(chain, question="q")

    assert _text_of(events) == NO_CONTEXT_ANSWER
    assert events[0] == {"type": "sources", "sources": []}
    assert events[-1] == {"type": "done"}
    assert chat.calls == [], "the model was asked to answer from nothing"


# =========================
# ask() and ask_stream() must not drift
# =========================

def test_both_paths_build_the_same_request():
    """A prompt change that reached only one of them would make the same
    question answer differently depending on the endpoint."""
    streamed = StreamingChat(["one"])
    blocking = StreamingChat(["one"])

    _events(RAGChain(Store(), client=streamed), question="q", language="Русский")
    RAGChain(Store(), client=blocking).ask("q", language="Русский", user_id="u1")

    stream_request = dict(streamed.calls[0])
    blocking_request = dict(blocking.calls[0])
    stream_request.pop("stream", None)

    assert stream_request == blocking_request


def test_both_paths_agree_on_sources():
    chain_a = RAGChain(Store(), client=StreamingChat(["x"]))
    chain_b = RAGChain(Store(), client=StreamingChat(["x"]))

    streamed = next(e for e in _events(chain_a, question="q") if e["type"] == "sources")
    blocking = chain_b.ask("q", user_id="u1")

    assert streamed["sources"] == blocking["sources"]


def test_ask_still_requires_a_user_id():
    chain = RAGChain(Store(), client=StreamingChat())

    with pytest.raises(ValueError, match="user_id is required"):
        chain.ask("q")


def test_ask_stream_also_requires_a_user_id():
    chain = RAGChain(Store(), client=StreamingChat())

    with pytest.raises(ValueError, match="user_id is required"):
        list(chain.ask_stream("q"))


# =========================
# The HTTP endpoint
# =========================

def _sse_events(response):
    events = []
    for line in response.iter_lines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
    return events


def _upload(api, content=b"RAG grounds answers in retrieved documents."):
    return api.post(
        "/upload",
        params={"user_id": "u1"},
        files={"file": ("doc.txt", content, "text/plain")},
    )


def test_the_endpoint_streams_server_sent_events(api):
    _upload(api)

    with api.stream(
        "POST", "/query/stream", json={"question": "what is rag", "user_id": "u1"}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _sse_events(response)

    assert [e["type"] for e in events] == ["sources", "token", "token", "done"]


def test_the_endpoint_disables_proxy_buffering(api):
    """Buffering the response would defeat the entire feature."""
    _upload(api)

    with api.stream(
        "POST", "/query/stream", json={"question": "q", "user_id": "u1"}
    ) as response:
        assert response.headers.get("cache-control") == "no-cache"
        assert response.headers.get("x-accel-buffering") == "no"
        list(response.iter_lines())


def test_the_endpoint_requires_the_api_key(api):
    with api.stream(
        "POST",
        "/query/stream",
        json={"question": "q", "user_id": "u1"},
        headers={"X-API-Key": "wrong"},
    ) as response:
        assert response.status_code == 401


def test_the_endpoint_validates_like_query(api):
    with api.stream("POST", "/query/stream", json={"question": "q"}) as response:
        assert response.status_code == 422


def test_a_rate_limit_before_the_stream_is_a_normal_429(api, monkeypatch):
    """Priming the generator in the handler is what buys this: after the first
    byte the status line is gone and a 429 could only be a stream event."""
    import httpx

    def boom(*args, **kwargs):
        raise RateLimitError(
            "slow down",
            response=httpx.Response(
                429, request=httpx.Request("POST", "https://api.openai.com/v1")
            ),
            body=None,
        )
        yield  # pragma: no cover - makes this a generator

    monkeypatch.setattr(api.app_state.rag_chain, "ask_stream", boom)

    with api.stream("POST", "/query/stream", json={"question": "q", "user_id": "u1"}) as r:
        assert r.status_code == 429
        assert r.headers.get("Retry-After")


def test_a_timeout_before_the_stream_is_a_normal_504(api, monkeypatch):
    import httpx

    def boom(*args, **kwargs):
        raise APITimeoutError(request=httpx.Request("POST", "https://api.openai.com/v1"))
        yield  # pragma: no cover

    monkeypatch.setattr(api.app_state.rag_chain, "ask_stream", boom)

    with api.stream("POST", "/query/stream", json={"question": "q", "user_id": "u1"}) as r:
        assert r.status_code == 504


def test_a_failure_mid_stream_is_reported_as_an_event(api, monkeypatch):
    def half_broken(*args, **kwargs):
        yield {"type": "sources", "sources": []}
        yield {"type": "token", "text": "start"}
        raise RuntimeError("the model hung up")

    monkeypatch.setattr(api.app_state.rag_chain, "ask_stream", half_broken)

    with api.stream("POST", "/query/stream", json={"question": "q", "user_id": "u1"}) as r:
        assert r.status_code == 200, "headers were already sent, so this cannot be an error code"
        events = _sse_events(r)

    assert events[-1]["type"] == "error"
    assert "the model hung up" not in json.dumps(events), "internals leaked to the client"


def test_query_still_answers_in_one_piece(api):
    """The streaming endpoint is an addition, not a replacement."""
    _upload(api)

    response = api.post("/query", json={"question": "q", "user_id": "u1"})

    assert response.status_code == 200
    assert response.json()["answer"]
