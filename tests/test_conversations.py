"""Follow-up questions.

"And the second one?" embeds to nothing useful: retrieval matched those literal
words rather than what the user meant, so the answer degraded exactly when the
conversation got going. A follow-up is now rewritten into a standalone question
for retrieval, while the answer still addresses the question as asked.
"""
from types import SimpleNamespace

import pytest
from langchain.schema import Document

from app.config import Settings
from app.rag.chain import CONDENSE_MAX_TOKENS, RAGChain


class Store:
    def __init__(self, docs=None):
        self.docs = docs if docs is not None else [Document(
            page_content="body", metadata={"source": "a.txt"}
        )]
        self.queries = []

    def similarity_search(self, query, k=4, filter=None):
        self.queries.append(query)
        return self.docs


class ScriptedChat:
    """Answers each call from a list, so condense and answer are separable."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                reply = outer.replies.pop(0) if outer.replies else "an answer"
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content=reply),
                            finish_reason="stop",
                        )
                    ],
                    usage=None,
                )

        self.chat = SimpleNamespace(completions=_Completions())


HISTORY = [
    {"question": "What did revenue do?", "answer": "It grew twelve percent."},
]


def _chain(replies=("standalone question", "an answer"), turns=6, docs=None):
    store = Store(docs)
    chat = ScriptedChat(replies)
    chain = RAGChain(store, client=chat, max_history_turns=turns)
    return chain, store, chat


# =========================
# A first question costs nothing new
# =========================

def test_no_history_means_no_condensing_call():
    """The common case must not pay for a second model round-trip."""
    chain, store, chat = _chain(replies=["an answer"])

    chain.ask("What did revenue do?", user_id="u1")

    assert len(chat.calls) == 1, "an extra call was made with no history to use"
    assert store.queries == ["What did revenue do?"]


@pytest.mark.parametrize("history", [None, [], ()])
def test_empty_history_is_treated_as_none(history):
    chain, store, chat = _chain(replies=["an answer"])

    chain.ask("q", user_id="u1", history=history)

    assert len(chat.calls) == 1
    assert store.queries == ["q"]


def test_history_is_ignored_when_the_feature_is_off():
    """max_history_turns=0 restores the single-turn behaviour exactly."""
    chain, store, chat = _chain(replies=["an answer"], turns=0)

    chain.ask("And the second?", user_id="u1", history=HISTORY)

    assert len(chat.calls) == 1
    assert store.queries == ["And the second?"]


# =========================
# Retrieval uses the standalone form
# =========================

def test_a_follow_up_is_condensed_before_retrieval():
    chain, store, chat = _chain(replies=["What did revenue do in Q2?", "an answer"])

    chain.ask("And in Q2?", user_id="u1", history=HISTORY)

    assert len(chat.calls) == 2, "the condensing call did not happen"
    assert store.queries == ["What did revenue do in Q2?"], (
        "retrieval still used the literal follow-up"
    )


def test_the_condensing_prompt_carries_the_conversation():
    chain, _, chat = _chain()

    chain.ask("And in Q2?", user_id="u1", history=HISTORY)

    condense_prompt = chat.calls[0]["messages"][1]["content"]
    assert "What did revenue do?" in condense_prompt
    assert "It grew twelve percent." in condense_prompt
    assert "And in Q2?" in condense_prompt


def test_condensing_is_bounded_and_deterministic():
    """It rewrites one sentence; it should not wander or cost much."""
    chain, _, chat = _chain()

    chain.ask("And in Q2?", user_id="u1", history=HISTORY)

    assert chat.calls[0]["max_tokens"] == CONDENSE_MAX_TOKENS
    assert chat.calls[0]["temperature"] == 0


def test_the_user_still_sees_their_own_question_answered():
    """Only retrieval uses the rewrite; the answer prompt keeps the original."""
    chain, _, chat = _chain(replies=["What did revenue do in Q2?", "an answer"])

    chain.ask("And in Q2?", user_id="u1", history=HISTORY)

    answer_prompt = chat.calls[1]["messages"][1]["content"]
    assert "And in Q2?" in answer_prompt


def test_the_answer_prompt_includes_the_conversation():
    chain, _, chat = _chain()

    chain.ask("And in Q2?", user_id="u1", history=HISTORY)

    answer_prompt = chat.calls[1]["messages"][1]["content"]
    assert "It grew twelve percent." in answer_prompt


# =========================
# Bounds
# =========================

def test_only_the_most_recent_turns_are_used():
    history = [
        {"question": f"q{i}", "answer": f"a{i}"} for i in range(10)
    ]
    chain, _, chat = _chain(turns=3)

    chain.ask("follow-up", user_id="u1", history=history)

    condense_prompt = chat.calls[0]["messages"][1]["content"]
    assert "q9" in condense_prompt and "q7" in condense_prompt
    assert "q6" not in condense_prompt, "more turns than max_history_turns were used"


def test_history_accepts_pydantic_turns():
    """The API hands the chain ChatTurn models, not dicts."""
    from app.models.schemas import ChatTurn

    chain, _, chat = _chain()

    chain.ask(
        "And in Q2?",
        user_id="u1",
        history=[ChatTurn(question="What did revenue do?", answer="It grew.")],
    )

    assert "It grew." in chat.calls[0]["messages"][1]["content"]


# =========================
# Failure of the extra call must not break the request
# =========================

class FailingCondense(ScriptedChat):
    def __init__(self):
        super().__init__(["an answer"])
        outer = self
        original = self.chat.completions.create

        class _Completions:
            def create(self, **kwargs):
                if not outer.calls:
                    outer.calls.append(kwargs)
                    raise RuntimeError("condensing is down")
                return original(**kwargs)

        self.chat = SimpleNamespace(completions=_Completions())


def test_a_failed_condense_falls_back_to_the_original_question(caplog):
    store = Store()
    chain = RAGChain(store, client=FailingCondense(), max_history_turns=6)

    with caplog.at_level("ERROR", logger="app.rag.chain"):
        result = chain.ask("And in Q2?", user_id="u1", history=HISTORY)

    assert store.queries == ["And in Q2?"], "the request should still have run"
    assert result["answer"]
    assert any("condense" in r.getMessage().lower() for r in caplog.records)


def test_an_empty_rewrite_falls_back_to_the_original_question():
    chain, store, _ = _chain(replies=["   ", "an answer"])

    chain.ask("And in Q2?", user_id="u1", history=HISTORY)

    assert store.queries == ["And in Q2?"]


# =========================
# The API contract
# =========================

def test_history_defaults_to_empty_on_the_request_model():
    from app.models.schemas import QueryRequest

    request = QueryRequest(question="hi", user_id="u1")

    assert request.history == []


def test_the_request_model_caps_history_length():
    from pydantic import ValidationError

    from app.models.schemas import QueryRequest

    turns = [{"question": "q", "answer": "a"} for _ in range(21)]

    with pytest.raises(ValidationError):
        QueryRequest(question="hi", user_id="u1", history=turns)


def test_a_turn_cannot_be_empty():
    from pydantic import ValidationError

    from app.models.schemas import ChatTurn

    with pytest.raises(ValidationError):
        ChatTurn(question="", answer="a")


def test_the_setting_is_range_checked():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(_env_file=None, openai_api_key="k", max_history_turns=-1)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, openai_api_key="k", max_history_turns=21)


def test_the_setting_reaches_the_chain(tmp_path, fake_openai_embeddings):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from tests.conftest import make_settings

    app = create_app(make_settings(tmp_path, max_history_turns=3))

    with TestClient(app):
        assert app.state.rag_chain.max_history_turns == 3


def test_history_travels_from_the_request_to_the_chain(api):
    """End to end: what the client sends is what the chain condenses on."""
    api.post(
        "/upload",
        params={"user_id": "u1"},
        files={"file": ("doc.txt", b"Revenue grew twelve percent last year.", "text/plain")},
    )

    response = api.post(
        "/query",
        json={
            "question": "And in Q2?",
            "user_id": "u1",
            "history": [{"question": "What did revenue do?", "answer": "It grew."}],
        },
    )

    assert response.status_code == 200, response.text
    sent = "".join(
        message["content"] for call in api.chat.calls for message in call["messages"]
    )
    assert "It grew." in sent, "the history never reached the model"
