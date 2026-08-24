"""/query end to end, with a stubbed chat client instead of OpenAI.

Before this file the only /query test that expected 200 avoided the OpenAI API
by accident: it queried an empty corpus, so RAGChain returned early and never
built a request. Nothing covered the path where documents exist.
"""
import pytest

ALICE_TEXT = b"The launch code for project Aurora is 4815162342 and it is confidential."
BOB_TEXT = b"Bob keeps notes about gardening, compost ratios and tomato varieties."


def _upload(api, content, filename="doc.txt", user_id="alice"):
    return api.post(
        "/upload",
        params={"user_id": user_id},
        files={"file": (filename, content, "text/plain")},
    )


def _ask(api, question="What is the launch code?", user_id="alice", language="Auto"):
    return api.post(
        "/query",
        json={"question": question, "language": language, "user_id": user_id},
    )


# =========================
# Happy path
# =========================

def test_query_returns_answer_and_sources(api):
    _upload(api, ALICE_TEXT, filename="secret.txt", user_id="alice")

    response = _ask(api)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"] == "This is a canned offline answer."
    assert [s["source"] for s in body["sources"]] == ["secret.txt"]
    assert body["sources"][0]["id"] == 1
    assert "Aurora" in body["sources"][0]["preview"]


def test_query_sends_the_retrieved_context_to_the_model(api):
    _upload(api, ALICE_TEXT, filename="secret.txt", user_id="alice")

    _ask(api)

    assert len(api.chat.calls) == 1
    user_message = api.chat.calls[0]["messages"][1]["content"]
    assert "Aurora" in user_message
    assert "Source: secret.txt" in user_message


def test_empty_corpus_answers_without_calling_the_model(api):
    response = _ask(api)

    assert response.status_code == 200
    assert response.json()["sources"] == []
    assert api.chat.calls == [], "the model was called with no context"


# =========================
# Tenant scoping
# =========================

def test_query_never_returns_another_tenants_documents(api):
    _upload(api, ALICE_TEXT, filename="secret.txt", user_id="alice")
    _upload(api, BOB_TEXT, filename="garden.txt", user_id="bob")

    response = _ask(api, question="What is the launch code?", user_id="bob")

    assert response.status_code == 200
    sources = [s["source"] for s in response.json()["sources"]]
    assert sources == ["garden.txt"]

    # Alice's text must not have reached the prompt either.
    sent = "".join(m["content"] for call in api.chat.calls for m in call["messages"])
    assert "Aurora" not in sent
    assert "4815162342" not in sent


def test_query_for_a_tenant_with_nothing_indexed_is_empty(api):
    _upload(api, ALICE_TEXT, filename="secret.txt", user_id="alice")

    response = _ask(api, user_id="stranger")

    assert response.status_code == 200
    assert response.json()["sources"] == []
    assert api.chat.calls == []


# =========================
# Language selection reaches the prompt
# =========================

@pytest.mark.parametrize(
    "language,expected",
    [
        ("Русский", "русском"),
        ("Қазақша", "қазақ"),
        ("中文", "简体中文"),
    ],
)
def test_selected_language_reaches_the_system_prompt(api, language, expected):
    _upload(api, ALICE_TEXT, user_id="alice")

    _ask(api, language=language)

    system = api.chat.calls[0]["messages"][0]["content"]
    assert expected in system


def test_model_and_temperature_come_from_settings(api):
    _upload(api, ALICE_TEXT, user_id="alice")

    _ask(api)

    call = api.chat.calls[0]
    assert call["model"] == api.app_state.settings.model_name
    assert call["temperature"] == api.app_state.settings.temperature


def test_top_k_from_settings_bounds_retrieval(api):
    # Many small chunks, then confirm no more than top_k reach the prompt.
    body = ("Aurora paragraph number %d. " % 0).encode() + b"\n\n".join(
        f"Aurora paragraph number {i}.".encode() for i in range(1, 40)
    )
    _upload(api, body, filename="many.txt", user_id="alice")

    _ask(api)

    context = api.chat.calls[0]["messages"][1]["content"]
    assert context.count("Source: many.txt") <= api.app_state.settings.top_k_results
