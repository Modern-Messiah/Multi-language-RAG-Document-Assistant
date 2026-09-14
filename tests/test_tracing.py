"""Tests for Langfuse tracing and observability."""
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.tracing import (
    LangfuseTracer,
    NoOpGenerationHandle,
    NoOpSpanHandle,
    NoOpTraceHandle,
    NoOpTracer,
    get_tracer,
)
from tests.conftest import TEST_API_KEY, FakeChatClient, make_settings


def test_noop_tracer_handles_all_methods_safely():
    tracer = NoOpTracer()
    assert tracer.enabled is False

    trace = tracer.start_trace(
        request_id="req-123",
        user_id="user-1",
        question="What is this?",
        language="Russian",
    )
    assert isinstance(trace, NoOpTraceHandle)

    span = trace.span("retrieval", input={"query": "test"})
    assert isinstance(span, NoOpSpanHandle)
    span.end(output={"count": 1})

    gen = trace.generation("generation", model="gpt-4o", input=[])
    assert isinstance(gen, NoOpGenerationHandle)
    gen.end(output="answer", usage={"total_tokens": 10})

    trace.end(output={"answer": "done"})
    trace.update(tags=["test"])

    # None of these should raise
    tracer.record_feedback("req-123", rating="up", comment="good")
    tracer.flush()
    tracer.shutdown()


def test_get_tracer_returns_noop_when_disabled():
    settings = Settings(_env_file=None, openai_api_key="key", langfuse_enabled=False)
    tracer = get_tracer(settings)
    assert isinstance(tracer, NoOpTracer)


def test_get_tracer_returns_noop_when_keys_missing():
    settings = Settings(
        _env_file=None,
        openai_api_key="key",
        langfuse_enabled=True,
        langfuse_public_key="",
        langfuse_secret_key="",
    )
    tracer = get_tracer(settings)
    assert isinstance(tracer, NoOpTracer)


def test_langfuse_tracer_starts_trace_and_scores_feedback():
    mock_client = MagicMock()
    mock_trace = MagicMock()
    mock_client.trace.return_value = mock_trace
    mock_span = MagicMock()
    mock_trace.span.return_value = mock_span
    mock_gen = MagicMock()
    mock_trace.generation.return_value = mock_gen

    with patch("langfuse.Langfuse", return_value=mock_client):
        tracer = LangfuseTracer(
            public_key="pk-test",
            secret_key="sk-test",
            host="https://cloud.langfuse.com",
        )
        assert tracer.enabled is True

        trace = tracer.start_trace(
            request_id="req-999",
            user_id="u123",
            question="hello?",
            language="Auto",
            tags=["cli"],
        )
        mock_client.trace.assert_called_once_with(
            id="req-999",
            name="rag-query",
            user_id="u123",
            session_id=None,
            input={"question": "hello?"},
            tags=["cli", "Auto"],
            metadata={},
        )

        span = trace.span("retrieval", input={"q": "hello"})
        span.end(output={"chunks": 2})
        mock_trace.span.assert_called_once_with(name="retrieval", input={"q": "hello"})
        mock_span.end.assert_called_once_with(
            output={"chunks": 2}, level=None, status_message=None
        )

        gen = trace.generation("gen", model="gpt-4o", input=[])
        gen.end(output="hi", usage={"total_tokens": 15})
        mock_trace.generation.assert_called_once_with(
            name="gen", model="gpt-4o", input=[], model_parameters=None
        )
        mock_gen.end.assert_called_once_with(
            output="hi", usage={"total_tokens": 15}
        )

        trace.end(output={"answer": "hi"})
        mock_trace.update.assert_called_once_with(output={"answer": "hi"})

        # Record thumbs up feedback
        tracer.record_feedback("req-999", rating="up", comment="nice")
        mock_client.score.assert_called_once_with(
            trace_id="req-999",
            name="user_feedback",
            value=1.0,
            comment="nice",
        )

        # Record thumbs down feedback
        mock_client.reset_mock()
        tracer.record_feedback("req-999", rating="down")
        mock_client.score.assert_called_once_with(
            trace_id="req-999",
            name="user_feedback",
            value=0.0,
            comment=None,
        )

        tracer.flush()
        mock_client.flush.assert_called_once()

        tracer.shutdown()
        mock_client.shutdown.assert_called_once()


def test_feedback_endpoint_forwards_score_to_tracer(tmp_path, fake_openai_embeddings):
    settings = make_settings(tmp_path)
    app = create_app(settings)

    mock_tracer = MagicMock()
    with TestClient(app) as client:
        client.headers["X-API-Key"] = TEST_API_KEY
        app.state.tracer = mock_tracer

        payload = {
            "rating": "down",
            "user_id": "u1",
            "question": "test question",
            "answer": "test answer",
            "sources": [],
            "request_id": "req-xyz",
            "comment": "bad result",
        }
        resp = client.post("/feedback", json=payload)
        assert resp.status_code == 200

        mock_tracer.record_feedback.assert_called_once_with(
            request_id="req-xyz",
            rating="down",
            comment="bad result",
            user_id="u1",
        )


def test_query_starts_trace_on_active_tracer(tmp_path, fake_openai_embeddings):
    settings = make_settings(tmp_path)
    app = create_app(settings)

    mock_tracer = MagicMock()
    mock_trace = MagicMock()
    mock_tracer.start_trace.return_value = mock_trace

    with TestClient(app) as client:
        client.headers["X-API-Key"] = TEST_API_KEY
        app.state.tracer = mock_tracer
        app.state.rag_chain.client = FakeChatClient()

        resp = client.post(
            "/query",
            json={"question": "What is photovoltaic?", "user_id": "u1"},
            headers={"X-Request-ID": "trace-test-id"},
        )
        assert resp.status_code == 200

        mock_tracer.start_trace.assert_called_once_with(
            request_id="trace-test-id",
            user_id="u1",
            question="What is photovoltaic?",
            language="Auto",
            tags=["default"],
        )
