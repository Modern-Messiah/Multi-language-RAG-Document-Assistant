"""Langfuse Observability and Tracing for RAG.

Integrates with Langfuse (cloud.langfuse.com or self-hosted) to trace:
- Full query lifecycle (user_id, request_id, session_id, language, latency)
- Retrieval step (search query, chunks count, sources, scores)
- Generation step (model, prompt messages, temperature, answer, token usage)
- User feedback (thumbs up / thumbs down mapped to scores on traces)

When LANGFUSE_ENABLED is False (the default) or keys are not provided,
all tracing operations fall back to NoOp classes with zero network overhead.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =========================
# NoOp Handles (safe fallback)
# =========================

class NoOpSpanHandle:
    def end(
        self,
        output: Optional[Any] = None,
        level: Optional[str] = None,
        status_message: Optional[str] = None,
        **kwargs,
    ) -> None:
        pass


class NoOpGenerationHandle:
    def end(
        self,
        output: Optional[Any] = None,
        usage: Optional[Dict[str, int]] = None,
        level: Optional[str] = None,
        status_message: Optional[str] = None,
        **kwargs,
    ) -> None:
        pass


class NoOpTraceHandle:
    def span(self, name: str, input: Optional[Any] = None, **kwargs) -> NoOpSpanHandle:
        return NoOpSpanHandle()

    def generation(
        self,
        name: str,
        model: Optional[str] = None,
        input: Optional[Any] = None,
        model_parameters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> NoOpGenerationHandle:
        return NoOpGenerationHandle()

    def end(
        self,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        pass

    def update(self, **kwargs) -> None:
        pass


class NoOpTracer:
    enabled: bool = False

    def start_trace(
        self,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        question: Optional[str] = None,
        language: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> NoOpTraceHandle:
        return NoOpTraceHandle()

    def record_feedback(
        self,
        request_id: str,
        rating: str,
        comment: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        pass

    def flush(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


# =========================
# Active Langfuse Handles
# =========================

class LangfuseSpanHandle:
    def __init__(self, raw_span):
        self._span = raw_span

    def end(
        self,
        output: Optional[Any] = None,
        level: Optional[str] = None,
        status_message: Optional[str] = None,
        **kwargs,
    ) -> None:
        if not self._span:
            return
        try:
            self._span.end(
                output=output,
                level=level,
                status_message=status_message,
                **kwargs,
            )
        except Exception:
            logger.debug("Failed to end Langfuse span", exc_info=True)


class LangfuseGenerationHandle:
    def __init__(self, raw_generation):
        self._generation = raw_generation

    def end(
        self,
        output: Optional[Any] = None,
        usage: Optional[Dict[str, int]] = None,
        level: Optional[str] = None,
        status_message: Optional[str] = None,
        **kwargs,
    ) -> None:
        if not self._generation:
            return
        try:
            params: Dict[str, Any] = {}
            if output is not None:
                params["output"] = output
            if usage is not None:
                params["usage"] = usage
            if level is not None:
                params["level"] = level
            if status_message is not None:
                params["status_message"] = status_message
            self._generation.end(**params, **kwargs)
        except Exception:
            logger.debug("Failed to end Langfuse generation", exc_info=True)


class LangfuseTraceHandle:
    def __init__(self, raw_trace):
        self._trace = raw_trace

    def span(self, name: str, input: Optional[Any] = None, **kwargs) -> Any:
        if not self._trace:
            return NoOpSpanHandle()
        try:
            raw_span = self._trace.span(name=name, input=input, **kwargs)
            return LangfuseSpanHandle(raw_span)
        except Exception:
            logger.debug("Failed to create Langfuse span", exc_info=True)
            return NoOpSpanHandle()

    def generation(
        self,
        name: str,
        model: Optional[str] = None,
        input: Optional[Any] = None,
        model_parameters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        if not self._trace:
            return NoOpGenerationHandle()
        try:
            raw_gen = self._trace.generation(
                name=name,
                model=model,
                input=input,
                model_parameters=model_parameters,
                **kwargs,
            )
            return LangfuseGenerationHandle(raw_gen)
        except Exception:
            logger.debug("Failed to create Langfuse generation", exc_info=True)
            return NoOpGenerationHandle()

    def end(
        self,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if not self._trace:
            return
        try:
            params: Dict[str, Any] = {}
            if output is not None:
                params["output"] = output
            if metadata is not None:
                params["metadata"] = metadata
            self._trace.update(**params, **kwargs)
        except Exception:
            logger.debug("Failed to update Langfuse trace", exc_info=True)

    def update(self, **kwargs) -> None:
        if not self._trace:
            return
        try:
            self._trace.update(**kwargs)
        except Exception:
            logger.debug("Failed to update Langfuse trace", exc_info=True)


class LangfuseTracer:
    enabled: bool = True

    def __init__(self, public_key: str, secret_key: str, host: str):
        from langfuse import Langfuse

        self.client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        logger.info("Langfuse tracing initialized (host=%s)", host)

    def start_trace(
        self,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        question: Optional[str] = None,
        language: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        try:
            tag_list = list(tags or [])
            if language and language not in tag_list:
                tag_list.append(language)
            trace_metadata = dict(metadata or {})

            raw_trace = self.client.trace(
                id=request_id,
                name="rag-query",
                user_id=user_id,
                session_id=session_id,
                input={"question": question},
                tags=tag_list,
                metadata=trace_metadata,
            )
            return LangfuseTraceHandle(raw_trace)
        except Exception:
            logger.warning("Failed to start Langfuse trace", exc_info=True)
            return NoOpTraceHandle()

    def record_feedback(
        self,
        request_id: str,
        rating: str,
        comment: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        try:
            value = 1.0 if rating == "up" else 0.0
            self.client.score(
                trace_id=request_id,
                name="user_feedback",
                value=value,
                comment=comment,
            )
            logger.info("Recorded feedback to Langfuse trace %s (rating=%s)", request_id, rating)
        except Exception:
            logger.warning("Failed to send feedback score to Langfuse", exc_info=True)

    def flush(self) -> None:
        try:
            self.client.flush()
        except Exception:
            logger.debug("Failed to flush Langfuse", exc_info=True)

    def shutdown(self) -> None:
        try:
            self.client.shutdown()
        except Exception:
            logger.debug("Failed to shutdown Langfuse", exc_info=True)


def get_tracer(settings) -> Any:
    """Factory creating either LangfuseTracer or NoOpTracer based on settings."""
    if not getattr(settings, "langfuse_enabled", False):
        return NoOpTracer()

    public_key = getattr(settings, "langfuse_public_key", "")
    secret_key = getattr(settings, "langfuse_secret_key", "")
    host = getattr(settings, "langfuse_host", "https://cloud.langfuse.com") or "https://cloud.langfuse.com"

    if not public_key or not secret_key:
        logger.warning("LANGFUSE_ENABLED=true but keys are missing; tracing disabled")
        return NoOpTracer()

    try:
        return LangfuseTracer(public_key=public_key, secret_key=secret_key, host=host)
    except Exception:
        logger.exception("Failed to initialize Langfuse tracer; falling back to NoOp")
        return NoOpTracer()
