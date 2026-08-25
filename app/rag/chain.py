"""
RAG Chain: Retrieval-Augmented Generation
"""

import logging
import os
import re
from typing import Dict, List, Optional

import httpx
from langchain.schema import Document
from openai import OpenAI

from app.rag.languages import AUTO_LANGUAGE, LANG_RULES, rule_for

# LANG_RULES is re-exported: it lived here first, and the clients now read the
# same table from app.rag.languages instead of keeping their own copies.
__all__ = ["LANG_RULES", "RAGChain", "SYSTEM_PROMPT"]

logger = logging.getLogger(__name__)


# =========================
# Base system prompt
# =========================
SYSTEM_PROMPT = """
You are a professional Retrieval-Augmented Generation (RAG) assistant.

Rules:
- Use ONLY the provided context
- Do NOT hallucinate
- Do NOT include citations like [1], [2] in the answer text
- Sources will be shown separately
- If the answer is not present in the context, say you don't know
"""


class RAGChain:
    def __init__(
        self,
        vectorstore,
        model: str = None,
        top_k: int = None,
        temperature: float = None,
        client: OpenAI = None,
        api_key: Optional[str] = None,
        max_answer_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        base_url: str = "",
    ):
        """
        Args:
            vectorstore: Vector store with a similarity_search(query, k, filter) method
            model / top_k / temperature: explicit values; fall back to the
                MODEL_NAME / TOP_K_RESULTS / TEMPERATURE env vars, then defaults
            client: injectable OpenAI client (tests); built from api_key otherwise
            api_key: OpenAI API key, passed explicitly by the app from Settings.
                pydantic-settings reads .env WITHOUT exporting it into
                os.environ, so relying on the env var alone made a plain
                `uvicorn app.main:app` run with only a .env file fail at startup.
        """
        self.vectorstore = vectorstore
        self.model = model or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.top_k = int(top_k if top_k is not None else os.getenv("TOP_K_RESULTS", 5))
        self.temperature = float(
            temperature if temperature is not None else os.getenv("TEMPERATURE", 0)
        )
        self.max_answer_tokens = max_answer_tokens

        if client is not None:
            self.client = client
        else:
            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError("OPENAI_API_KEY not found")
            options = {}
            if timeout is not None:
                # The SDK's own default is a 600 s read timeout, far longer
                # than any of our clients will wait.
                options["timeout"] = timeout
            if max_retries is not None:
                options["max_retries"] = max_retries
            if base_url:
                options["base_url"] = base_url
            self.client = OpenAI(
                api_key=key,
                http_client=httpx.Client(trust_env=False),
                **options,
            )

    # =========================
    # Cost accounting
    # =========================
    def _log_usage(self, response, user_id: str) -> None:
        """Record what the answer cost, and whether the cap truncated it.

        Nothing measured spend per tenant before: response.usage was read off
        the wire and thrown away. finish_reason is logged alongside because
        "length" is how a max_answer_tokens cap that is set too low looks from
        the outside - a sentence that stops mid-word.
        """
        usage = getattr(response, "usage", None)
        choices = getattr(response, "choices", None) or []
        finish_reason = getattr(choices[0], "finish_reason", None) if choices else None

        logger.info(
            "chat completion user_id=%s model=%s prompt_tokens=%s "
            "completion_tokens=%s total_tokens=%s finish_reason=%s",
            user_id,
            self.model,
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
            getattr(usage, "total_tokens", None),
            finish_reason,
        )

        if finish_reason == "length":
            logger.warning(
                "Answer for user_id=%s was truncated at MAX_ANSWER_TOKENS=%s",
                user_id,
                self.max_answer_tokens,
            )

    # =========================
    # Build context
    # =========================
    def _build_context(self, docs: List[Document]) -> str:
        parts = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            parts.append(f"Source: {source}\n{doc.page_content}")
        return "\n\n".join(parts)

    # =========================
    # Strip [1], [2], etc.
    # =========================
    def _strip_citations(self, text: str) -> str:
        return re.sub(r"\[\d+\]", "", text).strip()

    # =========================
    # Main RAG method
    # =========================
    def ask(
        self, question: str, language: str = AUTO_LANGUAGE, user_id: str = None
    ) -> Dict:
        # A falsy user_id used to mean "no filter", i.e. search every tenant's
        # documents. No caller wants that, so make it impossible rather than
        # leaving a cross-tenant read one missing argument away.
        if not user_id:
            raise ValueError("user_id is required for retrieval")

        filter_dict = {"user_id": user_id}
        
        docs = self.vectorstore.similarity_search(
            question, k=self.top_k, filter=filter_dict
        )

        if not docs:
            return {
                "answer": "No relevant information found.",
                "sources": []
            }

        context = self._build_context(docs)

        lang_rule = rule_for(language)

        system_prompt = f"""
{SYSTEM_PROMPT}

Language rule:
- {lang_rule}
"""

        user_prompt = f"""
Context:
{context}

Question:
{question}

Answer:
"""

        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
        }
        if self.max_answer_tokens is not None:
            request["max_tokens"] = self.max_answer_tokens

        response = self.client.chat.completions.create(**request)
        self._log_usage(response, user_id)

        raw_answer = response.choices[0].message.content.strip()
        answer = self._strip_citations(raw_answer)

        # =========================
        # Collect unique sources
        # =========================
        sources = []
        seen = set()
        sid = 1

        for doc in docs:
            src = doc.metadata.get("source", "unknown")
            if src in seen:
                continue
            seen.add(src)

            sources.append({
                "id": sid,
                "source": src,
                "preview": doc.page_content[:200]
            })
            sid += 1

        return {
            "answer": answer,
            "sources": sources
        }
