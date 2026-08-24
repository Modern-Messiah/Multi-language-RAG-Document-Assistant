"""
RAG Chain: Retrieval-Augmented Generation
"""

import os
import re
from typing import Dict, List, Optional

import httpx
from langchain.schema import Document
from openai import OpenAI

# =========================
# Language rules
# =========================
LANG_RULES = {
    "English": "Answer strictly in English.",
    "Русский": "Отвечай строго на русском языке.",
    "Қазақша": "Жауапты қатаң түрде қазақ тілінде бер.",
    "Français": "Réponds strictement en français.",
    "Deutsch": "Antworte ausschließlich auf Deutsch.",
    "Español": "Responde estrictamente en español.",
    "中文": "请严格使用简体中文回答。",
    "日本語": "必ず日本語で回答してください。"
}


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

        if client is not None:
            self.client = client
        else:
            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError("OPENAI_API_KEY not found")
            self.client = OpenAI(
                api_key=key,
                http_client=httpx.Client(trust_env=False)
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
    def ask(self, question: str, language: str = "Auto", user_id: str = None) -> Dict:
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

        if language == "Auto":
            lang_rule = (
                "Answer in the same language as the user's question. "
                "If the context is in another language, translate the answer."
            )
        else:
            lang_rule = LANG_RULES.get(
                language,
                "Answer in the same language as the user's question."
            )

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

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
        )

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
