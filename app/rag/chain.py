"""
RAG Chain: Retrieval-Augmented Generation
"""

from typing import List, Dict
from langchain.schema import Document
from openai import OpenAI
import httpx
import os


# =========================
# System Prompt
# =========================
SYSTEM_PROMPT = """
You are a Retrieval-Augmented Generation (RAG) assistant.

Rules:
- Answer ONLY using the provided context
- If the user asks in Russian, answer in Russian
- If the user asks in English, answer in English
- If the context is in English and the question is in Russian, translate the answer
- If the answer is not in the context, say that you don't know
- Do NOT hallucinate
- Cite sources using [number]
"""


class RAGChain:
    def __init__(
        self,
        vectorstore,
        model: str = "gpt-4o-mini",
        top_k: int = 3,
    ):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found")

        self.vectorstore = vectorstore
        self.top_k = top_k

        # OpenAI client (without proxy)
        self.client = OpenAI(
            http_client=httpx.Client(trust_env=False)
        )

        self.model = os.getenv("MODEL_NAME", model)
        self.temperature = float(os.getenv("TEMPERATURE", 0))

    # =========================
    # Build context from docs
    # =========================
    def _build_context(self, docs: List[Document]) -> str:
        context_parts = []

        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            context_parts.append(
                f"[{i}] Source: {source}\n{doc.page_content}"
            )

        return "\n\n".join(context_parts)

    # =========================
    # Main RAG method
    # =========================
    def ask(self, question: str, language: str = "Auto") -> Dict:
        docs = self.vectorstore.similarity_search(
            question, k=self.top_k
        )

        if not docs:
            return {
                "answer": "No relevant information found.",
                "sources": []
            }

        context = self._build_context(docs)

        if language == "English":
            lang_rule = "Answer strictly in English."
        elif language == "Русский":
            lang_rule = "Отвечай строго на русском языке."
        else:
            lang_rule = (
                "Answer in the same language as the user's question. "
                "If context is English and question is Russian, translate."
            )

        system_prompt = f"""
    You are a professional RAG assistant.

    Rules:
    - Use ONLY the provided context
    - Do NOT hallucinate
    - Cite sources using [number]
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

        answer = response.choices[0].message.content.strip()

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
