"""
RAG Chain: Retrieval-Augmented Generation
"""

from typing import List, Dict
from langchain.schema import Document
from openai import OpenAI
import httpx
import os
import re


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
        model: str = "gpt-4o-mini",
        top_k: int = 3,
    ):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found")

        self.vectorstore = vectorstore
        self.top_k = top_k

        self.client = OpenAI(
            http_client=httpx.Client(trust_env=False)
        )

        self.model = os.getenv("MODEL_NAME", model)
        self.temperature = float(os.getenv("TEMPERATURE", 0))

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
