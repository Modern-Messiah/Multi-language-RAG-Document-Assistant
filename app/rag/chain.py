"""
RAG Chain: Retrieval-Augmented Generation
"""

from typing import List, Dict
from langchain.schema import Document
from openai import OpenAI
import httpx
import os


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

    def _build_context(self, docs: List[Document]) -> str:
        """
        Build context block from retrieved documents
        """
        context_parts = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            context_parts.append(
                f"[{i}] Source: {source}\n{doc.page_content}"
            )
        return "\n\n".join(context_parts)

    def ask(self, question: str) -> Dict:
        """
        Ask a question using RAG
        Returns answer + sources
        """
        # 1. Retrieve
        docs = self.vectorstore.similarity_search(
            question, k=self.top_k
        )

        if not docs:
            return {
                "answer": "No relevant information found.",
                "sources": []
            }

        # 2. Build context
        context = self._build_context(docs)

        # 3. Prompt
        prompt = f"""
You are an AI assistant answering questions based ONLY on the provided context.

Context:
{context}

Question:
{question}

Rules:
- Use only the information from context
- If the answer is not in the context, say you don't know
- Cite sources using [number]

Answer:
"""

        # 4. Generate
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )


        answer = response.choices[0].message.content.strip()

        # 5. Collect sources
        sources = []
        for i, doc in enumerate(docs, start=1):
            sources.append({
                "id": i,
                "source": doc.metadata.get("source", "unknown"),
                "preview": doc.page_content[:120]
            })

        return {
            "answer": answer,
            "sources": sources
        }
