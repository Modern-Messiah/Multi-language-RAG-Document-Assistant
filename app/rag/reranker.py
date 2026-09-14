"""Reranking candidate chunks for high-precision RAG.

Dense retrieval (Bi-Encoder embeddings) quickly finds candidate chunks from
the vector store based on cosine distance. However, Bi-Encoders compute vectors
for queries and passages independently, missing fine-grained token-level
cross-attention.

A Reranker (Cross-Encoder) evaluates the query and candidate document together:
    [CLS] query [SEP] document [SEP]
enabling all-to-all cross-attention across query and passage tokens. This
significantly improves precision, especially on dense numerical data, tables,
and multi-entity disambiguation.

Supported providers:
- 'flashrank': Ultra-lightweight local ONNX-based reranker (runs in ~10-20ms on CPU).
- 'cohere': Cohere Rerank API (v3.5 / multilingual) via direct HTTP.
- 'none': Passthrough (returns candidates without reranking).
"""
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import httpx
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

DEFAULT_FLASHRANK_MODEL = "ms-marco-TinyBERT-L-2-v2"
DEFAULT_COHERE_MODEL = "rerank-v3.5"
COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"


class BaseReranker(ABC):
    """Abstract interface for rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int = 5,
    ) -> List[Document]:
        """Rerank candidate documents given a query and return top_n items."""
        pass


class NoOpReranker(BaseReranker):
    """Passthrough reranker that preserves existing vector search ranking."""

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int = 5,
    ) -> List[Document]:
        return documents[:top_n]


class FlashRankReranker(BaseReranker):
    """Local, high-speed ONNX cross-encoder using FlashRank."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or DEFAULT_FLASHRANK_MODEL
        self._ranker = None
        self._init_error = None

    def _get_ranker(self):
        if self._ranker is None and self._init_error is None:
            try:
                from flashrank import Ranker
                self._ranker = Ranker(model_name=self.model_name)
                logger.info("Initialized FlashRank reranker (model=%s)", self.model_name)
            except Exception as err:
                self._init_error = err
                logger.warning("Failed to initialize FlashRank (%s); falling back to NoOp", err)
        return self._ranker

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int = 5,
    ) -> List[Document]:
        if not documents or top_n <= 0:
            return []

        ranker = self._get_ranker()
        if ranker is None:
            return documents[:top_n]

        try:
            from flashrank import RerankRequest

            passages = [
                {"id": idx, "text": doc.page_content, "meta": dict(doc.metadata)}
                for idx, doc in enumerate(documents)
            ]
            request = RerankRequest(query=query, passages=passages)
            results = ranker.rerank(request)

            reranked: List[Document] = []
            for item in results[:top_n]:
                orig_idx = item["id"]
                score = float(item.get("score", 0.0))
                orig_doc = documents[orig_idx]
                new_meta = dict(orig_doc.metadata)
                new_meta["rerank_score"] = score
                reranked.append(
                    Document(
                        page_content=orig_doc.page_content,
                        metadata=new_meta,
                    )
                )
            logger.info(
                "FlashRank reranked %d candidates -> %d top results (best_score=%.4f)",
                len(documents),
                len(reranked),
                reranked[0].metadata.get("rerank_score", 0.0) if reranked else 0.0,
            )
            return reranked
        except Exception as exc:
            logger.warning("FlashRank reranking failed: %s; using vector order", exc)
            return documents[:top_n]


class CohereReranker(BaseReranker):
    """API-based cross-encoder reranking via Cohere."""

    def __init__(
        self,
        api_key: str,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 10.0,
    ):
        self.api_key = (api_key or "").strip()
        self.model = model or DEFAULT_COHERE_MODEL
        self.base_url = (base_url or COHERE_RERANK_URL).strip()
        self.timeout = timeout

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int = 5,
    ) -> List[Document]:
        if not documents or top_n <= 0:
            return []
        if not self.api_key:
            logger.warning("CohereReranker has no API key; falling back to vector order")
            return documents[:top_n]

        payload = {
            "model": self.model,
            "query": query,
            "documents": [doc.page_content for doc in documents],
            "top_n": min(top_n, len(documents)),
            "return_documents": False,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "MultiLanguageRAG/1.0",
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(self.base_url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()

            results = data.get("results", [])
            reranked: List[Document] = []
            for item in results:
                idx = item.get("index")
                score = float(item.get("relevance_score", 0.0))
                if idx is not None and 0 <= idx < len(documents):
                    orig_doc = documents[idx]
                    new_meta = dict(orig_doc.metadata)
                    new_meta["rerank_score"] = score
                    reranked.append(
                        Document(
                            page_content=orig_doc.page_content,
                            metadata=new_meta,
                        )
                    )
            logger.info(
                "Cohere reranked %d candidates -> %d results (best_score=%.4f)",
                len(documents),
                len(reranked),
                reranked[0].metadata.get("rerank_score", 0.0) if reranked else 0.0,
            )
            return reranked if reranked else documents[:top_n]
        except Exception as exc:
            logger.warning("Cohere rerank failed (%s); falling back to vector order", exc)
            return documents[:top_n]


def get_reranker(
    settings=None,
    provider_override: Optional[str] = None,
    api_key_override: Optional[str] = None,
    model_override: Optional[str] = None,
) -> BaseReranker:
    """Factory to instantiate the appropriate reranker based on configuration or overrides."""
    enabled = getattr(settings, "reranker_enabled", False)
    provider = (
        provider_override
        if provider_override is not None
        else getattr(settings, "reranker_provider", "none")
    )
    provider = (provider or "none").strip().lower()

    if not enabled and not provider_override:
        return NoOpReranker()

    if provider == "flashrank":
        model = model_override or getattr(settings, "reranker_model", "")
        return FlashRankReranker(model_name=model or None)

    if provider == "cohere":
        api_key = api_key_override or getattr(settings, "reranker_api_key", "")
        model = model_override or getattr(settings, "reranker_model", "")
        return CohereReranker(api_key=api_key, model=model or None)

    return NoOpReranker()
