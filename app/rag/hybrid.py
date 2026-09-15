"""Hybrid retrieval combining dense vector search and sparse BM25 retrieval.

Dense retrieval captures high-level semantic meaning and paraphrased questions,
while sparse BM25 retrieval captures exact keywords, rare terms, names, and numbers.

Results from both retrievers are merged using Reciprocal Rank Fusion (RRF):
    RRF_score(d) = sum( weight_i / (k_rrf + rank_i(d)) )

This ensures documents that appear in either list (and especially those ranking
high in both) are boosted effectively into the top candidate pool before
cross-encoder reranking.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from app.rag.bm25 import BM25Index

logger = logging.getLogger(__name__)

DEFAULT_RRF_K = 60
DEFAULT_DENSE_WEIGHT = 1.0
DEFAULT_BM25_WEIGHT = 1.0


def _document_key(doc: Document) -> str:
    """Generate a unique fingerprint for a chunk for deduplication."""
    file_hash = doc.metadata.get("file_hash", "")
    source = doc.metadata.get("source", "")
    # Fall back to start of content if no file hash exists
    content_prefix = (doc.page_content or "").strip()[:100]
    return f"{source}:{file_hash}:{content_prefix}"


def reciprocal_rank_fusion(
    ranked_lists: List[Tuple[List[Document], float]],
    k_rrf: int = DEFAULT_RRF_K,
) -> List[Document]:
    """Merge multiple ranked lists of documents using Reciprocal Rank Fusion (RRF).

    Args:
        ranked_lists: List of (documents, weight) pairs.
        k_rrf: Smoothing constant preventing early ranks from dominating (standard: 60).

    Returns:
        Deduplicated list of Document objects sorted descending by total RRF score.
    """
    if not ranked_lists:
        return []

    rrf_scores: Dict[str, float] = defaultdict(float)
    doc_registry: Dict[str, Document] = {}
    rank_details: Dict[str, Dict[str, int]] = defaultdict(dict)

    for list_idx, (doc_list, weight) in enumerate(ranked_lists):
        list_name = f"list_{list_idx}"
        for rank, doc in enumerate(doc_list, start=1):
            key = _document_key(doc)
            if key not in doc_registry:
                # Make a shallow copy of document with its metadata
                doc_registry[key] = Document(
                    page_content=doc.page_content,
                    metadata=dict(doc.metadata),
                )
            rrf_scores[key] += weight / (k_rrf + rank)
            rank_details[key][list_name] = rank

    # Sort documents by final RRF score descending
    sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

    fused_documents: List[Document] = []
    for key in sorted_keys:
        doc = doc_registry[key]
        doc.metadata["rrf_score"] = round(rrf_scores[key], 6)
        fused_documents.append(doc)

    return fused_documents


class HybridRetriever:
    """Coordinates dense vector search and sparse BM25 search."""

    def __init__(
        self,
        vectorstore: Any,
        bm25_index: BM25Index,
        embeddings_manager: Optional[Any] = None,
        dense_weight: float = DEFAULT_DENSE_WEIGHT,
        bm25_weight: float = DEFAULT_BM25_WEIGHT,
        k_rrf: int = DEFAULT_RRF_K,
        candidate_multiplier: int = 4,
    ):
        self.vectorstore = vectorstore
        self.bm25_index = bm25_index
        self.embeddings_manager = embeddings_manager
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.k_rrf = k_rrf
        self.candidate_multiplier = candidate_multiplier

    def retrieve(
        self,
        query: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        k: int = 5,
        dense_candidates: Optional[List[Document]] = None,
    ) -> List[Document]:
        """Perform hybrid retrieval combining dense and sparse search.

        Args:
            query: The user's query text.
            filter_dict: ChromaDB metadata filter (e.g. {'user_id': ...}).
            k: Target number of final candidates.
            dense_candidates: Pre-retrieved dense documents (optional).

        Returns:
            Merged and RRF-scored list of documents.
        """
        owner = (filter_dict or {}).get("user_id") if isinstance(filter_dict, dict) else None
        fetch_k = max(k * self.candidate_multiplier, 25)

        # 1. Dense search
        if dense_candidates is not None:
            dense_docs = dense_candidates
        else:
            try:
                dense_docs = self.vectorstore.similarity_search(
                    query=query,
                    k=fetch_k,
                    filter=filter_dict,
                )
            except Exception as err:
                logger.warning("Dense search in HybridRetriever failed (%s); continuing with BM25", err)
                dense_docs = []

        # 2. Sparse BM25 search
        # Lazy sync from Chroma if BM25 index has no documents for this owner yet
        if owner and self.embeddings_manager and getattr(self.embeddings_manager, "collection", None):
            if self.bm25_index.count_owner(owner) == 0:
                self.bm25_index.sync_from_chroma(self.embeddings_manager.collection, owner)

        bm25_results = self.bm25_index.search(
            query=query,
            owner=owner,
            k=fetch_k,
            filter_dict=filter_dict,
        )
        sparse_docs = [doc for doc, _ in bm25_results]

        logger.debug(
            "Hybrid retrieval for query '%s...': dense=%d, bm25=%d (fetch_k=%d)",
            query[:40],
            len(dense_docs),
            len(sparse_docs),
            fetch_k,
        )

        if not dense_docs and not sparse_docs:
            return []
        if not dense_docs:
            return sparse_docs[:fetch_k]
        if not sparse_docs:
            return dense_docs[:fetch_k]

        # 3. Fuse with RRF
        ranked_lists = [
            (dense_docs, self.dense_weight),
            (sparse_docs, self.bm25_weight),
        ]
        fused = reciprocal_rank_fusion(ranked_lists, k_rrf=self.k_rrf)
        return fused[:fetch_k]
