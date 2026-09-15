"""Okapi BM25 sparse retrieval index for high-precision keyword matching.

Dense retrieval (embeddings) captures semantic concepts, but often struggles
with exact entities, rare surnames, numbers, and inflected Russian terminology.
BM25 complements dense vectors by directly matching term frequencies (TF-IDF
variant with document length normalization).

This module provides a zero-external-dependency, thread-safe BM25Okapi
implementation optimized for multilingual (Russian, English, numerical) texts.
"""

import logging
import math
import re
import threading
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Regular expression splitting on whitespace and punctuation while preserving
# alphanumeric sequences in Cyrillic, Latin, and digits.
TOKEN_PATTERN = re.compile(r"(?u)\b\w+\b")


def tokenize_multilingual(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens supporting Russian and English."""
    if not text:
        return []
    return [t.lower() for t in TOKEN_PATTERN.findall(text)]


class BM25Index:
    """Thread-safe Okapi BM25 sparse retriever scoped per tenant (user_id).

    Parameters:
        k1: Controls term frequency saturation. Standard default is 1.5.
        b: Controls document length normalization penalty. Standard default is 0.75.
        epsilon: Floor for negative IDFs for ubiquitous terms.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        self._lock = threading.RLock()
        self._doc_counter = 0

        # Mapping: doc_id (int) -> Document
        self._docs: Dict[int, Document] = {}
        # Mapping: doc_id (int) -> List[str]
        self._doc_tokens: Dict[int, List[str]] = {}
        # Mapping: doc_id (int) -> int
        self._doc_lens: Dict[int, int] = {}
        # Term -> set of doc_ids
        self._inverted_index: Dict[str, Set[int]] = {}

        # Tenant tracking
        # owner -> set of doc_ids
        self._owner_docs: Dict[str, Set[int]] = {}
        # (owner, file_hash) -> set of doc_ids
        self._owner_hash_docs: Dict[Tuple[str, str], Set[int]] = {}

        # Cached statistics
        self._avgdl: float = 0.0
        self._idf_cache: Dict[str, float] = {}
        self._dirty: bool = False

    def _recalculate_statistics(self) -> None:
        """Recompute IDF and average document length after document mutations."""
        total_docs = len(self._docs)
        if total_docs == 0:
            self._avgdl = 0.0
            self._idf_cache.clear()
            self._dirty = False
            return

        self._avgdl = sum(self._doc_lens.values()) / float(total_docs)

        # Compute IDF for all terms in inverted index
        # Okapi BM25 formula: ln((N - n(q) + 0.5) / (n(q) + 0.5) + 1.0)
        new_idf: Dict[str, float] = {}
        negative_idfs: List[Tuple[str, float]] = []

        for term, doc_set in self._inverted_index.items():
            df = len(doc_set)
            val = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
            if val < 0:
                negative_idfs.append((term, val))
            else:
                new_idf[term] = val

        # Handle negative IDFs using epsilon floor based on average positive IDF
        if new_idf:
            avg_idf = sum(new_idf.values()) / len(new_idf)
            eps = self.epsilon * avg_idf
        else:
            eps = self.epsilon

        for term, _ in negative_idfs:
            new_idf[term] = eps

        self._idf_cache = new_idf
        self._dirty = False

    def add_documents(self, documents: List[Document], owner: Optional[str] = None) -> int:
        """Add a batch of documents to the BM25 index."""
        if not documents:
            return 0

        with self._lock:
            added = 0
            for doc in documents:
                text = doc.page_content or ""
                tokens = tokenize_multilingual(text)
                if not tokens:
                    continue

                doc_id = self._doc_counter
                self._doc_counter += 1

                self._docs[doc_id] = doc
                self._doc_tokens[doc_id] = tokens
                self._doc_lens[doc_id] = len(tokens)

                user_id = owner or doc.metadata.get("user_id") or ""
                file_hash = doc.metadata.get("file_hash") or ""

                if user_id:
                    self._owner_docs.setdefault(user_id, set()).add(doc_id)
                    if file_hash:
                        self._owner_hash_docs.setdefault((user_id, file_hash), set()).add(doc_id)

                for term in set(tokens):
                    self._inverted_index.setdefault(term, set()).add(doc_id)

                added += 1

            self._dirty = True
            self._recalculate_statistics()
            logger.debug("BM25 added %d documents (total in index: %d)", added, len(self._docs))
            return added

    def delete_by_file_hash(self, file_hash: str, owner: str) -> int:
        """Remove all chunks associated with a specific file hash for an owner."""
        with self._lock:
            key = (owner, file_hash)
            doc_ids = self._owner_hash_docs.pop(key, set())
            if not doc_ids:
                return 0

            for doc_id in doc_ids:
                self._docs.pop(doc_id, None)
                tokens = self._doc_tokens.pop(doc_id, [])
                self._doc_lens.pop(doc_id, None)

                for term in set(tokens):
                    term_docs = self._inverted_index.get(term)
                    if term_docs:
                        term_docs.discard(doc_id)
                        if not term_docs:
                            self._inverted_index.pop(term, None)

            owner_set = self._owner_docs.get(owner)
            if owner_set:
                owner_set.difference_update(doc_ids)

            self._dirty = True
            self._recalculate_statistics()
            logger.info("BM25 removed %d chunks for file_hash=%s owner=%s", len(doc_ids), file_hash, owner)
            return len(doc_ids)

    def clear_owner(self, owner: str) -> int:
        """Remove all documents belonging to an owner."""
        with self._lock:
            doc_ids = self._owner_docs.pop(owner, set())
            if not doc_ids:
                return 0

            # Clean up owner_hash_docs
            hash_keys = [k for k in self._owner_hash_docs if k[0] == owner]
            for k in hash_keys:
                self._owner_hash_docs.pop(k, None)

            for doc_id in doc_ids:
                self._docs.pop(doc_id, None)
                tokens = self._doc_tokens.pop(doc_id, [])
                self._doc_lens.pop(doc_id, None)

                for term in set(tokens):
                    term_docs = self._inverted_index.get(term)
                    if term_docs:
                        term_docs.discard(doc_id)
                        if not term_docs:
                            self._inverted_index.pop(term, None)

            self._dirty = True
            self._recalculate_statistics()
            logger.info("BM25 cleared %d documents for owner=%s", len(doc_ids), owner)
            return len(doc_ids)

    def count_owner(self, owner: str) -> int:
        """Return the number of documents indexed for a specific owner."""
        with self._lock:
            return len(self._owner_docs.get(owner, set()))

    def sync_from_chroma(self, collection: Any, owner: str) -> int:
        """Synchronize BM25 index with documents already present in ChromaDB for this owner."""
        if collection is None or not owner:
            return 0

        with self._lock:
            # If already indexed, skip fetching
            if len(self._owner_docs.get(owner, set())) > 0:
                return len(self._owner_docs[owner])

            try:
                res = collection.get(
                    where={"user_id": owner},
                    include=["documents", "metadatas"],
                )
                texts = res.get("documents") or []
                metadatas = res.get("metadatas") or []

                if not texts:
                    return 0

                docs: List[Document] = []
                for idx, text in enumerate(texts):
                    meta = dict(metadatas[idx] or {}) if idx < len(metadatas) else {}
                    meta["user_id"] = owner
                    docs.append(Document(page_content=text, metadata=meta))

                added = self.add_documents(docs, owner=owner)
                logger.info("BM25 synced %d documents from ChromaDB for owner=%s", added, owner)
                return added
            except Exception as err:
                logger.warning("BM25 sync from ChromaDB failed for owner=%s: %s", owner, err)
                return 0

    def search(
        self,
        query: str,
        owner: Optional[str] = None,
        k: int = 25,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Search for top_k documents using Okapi BM25.

        Returns:
            List of (Document, score) tuples sorted descending by BM25 score.
        """
        if not query or k <= 0:
            return []

        q_tokens = tokenize_multilingual(query)
        if not q_tokens:
            return []

        with self._lock:
            if self._dirty:
                self._recalculate_statistics()

            if not self._docs or self._avgdl <= 0:
                return []

            # Candidate filtering by owner
            candidate_ids: Optional[Set[int]] = None
            if owner:
                candidate_ids = self._owner_docs.get(owner)
                if not candidate_ids:
                    return []

            # Find matching document IDs containing at least one query term
            matched_ids: Set[int] = set()
            for qt in q_tokens:
                term_docs = self._inverted_index.get(qt)
                if term_docs:
                    if candidate_ids is not None:
                        matched_ids.update(term_docs & candidate_ids)
                    else:
                        matched_ids.update(term_docs)

            if not matched_ids:
                return []

            # Calculate BM25 scores for matching candidates
            scores: List[Tuple[float, int]] = []
            k1 = self.k1
            b = self.b
            avgdl = self._avgdl

            for doc_id in matched_ids:
                doc = self._docs[doc_id]

                # Apply metadata filter if provided
                if filter_dict:
                    meta = doc.metadata
                    match = True
                    for f_key, f_val in filter_dict.items():
                        if f_key == "user_id":
                            continue
                        if isinstance(f_val, dict) and "$in" in f_val:
                            if meta.get(f_key) not in f_val["$in"]:
                                match = False
                                break
                        elif meta.get(f_key) != f_val:
                            match = False
                            break
                    if not match:
                        continue

                tokens = self._doc_tokens[doc_id]
                doc_len = self._doc_lens[doc_id]
                tf = Counter(tokens)

                score = 0.0
                len_norm = k1 * (1.0 - b + b * (doc_len / avgdl))

                for qt in q_tokens:
                    count = tf.get(qt, 0)
                    if count > 0:
                        term_idf = self._idf_cache.get(qt, 0.0)
                        num = count * (k1 + 1.0)
                        denom = count + len_norm
                        score += term_idf * (num / denom)

                if score > 0.0:
                    scores.append((score, doc_id))

            scores.sort(key=lambda x: x[0], reverse=True)
            return [(self._docs[doc_id], score) for score, doc_id in scores[:k]]
