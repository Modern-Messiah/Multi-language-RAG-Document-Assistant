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

from app.rag.embeddings import DEFAULT_SPACE, distance_to_similarity, select_mmr
from app.rag.languages import AUTO_LANGUAGE, LANG_RULES, rule_for

# LANG_RULES is re-exported: it lived here first, and the clients now read the
# same table from app.rag.languages instead of keeping their own copies.
__all__ = ["LANG_RULES", "RAGChain", "SYSTEM_PROMPT"]

logger = logging.getLogger(__name__)

# A standalone question is one sentence; this only has to stop a runaway.
CONDENSE_MAX_TOKENS = 200

# How many candidates MMR gets to choose from, as a multiple of top_k. With no
# slack it has nothing to trade relevance against.
MMR_FETCH_MULTIPLIER = 4


NO_CONTEXT_ANSWER = "No relevant information found."

# A citation marker: an opening bracket, digits, a closing bracket.
_CITATION = re.compile(r"\[\d+\]")
# A prefix of one that has not finished arriving yet.
_PARTIAL_CITATION = re.compile(r"\[\d*$")


class CitationStripper:
    """Removes [1]-style markers from text arriving in pieces.

    The prompt asks the model not to emit them and ask() strips whatever slips
    through, but a stream cannot: "[1]" can arrive as "[", "1", "]" across
    three chunks, and a regex applied per chunk would pass all three through.

    Anything that might still turn into a marker is held back until the next
    chunk settles it, so at most a few characters are ever delayed.
    """

    def __init__(self):
        self._pending = ""

    def feed(self, text: str) -> str:
        """Text safe to emit now."""
        buffer = _CITATION.sub("", self._pending + text)

        held = _PARTIAL_CITATION.search(buffer)
        if held:
            self._pending = buffer[held.start():]
            return buffer[: held.start()]

        self._pending = ""
        return buffer

    def flush(self) -> str:
        """Whatever was held back, once the stream is over.

        An unterminated "[12" was never a citation, so it is real text and the
        user should see it.
        """
        remaining, self._pending = self._pending, ""
        return remaining


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
        relevance_threshold: float = 0.0,
        max_history_turns: int = 0,
        mmr_lambda: float = 1.0,
        embeddings_manager=None,
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
        # Cosine similarity below which a chunk is not worth putting in the
        # prompt. 0.0 keeps every candidate, which is the old behaviour.
        self.relevance_threshold = relevance_threshold
        # How many past exchanges inform retrieval and the answer. 0 disables
        # multi-turn entirely, which is how the chain behaved before.
        self.max_history_turns = max_history_turns
        # 1.0 is pure relevance, i.e. MMR off. Lower trades some relevance for
        # coverage of more than one passage.
        self.mmr_lambda = mmr_lambda
        # MMR needs candidate vectors, and only the manager can produce them.
        self.embeddings_manager = embeddings_manager
        self._warned_about_mmr = False

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
    def _log_usage(self, response, user_id: str, model: str = None) -> None:
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
            model or self.model,
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
    # Retrieval
    # =========================
    def _retrieve(self, question: str, filter_dict: dict) -> List[Document]:
        """Fetch candidates, dropping any that are not actually relevant.

        Plain similarity_search returns k chunks whether or not anything in the
        corpus has to do with the question, so asking about a topic the user
        never uploaded still filled the prompt with their nearest unrelated
        paragraphs. Whether the model then says "I don't know" or quietly
        answers from that noise is left entirely to the prompt.

        Scores are logged on every query even when filtering is off, so the
        threshold can be chosen from data rather than guessed.
        """
        if self._mmr_enabled():
            return self._retrieve_diverse(question, filter_dict)

        scored = self._search_with_scores(question, filter_dict)
        if scored is None:
            # A vector store without the scored API (or an injected double).
            return self.vectorstore.similarity_search(
                question, k=self.top_k, filter=filter_dict
            )

        space = self._index_space()
        similarities = []
        for document, distance in scored:
            similarity = distance_to_similarity(distance, space)
            similarities.append((document, similarity))

        if any(similarity is None for _, similarity in similarities):
            # Unknown metric: comparing against a scale that does not apply
            # would discard the best matches, so keep everything and say why.
            logger.warning(
                "Unknown index space %r - relevance filtering is disabled", space
            )
            return [document for document, _ in similarities]

        if similarities:
            logger.info(
                "retrieval space=%s candidates=%d similarity_best=%.3f "
                "similarity_worst=%.3f threshold=%s",
                space,
                len(similarities),
                max(s for _, s in similarities),
                min(s for _, s in similarities),
                self.relevance_threshold,
            )

        if not self.relevance_threshold:
            return [document for document, _ in similarities]

        kept = [
            document
            for document, similarity in similarities
            if similarity >= self.relevance_threshold
        ]
        dropped = len(similarities) - len(kept)
        if dropped:
            logger.info(
                "dropped %d/%d chunk(s) below RELEVANCE_THRESHOLD=%s",
                dropped,
                len(similarities),
                self.relevance_threshold,
            )
        return kept

    def _mmr_enabled(self) -> bool:
        """MMR needs candidate vectors, which only the manager can hand over.

        Without one - every injected test double, and any store that is not
        ChromaDB - retrieval stays on the plain scored path. Warned about once
        rather than silently, because a knob that appears to be set and is not
        is worse than one that is off.
        """
        if self.mmr_lambda >= 1.0:
            return False
        if self.embeddings_manager is None:
            if not self._warned_about_mmr:
                logger.warning(
                    "MMR_LAMBDA=%s but no embeddings manager was provided - "
                    "diversity re-ranking is inactive",
                    self.mmr_lambda,
                )
                self._warned_about_mmr = True
            return False
        return True

    def _retrieve_diverse(self, question: str, filter_dict: dict) -> List[Document]:
        """Threshold first, then pick a diverse subset of what survived.

        The order matters: MMR would otherwise spend one of its k slots on a
        chunk that is merely different, rather than different *and* relevant.
        """
        owner = filter_dict["user_id"]
        fetch_k = max(self.top_k, self.top_k * MMR_FETCH_MULTIPLIER)
        candidates = self.embeddings_manager.search_candidates(question, fetch_k, owner)
        if not candidates:
            return []

        space = self.embeddings_manager.index_space()
        scored = []
        for document, distance, vector in candidates:
            similarity = distance_to_similarity(distance, space)
            if similarity is None:
                logger.warning(
                    "Unknown index space %r - relevance filtering is disabled", space
                )
                similarity = 1.0
            scored.append((document, similarity, vector))

        logger.info(
            "retrieval space=%s candidates=%d similarity_best=%.3f "
            "similarity_worst=%.3f threshold=%s mmr_lambda=%s",
            space,
            len(scored),
            max(s for _, s, _ in scored),
            min(s for _, s, _ in scored),
            self.relevance_threshold,
            self.mmr_lambda,
        )

        if self.relevance_threshold:
            kept = [c for c in scored if c[1] >= self.relevance_threshold]
            if len(kept) != len(scored):
                logger.info(
                    "dropped %d/%d chunk(s) below RELEVANCE_THRESHOLD=%s",
                    len(scored) - len(kept),
                    len(scored),
                    self.relevance_threshold,
                )
            scored = kept

        return select_mmr(scored, self.top_k, self.mmr_lambda)

    def _search_with_scores(self, question: str, filter_dict: dict):
        """Scored search, or None when the store cannot do it."""
        search = getattr(self.vectorstore, "similarity_search_with_score", None)
        if search is None:
            return None
        try:
            return search(question, k=self.top_k, filter=filter_dict)
        except TypeError:
            # A double whose signature predates the filter argument.
            return None

    def _index_space(self) -> str:
        reader = getattr(self.vectorstore, "_collection", None)
        metadata = (getattr(reader, "metadata", None) or {}) if reader else {}
        return metadata.get("hnsw:space", DEFAULT_SPACE)

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
    # Conversation
    # =========================
    def _recent(self, history) -> List:
        """The last few turns, oldest first."""
        if not history:
            return []
        return list(history)[-self.max_history_turns:] if self.max_history_turns else []

    @staticmethod
    def _turn(entry):
        """Read a turn from either a pydantic model or a plain dict."""
        if isinstance(entry, dict):
            return entry.get("question", ""), entry.get("answer", "")
        return getattr(entry, "question", ""), getattr(entry, "answer", "")

    def _condense(self, question: str, history, client=None, model=None) -> str:
        """Rewrite a follow-up so it can stand on its own.

        "And the second one?" embeds to nothing useful: retrieval matched the
        literal words rather than what the user meant, so a follow-up pulled
        back unrelated chunks and the answer degraded exactly when the
        conversation got going.

        One extra model call, and only when there is history - a first question
        costs nothing new. If the call fails the original question is used, so
        a condensing hiccup degrades the answer instead of breaking the request.
        """
        recent = self._recent(history)
        if not recent:
            return question

        transcript = "\n".join(
            f"User: {q}\nAssistant: {a}" for q, a in map(self._turn, recent)
        )
        instruction = (
            "Rewrite the follow-up question as a standalone question in the "
            "same language, resolving pronouns and references from the "
            "conversation. Reply with the question only, nothing else. If it "
            "already stands alone, repeat it unchanged."
        )

        try:
            response = (client or self.client).chat.completions.create(
                model=model or self.model,
                messages=[
                    {"role": "system", "content": instruction},
                    {
                        "role": "user",
                        "content": f"Conversation:\n{transcript}\n\nFollow-up: {question}",
                    },
                ],
                temperature=0,
                max_tokens=CONDENSE_MAX_TOKENS,
            )
            rewritten = (response.choices[0].message.content or "").strip()
        except Exception:
            logger.exception("Could not condense the follow-up; using it as written")
            return question

        if not rewritten:
            return question

        logger.info("condensed follow-up %r -> %r", question, rewritten)
        return rewritten

    def _history_block(self, history) -> str:
        """The conversation so far, for the answering prompt."""
        recent = self._recent(history)
        if not recent:
            return ""
        lines = [
            f"User: {q}\nAssistant: {a}" for q, a in map(self._turn, recent)
        ]
        return "\n".join(lines)

    # =========================
    # Preparing a request
    # =========================
    def _collect_sources(self, docs: List[Document]) -> List[Dict]:
        """One entry per distinct source file, numbered from 1."""
        sources = []
        seen = set()

        for doc in docs:
            src = doc.metadata.get("source", "unknown")
            if src in seen:
                continue
            seen.add(src)
            sources.append({
                "id": len(sources) + 1,
                "source": src,
                "preview": doc.page_content[:200],
            })

        return sources

    def _prepare(self, question, language, user_id, history, client=None, model=None):
        """Retrieve and build the chat request.

        Shared by ask() and ask_stream() so the two cannot drift: a prompt
        change that reached only one of them would give the same question two
        different answers depending on which endpoint was called.

        Returns (request, sources); request is None when nothing was retrieved.
        """
        # A falsy user_id used to mean "no filter", i.e. search every tenant's
        # documents. No caller wants that, so make it impossible rather than
        # leaving a cross-tenant read one missing argument away.
        if not user_id:
            raise ValueError("user_id is required for retrieval")

        filter_dict = {"user_id": user_id}

        # Retrieve on the standalone form, answer the question as asked. The
        # condensing call goes on the caller's key too: it is their question
        # being rewritten, and billing half an exchange to each side would be
        # the strangest possible split.
        search_query = self._condense(question, history, client, model)
        docs = self._retrieve(search_query, filter_dict)

        if not docs:
            return None, []

        context = self._build_context(docs)
        lang_rule = rule_for(language)

        system_prompt = f"""
{SYSTEM_PROMPT}

Language rule:
- {lang_rule}
"""

        conversation = self._history_block(history)
        history_section = f"Conversation so far:\n{conversation}\n\n" if conversation else ""

        user_prompt = f"""
{history_section}Context:
{context}

Question:
{question}

Answer:
"""

        request = {
            "model": model or self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
        }
        if self.max_answer_tokens is not None:
            request["max_tokens"] = self.max_answer_tokens

        return request, self._collect_sources(docs)

    # =========================
    # Main RAG method
    # =========================
    def ask(
        self,
        question: str,
        language: str = AUTO_LANGUAGE,
        user_id: str = None,
        history=None,
        client=None,
        model=None,
    ) -> Dict:
        """Answer one question.

        `client` and `model` let a caller answer on their own API key with the
        model they chose; the chain's own are used when they are not given. The
        chain never keeps them - see app/byok.py for why.
        """
        request, sources = self._prepare(
            question, language, user_id, history, client, model
        )

        if request is None:
            return {"answer": NO_CONTEXT_ANSWER, "sources": [], "model": None}

        response = (client or self.client).chat.completions.create(**request)
        self._log_usage(response, user_id, model)

        raw_answer = (response.choices[0].message.content or "").strip()

        return {
            "answer": self._strip_citations(raw_answer),
            "sources": sources,
            "model": request["model"],
        }

    # =========================
    # Streaming
    # =========================
    def ask_stream(
        self,
        question: str,
        language: str = AUTO_LANGUAGE,
        user_id: str = None,
        history=None,
        client=None,
        model=None,
    ):
        """Yield the answer as it is generated.

        Five to fifteen seconds of a motionless spinner was the most visible
        latency in the product. Retrieval still happens up front - so a failure
        there is a normal HTTP error rather than something the client has to
        dig out of a half-delivered stream - and the sources go out first,
        since they are known before a single token is generated.

        Events are dicts, each one an SSE payload:
            {"type": "sources", "sources": [...]}
            {"type": "token", "text": "..."}
            {"type": "done"}
        """
        request, sources = self._prepare(
            question, language, user_id, history, client, model
        )

        if request is None:
            yield {"type": "sources", "sources": sources}
            yield {"type": "token", "text": NO_CONTEXT_ANSWER}
            yield {"type": "done"}
            return

        # The completion is opened BEFORE the first event. create() returns
        # once the response headers arrive, so an upstream refusal - a rejected
        # key above all, which is the commonest mistake when a caller brings
        # their own - raises while the handler is still priming this generator
        # and can answer with a real status code. Yielding sources first
        # committed a 200 to the wire and left the refusal to be dug out of the
        # stream as a mid-answer error event. Token order is unchanged: nothing
        # can be generated before the request is sent either way.
        stream = (client or self.client).chat.completions.create(**request, stream=True)

        yield {"type": "sources", "sources": sources}

        stripper = CitationStripper()

        finish_reason = None
        for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            finish_reason = getattr(choices[0], "finish_reason", None) or finish_reason
            delta = getattr(getattr(choices[0], "delta", None), "content", None)
            if not delta:
                continue
            text = stripper.feed(delta)
            if text:
                yield {"type": "token", "text": text}

        tail = stripper.flush()
        if tail:
            yield {"type": "token", "text": tail}

        # A streamed response carries no usage block, so the per-tenant cost
        # line that ask() logs is not available here; finish_reason is.
        logger.info(
            "streamed completion user_id=%s model=%s finish_reason=%s",
            user_id,
            model or self.model,
            finish_reason,
        )
        if finish_reason == "length":
            logger.warning(
                "Answer for user_id=%s was truncated at MAX_ANSWER_TOKENS=%s",
                user_id,
                self.max_answer_tokens,
            )

        yield {"type": "done"}
