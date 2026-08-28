"""
Central application settings.

Single source of truth for configuration: every knob in .env.template maps
to a field here, and components receive values from this class instead of
reading os.environ themselves.
"""
from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),  # allow the model_name field
    )

    # --- Secrets ---
    openai_api_key: str
    # Shared secret for the X-API-Key header. Empty disables authentication
    # (development mode); a warning is logged at startup in that case.
    backend_api_key: str = ""

    # --- LLM / retrieval ---
    # min_length=1: an empty MODEL_NAME= in .env is falsy, and the chain's
    # `model or os.getenv(...)` fallback would silently swap in a different
    # model rather than surfacing the misconfiguration.
    model_name: str = Field(default="gpt-4o-mini", min_length=1)
    embedding_model: str = Field(default="text-embedding-3-small", min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_k_results: int = Field(default=5, ge=1)

    # Cosine similarity a chunk must reach to be worth putting in the prompt.
    # 0.0 (the default) keeps every candidate, which is what retrieval did
    # before this existed. Deliberately not defaulted to a guess: the right
    # number depends on the corpus and the embedding model, and picking one
    # blind risks discarding relevant context, which is far harder to notice
    # than including too much. Measure, then set it.
    relevance_threshold: float = Field(default=0.0, ge=0.0, le=1.0)

    # Past exchanges a follow-up question may draw on. Each turn adds prompt
    # tokens and the condensing call adds one request, so this is bounded.
    # 0 turns off multi-turn behaviour altogether.
    max_history_turns: int = Field(default=6, ge=0, le=20)

    # Diversity of the retrieved chunks. 1.0 ranks by relevance alone; lower
    # values cover more passages, which matters because overlapping chunks make
    # the neighbours of a strong match strong matches too.
    #
    # Ships at 1.0 - off - on measured grounds, not caution. The measurement in
    # DOCUMENTATION shows diversity buys recall on a question spanning two
    # documents only once precision has already dropped from 0.60 to 0.35: a
    # question about one topic then gets three unrelated chunks in its prompt.
    # Which mix a given deployment sees is not something this repository can
    # know, so the number is left to whoever can measure it.
    mmr_lambda: float = Field(default=1.0, ge=0.0, le=1.0)

    # Cap on generated answer length. Without one the model can produce an
    # unbounded completion at the operator's expense, and a long answer is
    # also what used to blow past Telegram's 4096-character message limit.
    max_answer_tokens: int = Field(default=1000, ge=1)

    # --- OpenAI transport ---
    # The SDK defaults to a 600 s read timeout while our own clients give up
    # after 60-120 s, so the server held threads for requests nobody was
    # waiting on any more. Keep this below the client timeouts.
    openai_timeout: float = Field(default=45.0, gt=0)
    openai_max_retries: int = Field(default=2, ge=0)
    # Set for Azure OpenAI or an OpenAI-compatible endpoint (vLLM, Ollama).
    # An explicit field is required because pydantic-settings reads .env
    # without exporting it, so the SDK's own env fallback never fires.
    openai_base_url: str = ""

    # --- Chunking ---
    chunk_size: int = Field(default=1000, ge=1)
    chunk_overlap: int = Field(default=200, ge=0)

    # --- Storage ---
    chroma_persist_dir: Path = Path("./data/chroma_db")
    collection_name: str = Field(default="documents", min_length=1)
    upload_dir: Path = Path("data/uploads")
    max_file_size: int = Field(default=30 * 1024 * 1024, ge=1)  # 30 MB

    # Which model providers a caller may name with their own key. Empty means
    # every provider in app/byok.py's table. The list exists because outbound
    # connections are the operator's business: a deployment behind egress rules
    # decides where its backend may talk, and that is not a caller's choice.
    allowed_model_providers: str = ""

    # --- Per-owner quotas ---
    # Until these existed the only bound was MAX_FILE_SIZE, on one file. One
    # user_id could fill the volume the vector store lives on, and every upload
    # is paid embedding calls with no ceiling. 0 disables a limit.
    #
    # These ship ON with generous defaults, unlike RELEVANCE_THRESHOLD and
    # MMR_LAMBDA, which ship off. The difference is how they fail: a wrong
    # threshold silently drops relevant context, while an exceeded quota is a
    # 413 with the numbers in it. A blind default is acceptable when overshoot
    # is loud.
    max_documents_per_user: int = Field(default=200, ge=0)
    max_bytes_per_user: int = Field(default=1024 * 1024 * 1024, ge=0)  # 1 GiB

    # --- Answer feedback ---
    # Ratings are stored only when a user presses a button, so nothing is
    # recorded unasked - but the record does contain their question and the
    # answer, which is the operator's decision to make, not this file's.
    # Setting this to false makes POST /feedback answer 404 and creates nothing
    # on disk.
    feedback_enabled: bool = True
    feedback_dir: Path = Path("data/feedback")
    # The file is append-only and never rotated here. At the cap, new ratings
    # are refused with 507 rather than filling the volume the vector store
    # lives on; move the file aside to start a fresh one.
    feedback_max_bytes: int = Field(default=10 * 1024 * 1024, ge=1)  # 10 MB

    @model_validator(mode="after")
    def _api_key_is_header_safe(self) -> "Settings":
        # HTTP header values are latin-1 on the wire while clients send UTF-8,
        # so a non-ASCII key never round-trips and could never match. Say so at
        # startup instead of rejecting every request with a puzzling 401.
        if not self.backend_api_key.isascii():
            raise ValueError(
                "BACKEND_API_KEY must contain only ASCII characters "
                "(HTTP headers cannot carry others)"
            )
        return self

    @model_validator(mode="after")
    def _chunk_overlap_fits_in_chunk(self) -> "Settings":
        # The splitter raises on overlap >= size too, but only once a document
        # is loaded. Fail fast at startup with a message that names the env vars.
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be smaller than "
                f"CHUNK_SIZE ({self.chunk_size})"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
