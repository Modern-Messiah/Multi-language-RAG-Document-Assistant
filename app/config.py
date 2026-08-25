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
