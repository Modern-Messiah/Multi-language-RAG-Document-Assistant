from typing import List, Optional

from pydantic import BaseModel, Field

# user_id doubles as a directory name and a ChromaDB metadata filter value, so
# the same constraint applies wherever it appears.
USER_ID_FIELD = Field(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class ChatTurn(BaseModel):
    """One exchange the client wants the assistant to remember."""

    question: str = Field(..., min_length=1, max_length=4000)
    answer: str = Field(..., min_length=1, max_length=8000)


class QueryRequest(BaseModel):
    # Upper bound as well as lower: the question is embedded and then sent to
    # the model, so an unbounded one is a cost and latency amplifier.
    question: str = Field(..., min_length=1, max_length=4000)
    language: str = "Auto"
    user_id: str = USER_ID_FIELD
    # Conversation history, held by the client rather than the server: the API
    # stays stateless, and nothing has to expire or be cleaned up. Capped here
    # against a client that would otherwise grow the prompt without limit;
    # MAX_HISTORY_TURNS decides how many of these are actually used.
    history: List[ChatTurn] = Field(default_factory=list, max_length=20)


class Source(BaseModel):
    id: int
    source: str
    preview: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]


class UploadResponse(BaseModel):
    """What /upload returns.

    Typed rather than a bare dict so the OpenAPI schema describes it, and so
    file_hash - the handle every other document endpoint takes - is part of
    the contract instead of an internal detail.
    """

    message: str
    filename: str
    chunks: int
    duplicate: bool
    file_hash: str
    # Set when this upload replaced an earlier revision stored under the same
    # filename. Two revisions used to coexist and answer questions together.
    replaced: bool = False


class DocumentSummary(BaseModel):
    """One indexed document, as a person thinks of it.

    Chunks are the storage unit; this folds them back into the file they came
    from.
    """

    file_hash: str
    source: str
    chunks: int
    type: Optional[str] = None
    pages: Optional[int] = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentSummary]
    total_chunks: int


class DeleteResponse(BaseModel):
    message: str
    file_hash: str
    chunks_removed: int


class ClearResponse(BaseModel):
    message: str


class ErrorResponse(BaseModel):
    """The shape of every error body, so clients can rely on it."""

    detail: str
