from typing import List, Literal, Optional

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


class FeedbackRequest(BaseModel):
    """A rating of one answer, sent back by the client that showed it.

    The client sends the exchange rather than the server remembering it: the API
    stays stateless, nothing has to expire, and a replica that did not serve the
    query can still take the rating. The cost is trusting the client's copy of
    its own conversation, which is the same trust the history field already
    requires.
    """

    rating: Literal["up", "down"]
    user_id: str = USER_ID_FIELD
    question: str = Field(..., min_length=1, max_length=4000)
    # The answer that was rated. Optional because a rating is still worth
    # keeping without it, and long answers are already split for Telegram.
    answer: str = Field(default="", max_length=8000)
    # Filenames as shown to the user, not chunk ids: a golden case is written
    # in terms of documents.
    sources: List[str] = Field(default_factory=list, max_length=20)
    # The id of the *rated* request, from its X-Request-ID header - the thread
    # back to the log lines that produced the answer. Optional: a rating whose
    # client did not keep the header is better recorded than dropped.
    request_id: Optional[str] = Field(
        default=None, max_length=64, pattern=r"^[A-Za-z0-9._-]+$"
    )
    comment: Optional[str] = Field(default=None, max_length=1000)
    language: str = Field(default="", max_length=32)
    # Which surface produced it, so "the bot's answers are worse" becomes a
    # number. Constrained because it is written into a file operators read.
    client: str = Field(default="", max_length=16, pattern=r"^[a-z]*$")


class FeedbackResponse(BaseModel):
    message: str


class ErrorResponse(BaseModel):
    """The shape of every error body, so clients can rely on it."""

    detail: str
