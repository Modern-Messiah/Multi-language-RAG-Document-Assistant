from typing import List

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    # Upper bound as well as lower: the question is embedded and then sent to
    # the model, so an unbounded one is a cost and latency amplifier.
    question: str = Field(..., min_length=1, max_length=4000)
    language: str = "Auto"
    # Same constraints as the user_id query param: it is used as a metadata
    # filter value, so it must always be present and filter-safe.
    user_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class Source(BaseModel):
    id: int
    source: str
    preview: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
