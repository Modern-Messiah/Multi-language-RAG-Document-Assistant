from pydantic import BaseModel
from typing import List, Optional


class QueryRequest(BaseModel):
    question: str
    language: str = "Auto"
    user_id: Optional[str] = None


class Source(BaseModel):
    id: int
    source: str
    preview: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
