from pydantic import BaseModel
from typing import Optional


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    rerank: bool = True
    api_key: str = ""
    system_prompt: Optional[str] = None


class SearchResult(BaseModel):
    candidate_id: str
    name: Optional[str]
    email: Optional[str]
    summary: str
    score: float
    resume_file_url: str


class UpdateCandidateRequest(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
