from fastapi import APIRouter, Depends, HTTPException, Request
from openai import OpenAI
from app.auth import verify_token
from app.config import limiter
from app.schemas import SearchRequest, SearchResult
from app.services.search import query_pinecone, rerank_with_llm
from app.services.candidate import fetch_candidates_by_vector_ids
from embeddings import generate_embedding

router = APIRouter(prefix="/api")


@router.post("/search", response_model=list[SearchResult])
@limiter.limit("30/minute")
async def search_candidates(request: Request, body: SearchRequest, user=Depends(verify_token)) -> list[SearchResult]:
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if body.top_k < 1:
        raise HTTPException(status_code=400, detail="top_k must be at least 1")
    if not body.api_key:
        raise HTTPException(status_code=400, detail="api_key is required")

    per_request_client = OpenAI(api_key=body.api_key)

    try:
        query_vector = generate_embedding(query)
        matches = query_pinecone(query_vector, body.top_k)
        if not matches:
            return []

        vector_ids = [match["id"] for match in matches if match.get("id")]
        candidates = fetch_candidates_by_vector_ids(vector_ids, user["uid"])
        if not candidates:
            return []

        scores_by_vector_id = {
            match["id"]: float(match.get("score", 0.0))
            for match in matches if match.get("id")
        }
        candidates_by_vector_id = {
            c.get("vector_id"): c for c in candidates if c.get("vector_id")
        }

        ordered_candidates = []
        for vector_id in vector_ids:
            candidate = candidates_by_vector_id.get(vector_id)
            if candidate:
                enriched = candidate.copy()
                enriched["score"] = scores_by_vector_id.get(vector_id, 0.0)
                ordered_candidates.append(enriched)

        ranked_candidates = ordered_candidates
        if body.rerank and ordered_candidates:
            ranked_candidates = rerank_with_llm(
                query,
                ordered_candidates,
                client=per_request_client,
                system_prompt=body.system_prompt,
            )

        return [
            SearchResult(
                candidate_id=str(c.get("id")),
                name=c.get("name"),
                email=c.get("email"),
                summary=c.get("summary") or "",
                score=float(c.get("relevance_score", c.get("score", 0.0))),
                resume_file_url=c.get("resume_file_url") or ""
            )
            for c in ranked_candidates
        ]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc
