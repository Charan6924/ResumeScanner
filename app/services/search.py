import json
from typing import Optional
from app.config import pinecone_index, openai_client, supabase


def query_pinecone(query_vector: list[float], top_k: int) -> list[dict]:
    if not query_vector:
        return []

    response = pinecone_index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    matches = getattr(response, "matches", None)
    if matches is None and isinstance(response, dict):
        matches = response.get("matches", [])

    normalized_matches = []
    for match in matches or []:
        if isinstance(match, dict):
            normalized_matches.append({
                "id": match.get("id"),
                "score": float(match.get("score", 0.0)),
                "metadata": match.get("metadata", {}) or {}
            })
        else:
            normalized_matches.append({
                "id": getattr(match, "id", None),
                "score": float(getattr(match, "score", 0.0)),
                "metadata": getattr(match, "metadata", {}) or {}
            })

    return normalized_matches


def rerank_with_llm(query: str, candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []

    candidate_summaries = "\n\n".join([
        f"Candidate {i}:\n{c.get('summary', 'No summary available')}"
        for i, c in enumerate(candidates)
    ])

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a recruiter assistant. Score each candidate's relevance to the search query. Return ONLY a JSON array of objects with 'index' (integer) and 'score' (float 0.0-1.0) fields, sorted by score descending."
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nCandidates:\n{candidate_summaries}\n\nReturn JSON array with index and score for each candidate, sorted by relevance."
            }
        ],
        response_format={"type": "json_object"},
        max_tokens=500
    )

    scores_raw = json.loads(response.choices[0].message.content)

    if isinstance(scores_raw, dict) and "results" in scores_raw:
        scores = scores_raw["results"]
    elif isinstance(scores_raw, list):
        scores = scores_raw
    else:
        for c in candidates:
            c['relevance_score'] = 0.5
        return candidates

    scored_candidates = []
    for item in scores:
        idx = item.get('index', 0)
        score = item.get('score', 0.0)
        if 0 <= idx < len(candidates):
            candidate = candidates[idx].copy()
            candidate['relevance_score'] = score
            scored_candidates.append(candidate)

    scored_indices = {item.get('index') for item in scores if 'index' in item}
    for i, c in enumerate(candidates):
        if i not in scored_indices:
            candidate = c.copy()
            candidate['relevance_score'] = 0.0
            scored_candidates.append(candidate)

    scored_candidates.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
    return scored_candidates


def find_duplicate_resume(embedding: list[float], threshold: float = 0.95) -> Optional[str]:
    if not embedding:
        return None

    try:
        matches = query_pinecone(embedding, top_k=1)
        if not matches:
            return None

        best_match = matches[0]
        score = float(best_match.get("score", 0.0))
        vector_id = best_match.get("id")

        if score < threshold or not vector_id:
            return None

        result = (
            supabase
            .table("candidates")
            .select("id")
            .eq("vector_id", vector_id)
            .limit(1)
            .execute()
        )

        if result.data:
            return str(result.data[0]["id"])

        return None
    except Exception:
        return None
