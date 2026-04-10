from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
import PyPDF2
import io
import os
import uuid
from supabase import create_client, Client
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
from pinecone import Pinecone
from embeddings import generate_embedding

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    rerank: bool = True

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

load_dotenv()

app = FastAPI()
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase = create_client(str(url),str(key))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("text022026")


@app.get("/api/candidates/{id}")
async def get_candidate_by_id(id: str):
    """Get single candidate by ID"""
    result = supabase.table("candidates").select("*").eq("id", id).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Candidate not found")

    return result.data[0]


@app.get("/api/candidates")
async def list_candidates(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100)
):
    """List all candidates with pagination"""
    start = (page - 1) * limit
    end = start + limit - 1

    result = (
        supabase
        .table("candidates")
        .select("*", count="exact")
        .order("id", desc=True)
        .range(start, end)
        .execute()
    )

    total = result.count if result.count is not None else len(result.data)

    return {
        "page": page,
        "limit": limit,
        "total": total,
        "data": result.data
    }

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF"""
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

@app.post("/api/upload-resume")
async def upload_resume(
    file: UploadFile = File(...),
    name: str = None,
    email: str = None
):
    """Upload and process a resume PDF"""
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files supported")
    
    # Read file
    file_content = await file.read()
    
    raw_text = extract_text_from_pdf(file_content)
    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="Failed to extract text from PDF")
    
    #generate summary using OpenAI API 
    client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Summarize resumes concisely."},
        {"role": "user", "content": f"Summarize this resume:\n\n{raw_text}"}
    ],
    max_tokens=200
)
    summary = response.choices[0].message.content

    # Generate embedding and upsert to Pinecone
    vector = generate_embedding(raw_text)
    vector_id = str(uuid.uuid4())
    pinecone_index.upsert(vectors=[{
        "id": vector_id,
        "values": vector,
        "metadata": {"filename": file.filename, "name": name or "", "email": email or ""}
    }]) 

    # Upload resume to Supabase storage bucket
    storage_path = f"resumes/{datetime.now().timestamp()}_{file.filename}"
    supabase.storage.from_("resumes").upload(
        storage_path,
        file_content,
        {"content-type": "application/pdf"}
    )
    resume_url = supabase.storage.from_("resumes").get_public_url(storage_path)

    #store metadata in Supabase database
    result = supabase.table("candidates").insert({
    "name": name,
    "email": email,
    "resume_file_url": resume_url,
    "raw_text": raw_text,
    "summary": summary,
    "vector_id": vector_id}).execute()

    candidate_id = result.data[0]["id"]

    return {
        "message": "Resume uploaded successfully",
        "candidate_id": candidate_id,
        "filename": file.filename,
        "resume_url": resume_url,
        "vector_id": vector_id
    }

@app.post("/api/search", response_model=list[SearchResult])
async def search_candidates(request: SearchRequest) -> list[SearchResult]:
    """
    Semantic search for candidates using natural language queries.

    This endpoint enables recruiters to find candidates using queries like
    "Python developers with 5 years experience in machine learning" instead
    of rigid keyword filters.

    Flow:
    1. Convert the query text into a vector embedding using OpenAI
    2. Query Pinecone for the top_k most similar resume vectors
    3. Fetch full candidate data from Supabase for matched vector IDs
    4. If rerank=True, use LLM to re-rank results for better relevance
    5. Return sorted list of candidates with similarity scores

    Args:
        request: SearchRequest containing:
            - query: Natural language search query
            - top_k: Number of results to return (default 10)
            - rerank: Whether to use LLM re-ranking (default True)

    Returns:
        List of SearchResult objects sorted by relevance score (highest first)

    Raises:
        HTTPException 400: If query is empty
        HTTPException 500: If embedding generation or search fails
    """
    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if request.top_k < 1:
        raise HTTPException(status_code=400, detail="top_k must be at least 1")

    try:
        query_vector = generate_embedding(query)
        matches = query_pinecone(query_vector, request.top_k)

        if not matches:
            return []

        vector_ids = [match["id"] for match in matches if match.get("id")]
        candidates = fetch_candidates_by_vector_ids(vector_ids)

        if not candidates:
            return []

        scores_by_vector_id = {
            match["id"]: float(match.get("score", 0.0))
            for match in matches
            if match.get("id")
        }
        candidates_by_vector_id = {
            candidate.get("vector_id"): candidate
            for candidate in candidates
            if candidate.get("vector_id")
        }

        ordered_candidates = []
        for vector_id in vector_ids:
            candidate = candidates_by_vector_id.get(vector_id)
            if candidate:
                enriched_candidate = candidate.copy()
                enriched_candidate["score"] = scores_by_vector_id.get(vector_id, 0.0)
                ordered_candidates.append(enriched_candidate)

        ranked_candidates = ordered_candidates
        if request.rerank and ordered_candidates:
            try:
                ranked_candidates = rerank_with_llm(query, ordered_candidates)
            except Exception:
                ranked_candidates = ordered_candidates

        return [
            SearchResult(
                candidate_id=str(candidate.get("id")),
                name=candidate.get("name"),
                email=candidate.get("email"),
                summary=candidate.get("summary") or "",
                score=float(candidate.get("relevance_score", candidate.get("score", 0.0))),
                resume_file_url=candidate.get("resume_file_url") or ""
            )
            for candidate in ranked_candidates
        ]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc


def query_pinecone(query_vector: list[float], top_k: int) -> list[dict]:
    """
    Query Pinecone index for similar vectors.

    Takes a query vector and returns the top_k most similar vectors
    from the Pinecone index along with their metadata and scores.

    Args:
        query_vector: 3072-dimension embedding vector from OpenAI
        top_k: Maximum number of results to return

    Returns:
        List of dicts containing:
            - id: The vector_id (matches candidate's vector_id in Supabase)
            - score: Cosine similarity score (0.0 to 1.0)
            - metadata: Dict with filename, name, email
    """
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


def fetch_candidates_by_vector_ids(vector_ids: list[str]) -> list[dict]:
    """
    Fetch full candidate records from Supabase by their vector IDs.

    Takes a list of vector_ids returned from Pinecone and retrieves
    the complete candidate records from the Supabase database.

    Args:
        vector_ids: List of vector_id strings to look up

    Returns:
        List of candidate dicts from Supabase, each containing:
            - id, name, email, resume_file_url, raw_text, summary, vector_id
    """
    if not vector_ids:
        return []

    result = supabase.table("candidates").select("id, name, email, resume_file_url, raw_text, summary, vector_id").in_("vector_id", vector_ids).execute()

    return result.data


def rerank_with_llm(query: str, candidates: list[dict]) -> list[dict]:
    """
    Re-rank search results using LLM for improved relevance.

    Vector similarity finds semantically related resumes, but may not
    perfectly match the recruiter's intent. This function uses GPT to
    analyze each candidate's summary against the original query and
    re-order results by true relevance.

    Args:
        query: The original natural language search query
        candidates: List of candidate dicts with at least 'summary' field

    Returns:
        Same list of candidates re-ordered by LLM-determined relevance,
        with an added 'relevance_score' field (0.0 to 1.0)
    """
    if not candidates:
        return []

    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    # Build a prompt to score all candidates at once
    candidate_summaries = "\n\n".join([
        f"Candidate {i}:\n{c.get('summary', 'No summary available')}"
        for i, c in enumerate(candidates)
    ])

    response = client.chat.completions.create(
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

    # Parse the LLM response
    import json
    scores_raw = json.loads(response.choices[0].message.content)

    # Handle both {"results": [...]} and direct [...] formats
    if isinstance(scores_raw, dict) and "results" in scores_raw:
        scores = scores_raw["results"]
    elif isinstance(scores_raw, list):
        scores = scores_raw
    else:
        # Fallback: return original order with equal scores
        for c in candidates:
            c['relevance_score'] = 0.5
        return candidates

    # Create scored candidates list
    scored_candidates = []
    for item in scores:
        idx = item.get('index', 0)
        score = item.get('score', 0.0)
        if 0 <= idx < len(candidates):
            candidate = candidates[idx].copy()
            candidate['relevance_score'] = score
            scored_candidates.append(candidate)

    # Add any candidates that weren't scored
    scored_indices = {item.get('index') for item in scores if 'index' in item}
    for i, c in enumerate(candidates):
        if i not in scored_indices:
            candidate = c.copy()
            candidate['relevance_score'] = 0.0
            scored_candidates.append(candidate)

    # Sort by relevance score descending
    scored_candidates.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

    return scored_candidates


@app.put("/api/candidates/{id}")
async def update_candidate(id: str, request: UpdateCandidateRequest) -> dict:
    """
    Update a candidate's metadata (name and/or email).

    Only updates the fields provided in the request. Does not allow
    updating the resume itself - for that, delete and re-upload.
    Also updates the metadata in Pinecone to keep it in sync.

    Args:
        id: The candidate's Supabase ID
        request: UpdateCandidateRequest with optional name and email fields

    Returns:
        Updated candidate record from Supabase

    Raises:
        HTTPException 400: If no fields provided to update
        HTTPException 404: If candidate not found
    """
    update_data = request.model_dump(exclude_none=True)

    if not update_data:
        raise HTTPException(status_code=400, detail="Provide at least one field to update")

    try:
        existing_result = supabase.table("candidates").select("*").eq("id", id).execute()
        if not existing_result.data:
            raise HTTPException(status_code=404, detail="Candidate not found")

        updated_result = (
            supabase
            .table("candidates")
            .update(update_data)
            .eq("id", id)
            .execute()
        )

        if not updated_result.data:
            raise HTTPException(status_code=404, detail="Candidate not found")

        updated_candidate = updated_result.data[0]

        vector_id = updated_candidate.get("vector_id")
        if vector_id:
            pinecone_index.update(
                id=vector_id,
                set_metadata={
                    "name": updated_candidate.get("name") or "",
                    "email": updated_candidate.get("email") or ""
                }
            )

        return updated_candidate
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to update candidate: {exc}") from exc


@app.delete("/api/candidates/{id}")
async def delete_candidate(id: str) -> dict:
    """
    Delete a candidate and all associated data.

    Removes the candidate from all storage locations:
    1. Delete vector from Pinecone index
    2. Delete PDF file from Supabase storage bucket
    3. Delete candidate record from Supabase database

    Operations should be performed in this order so that if any step
    fails, the data can still be cleaned up manually.

    Args:
        id: The candidate's Supabase ID

    Returns:
        Confirmation message with deleted candidate ID

    Raises:
        HTTPException 404: If candidate not found
        HTTPException 500: If deletion from any service fails
    """
    try:
        result = supabase.table("candidates").select("*").eq("id", id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Candidate not found")

        candidate = result.data[0]
        vector_id = candidate.get("vector_id")
        resume_url = candidate.get("resume_file_url") or ""

        if vector_id:
            pinecone_index.delete(ids=[vector_id])

        if resume_url:
            from urllib.parse import unquote

            storage_path = None
            public_markers = [
                "/storage/v1/object/public/resumes/",
                "/object/public/resumes/",
                "/resumes/"
            ]

            for marker in public_markers:
                if marker in resume_url:
                    storage_path = resume_url.split(marker, 1)[1]
                    break

            if storage_path:
                storage_path = unquote(storage_path)
                supabase.storage.from_("resumes").remove([storage_path])

        delete_result = supabase.table("candidates").delete().eq("id", id).execute()
        if hasattr(delete_result, "data") and delete_result.data == []:
            raise HTTPException(status_code=404, detail="Candidate not found")

        return {
            "message": "Candidate deleted successfully",
            "candidate_id": id
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete candidate: {exc}") from exc

def find_duplicate_resume(embedding: list[float], threshold: float = 0.95) -> Optional[str]:
    """
    Check if a similar resume already exists using vector similarity.

    Queries Pinecone with the new resume's embedding to find if there's
    an existing resume with similarity above the threshold. This catches
    both exact duplicates and near-duplicates (e.g., slightly reformatted
    versions of the same resume).

    Should be called during upload flow before upserting to Pinecone.

    Args:
        embedding: 3072-dimension vector of the new resume
        threshold: Minimum similarity score to consider a duplicate (default 0.95)

    Returns:
        candidate_id of existing resume if duplicate found, None otherwise
    """
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
