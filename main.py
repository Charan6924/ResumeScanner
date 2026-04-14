from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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
import firebase_admin
from firebase_admin import auth, credentials

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

firebase_cred = credentials.Certificate(os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH"))
firebase_admin.initialize_app(firebase_cred)

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        decoded = auth.verify_id_token(credentials.credentials)
        return decoded
    except Exception as e:
        print(f"Token verification failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")

app = FastAPI()
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase = create_client(str(url),str(key))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("text022026")


@app.get("/api/candidates/{id}")
async def get_candidate_by_id(id: str, user=Depends(verify_token)):
    """Get single candidate by ID"""
    result = supabase.table("candidates").select("*").eq("id", id).eq("user_id", user["uid"]).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Candidate not found")

    return result.data[0]


@app.get("/api/candidates")
async def list_candidates(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    user=Depends(verify_token)
):
    """List all candidates with pagination"""
    start = (page - 1) * limit
    end = start + limit - 1

    result = (
        supabase
        .table("candidates")
        .select("*", count="exact")
        .eq("user_id", user["uid"])
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
    email: str = None,
    user = Depends(verify_token)
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
    "vector_id": vector_id,
    "user_id": user["uid"]}).execute()

    candidate_id = result.data[0]["id"]

    return {
        "message": "Resume uploaded successfully",
        "candidate_id": candidate_id,
        "filename": file.filename,
        "resume_url": resume_url,
        "vector_id": vector_id
    }

@app.post("/api/search", response_model=list[SearchResult])
async def search_candidates(request: SearchRequest, user=Depends(verify_token)) -> list[SearchResult]:
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
    raise NotImplementedError


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
    raise NotImplementedError


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
async def update_candidate(id: str, request: UpdateCandidateRequest, user=Depends(verify_token)) -> dict:
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
    raise NotImplementedError


@app.delete("/api/candidates/{id}")
async def delete_candidate(id: str, user=Depends(verify_token)) -> dict:
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
    raise NotImplementedError

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
    raise NotImplementedError
