from fastapi import FastAPI, UploadFile, File, HTTPException, Query
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
    
    # TODO: Process resume (parse, store in DB, create embeddings)
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
