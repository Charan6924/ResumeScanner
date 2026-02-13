from fastapi import FastAPI, UploadFile, File, HTTPException
import PyPDF2
import io
import os
from supabase import create_client, Client
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime

load_dotenv()

app = FastAPI()
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase = create_client(str(url),str(key))

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
    "summary": summary}).execute()

    candidate_id = result.data[0]["id"]

    return {
        "message": "Resume uploaded successfully",
        "candidate_id": candidate_id,
        "filename": file.filename,
        "resume_url": resume_url
    }
