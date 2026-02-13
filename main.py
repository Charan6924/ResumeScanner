from fastapi import FastAPI, UploadFile, File, HTTPException
import PyPDF2
import io
app = FastAPI()


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
    
    return {
        "message": "Resume uploaded successfully",
        "filename": file.filename,
        "name": name,
        "email": email,
        "size": len(file_content)
    }