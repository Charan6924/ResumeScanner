from fastapi import FastAPI, UploadFile, File, HTTPException

app = FastAPI()

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
    
    return {
        "message": "Resume uploaded successfully",
        "filename": file.filename,
        "name": name,
        "email": email,
        "size": len(file_content)
    }