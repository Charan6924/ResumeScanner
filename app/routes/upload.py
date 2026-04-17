import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from app.auth import verify_token
from app.config import supabase, pinecone_index, openai_client, limiter
from app.services.pdf import extract_text_from_pdf
from app.services.search import find_duplicate_resume
from embeddings import generate_embedding

router = APIRouter(prefix="/api")


@router.post("/upload-resume")
@limiter.limit("20/minute")
async def upload_resume(
    request: Request,
    file: UploadFile = File(...),
    name: str = None,
    email: str = None,
    user=Depends(verify_token)
):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    file_content = await file.read()
    raw_text = extract_text_from_pdf(file_content)
    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="Failed to extract text from PDF")

    vector_for_dedup = generate_embedding(raw_text)
    duplicate_id = find_duplicate_resume(vector_for_dedup)

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summarize resumes concisely."},
            {"role": "user", "content": f"Summarize this resume:\n\n{raw_text}"}
        ],
        max_tokens=200
    )
    summary = response.choices[0].message.content

    vector_id = str(uuid.uuid4())
    pinecone_index.upsert(vectors=[{
        "id": vector_id,
        "values": vector_for_dedup,
        "metadata": {"filename": file.filename, "name": name or "", "email": email or ""}
    }])

    storage_path = f"resumes/{datetime.now().timestamp()}_{file.filename}"
    supabase.storage.from_("resumes").upload(storage_path, file_content, {"content-type": "application/pdf"})
    resume_url = supabase.storage.from_("resumes").get_public_url(storage_path)

    result = supabase.table("candidates").insert({
        "name": name,
        "email": email,
        "resume_file_url": resume_url,
        "raw_text": raw_text,
        "summary": summary,
        "vector_id": vector_id,
        "user_id": user["uid"]
    }).execute()

    candidate_id = result.data[0]["id"]
    response_data = {
        "message": "Resume uploaded successfully",
        "candidate_id": candidate_id,
        "filename": file.filename,
        "resume_url": resume_url,
        "vector_id": vector_id
    }
    if duplicate_id:
        response_data["duplicate_of"] = duplicate_id
        response_data["message"] = "Resume uploaded (possible duplicate detected)"
    return response_data


@router.post("/upload-resumes/bulk")
@limiter.limit("10/minute")
async def bulk_upload_resumes(
    request: Request,
    files: list[UploadFile] = File(...),
    user=Depends(verify_token)
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 files per bulk upload")

    results = []
    for file in files:
        try:
            if not file.filename.endswith(".pdf"):
                results.append({"filename": file.filename, "error": "Only PDF files supported"})
                continue

            file_content = await file.read()
            raw_text = extract_text_from_pdf(file_content)
            if not raw_text.strip():
                results.append({"filename": file.filename, "error": "Failed to extract text from PDF"})
                continue

            vector_for_dedup = generate_embedding(raw_text)
            duplicate_id = find_duplicate_resume(vector_for_dedup)

            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Summarize resumes concisely."},
                    {"role": "user", "content": f"Summarize this resume:\n\n{raw_text}"}
                ],
                max_tokens=200
            )
            summary = resp.choices[0].message.content

            vector_id = str(uuid.uuid4())
            pinecone_index.upsert(vectors=[{
                "id": vector_id,
                "values": vector_for_dedup,
                "metadata": {"filename": file.filename, "name": "", "email": ""}
            }])

            storage_path = f"resumes/{datetime.now().timestamp()}_{file.filename}"
            supabase.storage.from_("resumes").upload(storage_path, file_content, {"content-type": "application/pdf"})
            resume_url = supabase.storage.from_("resumes").get_public_url(storage_path)

            db_result = supabase.table("candidates").insert({
                "name": None,
                "email": None,
                "resume_file_url": resume_url,
                "raw_text": raw_text,
                "summary": summary,
                "vector_id": vector_id,
                "user_id": user["uid"]
            }).execute()

            candidate_id = db_result.data[0]["id"]
            entry = {
                "filename": file.filename,
                "candidate_id": candidate_id,
                "vector_id": vector_id,
                "resume_url": resume_url
            }
            if duplicate_id:
                entry["duplicate_of"] = duplicate_id
            results.append(entry)
        except Exception as exc:
            results.append({"filename": file.filename, "error": str(exc)})

    return {"uploaded": len([r for r in results if "candidate_id" in r]), "results": results}
