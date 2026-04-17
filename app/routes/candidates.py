from urllib.parse import unquote
from fastapi import APIRouter, Depends, HTTPException, Query
from app.auth import verify_token
from app.schemas import UpdateCandidateRequest
from app.config import supabase, pinecone_index

router = APIRouter(prefix="/api")


@router.get("/candidates/{id}")
async def get_candidate_by_id(id: str, user=Depends(verify_token)):
    result = supabase.table("candidates").select("*").eq("id", id).eq("user_id", user["uid"]).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return result.data[0]


@router.get("/candidates")
async def list_candidates(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    user=Depends(verify_token)
):
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
    return {"page": page, "limit": limit, "total": total, "data": result.data}


@router.put("/candidates/{id}")
async def update_candidate(id: str, request: UpdateCandidateRequest, user=Depends(verify_token)) -> dict:
    update_data = request.model_dump(exclude_none=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="Provide at least one field to update")

    try:
        existing = supabase.table("candidates").select("*").eq("id", id).eq("user_id", user["uid"]).execute()
        if not existing.data:
            raise HTTPException(status_code=404, detail="Candidate not found")

        updated = (
            supabase.table("candidates")
            .update(update_data)
            .eq("id", id)
            .eq("user_id", user["uid"])
            .execute()
        )
        if not updated.data:
            raise HTTPException(status_code=404, detail="Candidate not found")

        updated_candidate = updated.data[0]
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


@router.delete("/candidates/{id}")
async def delete_candidate(id: str, user=Depends(verify_token)) -> dict:
    try:
        result = supabase.table("candidates").select("*").eq("id", id).eq("user_id", user["uid"]).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Candidate not found")

        candidate = result.data[0]
        vector_id = candidate.get("vector_id")
        resume_url = candidate.get("resume_file_url") or ""

        if vector_id:
            pinecone_index.delete(ids=[vector_id])

        if resume_url:
            storage_path = None
            for marker in ["/storage/v1/object/public/resumes/", "/object/public/resumes/", "/resumes/"]:
                if marker in resume_url:
                    storage_path = unquote(resume_url.split(marker, 1)[1])
                    break
            if storage_path:
                supabase.storage.from_("resumes").remove([storage_path])

        delete_result = supabase.table("candidates").delete().eq("id", id).eq("user_id", user["uid"]).execute()
        if hasattr(delete_result, "data") and delete_result.data == []:
            raise HTTPException(status_code=404, detail="Candidate not found")

        return {"message": "Candidate deleted successfully", "candidate_id": id}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete candidate: {exc}") from exc
