from app.config import supabase


def fetch_candidates_by_vector_ids(vector_ids: list[str], user_id: str) -> list[dict]:
    if not vector_ids:
        return []

    result = (
        supabase.table("candidates")
        .select("id, name, email, resume_file_url, raw_text, summary, vector_id")
        .in_("vector_id", vector_ids)
        .eq("user_id", user_id)
        .execute()
    )

    return result.data
