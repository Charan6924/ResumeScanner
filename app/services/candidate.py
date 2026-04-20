from app.config import supabase
import re

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

    return result.data #type: ignore

def extract_name_and_email(raw_text: str) -> tuple[str | None, str | None]:
    # Email — regex is reliable
    email_match = re.search(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
        raw_text
    )
    email = email_match.group(0).lower() if email_match else None

    # Name — first non-empty line that looks like a real name
    name = None
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Must be 2-4 words, only letters/hyphens/apostrophes, no digits
        if re.fullmatch(r"[A-Za-z][A-Za-z'\-]+(?: [A-Za-z][A-Za-z'\-]+){1,3}", line):
            name = line
            break

    return name, email
