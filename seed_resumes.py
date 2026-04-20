"""
Seed resumes from a local folder of PDFs into the database.

Usage:
    uv run python seed_resumes.py
"""
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from app.config import supabase, pinecone_index, openai_client
from app.services.pdf import extract_text_from_pdf
from app.services.search import find_duplicate_resume
from embeddings import generate_embedding


@dataclass
class SeedConfig:
    folder: Path = Path("./resumes")
    user_id: str = ""        # your Firebase UID
    limit: int = 20          # 0 = process all PDFs in folder


CONFIG = SeedConfig(
    folder=Path(os.getenv("SEED_FOLDER", "")),
    user_id=os.getenv("SEED_USER_ID", ""),
    limit=20,
)


def extract_name_and_email(raw_text: str) -> tuple[str | None, str | None]:
    email_match = re.search(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
        raw_text
    )
    email = email_match.group(0).lower() if email_match else None

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract the full name of the candidate from the top of this resume. "
                    "Reply with ONLY the name, nothing else. "
                    "If you cannot determine the name, reply with null."
                ),
            },
            {"role": "user", "content": raw_text[:800]},
        ],
        max_tokens=20,
    )
    raw_name = response.choices[0].message.content.strip()
    name = None if raw_name.lower() in ("null", "none", "", "unknown") else raw_name

    return name, email


def process_resume(pdf_path: Path, user_id: str) -> dict | None:
    filename = pdf_path.name
    file_content = pdf_path.read_bytes()

    if len(file_content) > 10 * 1024 * 1024:
        print(f"  SKIP {filename} — exceeds 10MB")
        return None

    raw_text = extract_text_from_pdf(file_content)
    if not raw_text.strip():
        print(f"  SKIP {filename} — no extractable text")
        return None

    embedding = generate_embedding(raw_text)
    duplicate_id = find_duplicate_resume(embedding)
    if duplicate_id:
        print(f"  SKIP {filename} — duplicate of {duplicate_id}")
        return None

    name, email = extract_name_and_email(raw_text)
    print(f"  name={name!r}  email={email!r}")

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summarize resumes concisely."},
            {"role": "user", "content": f"Summarize this resume:\n\n{raw_text}"},
        ],
        max_tokens=200,
    )
    summary = response.choices[0].message.content

    vector_id = str(uuid.uuid4())
    pinecone_index.upsert(vectors=[{
        "id": vector_id,
        "values": embedding,
        "metadata": {"filename": filename, "name": name or "", "email": email or ""},
    }])

    storage_path = f"{user_id}/{vector_id}.pdf"
    supabase.storage.from_("resumes").upload(
        storage_path, file_content, {"content-type": "application/pdf"}
    )
    resume_url = supabase.storage.from_("resumes").get_public_url(storage_path)

    result = supabase.table("candidates").insert({
        "name": name,
        "email": email,
        "resume_file_url": resume_url,
        "raw_text": raw_text,
        "summary": summary,
        "vector_id": vector_id,
        "user_id": user_id,
    }).execute()

    return result.data[0]


def main():
    cfg = CONFIG

    if not cfg.user_id:
        print("ERROR: set user_id in SeedConfig")
        print("Find it in Firebase Console → Authentication → your user → UID")
        return

    if not cfg.folder.exists() or not cfg.folder.is_dir():
        print(f"ERROR: folder not found: {cfg.folder}")
        return

    pdfs = sorted(cfg.folder.glob("*.pdf"))
    if cfg.limit > 0:
        pdfs = pdfs[:cfg.limit]

    if not pdfs:
        print(f"No PDFs found in {cfg.folder}")
        return

    print(f"Seeding as user_id={cfg.user_id}")
    print(f"Found {len(pdfs)} PDFs in {cfg.folder}\n")

    success, skipped, failed = 0, 0, 0
    for i, pdf_path in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] {pdf_path.name}")
        try:
            result = process_resume(pdf_path, cfg.user_id)
            if result:
                print(f"  OK  candidate_id={result['id']}")
                success += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  ERR {e}")
            failed += 1

    print(f"\nDone — {success} inserted, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
