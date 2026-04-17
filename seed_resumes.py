"""
Seed resumes from HuggingFace dataset d4rk3r/resumes-raw-pdf into the database.

Usage:
    uv run python seed_resumes.py           # seeds 20 resumes
    uv run python seed_resumes.py --limit 5  # seeds 5 resumes
    uv run python seed_resumes.py --limit 0  # seeds ALL (expensive!)

Each resume calls OpenAI twice (embedding + summary). At ~$0.002/resume, 20 resumes ≈ $0.04.
"""
import argparse
import uuid
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import HfFileSystem
from app.config import supabase, pinecone_index, openai_client
from app.services.pdf import extract_text_from_pdf
from app.services.search import find_duplicate_resume
from embeddings import generate_embedding

DATASET_REPO = "datasets/d4rk3r/resumes-raw-pdf"
SEED_FOLDER = "it-domain"  # change to "all-domains" for broader set
SEED_USER_ID = ""  # set this to your Firebase UID, or pass --user-id


def list_pdf_paths(fs: HfFileSystem, limit: int) -> list[str]:
    folder = f"{DATASET_REPO}/{SEED_FOLDER}"
    all_files = fs.ls(folder, detail=False)
    pdfs = [f for f in all_files if f.endswith(".pdf")]
    return pdfs[:limit] if limit > 0 else pdfs


def process_resume(fs, hf_path: str, user_id: str) -> dict | None:
    filename = hf_path.split("/")[-1]

    with fs.open(hf_path, "rb") as f:
        file_content = f.read()

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
        "values": embedding,
        "metadata": {"filename": filename, "name": "", "email": ""}
    }])

    storage_path = f"resumes/{datetime.now().timestamp()}_{filename}"
    supabase.storage.from_("resumes").upload(
        storage_path, file_content, {"content-type": "application/pdf"}
    )
    resume_url = supabase.storage.from_("resumes").get_public_url(storage_path)

    result = supabase.table("candidates").insert({
        "name": None,
        "email": None,
        "resume_file_url": resume_url,
        "raw_text": raw_text,
        "summary": summary,
        "vector_id": vector_id,
        "user_id": user_id,
    }).execute()

    return result.data[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20,
                        help="Number of resumes to seed (0 = all 1940)")
    parser.add_argument("--user-id", type=str, default=SEED_USER_ID,
                        help="Firebase UID to assign as owner of seeded candidates")
    args = parser.parse_args()

    if not args.user_id:
        print("ERROR: provide your Firebase UID via --user-id <uid>")
        print("Find it in Firebase Console → Authentication → your user → UID")
        return

    user_id = args.user_id
    fs = HfFileSystem()
    print(f"Seeding as user_id={user_id}")
    print(f"Listing PDFs in {DATASET_REPO}/{SEED_FOLDER}...")
    paths = list_pdf_paths(fs, args.limit)
    print(f"Found {len(paths)} PDFs to process\n")

    success, skipped, failed = 0, 0, 0
    for i, path in enumerate(paths, 1):
        filename = path.split("/")[-1]
        print(f"[{i}/{len(paths)}] {filename}")
        try:
            result = process_resume(fs, path, user_id)
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
