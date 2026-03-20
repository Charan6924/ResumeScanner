# Resume Screening API

AI-powered resume screening backend. Recruiters can search candidates using natural language queries instead of rigid filters.

## Tech Stack

- **FastAPI** - Async Python web framework
- **Supabase** - PostgreSQL database + file storage
- **OpenAI** - GPT-4o-mini for summaries, text-embedding-3-large for vectors
- **Pinecone** - Vector database for semantic search

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- Supabase project with `candidates` table and `resumes` storage bucket
- Pinecone index (dimension: 3072, metric: cosine)
- OpenAI API key

### Environment Variables

Create a `.env` file:

```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
```

### Install & Run

```bash
uv sync
uvicorn main:app --reload
```

## API Endpoints

### Upload Resume
```
POST /api/upload-resume
```
Upload a PDF resume. Extracts text, generates summary and embedding, stores in Supabase and Pinecone.

**Form Data:**
- `file` (required): PDF file
- `name` (optional): Candidate name
- `email` (optional): Candidate email

### Search Candidates
```
POST /api/search
```
Semantic search using natural language queries.

**Body:**
```json
{
  "query": "Python developers with ML experience",
  "top_k": 10,
  "rerank": true
}
```

### List Candidates
```
GET /api/candidates?page=1&limit=10
```
Paginated list of all candidates.

### Get Candidate
```
GET /api/candidates/{id}
```
Get a single candidate by ID.

### Update Candidate
```
PUT /api/candidates/{id}
```
Update candidate metadata (name/email).

**Body:**
```json
{
  "name": "New Name",
  "email": "new@email.com"
}
```

### Delete Candidate
```
DELETE /api/candidates/{id}
```
Remove candidate from database, storage, and vector index.

## Database Schema

**Supabase `candidates` table:**

| Column | Type | Description |
|--------|------|-------------|
| id | uuid | Primary key |
| name | text | Candidate name |
| email | text | Candidate email |
| resume_file_url | text | Public URL to PDF |
| raw_text | text | Extracted text from PDF |
| summary | text | AI-generated summary |
| vector_id | text | Pinecone vector ID |

## Architecture

```
Upload Flow:
PDF → Text Extraction → OpenAI Summary → Embedding → Pinecone + Supabase

Search Flow:
Query → Embedding → Pinecone Search → Fetch from Supabase → LLM Re-rank
```
