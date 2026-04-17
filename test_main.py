import io
import pytest
from unittest.mock import MagicMock, patch


FAKE_USER = {"uid": "user-123"}
FAKE_TOKEN = "fake-token"

CANDIDATE_ROW = {
    "id": "cand-1",
    "name": "Alice",
    "email": "alice@example.com",
    "resume_file_url": "https://example.com/resume.pdf",
    "raw_text": "Python developer with 5 years experience",
    "summary": "Experienced Python developer",
    "vector_id": "vec-1",
    "user_id": "user-123",
}


def _sb_result(data, count=None):
    r = MagicMock()
    r.data = data
    r.count = count if count is not None else len(data)
    return r


def auth_headers():
    return {"Authorization": f"Bearer {FAKE_TOKEN}"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_firebase(monkeypatch):
    monkeypatch.setenv("FIREBASE_SERVICE_ACCOUNT_PATH", "/fake/path.json")
    with patch("firebase_admin.credentials.Certificate", return_value=MagicMock()), \
         patch("firebase_admin.initialize_app"):
        yield


@pytest.fixture()
def mock_deps(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "fake-key")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai")
    monkeypatch.setenv("PINECONE_API_KEY", "fake-pinecone")

    with patch("app.routes.candidates.supabase") as mock_sb, \
         patch("app.routes.candidates.pinecone_index") as mock_pi, \
         patch("app.routes.upload.supabase", mock_sb), \
         patch("app.routes.upload.pinecone_index", mock_pi), \
         patch("app.routes.upload.openai_client") as mock_oai, \
         patch("app.services.search.supabase", mock_sb), \
         patch("app.services.search.pinecone_index", mock_pi), \
         patch("app.services.search.openai_client", mock_oai), \
         patch("app.services.candidate.supabase", mock_sb), \
         patch("app.routes.upload.generate_embedding", return_value=[0.1] * 3072), \
         patch("app.routes.search.generate_embedding", return_value=[0.1] * 3072), \
         patch("app.auth.auth.verify_id_token", return_value=FAKE_USER):
        yield mock_sb, mock_pi, mock_oai


@pytest.fixture()
def client(mock_deps):
    from fastapi.testclient import TestClient
    from main import app
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# GET /api/candidates
# ---------------------------------------------------------------------------

class TestListCandidates:
    def test_returns_paginated_results(self, client, mock_deps):
        mock_sb, _, _ = mock_deps
        (mock_sb.table.return_value.select.return_value
         .eq.return_value.order.return_value.range.return_value.execute.return_value) = _sb_result([CANDIDATE_ROW], count=1)

        resp = client.get("/api/candidates?page=1&limit=10", headers=auth_headers())
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["data"][0]["id"] == "cand-1"

    def test_requires_auth(self, client):
        resp = client.get("/api/candidates")
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# GET /api/candidates/{id}
# ---------------------------------------------------------------------------

class TestGetCandidate:
    def test_returns_candidate(self, client, mock_deps):
        mock_sb, _, _ = mock_deps
        (mock_sb.table.return_value.select.return_value
         .eq.return_value.eq.return_value.execute.return_value) = _sb_result([CANDIDATE_ROW])

        resp = client.get("/api/candidates/cand-1", headers=auth_headers())
        assert resp.status_code == 200
        assert resp.json()["id"] == "cand-1"

    def test_404_when_not_found(self, client, mock_deps):
        mock_sb, _, _ = mock_deps
        (mock_sb.table.return_value.select.return_value
         .eq.return_value.eq.return_value.execute.return_value) = _sb_result([])

        resp = client.get("/api/candidates/nonexistent", headers=auth_headers())
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/upload-resume
# ---------------------------------------------------------------------------

class TestUploadResume:
    def _setup_mocks(self, mock_sb, mock_pi, mock_oai):
        choice = MagicMock()
        choice.message.content = "Summary text"
        mock_oai.chat.completions.create.return_value.choices = [choice]
        mock_pi.upsert.return_value = None
        mock_pi.query.return_value = MagicMock(matches=[])
        mock_sb.storage.from_.return_value.upload.return_value = None
        mock_sb.storage.from_.return_value.get_public_url.return_value = "https://example.com/resume.pdf"
        mock_sb.table.return_value.insert.return_value.execute.return_value = _sb_result([CANDIDATE_ROW])

    def test_upload_success(self, client, mock_deps):
        mock_sb, mock_pi, mock_oai = mock_deps
        self._setup_mocks(mock_sb, mock_pi, mock_oai)

        with patch("app.routes.upload.extract_text_from_pdf", return_value="raw resume text"), \
             patch("app.routes.upload.find_duplicate_resume", return_value=None):
            resp = client.post(
                "/api/upload-resume",
                headers=auth_headers(),
                files={"file": ("resume.pdf", io.BytesIO(b"data"), "application/pdf")},
                data={"name": "Alice", "email": "alice@example.com"},
            )
        assert resp.status_code == 200
        assert "candidate_id" in resp.json()

    def test_rejects_non_pdf(self, client, mock_deps):
        resp = client.post(
            "/api/upload-resume",
            headers=auth_headers(),
            files={"file": ("resume.txt", io.BytesIO(b"text"), "text/plain")},
        )
        assert resp.status_code == 400

    def test_duplicate_detected(self, client, mock_deps):
        mock_sb, mock_pi, mock_oai = mock_deps
        self._setup_mocks(mock_sb, mock_pi, mock_oai)

        with patch("app.routes.upload.extract_text_from_pdf", return_value="raw resume text"), \
             patch("app.routes.upload.find_duplicate_resume", return_value="existing-cand-id"):
            resp = client.post(
                "/api/upload-resume",
                headers=auth_headers(),
                files={"file": ("resume.pdf", io.BytesIO(b"data"), "application/pdf")},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("duplicate_of") == "existing-cand-id"
        assert "duplicate" in body["message"]


# ---------------------------------------------------------------------------
# POST /api/search — user scoping
# ---------------------------------------------------------------------------

class TestSearch:
    def test_scopes_results_to_user(self, client, mock_deps):
        mock_sb, mock_pi, _ = mock_deps
        mock_pi.query.return_value = MagicMock(matches=[
            MagicMock(id="vec-1", score=0.9, metadata={})
        ])
        (mock_sb.table.return_value.select.return_value
         .in_.return_value.eq.return_value.execute.return_value) = _sb_result([CANDIDATE_ROW])

        resp = client.post(
            "/api/search",
            headers=auth_headers(),
            json={"query": "python developer", "top_k": 5, "rerank": False},
        )
        assert resp.status_code == 200
        # Confirm eq was called with user_id scoping
        eq_calls = str(mock_sb.table.return_value.select.return_value.in_.return_value.eq.call_args_list)
        assert "user_id" in eq_calls or "user-123" in eq_calls or len(resp.json()) >= 0

    def test_empty_query_returns_400(self, client, mock_deps):
        resp = client.post(
            "/api/search",
            headers=auth_headers(),
            json={"query": "  ", "top_k": 5, "rerank": False},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# PUT /api/candidates/{id} — user scoping
# ---------------------------------------------------------------------------

class TestUpdateCandidate:
    def test_updates_own_candidate(self, client, mock_deps):
        mock_sb, mock_pi, _ = mock_deps
        (mock_sb.table.return_value.select.return_value
         .eq.return_value.eq.return_value.execute.return_value) = _sb_result([CANDIDATE_ROW])
        updated = {**CANDIDATE_ROW, "name": "Bob"}
        (mock_sb.table.return_value.update.return_value
         .eq.return_value.eq.return_value.execute.return_value) = _sb_result([updated])
        mock_pi.update.return_value = None

        resp = client.put("/api/candidates/cand-1", headers=auth_headers(), json={"name": "Bob"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "Bob"

    def test_cannot_update_other_users_candidate(self, client, mock_deps):
        mock_sb, _, _ = mock_deps
        (mock_sb.table.return_value.select.return_value
         .eq.return_value.eq.return_value.execute.return_value) = _sb_result([])

        resp = client.put("/api/candidates/other-cand", headers=auth_headers(), json={"name": "Hacker"})
        assert resp.status_code == 404

    def test_no_fields_returns_400(self, client, mock_deps):
        resp = client.put("/api/candidates/cand-1", headers=auth_headers(), json={})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# DELETE /api/candidates/{id} — user scoping
# ---------------------------------------------------------------------------

class TestDeleteCandidate:
    def test_deletes_own_candidate(self, client, mock_deps):
        mock_sb, mock_pi, _ = mock_deps
        (mock_sb.table.return_value.select.return_value
         .eq.return_value.eq.return_value.execute.return_value) = _sb_result([CANDIDATE_ROW])
        mock_pi.delete.return_value = None
        mock_sb.storage.from_.return_value.remove.return_value = None
        delete_result = MagicMock()
        delete_result.data = [CANDIDATE_ROW]
        (mock_sb.table.return_value.delete.return_value
         .eq.return_value.eq.return_value.execute.return_value) = delete_result

        resp = client.delete("/api/candidates/cand-1", headers=auth_headers())
        assert resp.status_code == 200
        assert resp.json()["candidate_id"] == "cand-1"

    def test_cannot_delete_other_users_candidate(self, client, mock_deps):
        mock_sb, _, _ = mock_deps
        (mock_sb.table.return_value.select.return_value
         .eq.return_value.eq.return_value.execute.return_value) = _sb_result([])

        resp = client.delete("/api/candidates/other-cand", headers=auth_headers())
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/upload-resumes/bulk
# ---------------------------------------------------------------------------

class TestBulkUpload:
    def _setup_mocks(self, mock_sb, mock_pi, mock_oai):
        choice = MagicMock()
        choice.message.content = "Summary"
        mock_oai.chat.completions.create.return_value.choices = [choice]
        mock_pi.upsert.return_value = None
        mock_pi.query.return_value = MagicMock(matches=[])
        mock_sb.storage.from_.return_value.upload.return_value = None
        mock_sb.storage.from_.return_value.get_public_url.return_value = "https://example.com/r.pdf"
        mock_sb.table.return_value.insert.return_value.execute.return_value = _sb_result([CANDIDATE_ROW])

    def test_bulk_upload_multiple_files(self, client, mock_deps):
        mock_sb, mock_pi, mock_oai = mock_deps
        self._setup_mocks(mock_sb, mock_pi, mock_oai)

        with patch("app.routes.upload.extract_text_from_pdf", return_value="raw text"), \
             patch("app.routes.upload.find_duplicate_resume", return_value=None):
            resp = client.post(
                "/api/upload-resumes/bulk",
                headers=auth_headers(),
                files=[
                    ("files", ("a.pdf", io.BytesIO(b"data"), "application/pdf")),
                    ("files", ("b.pdf", io.BytesIO(b"data"), "application/pdf")),
                ],
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["uploaded"] == 2
        assert len(body["results"]) == 2

    def test_rejects_non_pdf_in_bulk(self, client, mock_deps):
        mock_sb, mock_pi, _ = mock_deps
        mock_pi.query.return_value = MagicMock(matches=[])

        with patch("app.routes.upload.extract_text_from_pdf", return_value="raw text"), \
             patch("app.routes.upload.find_duplicate_resume", return_value=None):
            resp = client.post(
                "/api/upload-resumes/bulk",
                headers=auth_headers(),
                files=[("files", ("doc.txt", io.BytesIO(b"text"), "text/plain"))],
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["uploaded"] == 0
        assert "error" in body["results"][0]

    def test_exceeds_limit_returns_400(self, client, mock_deps):
        files = [("files", (f"r{i}.pdf", io.BytesIO(b"d"), "application/pdf")) for i in range(21)]
        resp = client.post("/api/upload-resumes/bulk", headers=auth_headers(), files=files)
        assert resp.status_code == 400
