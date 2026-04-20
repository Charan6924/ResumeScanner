"""
Tests for the three critical fixes:
  1. Empty-text guard in upload_resume must fire before extract_name_and_email
  2. seed_resumes.py CONFIG must read user_id/folder from env vars
  3. upload_resume must not accept name/email as form parameters
"""
import io
import os
import sys
import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest


FAKE_USER = {"uid": "user-123"}
FAKE_TOKEN = "fake-token"


def auth_headers():
    return {"Authorization": f"Bearer {FAKE_TOKEN}"}


def _sb_result(data):
    r = MagicMock()
    r.data = data
    r.count = len(data)
    return r


CANDIDATE_ROW = {
    "id": "cand-1",
    "name": "Alice Smith",
    "email": "alice@example.com",
    "resume_file_url": "https://example.com/resume.pdf",
    "raw_text": "Alice Smith\nalice@example.com\nPython developer",
    "summary": "Experienced Python developer",
    "vector_id": "vec-1",
    "user_id": "user-123",
}


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
# Fix 1: Empty-text guard fires BEFORE extract_name_and_email
# ---------------------------------------------------------------------------

class TestEmptyTextGuard:
    def test_empty_pdf_returns_400_without_calling_extract(self, client, mock_deps):
        """extract_name_and_email must NOT be called when raw_text is empty."""
        with patch("app.routes.upload.extract_text_from_pdf", return_value=""), \
             patch("app.routes.upload.extract_name_and_email") as mock_extract, \
             patch("app.routes.upload.find_duplicate_resume", return_value=None):
            resp = client.post(
                "/api/upload-resume",
                headers=auth_headers(),
                files={"file": ("resume.pdf", io.BytesIO(b"%PDF empty"), "application/pdf")},
            )
        assert resp.status_code == 400
        assert "text" in resp.json()["detail"].lower()
        mock_extract.assert_not_called()

    def test_whitespace_only_pdf_returns_400(self, client, mock_deps):
        """Whitespace-only text (strip() == '') must be caught by the guard."""
        with patch("app.routes.upload.extract_text_from_pdf", return_value="   \n\t  "), \
             patch("app.routes.upload.extract_name_and_email") as mock_extract, \
             patch("app.routes.upload.find_duplicate_resume", return_value=None):
            resp = client.post(
                "/api/upload-resume",
                headers=auth_headers(),
                files={"file": ("resume.pdf", io.BytesIO(b"%PDF ws"), "application/pdf")},
            )
        assert resp.status_code == 400
        mock_extract.assert_not_called()

    def test_none_raw_text_returns_400_not_500(self, client, mock_deps):
        """If extract_text_from_pdf returns None the route must 400, not 500."""
        with patch("app.routes.upload.extract_text_from_pdf", return_value=None), \
             patch("app.routes.upload.extract_name_and_email") as mock_extract, \
             patch("app.routes.upload.find_duplicate_resume", return_value=None):
            resp = client.post(
                "/api/upload-resume",
                headers=auth_headers(),
                files={"file": ("resume.pdf", io.BytesIO(b"%PDF none"), "application/pdf")},
            )
        assert resp.status_code == 400
        mock_extract.assert_not_called()

    def test_valid_text_does_call_extract(self, client, mock_deps):
        """Sanity check: extract_name_and_email IS called when text is valid."""
        mock_sb, mock_pi, mock_oai = mock_deps
        choice = MagicMock()
        choice.message.content = "Summary"
        mock_oai.chat.completions.create.return_value.choices = [choice]
        mock_pi.query.return_value = MagicMock(matches=[])
        mock_sb.storage.from_.return_value.upload.return_value = None
        mock_sb.storage.from_.return_value.get_public_url.return_value = "https://x.com/r.pdf"
        mock_sb.table.return_value.insert.return_value.execute.return_value = _sb_result([CANDIDATE_ROW])

        with patch("app.routes.upload.extract_text_from_pdf", return_value="Alice Smith\nalice@example.com\nPython dev"), \
             patch("app.routes.upload.extract_name_and_email", return_value=("Alice Smith", "alice@example.com")) as mock_extract, \
             patch("app.routes.upload.find_duplicate_resume", return_value=None):
            resp = client.post(
                "/api/upload-resume",
                headers=auth_headers(),
                files={"file": ("resume.pdf", io.BytesIO(b"data"), "application/pdf")},
            )
        assert resp.status_code == 200
        mock_extract.assert_called_once()


# ---------------------------------------------------------------------------
# Fix 2: seed_resumes CONFIG reads from env vars
# ---------------------------------------------------------------------------

class TestSeedConfig:
    def _reload_seed_module(self, extra_env: dict):
        for key in list(sys.modules):
            if "seed_resumes" in key:
                del sys.modules[key]

        stubs = {
            "app.config": MagicMock(),
            "app.services.pdf": MagicMock(),
            "app.services.search": MagicMock(),
            "embeddings": MagicMock(),
            "dotenv": MagicMock(),
        }
        stubs["dotenv"].load_dotenv = MagicMock()

        with patch.dict(sys.modules, stubs), \
             patch.dict(os.environ, extra_env, clear=False):
            import seed_resumes
            importlib.reload(seed_resumes)
            return seed_resumes

    def test_user_id_read_from_env(self):
        mod = self._reload_seed_module({
            "SEED_USER_ID": "env-uid-abc123",
            "SEED_FOLDER": "/tmp/resumes",
        })
        assert mod.CONFIG.user_id == "env-uid-abc123"

    def test_folder_read_from_env(self):
        mod = self._reload_seed_module({
            "SEED_USER_ID": "uid-xyz",
            "SEED_FOLDER": "/tmp/my_resumes",
        })
        assert mod.CONFIG.folder == Path("/tmp/my_resumes")

    def test_hardcoded_uid_not_present(self):
        """The hardcoded UID must not appear anywhere in the source."""
        source = Path("/Users/charan/Documents/ResumeProject/seed_resumes.py").read_text()
        assert "WXfvb86hWPMqISd1IhYBFX4sQxf1" not in source

    def test_hardcoded_desktop_path_not_present(self):
        """The hardcoded desktop path must not appear in the source."""
        source = Path("/Users/charan/Documents/ResumeProject/seed_resumes.py").read_text()
        assert "/Users/charan/Desktop/Resumes" not in source

    def test_missing_seed_user_id_leaves_user_id_empty(self):
        env = {k: v for k, v in os.environ.items() if k != "SEED_USER_ID"}
        mod = self._reload_seed_module({"SEED_FOLDER": "/tmp"})
        assert mod.CONFIG.user_id == "" or mod.CONFIG.user_id is None

    def test_env_example_contains_seed_vars(self):
        env_example = Path("/Users/charan/Documents/ResumeProject/.env.example").read_text()
        assert "SEED_USER_ID" in env_example
        assert "SEED_FOLDER" in env_example


# ---------------------------------------------------------------------------
# Fix 3: Dead name/email parameters removed from upload_resume signature
# ---------------------------------------------------------------------------

class TestDeadParameters:
    def test_upload_resume_signature_has_no_name_param(self):
        import inspect
        from app.routes.upload import upload_resume
        sig = inspect.signature(upload_resume)
        assert "name" not in sig.parameters, \
            "upload_resume must not accept a 'name' form parameter"

    def test_upload_resume_signature_has_no_email_param(self):
        import inspect
        from app.routes.upload import upload_resume
        sig = inspect.signature(upload_resume)
        assert "email" not in sig.parameters, \
            "upload_resume must not accept an 'email' form parameter"

    def test_name_email_derived_from_pdf_not_form(self, client, mock_deps):
        """Values inserted into DB must come from extract_name_and_email, not form data."""
        mock_sb, mock_pi, mock_oai = mock_deps
        choice = MagicMock()
        choice.message.content = "Summary"
        mock_oai.chat.completions.create.return_value.choices = [choice]
        mock_pi.query.return_value = MagicMock(matches=[])
        mock_sb.storage.from_.return_value.upload.return_value = None
        mock_sb.storage.from_.return_value.get_public_url.return_value = "https://x.com/r.pdf"
        mock_sb.table.return_value.insert.return_value.execute.return_value = _sb_result([CANDIDATE_ROW])

        with patch("app.routes.upload.extract_text_from_pdf", return_value="Bob Jones\nbob@example.com\nEngineer"), \
             patch("app.routes.upload.extract_name_and_email", return_value=("Bob Jones", "bob@example.com")), \
             patch("app.routes.upload.find_duplicate_resume", return_value=None):
            resp = client.post(
                "/api/upload-resume",
                headers=auth_headers(),
                files={"file": ("resume.pdf", io.BytesIO(b"data"), "application/pdf")},
            )
        assert resp.status_code == 200
        insert_call = mock_sb.table.return_value.insert.call_args
        inserted = insert_call[0][0]
        assert inserted["name"] == "Bob Jones"
        assert inserted["email"] == "bob@example.com"
