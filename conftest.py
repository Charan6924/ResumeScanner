"""
Stub out external SDK modules before main.py is imported so tests work
without real credentials and without the system Python having those packages.
"""
import sys
from unittest.mock import MagicMock

_STUBS = [
    "firebase_admin",
    "firebase_admin.auth",
    "firebase_admin.credentials",
    "supabase",
    "pinecone",
    "openai",
    "PyPDF2",
    "embeddings",
    "slowapi",
    "slowapi.util",
    "slowapi.errors",
]

for _mod in _STUBS:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import firebase_admin
firebase_admin.initialize_app = MagicMock()
firebase_admin.credentials = MagicMock()
firebase_admin.credentials.Certificate = MagicMock(return_value=MagicMock())
firebase_admin.auth = MagicMock()

import supabase as _sb
_sb.create_client = MagicMock(return_value=MagicMock())

import pinecone as _pc
_pc.Pinecone = MagicMock(return_value=MagicMock())

import openai as _oai
_oai.OpenAI = MagicMock(return_value=MagicMock())

# Replace the slowapi mock with a real exception class so FastAPI can register it
class _RateLimitExceeded(Exception):
    pass

sys.modules["slowapi.errors"].RateLimitExceeded = _RateLimitExceeded

# Make limiter.limit() act as a no-op decorator so route functions aren't replaced
def _identity_decorator(*args, **kwargs):
    def decorator(fn):
        return fn
    return decorator

_limiter_instance = MagicMock()
_limiter_instance.limit = _identity_decorator

_limiter_class = MagicMock(return_value=_limiter_instance)

sys.modules["slowapi"].Limiter = _limiter_class
sys.modules["slowapi"]._rate_limit_exceeded_handler = MagicMock()
sys.modules["slowapi.util"].get_remote_address = MagicMock()
