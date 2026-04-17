import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from supabase import create_client
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()

supabase = create_client(str(os.getenv("SUPABASE_URL")), str(os.getenv("SUPABASE_KEY")))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("text022026")
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

limiter = Limiter(key_func=get_remote_address)

__all__ = [
    "supabase",
    "pinecone_index",
    "openai_client",
    "limiter",
    "_rate_limit_exceeded_handler",
    "RateLimitExceeded",
]
