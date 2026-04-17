import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import limiter, _rate_limit_exceeded_handler, RateLimitExceeded
from app.routes import candidates, upload, search

_origins_env = os.getenv("ALLOWED_ORIGINS", "")
allowed_origins = [o.strip() for o in _origins_env.split(",") if o.strip()] or ["*"]

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(candidates.router)
app.include_router(upload.router)
app.include_router(search.router)
