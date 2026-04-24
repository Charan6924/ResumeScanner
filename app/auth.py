import os
import json
import firebase_admin
from firebase_admin import auth, credentials
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

_json_str = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
if _json_str:
    firebase_cred = credentials.Certificate(json.loads(_json_str))
else:
    firebase_cred = credentials.Certificate(os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH"))
firebase_admin.initialize_app(firebase_cred)

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        decoded = auth.verify_id_token(credentials.credentials)
        return decoded
    except Exception as e:
        print(f"Token verification failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")
