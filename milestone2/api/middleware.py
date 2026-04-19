import time
from fastapi import Request, HTTPException, Security
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from milestone2.config import API_SECRET_KEY
from milestone2.logger import logger_api

# Define that we expect "X-API-Key" in the header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key_header: str = Security(api_key_header)):
    # If NO API key is set in .env, we assume testing environment and allow access
    if not API_SECRET_KEY:
        return True
        
    if api_key_header == API_SECRET_KEY:
        return True
        
    raise HTTPException(
        status_code=403, 
        detail="Could not validate credentials. Invalid or missing X-API-Key header."
    )


class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        try:
            response = await call_next(request)
        except Exception as e:
            logger_api.error(f"Unhandled Exception caught by middleware: {e}")
            raise
            
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
