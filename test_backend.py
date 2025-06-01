#!/usr/bin/env python3
"""
Minimal AgenticSeek Backend Test Server
This tests if the basic FastAPI setup works before loading the full backend
"""

import os
import sys
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import redis

app = FastAPI(title="AgenticSeek Backend Test", version="1.0.0")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "AgenticSeek Backend Test Server Running!", "status": "ok"}

@app.get("/health")
async def health_check():
    status = {"backend": "running", "redis": "unknown", "lm_studio": "unknown"}
    
    # Test Redis connection
    try:
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        status["redis"] = "connected"
    except Exception as e:
        status["redis"] = f"error: {str(e)}"
    
    # Test LM Studio connection
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get("http://192.168.1.37:1234/v1/models", timeout=5.0)
            if response.status_code == 200:
                status["lm_studio"] = "connected"
            else:
                status["lm_studio"] = f"http_error: {response.status_code}"
    except Exception as e:
        status["lm_studio"] = f"error: {str(e)}"
    
    return status

@app.get("/test-chat")
async def test_chat():
    """Test endpoint to verify the backend is working"""
    return {
        "response": "Hello from AgenticSeek backend! 🤖",
        "agent": "test",
        "timestamp": "2025-05-31T11:00:00Z"
    }

if __name__ == "__main__":
    print("🚀 Starting AgenticSeek Backend Test Server...")
    print("📍 Backend URL: http://localhost:8000")
    print("🔗 Health Check: http://localhost:8000/health")
    print("💬 Test Chat: http://localhost:8000/test-chat")
    print("📚 API Docs: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")