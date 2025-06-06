#!/usr/bin/env python3
"""
Debug API startup issues - find the real problem
"""

import sys
import os
sys.path.append('.')

try:
    print("🔍 Testing API import...")
    from api import api
    print("✅ API imports successfully")
    
    print("🔍 Testing minimal FastAPI...")
    from fastapi import FastAPI
    test_app = FastAPI()
    
    @test_app.get("/test")
    def test_endpoint():
        return {"status": "test works"}
    
    print("✅ Minimal FastAPI works")
    
    print("🔍 Testing API routes...")
    for route in api.routes:
        print(f"  Route: {route}")
    
    print("🔍 Starting API with detailed logging...")
    import uvicorn
    uvicorn.run(api, host="127.0.0.1", port=8003, log_level="debug")
    
except Exception as e:
    print(f"❌ API debug failed: {e}")
    import traceback
    traceback.print_exc()