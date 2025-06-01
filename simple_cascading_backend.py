#!/usr/bin/env python3
"""
Simple Cascading Backend for AgenticSeek
Implements basic cascading without heavy dependencies
"""

import os
import sys
import httpx
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import redis
from datetime import datetime
from pathlib import Path

# Import model manager and config manager
from model_manager import model_manager
from config_manager import config_manager

app = FastAPI(title="AgenticSeek Simple Cascading Backend", version="1.0.0")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage for chat sessions
chat_sessions = {}
latest_response = None

# Simple provider configurations
PROVIDERS = [
    {
        "name": "LM Studio",
        "type": "lm_studio", 
        "url": "http://192.168.1.37:1234",
        "test_endpoint": "/v1/models",
        "is_local": True
    },
    {
        "name": "Ollama",
        "type": "ollama",
        "url": "http://127.0.0.1:11434", 
        "test_endpoint": "/api/tags",
        "is_local": True
    },
    {
        "name": "Test Provider",
        "type": "test",
        "url": "localhost",
        "test_endpoint": "",
        "is_local": True
    }
]

async def test_provider_async(provider):
    """Test if a provider is available"""
    if provider["type"] == "test":
        return True
        
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{provider['url']}{provider['test_endpoint']}", timeout=3.0)
            return response.status_code == 200
    except:
        return False

async def check_all_providers_parallel():
    """Check all providers in parallel"""
    tasks = []
    for provider in PROVIDERS:
        task = asyncio.create_task(test_provider_async(provider))
        tasks.append((provider, task))
    
    results = []
    for provider, task in tasks:
        try:
            is_available = await task
            results.append({
                "provider": provider,
                "available": is_available
            })
        except Exception as e:
            results.append({
                "provider": provider,
                "available": False,
                "error": str(e)
            })
    
    return results

async def get_response_from_provider(provider, message):
    """Get response from a specific provider"""
    if provider["type"] == "lm_studio":
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{provider['url']}/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": message}],
                        "temperature": 0.7,
                        "max_tokens": 1000
                    },
                    timeout=30.0
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            raise Exception(f"LM Studio error: {str(e)}")
    
    elif provider["type"] == "ollama":
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{provider['url']}/api/generate",
                    json={
                        "model": "deepseek-r1:14b",
                        "prompt": message,
                        "stream": False
                    },
                    timeout=30.0
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get("response", "")
        except Exception as e:
            raise Exception(f"Ollama error: {str(e)}")
    
    elif provider["type"] == "test":
        await asyncio.sleep(0.5)  # Simulate processing
        return f"Test response from {provider['name']}: I received your message '{message}'"
    
    raise Exception(f"Unknown provider type: {provider['type']}")

@app.get("/")
async def root():
    return {"message": "AgenticSeek Simple Cascading Backend Running!", "status": "ok"}

@app.get("/health")
async def health_check():
    status = {"backend": "running", "redis": "unknown", "providers": []}
    
    # Test Redis connection
    try:
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        status["redis"] = "connected"
    except Exception as e:
        status["redis"] = f"error: {str(e)}"
    
    # Test all providers in parallel
    provider_results = await check_all_providers_parallel()
    
    for result in provider_results:
        provider = result["provider"]
        status["providers"].append({
            "name": provider["name"],
            "type": provider["type"],
            "url": provider["url"],
            "is_local": provider["is_local"],
            "status": "available" if result["available"] else "unavailable",
            "error": result.get("error")
        })
    
    return status

@app.post("/query")
async def handle_query(request: Request):
    """Handle chat queries with cascading fallback"""
    global latest_response
    
    try:
        data = await request.json()
        user_message = data.get("message", "")
        session_id = data.get("session_id", "default")
        
        print(f"📝 Received query: {user_message}")
        
        # Try providers in order
        provider_results = await check_all_providers_parallel()
        
        # Sort by availability and try each provider
        available_providers = [r for r in provider_results if r["available"]]
        
        llm_response = None
        used_provider = None
        
        for result in available_providers:
            provider = result["provider"]
            try:
                print(f"🔄 Trying {provider['name']}...")
                llm_response = await get_response_from_provider(provider, user_message)
                used_provider = provider
                print(f"✅ Success with {provider['name']}")
                break
            except Exception as e:
                print(f"❌ {provider['name']} failed: {str(e)}")
                continue
        
        if not llm_response:
            llm_response = f"I'm sorry, I'm having trouble connecting to the AI models right now. Your message was: '{user_message}'"
            used_provider = {"name": "Error Handler", "type": "error"}
        
        # Create response matching AgenticSeek frontend expectations
        response = {
            "answer": llm_response,
            "agent_name": f"AgenticSeek_AI ({used_provider['name']})",
            "blocks": {
                "0": {
                    "tool_type": "llm_response",
                    "block": f"Used {used_provider['name']} provider",
                    "feedback": f"Response generated successfully via {used_provider['type']}",
                    "success": True
                }
            },
            "done": True,
            "status": "completed",
            "uid": session_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store response
        latest_response = response
        
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
        chat_sessions[session_id].append({
            "user": user_message,
            "assistant": response["answer"],
            "timestamp": response["timestamp"]
        })
        
        print(f"✅ Generated response: {response['answer'][:50]}...")
        return response
        
    except Exception as e:
        print(f"❌ Error processing query: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing query: {str(e)}"}
        )

@app.get("/latest_answer")
async def get_latest_answer():
    """Get the latest response"""
    global latest_response
    
    if latest_response:
        return latest_response
    else:
        return {
            "answer": "No recent messages yet. Try sending a message first!",
            "agent_name": "system",
            "blocks": {
                "0": {
                    "tool_type": "system_message",
                    "block": "Waiting for user input",
                    "feedback": "System ready",
                    "success": True
                }
            },
            "done": True,
            "status": "waiting",
            "uid": "system",
            "timestamp": datetime.now().isoformat()
        }

# Create screenshots directory if it doesn't exist
screenshots_dir = Path("screenshots")
screenshots_dir.mkdir(exist_ok=True)

@app.get("/screenshots/{filename}")
async def get_screenshot(filename: str):
    """Serve screenshot files"""
    return JSONResponse(
        status_code=404,
        content={"error": f"Screenshot {filename} not found"}
    )

# Model Management API Endpoints

@app.get("/models/catalog")
async def get_models_catalog():
    """Get catalog of available models for all providers"""
    try:
        catalog = await model_manager.get_available_models_catalog()
        return {
            "success": True,
            "catalog": catalog,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/models/installed")
async def get_installed_models():
    """Get list of installed models"""
    try:
        models = await model_manager.get_installed_models()
        return {
            "success": True,
            "models": [model.to_dict() for model in models],
            "count": len(models),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/models/download")
async def download_model(request: Request):
    """Download a model"""
    try:
        data = await request.json()
        model_name = data.get("model_name")
        provider = data.get("provider")
        
        if not model_name or not provider:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "model_name and provider are required"}
            )
        
        print(f"🔽 Starting download: {model_name} ({provider})")
        
        # Start download in background
        success = await model_manager.download_model(model_name, provider)
        
        return {
            "success": success,
            "message": f"Download {'started' if success else 'failed'} for {model_name}",
            "model_name": model_name,
            "provider": provider,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.delete("/models/{provider}/{model_name}")
async def delete_model(provider: str, model_name: str):
    """Delete a model"""
    try:
        print(f"🗑️ Deleting model: {model_name} ({provider})")
        
        success = await model_manager.delete_model(model_name, provider)
        
        return {
            "success": success,
            "message": f"Model {'deleted' if success else 'deletion failed'}: {model_name}",
            "model_name": model_name,
            "provider": provider,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/models/storage")
async def get_storage_info():
    """Get storage information"""
    try:
        storage_info = model_manager.get_storage_info()
        return {
            "success": True,
            "storage": storage_info,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

# Configuration Management API Endpoints

@app.get("/config/providers")
async def get_provider_configs():
    """Get all provider configurations"""
    try:
        providers = config_manager.get_provider_configs()
        return {
            "success": True,
            "providers": [p.to_dict() for p in providers],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/config/current")
async def get_current_config():
    """Get current configuration"""
    try:
        config = config_manager.get_current_config()
        return {
            "success": True,
            "config": config,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/config/api-keys")
async def get_api_keys_status():
    """Get API keys status"""
    try:
        api_keys = config_manager.get_api_keys_status()
        return {
            "success": True,
            "api_keys": [key.to_dict() for key in api_keys],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/config/api-key")
async def set_api_key(request: Request):
    """Set API key for a provider"""
    try:
        data = await request.json()
        provider = data.get("provider")
        api_key = data.get("api_key")
        
        if not provider or not api_key:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "provider and api_key are required"}
            )
        
        success = config_manager.set_api_key(provider, api_key)
        
        return {
            "success": success,
            "message": f"API key {'set' if success else 'failed to set'} for {provider}",
            "provider": provider,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/config/provider")
async def update_provider_config(request: Request):
    """Update provider configuration"""
    try:
        data = await request.json()
        provider_role = data.get("role")  # "main", "fallback_local", "fallback_api"
        provider_name = data.get("provider_name")
        model = data.get("model")
        server_address = data.get("server_address")
        
        if not all([provider_role, provider_name, model, server_address]):
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "role, provider_name, model, and server_address are required"}
            )
        
        success = config_manager.update_provider_config(provider_role, provider_name, model, server_address)
        
        return {
            "success": success,
            "message": f"Provider configuration {'updated' if success else 'failed to update'}",
            "provider_role": provider_role,
            "provider_name": provider_name,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/config/models/{provider}")
async def get_available_models(provider: str):
    """Get available models for a provider"""
    try:
        models = config_manager.get_available_models_for_provider(provider)
        return {
            "success": True,
            "provider": provider,
            "models": models,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

if __name__ == "__main__":
    print("🚀 Starting AgenticSeek Simple Cascading Backend...")
    print("📍 Backend URL: http://localhost:8000")
    print("🔗 Health Check: http://localhost:8000/health")
    print("💬 Chat Endpoint: POST http://localhost:8000/query")
    print("📊 Latest Answer: GET http://localhost:8000/latest_answer")
    print("📚 API Docs: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")