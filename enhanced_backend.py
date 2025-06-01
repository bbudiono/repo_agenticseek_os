#!/usr/bin/env python3
"""
Enhanced AgenticSeek Backend Test Server
This provides all the missing endpoints that the React frontend expects
"""

import os
import sys
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import redis
import asyncio
from datetime import datetime
from pathlib import Path

# Add the sources directory to the path
sys.path.append('sources')
from cascading_provider import CascadingProvider

app = FastAPI(title="AgenticSeek Backend Test", version="1.0.0")

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

# Initialize cascading provider system
llm_provider = CascadingProvider()
llm_provider.setup_shared_models()

@app.get("/")
async def root():
    return {"message": "AgenticSeek Backend Test Server Running!", "status": "ok"}

@app.get("/health")
async def health_check():
    status = {"backend": "running", "redis": "unknown", "llm_providers": []}
    
    # Test Redis connection
    try:
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        status["redis"] = "connected"
    except Exception as e:
        status["redis"] = f"error: {str(e)}"
    
    # Test LLM providers in parallel
    try:
        provider_statuses = await llm_provider.check_all_providers_parallel()
        for provider_status in provider_statuses:
            config = provider_status['config']
            status["llm_providers"].append({
                "name": config['description'],
                "type": config['name'],
                "model": config['model'],
                "is_local": config['is_local'],
                "status": "available" if provider_status['available'] else "unavailable"
            })
    except Exception as e:
        status["llm_providers"].append({
            "error": f"Failed to check providers: {str(e)}"
        })
    
    # Add current active provider info
    status["active_provider"] = llm_provider.get_current_provider_info()
    
    return status

@app.post("/query")
async def handle_query(request: Request):
    """Handle chat queries from the frontend"""
    global latest_response
    
    try:
        data = await request.json()
        user_message = data.get("message", "")
        session_id = data.get("session_id", "default")
        
        print(f"📝 Received query: {user_message}")
        
        # Prepare message history for LLM
        history = [{"role": "user", "content": user_message}]
        
        # Get response from cascading provider
        try:
            llm_response = llm_provider.respond_with_fallback(history, verbose=False)
            provider_info = llm_provider.get_current_provider_info()
            
            # Create response matching AgenticSeek frontend expectations
            response = {
                "answer": llm_response,
                "agent_name": f"AgenticSeek_AI ({provider_info['description']})",
                "blocks": {
                    "0": {
                        "tool_type": "llm_response",
                        "block": f"Used {provider_info['description']} with model {provider_info['model']}",
                        "feedback": f"Response generated successfully via {provider_info['name']}",
                        "success": True
                    }
                },
                "done": True,
                "status": "completed",
                "uid": session_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ LLM Error: {str(e)}")
            # Fallback to test response if all providers fail
            response = {
                "answer": f"I'm sorry, I'm having trouble connecting to the AI models right now. Your message was: '{user_message}'. Please check the provider configuration.",
                "agent_name": "AgenticSeek_AI (Error Mode)",
                "blocks": {
                    "0": {
                        "tool_type": "error_response",
                        "block": f"Error: {str(e)}",
                        "feedback": "All LLM providers failed",
                        "success": False
                    }
                },
                "done": True,
                "status": "error",
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
    file_path = screenshots_dir / filename
    
    if file_path.exists():
        return FileResponse(file_path)
    else:
        # Return a placeholder response
        return JSONResponse(
            status_code=404,
            content={"error": f"Screenshot {filename} not found"}
        )

@app.get("/config/providers")
async def get_providers():
    """Get all configured providers"""
    try:
        provider_statuses = await llm_provider.check_all_providers_parallel()
        providers = []
        for provider_status in provider_statuses:
            config = provider_status['config']
            providers.append({
                "name": config['name'],
                "display_name": config['description'],
                "model": config['model'],
                "is_local": config['is_local'],
                "is_available": provider_status['available'],
                "status": "active" if provider_status['available'] else "inactive"
            })
        
        return {
            "success": True,
            "providers": providers
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "providers": []
        }

@app.get("/config/api-keys")
async def get_api_keys():
    """Get API key configuration status"""
    return {
        "success": True,
        "api_keys": [
            {"provider": "anthropic", "is_set": True, "key_name": "ANTHROPIC_API_KEY"},
            {"provider": "openai", "is_set": False, "key_name": "OPENAI_API_KEY"},
            {"provider": "deepseek", "is_set": False, "key_name": "DEEPSEEK_API_KEY"},
            {"provider": "google", "is_set": False, "key_name": "GOOGLE_API_KEY"}
        ]
    }

@app.get("/config/models/{provider}")
async def get_models_for_provider(provider: str):
    """Get available models for a specific provider"""
    
    # Mock model lists for different providers - in a real implementation, 
    # these would come from API calls to the actual providers
    provider_models = {
        "anthropic": [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022", 
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ],
        "openai": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo"
        ],
        "deepseek": [
            "deepseek-chat",
            "deepseek-coder"
        ],
        "google": [
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-pro"
        ],
        "lm_studio": [
            "any"  # LM Studio uses "any" for auto-selection
        ],
        "ollama": [
            "deepseek-r1:14b",
            "llama3.2:3b",
            "qwen2.5:7b",
            "codellama:7b"
        ]
    }
    
    models = provider_models.get(provider, [])
    
    return {
        "success": True,
        "provider": provider,
        "models": models
    }

@app.get("/models/catalog")
async def get_model_catalog():
    """Get the complete model catalog"""
    try:
        catalog = {}
        providers = ["anthropic", "openai", "deepseek", "google", "lm_studio", "ollama"]
        
        for provider in providers:
            # Get models for each provider
            models_response = await get_models_for_provider(provider)
            if models_response["success"]:
                catalog[provider] = models_response["models"]
        
        return {
            "success": True,
            "catalog": catalog
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "catalog": {}
        }

@app.get("/models/installed")
async def get_installed_models():
    """Get locally installed models"""
    return {
        "success": True,
        "models": [
            {"name": "deepseek-r1:14b", "provider": "ollama", "size": "8.1GB"},
            {"name": "llama3.2:3b", "provider": "ollama", "size": "2.0GB"}
        ]
    }

@app.post("/config/provider")
async def update_provider(request: Request):
    """Update the active provider configuration"""
    try:
        data = await request.json()
        provider_name = data.get("provider_name")
        model = data.get("model") 
        server_address = data.get("server_address")
        role = data.get("role", "main")
        
        # In a real implementation, this would update the actual provider configuration
        print(f"🔄 Provider update request: {provider_name}/{model} at {server_address}")
        
        return {
            "success": True,
            "message": f"Provider updated to {provider_name}/{model}",
            "provider": provider_name,
            "model": model
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/models/download")
async def download_model(request: Request):
    """Start downloading a model"""
    try:
        data = await request.json()
        model_name = data.get("model_name")
        provider = data.get("provider")
        download_url = data.get("download_url")
        target_path = data.get("target_path")
        
        # Mock download start - in real implementation this would start actual download
        print(f"📥 Starting download: {model_name} from {provider}")
        
        return {
            "success": True,
            "message": f"Started download of {model_name}",
            "model_name": model_name,
            "provider": provider
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/test-chat")
async def test_chat():
    """Test endpoint to verify the backend is working"""
    return {
        "response": "Hello from AgenticSeek backend! 🤖",
        "agent": "test",
        "timestamp": "2025-05-31T11:00:00Z"
    }

if __name__ == "__main__":
    print("🚀 Starting Enhanced AgenticSeek Backend Test Server...")
    print("📍 Backend URL: http://localhost:8000")
    print("🔗 Health Check: http://localhost:8000/health")
    print("💬 Chat Endpoint: POST http://localhost:8000/query")
    print("📊 Latest Answer: GET http://localhost:8000/latest_answer")
    print("📚 API Docs: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")