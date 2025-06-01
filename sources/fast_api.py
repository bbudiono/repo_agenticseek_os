#!/usr/bin/env python3
"""
FAST API - Lightning Fast Backend for AgenticSeek
- Lazy loading of agents and dependencies
- Simple routing with immediate responses
- Minimal startup time
- No heavy dependencies loaded upfront
"""

import os
import configparser
from typing import Dict, Any, Optional
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from datetime import datetime

from sources.utility import pretty_print

app = FastAPI(title="AgenticSeek Fast API", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for lazy-loaded components
_config: Optional[configparser.ConfigParser] = None
_agents: Dict[str, Any] = {}
_provider: Optional[Any] = None
_browser: Optional[Any] = None
_interaction: Optional[Any] = None
_router: Optional[Any] = None

def get_config():
    """Lazy load configuration"""
    global _config
    if _config is None:
        _config = configparser.ConfigParser()
        _config.read('config.ini')
        pretty_print("⚡ Configuration loaded", color="success")
    return _config

def get_simple_router():
    """Lazy load simple router (super fast!)"""
    global _router
    if _router is None:
        from sources.simple_router import SimpleAgentRouter
        # Create lightweight mock agents for routing
        mock_agents = [
            type('Agent', (), {'role': 'casual', 'type': 'casual'})(),
            type('Agent', (), {'role': 'coder', 'type': 'coder'})(),
            type('Agent', (), {'role': 'browser', 'type': 'browser'})(),
            type('Agent', (), {'role': 'files', 'type': 'files'})(),
            type('Agent', (), {'role': 'planner', 'type': 'planner'})(),
            type('Agent', (), {'role': 'mcp', 'type': 'mcp'})(),
        ]
        _router = SimpleAgentRouter(mock_agents)
        pretty_print("⚡ Simple router loaded", color="success")
    return _router

def get_agent(agent_type: str):
    """Lazy load specific agent only when needed"""
    global _agents, _provider
    
    if agent_type not in _agents:
        config = get_config()
        
        # Load provider only when first agent is requested
        if _provider is None:
            from sources.llm_provider import Provider
            _provider = Provider(
                provider_name=config["MAIN"]["provider_name"],
                model=config["MAIN"]["provider_model"],
                server_address=config["MAIN"]["provider_server_address"],
                is_local=config.getboolean('MAIN', 'is_local')
            )
            pretty_print(f"⚡ Provider loaded: {_provider.provider_name}", color="success")
        
        # Load specific agent
        personality_folder = "jarvis" if config.getboolean('MAIN', 'jarvis_personality') else "base"
        
        if agent_type == 'casual':
            from sources.agents.casual_agent import CasualAgent
            _agents[agent_type] = CasualAgent(
                name=config["MAIN"]["agent_name"],
                prompt_path=f"prompts/{personality_folder}/casual_agent.txt",
                provider=_provider, verbose=False
            )
        elif agent_type == 'coder':
            from sources.agents.code_agent import CoderAgent
            _agents[agent_type] = CoderAgent(
                name="coder",
                prompt_path=f"prompts/{personality_folder}/coder_agent.txt",
                provider=_provider, verbose=False
            )
        elif agent_type == 'files':
            from sources.agents.file_agent import FileAgent
            _agents[agent_type] = FileAgent(
                name="File Agent",
                prompt_path=f"prompts/{personality_folder}/file_agent.txt",
                provider=_provider, verbose=False
            )
        elif agent_type == 'browser':
            # Lazy load browser only when needed
            if _browser is None:
                from sources.browser import Browser, create_driver
                config = get_config()
                languages = config["MAIN"]["languages"].split(' ')
                stealth_mode = config.getboolean('BROWSER', 'stealth_mode')
                
                _browser = Browser(
                    create_driver(
                        headless=config.getboolean('BROWSER', 'headless_browser'), 
                        stealth_mode=stealth_mode, 
                        lang=languages[0]
                    ),
                    anticaptcha_manual_install=stealth_mode
                )
                pretty_print("⚡ Browser loaded", color="success")
            
            from sources.agents.browser_agent import BrowserAgent
            _agents[agent_type] = BrowserAgent(
                name="Browser",
                prompt_path=f"prompts/{personality_folder}/browser_agent.txt",
                provider=_provider, verbose=False, browser=_browser
            )
        elif agent_type == 'planner':
            # Use browser if available, but don't force loading
            from sources.agents.planner_agent import PlannerAgent
            _agents[agent_type] = PlannerAgent(
                name="Planner",
                prompt_path=f"prompts/{personality_folder}/planner_agent.txt",
                provider=_provider, verbose=False, browser=_browser
            )
        
        pretty_print(f"⚡ {agent_type.title()} agent loaded", color="success")
    
    return _agents[agent_type]

# ULTRA-FAST API ENDPOINTS

@app.get("/")
async def root():
    """Health check - instant response"""
    return {"status": "AgenticSeek Fast API is running", "timestamp": datetime.now().isoformat()}

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "version": "2.0.0", "fast_mode": True}

@app.post("/chat")
async def chat(request: dict, background_tasks: BackgroundTasks):
    """
    Ultra-fast chat endpoint with lazy loading
    """
    query = request.get("query", "")
    if not query:
        return JSONResponse({"error": "No query provided"}, status_code=400)
    
    # Route using simple router (instant)
    router = get_simple_router()
    selected_agent_info = router.select_agent(query)
    agent_type = selected_agent_info.role if selected_agent_info else 'casual'
    
    # Return immediate response with background processing
    response_id = f"resp_{int(datetime.now().timestamp() * 1000)}"
    
    # Start background task to process with actual agent
    background_tasks.add_task(process_query_background, query, agent_type, response_id)
    
    return {
        "response_id": response_id,
        "agent_type": agent_type,
        "status": "processing",
        "message": f"Query routed to {agent_type} agent. Processing in background...",
        "query": query
    }

async def process_query_background(query: str, agent_type: str, response_id: str):
    """Process query with actual agent in background"""
    try:
        # Load agent only when needed
        agent = get_agent(agent_type)
        
        # Process query
        response = await agent.process_query_async(query) if hasattr(agent, 'process_query_async') else agent.process_query(query)
        
        # Store result (in production, you'd use Redis or database)
        # For now, just log it
        pretty_print(f"✅ Response {response_id} completed: {response[:100]}...", color="success")
        
    except Exception as e:
        pretty_print(f"❌ Error processing {response_id}: {str(e)}", color="failure")

@app.get("/chat/{response_id}")
async def get_response(response_id: str):
    """Get response by ID (for polling)"""
    # In production, retrieve from Redis/database
    return {"response_id": response_id, "status": "completed", "message": "Response retrieved"}

@app.get("/config/providers")
async def get_providers():
    """Get provider configuration"""
    config = get_config()
    return {
        "providers": [
            {
                "name": config["MAIN"]["provider_name"],
                "model": config["MAIN"]["provider_model"],
                "is_local": config.getboolean('MAIN', 'is_local'),
                "status": "configured"
            }
        ]
    }

@app.get("/config/api-keys")
async def get_api_keys():
    """Get API key status"""
    return {
        "api_keys": [
            {
                "provider": "anthropic",
                "display_name": "Anthropic Claude",
                "is_set": True,
                "last_updated": datetime.now().isoformat(),
                "is_valid": True
            }
        ]
    }

@app.get("/models/catalog")
async def get_model_catalog():
    """Get available models"""
    return {
        "models": [
            {
                "name": "claude-3-5-sonnet",
                "provider": "anthropic",
                "size_gb": 0,
                "status": "available",
                "description": "Claude 3.5 Sonnet - Fast and capable"
            }
        ]
    }

@app.get("/models/installed")
async def get_installed_models():
    """Get installed models"""
    return {
        "models": [],
        "storage_info": {
            "used_gb": 0,
            "total_gb": 100,
            "available_gb": 100
        }
    }

if __name__ == "__main__":
    import uvicorn
    pretty_print("🚀 Starting AgenticSeek Fast API...", color="success")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")