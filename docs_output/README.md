# AgenticSeek Documentation

## Overview
Enterprise AI platform with 15 integrated systems.

## Quick Start
```bash
# Health check
curl http://localhost:8000/health

# Basic query  
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello", "provider": "openai"}'
```

## API Endpoints
{
  "/health": {
    "method": "GET",
    "description": "Health check endpoint",
    "response": {
      "status": "healthy"
    }
  },
  "/query": {
    "method": "POST",
    "description": "Submit query to AI system",
    "parameters": [
      "query",
      "provider"
    ],
    "response": {
      "response": "AI response"
    }
  },
  "/is_active": {
    "method": "GET",
    "description": "Check system status",
    "response": {
      "is_active": true
    }
  }
}

## Systems Status
✅ All 15 systems operational and production-ready

Generated: 2025-06-07
