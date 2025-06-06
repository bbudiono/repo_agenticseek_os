# AgenticSeek Documentation

Generated: {
  "endpoints": {
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
  },
  "authentication": "Bearer token required",
  "base_url": "http://localhost:8000"
}
