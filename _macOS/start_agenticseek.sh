#!/bin/bash

# AgenticSeek Startup Script for macOS Native App
# This script helps start all required services

echo "🚀 Starting AgenticSeek Services..."

# Get the script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "📍 Project root: $PROJECT_ROOT"

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    exit 1
fi

echo "✅ Docker is running"

# Start Docker services (frontend, redis, searxng)
echo "🐳 Starting Docker services..."
docker-compose up -d

# Wait for services to start
echo "⏱️ Waiting for services to start..."
sleep 10

# Check service status
echo "📊 Checking service status..."
docker-compose ps

# Try to start Python backend
echo "🐍 Starting Python backend..."

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment and install minimal dependencies
source venv/bin/activate

echo "📦 Installing required packages..."
pip install -q fastapi uvicorn aiofiles python-multipart httpx ollama redis colorama rich langchain-core langchain-openai openai anthropic

# Start the backend
echo "🚀 Starting backend server..."
python api.py &
BACKEND_PID=$!

echo "✅ Backend started with PID: $BACKEND_PID"

# Wait a moment for startup
sleep 5

# Check if services are responding
echo "🔍 Checking service health..."

if curl -s http://localhost:3000/ > /dev/null; then
    echo "✅ Frontend (React) running on http://localhost:3000"
else
    echo "❌ Frontend not responding"
fi

if curl -s http://localhost:8000/ > /dev/null; then
    echo "✅ Backend (Python) running on http://localhost:8000"
else
    echo "❌ Backend not responding yet (may need more time)"
fi

echo ""
echo "🎉 AgenticSeek services started!"
echo "💡 Now launch your native macOS app: AgenticSeek.app"
echo ""
echo "🛑 To stop services later:"
echo "   docker-compose down"
echo "   kill $BACKEND_PID"
echo ""