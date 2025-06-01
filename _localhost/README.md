# AgenticSeek Web Application (localhost)

This directory contains the original AgenticSeek web application components that run on localhost.

## Components

- **frontend/**: React web application frontend
- **api.py**: FastAPI backend server
- **docker-compose.yml**: Docker configuration for services
- **requirements.txt**: Python dependencies

## Running the Web Application

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the backend:**
   ```bash
   python api.py
   ```

3. **Start the frontend:**
   ```bash
   cd frontend/agentic-seek-front
   npm install
   npm start
   ```

4. **Or use Docker:**
   ```bash
   docker-compose up
   ```

## Access

- Frontend: http://localhost:3000
- Backend API: http://localhost:8001
- Documentation: http://localhost:8001/docs

## Note

This is the original web-based interface. The main repository now focuses on the native macOS application with .cursorrules compliance.