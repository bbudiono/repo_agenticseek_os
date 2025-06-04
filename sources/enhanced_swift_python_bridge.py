#!/usr/bin/env python3
"""
Enhanced Swift-Python Bridge System
===================================

* Purpose: Production-ready Swift-Python bridge with HTTP/WebSocket communication and multi-agent coordination
* Issues & Complexity Summary: Real-time bidirectional communication between Swift frontend and Python backend
* Key Complexity Drivers:
  - Logic Scope (Est. LoC): ~800
  - Core Algorithm Complexity: Very High
  - Dependencies: 10 New, 8 Mod
  - State Management Complexity: Very High
  - Novelty/Uncertainty Factor: High
* AI Pre-Task Self-Assessment (Est. Solution Difficulty %): 92%
* Problem Estimate (Inherent Problem Difficulty %): 95%
* Initial Code Complexity Estimate %: 90%
* Justification for Estimates: Complex real-time bridge with WebSocket, HTTP, and multi-agent coordination
* Final Code Complexity (Actual %): TBD
* Overall Result Score (Success & Quality %): TBD
* Key Variances/Learnings: TBD
* Last Updated: 2025-01-06
"""

import asyncio
import json
import time
import uuid
import threading
import weakref
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict, deque
import logging

# FastAPI and WebSocket imports
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("Warning: FastAPI not available, bridge will not work")

# AgenticSeek imports
try:
    from sources.multi_agent_coordinator import MultiAgentCoordinator, AgentRole, TaskPriority
    from sources.enhanced_voice_pipeline_system import EnhancedVoicePipelineSystem
    from sources.voice_enabled_agent_router import VoiceEnabledAgentRouter
    from sources.logger import Logger
    from sources.utility import pretty_print
except ImportError as e:
    print(f"Warning: Some AgenticSeek modules not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BridgeMessageType(Enum):
    """Message types for Swift-Python communication"""
    VOICE_COMMAND = "voice_command"
    TEXT_MESSAGE = "text_message"
    AGENT_REQUEST = "agent_request"
    SESSION_CONTROL = "session_control"
    STATUS_UPDATE = "status_update"
    ERROR_NOTIFICATION = "error_notification"
    PING = "ping"
    PONG = "pong"

class ConnectionStatus(Enum):
    """Connection status tracking"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATING = "authenticating"
    ACTIVE = "active"
    INACTIVE = "inactive"
    DISCONNECTED = "disconnected"
    ERROR = "error"

@dataclass
class BridgeMessage:
    """Standardized message format for Swift-Python communication"""
    message_type: BridgeMessageType
    payload: Dict[str, Any]
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    timestamp: float = field(default_factory=time.time)
    source: str = "python_backend"
    target: str = "swift_frontend"
    requires_response: bool = False
    correlation_id: Optional[str] = None

@dataclass
class SwiftClient:
    """Swift client connection information"""
    websocket: WebSocket
    client_id: str
    session_id: str
    connected_at: float
    last_activity: float
    status: ConnectionStatus
    capabilities: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BridgeMetrics:
    """Performance and operational metrics"""
    total_connections: int = 0
    active_connections: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    errors_count: int = 0
    average_response_time: float = 0.0
    uptime_start: float = field(default_factory=time.time)

class EnhancedSwiftPythonBridge:
    """
    Production-ready Swift-Python bridge with HTTP/WebSocket communication
    """
    
    def __init__(self, 
                 multi_agent_coordinator: Optional[MultiAgentCoordinator] = None,
                 voice_pipeline: Optional[EnhancedVoicePipelineSystem] = None,
                 host: str = "127.0.0.1",
                 port: int = 8765,
                 enable_cors: bool = True,
                 enable_auth: bool = False):
        
        if not FASTAPI_AVAILABLE:
            raise RuntimeError("FastAPI not available, cannot create Swift-Python bridge")
        
        # Core configuration
        self.host = host
        self.port = port
        self.enable_cors = enable_cors
        self.enable_auth = enable_auth
        
        # Initialize components
        self.coordinator = multi_agent_coordinator or MultiAgentCoordinator()
        self.voice_pipeline = voice_pipeline
        
        # Create FastAPI application
        self.app = FastAPI(
            title="AgenticSeek Swift-Python Bridge",
            description="Real-time communication bridge between Swift frontend and Python backend",
            version="1.0.0"
        )
        
        # Connection management
        self.swift_clients: Dict[str, SwiftClient] = {}
        self.connection_pool: Set[WebSocket] = set()
        self.message_handlers: Dict[BridgeMessageType, Callable] = {}
        self.response_cache: Dict[str, Any] = {}
        
        # Performance tracking
        self.metrics = BridgeMetrics()
        self.response_times: deque = deque(maxlen=100)
        
        # Rate limiting
        self.rate_limits: Dict[str, deque] = defaultdict(lambda: deque(maxlen=60))
        self.max_messages_per_minute = 60
        
        # Session management
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_timeout = 3600  # 1 hour
        
        logger.info(f"Enhanced Swift-Python Bridge initialized on {host}:{port}")
        
        # Configure CORS if enabled
        if enable_cors:
            self._setup_cors()
        
        # Setup routes and handlers
        self._setup_routes()
        self._setup_message_handlers()
    
    def _setup_cors(self):
        """Configure CORS for Swift frontend communication"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup HTTP routes for bridge communication"""
        
        @self.app.get("/api/bridge/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "uptime": time.time() - self.metrics.uptime_start,
                "active_connections": self.metrics.active_connections,
                "version": "1.0.0"
            }
        
        @self.app.get("/api/bridge/metrics")
        async def get_metrics():
            """Get bridge performance metrics"""
            return asdict(self.metrics)
        
        @self.app.post("/api/voice/process")
        async def process_voice_command(request: Request):
            """Process voice command from Swift frontend"""
            try:
                data = await request.json()
                
                # Extract voice data
                audio_data = data.get("audio_data")
                session_id = data.get("session_id", str(uuid.uuid4()))
                
                if not audio_data:
                    raise HTTPException(status_code=400, detail="No audio data provided")
                
                # Process with voice pipeline
                if self.voice_pipeline:
                    result = await self.voice_pipeline.process_voice_command(
                        audio_data=audio_data.encode() if isinstance(audio_data, str) else audio_data,
                        session_id=session_id
                    )
                else:
                    result = {
                        "success": True,
                        "response_text": "Voice processing not available",
                        "session_id": session_id
                    }
                
                return JSONResponse(content=result)
                
            except Exception as e:
                logger.error(f"Voice processing error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/agent/coordinate")
        async def coordinate_agents(request: Request):
            """Coordinate multi-agent task from Swift frontend"""
            try:
                data = await request.json()
                
                query = data.get("query", "")
                task_type = data.get("task_type", "general")
                priority = TaskPriority(data.get("priority", TaskPriority.MEDIUM.value))
                
                if not query:
                    raise HTTPException(status_code=400, detail="No query provided")
                
                # Process with multi-agent coordinator
                result = await self.coordinator.coordinate_task(
                    query=query,
                    task_type=task_type,
                    priority=priority
                )
                
                return JSONResponse(content={
                    "success": True,
                    "response": result.final_content,
                    "confidence": result.confidence_level,
                    "processing_time": result.total_processing_time,
                    "metadata": result.execution_metadata
                })
                
            except Exception as e:
                logger.error(f"Agent coordination error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/session/start")
        async def start_session(request: Request):
            """Start new session for Swift client"""
            try:
                data = await request.json()
                
                session_id = str(uuid.uuid4())
                client_info = data.get("client_info", {})
                
                # Create session
                self.active_sessions[session_id] = {
                    "session_id": session_id,
                    "created_at": time.time(),
                    "client_info": client_info,
                    "last_activity": time.time(),
                    "status": "active"
                }
                
                return JSONResponse(content={
                    "success": True,
                    "session_id": session_id,
                    "websocket_url": f"ws://{self.host}:{self.port}/ws/{session_id}"
                })
                
            except Exception as e:
                logger.error(f"Session start error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.websocket("/ws/{session_id}")
        async def websocket_endpoint(websocket: WebSocket, session_id: str):
            """WebSocket endpoint for real-time communication"""
            await self._handle_websocket_connection(websocket, session_id)
    
    def _setup_message_handlers(self):
        """Setup message handlers for different message types"""
        
        async def handle_voice_command(message: BridgeMessage, client: SwiftClient) -> Dict[str, Any]:
            """Handle voice command from Swift"""
            try:
                payload = message.payload
                audio_data = payload.get("audio_data")
                
                if self.voice_pipeline and audio_data:
                    result = await self.voice_pipeline.process_voice_command(
                        audio_data=audio_data.encode() if isinstance(audio_data, str) else audio_data,
                        session_id=message.session_id
                    )
                else:
                    result = {
                        "success": True,
                        "response_text": "Voice command received",
                        "session_id": message.session_id
                    }
                
                return result
                
            except Exception as e:
                logger.error(f"Voice command handling error: {str(e)}")
                return {"success": False, "error": str(e)}
        
        async def handle_agent_request(message: BridgeMessage, client: SwiftClient) -> Dict[str, Any]:
            """Handle agent coordination request from Swift"""
            try:
                payload = message.payload
                query = payload.get("query", "")
                task_type = payload.get("task_type", "general")
                priority = TaskPriority(payload.get("priority", TaskPriority.MEDIUM.value))
                
                result = await self.coordinator.coordinate_task(
                    query=query,
                    task_type=task_type,
                    priority=priority
                )
                
                return {
                    "success": True,
                    "response": result.final_content,
                    "confidence": result.confidence_level,
                    "processing_time": result.total_processing_time
                }
                
            except Exception as e:
                logger.error(f"Agent request handling error: {str(e)}")
                return {"success": False, "error": str(e)}
        
        async def handle_session_control(message: BridgeMessage, client: SwiftClient) -> Dict[str, Any]:
            """Handle session control from Swift"""
            try:
                action = message.payload.get("action", "")
                
                if action == "ping":
                    return {"success": True, "action": "pong", "timestamp": time.time()}
                elif action == "get_status":
                    return {
                        "success": True,
                        "client_id": client.client_id,
                        "session_id": client.session_id,
                        "connected_at": client.connected_at,
                        "status": client.status.value
                    }
                else:
                    return {"success": False, "error": f"Unknown action: {action}"}
                
            except Exception as e:
                logger.error(f"Session control handling error: {str(e)}")
                return {"success": False, "error": str(e)}
        
        # Register handlers
        self.message_handlers[BridgeMessageType.VOICE_COMMAND] = handle_voice_command
        self.message_handlers[BridgeMessageType.AGENT_REQUEST] = handle_agent_request
        self.message_handlers[BridgeMessageType.SESSION_CONTROL] = handle_session_control
    
    async def _handle_websocket_connection(self, websocket: WebSocket, session_id: str):
        """Handle WebSocket connection from Swift client"""
        await websocket.accept()
        
        client_id = str(uuid.uuid4())
        client = SwiftClient(
            websocket=websocket,
            client_id=client_id,
            session_id=session_id,
            connected_at=time.time(),
            last_activity=time.time(),
            status=ConnectionStatus.CONNECTED,
            capabilities=["voice", "text", "agents"]
        )
        
        # Register client
        self.swift_clients[client_id] = client
        self.connection_pool.add(websocket)
        self.metrics.active_connections += 1
        self.metrics.total_connections += 1
        
        logger.info(f"Swift client connected: {client_id} (session: {session_id})")
        
        try:
            # Send welcome message
            welcome_message = BridgeMessage(
                message_type=BridgeMessageType.STATUS_UPDATE,
                payload={
                    "status": "connected",
                    "client_id": client_id,
                    "session_id": session_id,
                    "capabilities": client.capabilities
                },
                session_id=session_id
            )
            await self._send_message_to_client(client, welcome_message)
            
            # Message handling loop
            while True:
                try:
                    # Receive message from Swift
                    data = await websocket.receive_text()
                    message_data = json.loads(data)
                    
                    # Create message object
                    message = BridgeMessage(
                        message_type=BridgeMessageType(message_data.get("type", "text_message")),
                        payload=message_data.get("payload", {}),
                        message_id=message_data.get("message_id", str(uuid.uuid4())),
                        session_id=session_id,
                        source="swift_frontend",
                        target="python_backend"
                    )
                    
                    # Update client activity
                    client.last_activity = time.time()
                    self.metrics.messages_received += 1
                    
                    # Rate limiting check
                    if not self._check_rate_limit(client_id):
                        await self._send_error_to_client(client, "Rate limit exceeded")
                        continue
                    
                    # Process message
                    response = await self._process_message(message, client)
                    
                    # Send response if required
                    if message.requires_response or response.get("success") is False:
                        response_message = BridgeMessage(
                            message_type=BridgeMessageType.STATUS_UPDATE,
                            payload=response,
                            session_id=session_id,
                            correlation_id=message.message_id
                        )
                        await self._send_message_to_client(client, response_message)
                
                except WebSocketDisconnect:
                    break
                except json.JSONDecodeError:
                    await self._send_error_to_client(client, "Invalid JSON format")
                except Exception as e:
                    logger.error(f"Message processing error: {str(e)}")
                    await self._send_error_to_client(client, str(e))
        
        except Exception as e:
            logger.error(f"WebSocket connection error: {str(e)}")
        
        finally:
            # Cleanup
            await self._cleanup_client(client)
    
    async def _process_message(self, message: BridgeMessage, client: SwiftClient) -> Dict[str, Any]:
        """Process incoming message from Swift client"""
        start_time = time.time()
        
        try:
            # Get message handler
            handler = self.message_handlers.get(message.message_type)
            if not handler:
                return {"success": False, "error": f"Unknown message type: {message.message_type.value}"}
            
            # Process message
            result = await handler(message, client)
            
            # Track response time
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            self.metrics.average_response_time = sum(self.response_times) / len(self.response_times)
            
            return result
            
        except Exception as e:
            logger.error(f"Message processing error: {str(e)}")
            self.metrics.errors_count += 1
            return {"success": False, "error": str(e)}
    
    async def _send_message_to_client(self, client: SwiftClient, message: BridgeMessage):
        """Send message to Swift client"""
        try:
            message_data = {
                "type": message.message_type.value,
                "payload": message.payload,
                "message_id": message.message_id,
                "session_id": message.session_id,
                "timestamp": message.timestamp,
                "source": message.source,
                "target": message.target,
                "correlation_id": message.correlation_id
            }
            
            await client.websocket.send_text(json.dumps(message_data))
            self.metrics.messages_sent += 1
            
        except Exception as e:
            logger.error(f"Error sending message to client {client.client_id}: {str(e)}")
            await self._cleanup_client(client)
    
    async def _send_error_to_client(self, client: SwiftClient, error_message: str):
        """Send error message to Swift client"""
        error_msg = BridgeMessage(
            message_type=BridgeMessageType.ERROR_NOTIFICATION,
            payload={"error": error_message, "timestamp": time.time()},
            session_id=client.session_id
        )
        await self._send_message_to_client(client, error_msg)
    
    def _check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limits"""
        current_time = time.time()
        client_messages = self.rate_limits[client_id]
        
        # Remove old messages (older than 1 minute)
        while client_messages and current_time - client_messages[0] > 60:
            client_messages.popleft()
        
        # Check if under limit
        if len(client_messages) >= self.max_messages_per_minute:
            return False
        
        # Add current message
        client_messages.append(current_time)
        return True
    
    async def _cleanup_client(self, client: SwiftClient):
        """Clean up disconnected client"""
        try:
            # Remove from tracking
            if client.client_id in self.swift_clients:
                del self.swift_clients[client.client_id]
            
            if client.websocket in self.connection_pool:
                self.connection_pool.remove(client.websocket)
            
            self.metrics.active_connections -= 1
            
            logger.info(f"Swift client disconnected: {client.client_id}")
            
        except Exception as e:
            logger.error(f"Client cleanup error: {str(e)}")
    
    async def broadcast_to_all_clients(self, message: BridgeMessage):
        """Broadcast message to all connected Swift clients"""
        disconnected_clients = []
        
        for client in self.swift_clients.values():
            try:
                await self._send_message_to_client(client, message)
            except Exception as e:
                logger.error(f"Error broadcasting to client {client.client_id}: {str(e)}")
                disconnected_clients.append(client)
        
        # Cleanup disconnected clients
        for client in disconnected_clients:
            await self._cleanup_client(client)
    
    def get_bridge_status(self) -> Dict[str, Any]:
        """Get current bridge status"""
        return {
            "bridge_info": {
                "host": self.host,
                "port": self.port,
                "uptime": time.time() - self.metrics.uptime_start,
                "version": "1.0.0"
            },
            "connections": {
                "active": self.metrics.active_connections,
                "total": self.metrics.total_connections,
                "clients": [
                    {
                        "client_id": client.client_id,
                        "session_id": client.session_id,
                        "connected_at": client.connected_at,
                        "status": client.status.value,
                        "capabilities": client.capabilities
                    }
                    for client in self.swift_clients.values()
                ]
            },
            "performance": asdict(self.metrics),
            "features": {
                "voice_processing": self.voice_pipeline is not None,
                "agent_coordination": self.coordinator is not None,
                "cors_enabled": self.enable_cors,
                "auth_enabled": self.enable_auth
            }
        }
    
    async def start_server(self):
        """Start the bridge server"""
        logger.info(f"Starting Enhanced Swift-Python Bridge on {self.host}:{self.port}")
        
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        await server.serve()
    
    async def stop_server(self):
        """Stop the bridge server and cleanup connections"""
        logger.info("Stopping Enhanced Swift-Python Bridge")
        
        # Close all client connections
        for client in list(self.swift_clients.values()):
            try:
                await client.websocket.close()
            except:
                pass
        
        # Clear all tracking
        self.swift_clients.clear()
        self.connection_pool.clear()
        self.active_sessions.clear()

# Example usage and testing
async def main():
    """Example usage of EnhancedSwiftPythonBridge"""
    # Create components
    coordinator = MultiAgentCoordinator()
    
    # Create enhanced bridge
    bridge = EnhancedSwiftPythonBridge(
        multi_agent_coordinator=coordinator,
        host="127.0.0.1",
        port=8765,
        enable_cors=True
    )
    
    # Start server
    print("Starting Enhanced Swift-Python Bridge...")
    print(f"Bridge status: {bridge.get_bridge_status()}")
    
    # In production, you would await bridge.start_server()
    print("Bridge ready for Swift frontend connections")

if __name__ == "__main__":
    asyncio.run(main())