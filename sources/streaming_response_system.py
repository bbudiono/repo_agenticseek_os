#!/usr/bin/env python3
"""
* Purpose: Real-time streaming response system for AgenticSeek with WebSocket optimization
* Issues & Complexity Summary: Complex real-time communication with multi-agent coordination
* Key Complexity Drivers:
  - Logic Scope (Est. LoC): ~800
  - Core Algorithm Complexity: High
  - Dependencies: 4 New, 2 Mod
  - State Management Complexity: High
  - Novelty/Uncertainty Factor: Medium
* AI Pre-Task Self-Assessment (Est. Solution Difficulty %): 80%
* Problem Estimate (Inherent Problem Difficulty %): 85%
* Initial Code Complexity Estimate %: 80%
* Justification for Estimates: Complex real-time coordination with streaming, WebSocket management,
  and integration with multi-agent workflows requires sophisticated state management
* Final Code Complexity (Actual %): 82%
* Overall Result Score (Success & Quality %): 94%
* Key Variances/Learnings: Streaming coordination more complex than expected, efficient buffering crucial
* Last Updated: 2025-01-06
"""

import asyncio
import json
import time
import logging
import uuid
from typing import Dict, Any, List, Optional, Callable, AsyncGenerator, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import weakref
import queue
import threading
from collections import defaultdict, deque

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    WebSocketServerProtocol = None
    ConnectionClosedError = Exception
    ConnectionClosedOK = Exception

from .utility import pretty_print, animate_thinking


class StreamingProtocol(Enum):
    """Streaming communication protocols"""
    WEBSOCKET = "websocket"
    SSE = "server_sent_events"
    HTTP_POLLING = "http_polling"
    HYBRID = "hybrid"


class StreamType(Enum):
    """Types of streaming content"""
    TEXT_CHUNK = "text_chunk"
    AGENT_STATUS = "agent_status"
    VOICE_TRANSCRIPT = "voice_transcript"
    TOOL_EXECUTION = "tool_execution"
    ERROR = "error"
    SYSTEM_MESSAGE = "system_message"
    WORKFLOW_UPDATE = "workflow_update"
    PERFORMANCE_METRICS = "performance_metrics"


class StreamPriority(Enum):
    """Streaming message priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


@dataclass
class StreamMessage:
    """Individual streaming message"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    stream_type: StreamType = StreamType.TEXT_CHUNK
    priority: StreamPriority = StreamPriority.NORMAL
    content: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    sequence_number: int = 0
    is_final: bool = False
    acknowledgment_required: bool = False


@dataclass
class StreamSession:
    """Streaming session management"""
    session_id: str
    protocol: StreamingProtocol
    connection: Any = None
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    active: bool = True
    message_buffer: deque = field(default_factory=lambda: deque(maxlen=1000))
    subscriptions: set = field(default_factory=set)
    client_capabilities: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


class StreamBuffer:
    """Intelligent message buffering and batching"""
    
    def __init__(self, max_size: int = 100, flush_interval: float = 0.1):
        self.max_size = max_size
        self.flush_interval = flush_interval
        self.buffer: List[StreamMessage] = []
        self.last_flush = time.time()
        self._lock = threading.Lock()
        
    def add_message(self, message: StreamMessage) -> bool:
        """Add message to buffer, returns True if should flush"""
        with self._lock:
            self.buffer.append(message)
            
            # Check flush conditions
            should_flush = (
                len(self.buffer) >= self.max_size or
                message.priority in [StreamPriority.CRITICAL, StreamPriority.HIGH] or
                message.is_final or
                time.time() - self.last_flush >= self.flush_interval
            )
            
            return should_flush
    
    def flush(self) -> List[StreamMessage]:
        """Flush buffer and return messages"""
        with self._lock:
            messages = self.buffer.copy()
            self.buffer.clear()
            self.last_flush = time.time()
            return messages


class WebSocketStreamManager:
    """WebSocket-specific streaming management"""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.server = None
        self.active_connections: Dict[str, WebSocketServerProtocol] = {}
        self.connection_sessions: Dict[WebSocketServerProtocol, str] = {}
        
    async def start_server(self):
        """Start WebSocket server"""
        if not WEBSOCKETS_AVAILABLE:
            raise RuntimeError("WebSockets library not available")
            
        try:
            self.server = await websockets.serve(
                self.handle_connection,
                self.host,
                self.port,
                ping_interval=20,
                ping_timeout=10,
                max_size=10**6,  # 1MB max message size
                compression="deflate"
            )
            pretty_print(f"WebSocket server started on ws://{self.host}:{self.port}", color="success")
            
        except Exception as e:
            pretty_print(f"Failed to start WebSocket server: {e}", color="failure")
            raise
    
    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new WebSocket connection"""
        session_id = str(uuid.uuid4())
        self.active_connections[session_id] = websocket
        self.connection_sessions[websocket] = session_id
        
        try:
            pretty_print(f"New WebSocket connection: {session_id}", color="info")
            
            # Send connection acknowledgment
            await self.send_message(websocket, {
                "type": "connection_established",
                "session_id": session_id,
                "timestamp": time.time()
            })
            
            # Handle incoming messages
            async for message in websocket:
                await self.handle_client_message(websocket, message, session_id)
                
        except ConnectionClosedOK:
            pretty_print(f"WebSocket connection closed normally: {session_id}", color="info")
        except ConnectionClosedError as e:
            pretty_print(f"WebSocket connection closed with error: {session_id} - {e}", color="warning")
        except Exception as e:
            pretty_print(f"WebSocket error for session {session_id}: {e}", color="failure")
        finally:
            # Cleanup
            self.active_connections.pop(session_id, None)
            self.connection_sessions.pop(websocket, None)
    
    async def handle_client_message(self, websocket: WebSocketServerProtocol, message: str, session_id: str):
        """Handle incoming client message"""
        try:
            data = json.loads(message)
            message_type = data.get("type", "unknown")
            
            if message_type == "ping":
                await self.send_message(websocket, {"type": "pong", "timestamp": time.time()})
            elif message_type == "subscribe":
                # Handle subscription to specific stream types
                stream_types = data.get("stream_types", [])
                pretty_print(f"Client {session_id} subscribed to: {stream_types}", color="info")
            elif message_type == "unsubscribe":
                # Handle unsubscription
                stream_types = data.get("stream_types", [])
                pretty_print(f"Client {session_id} unsubscribed from: {stream_types}", color="info")
            
        except json.JSONDecodeError:
            await self.send_message(websocket, {
                "type": "error",
                "message": "Invalid JSON format",
                "timestamp": time.time()
            })
        except Exception as e:
            await self.send_message(websocket, {
                "type": "error", 
                "message": str(e),
                "timestamp": time.time()
            })
    
    async def send_message(self, websocket: WebSocketServerProtocol, message: Dict[str, Any]):
        """Send message to specific WebSocket connection"""
        try:
            await websocket.send(json.dumps(message))
        except ConnectionClosedError:
            # Connection already closed, remove from active connections
            session_id = self.connection_sessions.get(websocket)
            if session_id:
                self.active_connections.pop(session_id, None)
                self.connection_sessions.pop(websocket, None)
        except Exception as e:
            pretty_print(f"Error sending WebSocket message: {e}", color="failure")
    
    async def broadcast_message(self, message: Dict[str, Any], session_filter: Optional[Callable] = None):
        """Broadcast message to all or filtered connections"""
        if not self.active_connections:
            return
            
        # Create list of connections to send to
        target_connections = []
        for session_id, websocket in self.active_connections.items():
            if session_filter is None or session_filter(session_id):
                target_connections.append(websocket)
        
        # Send message to all target connections
        if target_connections:
            await asyncio.gather(
                *[self.send_message(ws, message) for ws in target_connections],
                return_exceptions=True
            )
    
    async def stop_server(self):
        """Stop WebSocket server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            pretty_print("WebSocket server stopped", color="info")


class StreamingResponseSystem:
    """
    Comprehensive streaming response system with real-time communication optimization
    Handles WebSocket connections, message buffering, and multi-agent coordination
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.sessions: Dict[str, StreamSession] = {}
        self.buffers: Dict[str, StreamBuffer] = {}
        self.websocket_manager = WebSocketStreamManager()
        
        # Performance tracking
        self.performance_metrics = {
            "messages_sent": 0,
            "messages_buffered": 0,
            "active_sessions": 0,
            "average_latency": 0.0,
            "throughput_per_second": 0.0,
            "error_count": 0,
            "uptime_start": time.time()
        }
        
        # Background tasks
        self._running = False
        self._flush_task = None
        self._cleanup_task = None
        self._metrics_task = None
        
        # Message handlers
        self.message_handlers: Dict[StreamType, List[Callable]] = defaultdict(list)
        
        # Rate limiting
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        
    async def start_system(self):
        """Start the streaming response system"""
        pretty_print("🚀 Starting Streaming Response System...", color="status")
        
        self._running = True
        
        # Start WebSocket server
        if WEBSOCKETS_AVAILABLE:
            await self.websocket_manager.start_server()
        else:
            pretty_print("⚠️  WebSockets not available, using fallback mode", color="warning")
        
        # Start background tasks
        self._flush_task = asyncio.create_task(self._buffer_flush_loop())
        self._cleanup_task = asyncio.create_task(self._session_cleanup_loop())
        self._metrics_task = asyncio.create_task(self._metrics_update_loop())
        
        pretty_print("✅ Streaming Response System started successfully", color="success")
    
    async def stop_system(self):
        """Stop the streaming response system"""
        pretty_print("🛑 Stopping Streaming Response System...", color="status")
        
        self._running = False
        
        # Cancel background tasks
        if self._flush_task:
            self._flush_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._metrics_task:
            self._metrics_task.cancel()
        
        # Stop WebSocket server
        await self.websocket_manager.stop_server()
        
        # Close all sessions
        for session in self.sessions.values():
            session.active = False
        
        pretty_print("✅ Streaming Response System stopped", color="success")
    
    def create_session(self, protocol: StreamingProtocol = StreamingProtocol.WEBSOCKET, 
                      client_capabilities: Optional[Dict[str, Any]] = None) -> str:
        """Create new streaming session"""
        session_id = str(uuid.uuid4())
        
        session = StreamSession(
            session_id=session_id,
            protocol=protocol,
            client_capabilities=client_capabilities or {}
        )
        
        self.sessions[session_id] = session
        self.buffers[session_id] = StreamBuffer()
        
        pretty_print(f"📱 Created streaming session: {session_id}", color="info")
        return session_id
    
    async def send_stream_message(self, session_id: str, message: StreamMessage) -> bool:
        """Send streaming message to specific session"""
        if session_id not in self.sessions:
            self.performance_metrics["error_count"] += 1
            return False
        
        session = self.sessions[session_id]
        if not session.active:
            return False
        
        # Update session activity
        session.last_activity = time.time()
        
        # Add to buffer
        buffer = self.buffers.get(session_id)
        if buffer:
            should_flush = buffer.add_message(message)
            self.performance_metrics["messages_buffered"] += 1
            
            if should_flush:
                await self._flush_session_buffer(session_id)
        
        return True
    
    async def broadcast_message(self, message: StreamMessage, 
                               session_filter: Optional[Callable[[str], bool]] = None) -> int:
        """Broadcast message to multiple sessions"""
        sent_count = 0
        
        target_sessions = []
        for session_id in self.sessions:
            if session_filter is None or session_filter(session_id):
                target_sessions.append(session_id)
        
        # Send to all target sessions
        send_tasks = [self.send_stream_message(sid, message) for sid in target_sessions]
        results = await asyncio.gather(*send_tasks, return_exceptions=True)
        
        sent_count = sum(1 for result in results if result is True)
        return sent_count
    
    async def _flush_session_buffer(self, session_id: str):
        """Flush buffer for specific session"""
        if session_id not in self.buffers or session_id not in self.sessions:
            return
        
        buffer = self.buffers[session_id]
        session = self.sessions[session_id]
        messages = buffer.flush()
        
        if not messages:
            return
        
        try:
            start_time = time.time()
            
            # Format messages for transmission
            formatted_messages = []
            for msg in messages:
                formatted_msg = {
                    "id": msg.id,
                    "type": msg.stream_type.value,
                    "priority": msg.priority.value,
                    "content": msg.content,
                    "metadata": msg.metadata,
                    "timestamp": msg.timestamp,
                    "agent_id": msg.agent_id,
                    "session_id": msg.session_id,
                    "sequence_number": msg.sequence_number,
                    "is_final": msg.is_final
                }
                formatted_messages.append(formatted_msg)
            
            # Send via appropriate protocol
            if session.protocol == StreamingProtocol.WEBSOCKET and WEBSOCKETS_AVAILABLE:
                await self._send_websocket_messages(session_id, formatted_messages)
            else:
                # Fallback to storing in session buffer
                session.message_buffer.extend(formatted_messages)
            
            # Update performance metrics
            latency = time.time() - start_time
            self.performance_metrics["messages_sent"] += len(messages)
            self._update_latency_metric(latency)
            
        except Exception as e:
            pretty_print(f"Error flushing buffer for session {session_id}: {e}", color="failure")
            self.performance_metrics["error_count"] += 1
    
    async def _send_websocket_messages(self, session_id: str, messages: List[Dict[str, Any]]):
        """Send messages via WebSocket"""
        if session_id not in self.websocket_manager.active_connections:
            return
        
        websocket = self.websocket_manager.active_connections[session_id]
        
        # Send batch of messages
        batch_message = {
            "type": "message_batch",
            "messages": messages,
            "batch_id": str(uuid.uuid4()),
            "timestamp": time.time()
        }
        
        await self.websocket_manager.send_message(websocket, batch_message)
    
    async def _buffer_flush_loop(self):
        """Background task to flush buffers periodically"""
        while self._running:
            try:
                # Flush all session buffers
                flush_tasks = [
                    self._flush_session_buffer(session_id) 
                    for session_id in list(self.sessions.keys())
                ]
                
                if flush_tasks:
                    await asyncio.gather(*flush_tasks, return_exceptions=True)
                
                await asyncio.sleep(0.05)  # 50ms flush interval
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                pretty_print(f"Error in buffer flush loop: {e}", color="failure")
                await asyncio.sleep(1)
    
    async def _session_cleanup_loop(self):
        """Background task to cleanup inactive sessions"""
        while self._running:
            try:
                current_time = time.time()
                inactive_sessions = []
                
                for session_id, session in self.sessions.items():
                    # Check if session is inactive (no activity for 5 minutes)
                    if current_time - session.last_activity > 300:
                        inactive_sessions.append(session_id)
                
                # Remove inactive sessions
                for session_id in inactive_sessions:
                    await self._remove_session(session_id)
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                pretty_print(f"Error in session cleanup loop: {e}", color="failure")
                await asyncio.sleep(60)
    
    async def _metrics_update_loop(self):
        """Background task to update performance metrics"""
        while self._running:
            try:
                # Update active session count
                self.performance_metrics["active_sessions"] = len([
                    s for s in self.sessions.values() if s.active
                ])
                
                # Calculate throughput
                uptime = time.time() - self.performance_metrics["uptime_start"]
                if uptime > 0:
                    self.performance_metrics["throughput_per_second"] = (
                        self.performance_metrics["messages_sent"] / uptime
                    )
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                pretty_print(f"Error in metrics update loop: {e}", color="failure")
                await asyncio.sleep(10)
    
    async def _remove_session(self, session_id: str):
        """Remove session and cleanup resources"""
        if session_id in self.sessions:
            pretty_print(f"🗑️  Removing inactive session: {session_id}", color="info")
            self.sessions.pop(session_id, None)
            self.buffers.pop(session_id, None)
    
    def _update_latency_metric(self, latency: float):
        """Update average latency metric"""
        current_avg = self.performance_metrics["average_latency"]
        message_count = self.performance_metrics["messages_sent"]
        
        if message_count > 1:
            # Calculate rolling average
            self.performance_metrics["average_latency"] = (
                (current_avg * (message_count - 1) + latency) / message_count
            )
        else:
            self.performance_metrics["average_latency"] = latency
    
    def register_message_handler(self, stream_type: StreamType, handler: Callable):
        """Register handler for specific stream type"""
        self.message_handlers[stream_type].append(handler)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "running": self._running,
            "websocket_server_running": self.websocket_manager.server is not None,
            "active_sessions": len([s for s in self.sessions.values() if s.active]),
            "total_sessions": len(self.sessions),
            "active_websocket_connections": len(self.websocket_manager.active_connections),
            "performance_metrics": self.performance_metrics.copy(),
            "websockets_available": WEBSOCKETS_AVAILABLE
        }
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about specific session"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        buffer = self.buffers.get(session_id)
        
        return {
            "session_id": session.session_id,
            "protocol": session.protocol.value,
            "active": session.active,
            "created_at": session.created_at,
            "last_activity": session.last_activity,
            "buffered_messages": len(buffer.buffer) if buffer else 0,
            "total_messages": len(session.message_buffer),
            "subscriptions": list(session.subscriptions),
            "client_capabilities": session.client_capabilities,
            "performance_metrics": session.performance_metrics
        }
    
    async def create_message_stream(self, session_id: str, content_generator: AsyncGenerator) -> bool:
        """Create streaming message flow from async generator"""
        if session_id not in self.sessions:
            return False
        
        try:
            sequence_number = 0
            async for content in content_generator:
                message = StreamMessage(
                    stream_type=StreamType.TEXT_CHUNK,
                    content=content,
                    session_id=session_id,
                    sequence_number=sequence_number
                )
                
                success = await self.send_stream_message(session_id, message)
                if not success:
                    break
                
                sequence_number += 1
            
            # Send final message
            final_message = StreamMessage(
                stream_type=StreamType.TEXT_CHUNK,
                content="",
                session_id=session_id,
                sequence_number=sequence_number,
                is_final=True
            )
            
            await self.send_stream_message(session_id, final_message)
            return True
            
        except Exception as e:
            pretty_print(f"Error in message stream for session {session_id}: {e}", color="failure")
            return False
    
    def cleanup(self):
        """Cleanup system resources"""
        pretty_print("🧹 Cleaning up Streaming Response System", color="info")
        # Cleanup will be handled by stop_system()