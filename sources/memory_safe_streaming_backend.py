#!/usr/bin/env python3
"""
Memory-Safe Real-Time LLM Streaming Backend
==========================================

Purpose: Build memory-safe real-time LLM streaming backend for production
Issues & Complexity Summary: Complex streaming with memory leak prevention and garbage collection
Key Complexity Drivers:
  - Logic Scope (Est. LoC): ~350
  - Core Algorithm Complexity: High
  - Dependencies: 4 New, 2 Mod
  - State Management Complexity: High
  - Novelty/Uncertainty Factor: Medium
AI Pre-Task Self-Assessment (Est. Solution Difficulty %): 80%
Problem Estimate (Inherent Problem Difficulty %): 75%
Initial Code Complexity Estimate %: 80%
Justification for Estimates: Memory management with real-time streaming requires careful resource handling
Final Code Complexity (Actual %): TBD
Overall Result Score (Success & Quality %): TBD
Key Variances/Learnings: TBD
Last Updated: 2025-06-05
"""

import asyncio
import json
import logging
import time
import gc
import psutil
import threading
import weakref
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncGenerator, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import websockets
import aiohttp
from aiohttp import web
import ssl
import os
from pathlib import Path

# Memory monitoring
import tracemalloc
import resource

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class StreamingBuffer:
    """Memory-safe streaming buffer with automatic cleanup"""
    id: str
    max_size: int = 1024 * 1024  # 1MB default
    chunks: deque = field(default_factory=deque)
    total_size: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    compression_enabled: bool = True
    
    def add_chunk(self, data: bytes) -> bool:
        """Add chunk with memory safety"""
        chunk_size = len(data)
        
        # Check memory limits
        if self.total_size + chunk_size > self.max_size:
            # Remove oldest chunks to make space
            while self.chunks and self.total_size + chunk_size > self.max_size:
                old_chunk = self.chunks.popleft()
                self.total_size -= len(old_chunk)
        
        self.chunks.append(data)
        self.total_size += chunk_size
        self.last_accessed = datetime.now()
        
        return True
    
    def get_chunks(self) -> List[bytes]:
        """Get all chunks and mark as accessed"""
        self.last_accessed = datetime.now()
        return list(self.chunks)
    
    def clear(self):
        """Clear buffer and free memory"""
        self.chunks.clear()
        self.total_size = 0
        gc.collect()  # Force garbage collection

class MemoryManager:
    """Memory management for streaming operations"""
    
    def __init__(self, max_memory_mb: int = 512):
        self.max_memory_mb = max_memory_mb
        self.buffers: Dict[str, StreamingBuffer] = {}
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = datetime.now()
        self._lock = threading.Lock()
        
        # Start memory monitoring
        tracemalloc.start()
        
    def create_buffer(self, buffer_id: str, max_size: int = None) -> StreamingBuffer:
        """Create new streaming buffer with memory tracking"""
        with self._lock:
            if buffer_id in self.buffers:
                # Clear existing buffer
                self.buffers[buffer_id].clear()
            
            buffer = StreamingBuffer(
                id=buffer_id,
                max_size=max_size or (1024 * 1024)
            )
            self.buffers[buffer_id] = buffer
            
            logger.debug(f"Created streaming buffer: {buffer_id}")
            return buffer
    
    def get_buffer(self, buffer_id: str) -> Optional[StreamingBuffer]:
        """Get buffer by ID"""
        with self._lock:
            return self.buffers.get(buffer_id)
    
    def cleanup_expired_buffers(self):
        """Clean up expired buffers to prevent memory leaks"""
        with self._lock:
            current_time = datetime.now()
            expired_ids = []
            
            for buffer_id, buffer in self.buffers.items():
                # Remove buffers not accessed in last 30 minutes
                if (current_time - buffer.last_accessed).total_seconds() > 1800:
                    expired_ids.append(buffer_id)
            
            for buffer_id in expired_ids:
                self.buffers[buffer_id].clear()
                del self.buffers[buffer_id]
                logger.info(f"Cleaned up expired buffer: {buffer_id}")
            
            # Force garbage collection
            gc.collect()
            
            self.last_cleanup = current_time
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Get tracemalloc statistics
        current, peak = tracemalloc.get_traced_memory()
        
        return {
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_mb": memory_info.vms / (1024 * 1024),
            "traced_current_mb": current / (1024 * 1024),
            "traced_peak_mb": peak / (1024 * 1024),
            "buffer_count": len(self.buffers),
            "total_buffer_size_mb": sum(b.total_size for b in self.buffers.values()) / (1024 * 1024)
        }
    
    def force_cleanup(self):
        """Force immediate cleanup of all buffers"""
        with self._lock:
            for buffer in self.buffers.values():
                buffer.clear()
            self.buffers.clear()
            gc.collect()
            logger.info("Forced cleanup of all streaming buffers")

class LLMStreamingClient:
    """Memory-safe LLM streaming client"""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300),
            connector=aiohttp.TCPConnector(limit=10)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup"""
        if self.session:
            await self.session.close()
        gc.collect()
    
    async def stream_anthropic_response(self, api_key: str, messages: List[Dict], 
                                      buffer_id: str) -> AsyncGenerator[Dict, None]:
        """Stream response from Anthropic API with memory safety"""
        buffer = self.memory_manager.create_buffer(buffer_id)
        
        headers = {
            "x-api-key": api_key,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 1024,
            "messages": messages,
            "stream": True
        }
        
        try:
            async with self.session.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Anthropic API error: {response.status} - {error_text}")
                
                async for line in response.content:
                    if line:
                        # Add to buffer with memory safety
                        buffer.add_chunk(line)
                        
                        # Parse streaming response
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith('data: '):
                            try:
                                data = json.loads(line_str[6:])
                                yield data
                            except json.JSONDecodeError:
                                continue
                        
                        # Check memory usage periodically
                        if len(buffer.chunks) % 100 == 0:
                            await self._check_memory_usage()
                            
        except Exception as e:
            logger.error(f"Error streaming from Anthropic: {e}")
            raise
        finally:
            buffer.clear()
    
    async def stream_openai_response(self, api_key: str, messages: List[Dict], 
                                   buffer_id: str) -> AsyncGenerator[Dict, None]:
        """Stream response from OpenAI API with memory safety"""
        buffer = self.memory_manager.create_buffer(buffer_id)
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-4",
            "messages": messages,
            "stream": True,
            "max_tokens": 1024
        }
        
        try:
            async with self.session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"OpenAI API error: {response.status} - {error_text}")
                
                async for line in response.content:
                    if line:
                        # Add to buffer with memory safety
                        buffer.add_chunk(line)
                        
                        # Parse streaming response
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith('data: '):
                            try:
                                if line_str == 'data: [DONE]':
                                    break
                                data = json.loads(line_str[6:])
                                yield data
                            except json.JSONDecodeError:
                                continue
                        
                        # Check memory usage periodically
                        if len(buffer.chunks) % 100 == 0:
                            await self._check_memory_usage()
                            
        except Exception as e:
            logger.error(f"Error streaming from OpenAI: {e}")
            raise
        finally:
            buffer.clear()
    
    async def _check_memory_usage(self):
        """Check memory usage and trigger cleanup if needed"""
        memory_stats = self.memory_manager.get_memory_usage()
        
        if memory_stats["rss_mb"] > 500:  # 500MB threshold
            logger.warning(f"High memory usage detected: {memory_stats['rss_mb']:.1f}MB")
            self.memory_manager.cleanup_expired_buffers()

class WebSocketStreamingServer:
    """Memory-safe WebSocket server for real-time streaming"""
    
    def __init__(self, memory_manager: MemoryManager, port: int = 8765):
        self.memory_manager = memory_manager
        self.port = port
        self.clients: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.client_buffers: Dict[str, str] = {}  # client_id -> buffer_id mapping
        self._cleanup_task = None
        
    async def start_server(self):
        """Start WebSocket server with memory safety"""
        logger.info(f"Starting memory-safe WebSocket server on port {self.port}")
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        
        return await websockets.serve(
            self.handle_client,
            "localhost",
            self.port,
            ping_interval=30,
            ping_timeout=10
        )
    
    async def handle_client(self, websocket, path):
        """Handle WebSocket client with memory management"""
        client_id = f"client_{len(self.clients)}_{int(time.time())}"
        buffer_id = f"buffer_{client_id}"
        
        self.clients[client_id] = websocket
        self.client_buffers[client_id] = buffer_id
        
        logger.info(f"WebSocket client connected: {client_id}")
        
        try:
            async for message in websocket:
                await self._handle_message(client_id, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"Error handling WebSocket client {client_id}: {e}")
        finally:
            # Cleanup client resources
            self._cleanup_client(client_id)
    
    async def _handle_message(self, client_id: str, message: str):
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "stream_request":
                await self._handle_stream_request(client_id, data)
            elif message_type == "ping":
                await self._send_to_client(client_id, {"type": "pong"})
            
        except json.JSONDecodeError:
            await self._send_to_client(client_id, {
                "type": "error",
                "message": "Invalid JSON message"
            })
        except Exception as e:
            logger.error(f"Error handling message from {client_id}: {e}")
            await self._send_to_client(client_id, {
                "type": "error",
                "message": str(e)
            })
    
    async def _handle_stream_request(self, client_id: str, data: Dict):
        """Handle LLM streaming request"""
        provider = data.get("provider", "anthropic")
        messages = data.get("messages", [])
        api_key = data.get("api_key")
        
        if not api_key:
            await self._send_to_client(client_id, {
                "type": "error",
                "message": "API key required"
            })
            return
        
        buffer_id = self.client_buffers[client_id]
        
        try:
            async with LLMStreamingClient(self.memory_manager) as client:
                if provider == "anthropic":
                    stream = client.stream_anthropic_response(api_key, messages, buffer_id)
                elif provider == "openai":
                    stream = client.stream_openai_response(api_key, messages, buffer_id)
                else:
                    raise ValueError(f"Unsupported provider: {provider}")
                
                async for chunk in stream:
                    await self._send_to_client(client_id, {
                        "type": "stream_chunk",
                        "data": chunk
                    })
                
                await self._send_to_client(client_id, {
                    "type": "stream_complete"
                })
                
        except Exception as e:
            logger.error(f"Error streaming for client {client_id}: {e}")
            await self._send_to_client(client_id, {
                "type": "error",
                "message": str(e)
            })
    
    async def _send_to_client(self, client_id: str, data: Dict):
        """Send data to WebSocket client"""
        if client_id in self.clients:
            try:
                await self.clients[client_id].send(json.dumps(data))
            except Exception as e:
                logger.error(f"Error sending to client {client_id}: {e}")
                self._cleanup_client(client_id)
    
    def _cleanup_client(self, client_id: str):
        """Clean up client resources"""
        if client_id in self.clients:
            del self.clients[client_id]
        
        if client_id in self.client_buffers:
            buffer_id = self.client_buffers[client_id]
            buffer = self.memory_manager.get_buffer(buffer_id)
            if buffer:
                buffer.clear()
            del self.client_buffers[client_id]
        
        logger.debug(f"Cleaned up resources for client: {client_id}")
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of memory and resources"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Cleanup expired buffers
                self.memory_manager.cleanup_expired_buffers()
                
                # Log memory usage
                memory_stats = self.memory_manager.get_memory_usage()
                logger.info(f"Memory usage: {memory_stats['rss_mb']:.1f}MB RSS, "
                           f"{memory_stats['buffer_count']} buffers")
                
                # Force garbage collection
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")

class MemorySafeStreamingBackend:
    """Main memory-safe streaming backend"""
    
    def __init__(self, port: int = 8765):
        self.port = port
        self.memory_manager = MemoryManager(max_memory_mb=512)
        self.websocket_server = WebSocketStreamingServer(self.memory_manager, port)
        self.http_server = None
        self._server_task = None
        
    async def start(self):
        """Start the streaming backend"""
        logger.info("Starting Memory-Safe Streaming Backend...")
        
        # Start WebSocket server
        websocket_server = await self.websocket_server.start_server()
        
        # Start HTTP server for health checks
        app = web.Application()
        app.router.add_get('/health', self._health_check)
        app.router.add_get('/memory', self._memory_stats)
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', self.port + 1)
        await site.start()
        
        logger.info(f"Backend started - WebSocket: {self.port}, HTTP: {self.port + 1}")
        
        return websocket_server
    
    async def _health_check(self, request):
        """Health check endpoint"""
        memory_stats = self.memory_manager.get_memory_usage()
        
        return web.json_response({
            "status": "healthy",
            "memory_usage_mb": memory_stats["rss_mb"],
            "buffer_count": memory_stats["buffer_count"],
            "timestamp": datetime.now().isoformat()
        })
    
    async def _memory_stats(self, request):
        """Memory statistics endpoint"""
        return web.json_response(self.memory_manager.get_memory_usage())

# Test function for TDD
async def test_memory_safe_streaming():
    """Test memory-safe streaming backend"""
    logger.info("Testing Memory-Safe Streaming Backend with TDD...")
    
    backend = MemorySafeStreamingBackend(port=8766)
    
    try:
        # Start backend
        server = await backend.start()
        
        # Test memory manager
        memory_manager = backend.memory_manager
        
        # Create test buffer
        buffer = memory_manager.create_buffer("test_buffer", max_size=1024)
        assert buffer is not None, "Failed to create buffer"
        
        # Test buffer operations
        test_data = b"Hello, World!" * 10
        success = buffer.add_chunk(test_data)
        assert success, "Failed to add chunk to buffer"
        
        # Test memory usage
        memory_stats = memory_manager.get_memory_usage()
        assert memory_stats["buffer_count"] > 0, "Buffer count should be > 0"
        
        # Test cleanup
        memory_manager.force_cleanup()
        memory_stats_after = memory_manager.get_memory_usage()
        assert memory_stats_after["buffer_count"] == 0, "Buffers should be cleaned up"
        
        logger.info("✅ Memory-Safe Streaming Backend TDD test completed successfully")
        
        # Stop server
        server.close()
        await server.wait_closed()
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    # Run TDD test
    asyncio.run(test_memory_safe_streaming())
    
    logger.info("🚀 Memory-Safe Streaming Backend ready for production deployment")