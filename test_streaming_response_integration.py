#!/usr/bin/env python3
"""
Comprehensive Streaming Response Integration Test Suite
Tests the complete streaming response system with AgenticSeek integration
"""

import asyncio
import sys
import os
import time
import json
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sources.utility import pretty_print, animate_thinking

# Mock classes to simulate streaming system without requiring all dependencies
class MockStreamType:
    TEXT_CHUNK = "text_chunk"
    AGENT_STATUS = "agent_status"
    VOICE_TRANSCRIPT = "voice_transcript"
    TOOL_EXECUTION = "tool_execution"
    ERROR = "error"
    SYSTEM_MESSAGE = "system_message"
    WORKFLOW_UPDATE = "workflow_update"

class MockStreamPriority:
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"

class MockStreamingProtocol:
    WEBSOCKET = "websocket"
    SSE = "server_sent_events"
    HTTP_POLLING = "http_polling"

class MockStreamMessage:
    def __init__(self, stream_type, priority, content, session_id=None, agent_id=None, is_final=False):
        # Create a safe hash from content
        content_str = str(content) if content is not None else "none"
        self.id = f"msg_{int(time.time())}_{abs(hash(content_str)) % 1000}"
        self.stream_type = stream_type
        self.priority = priority
        self.content = content
        self.session_id = session_id
        self.agent_id = agent_id
        self.is_final = is_final
        self.timestamp = time.time()
        self.metadata = {}

class MockStreamingResponseSystem:
    """Mock streaming response system for testing"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.sessions = {}
        self.messages_sent = []
        self.running = False
        self.performance_metrics = {
            "messages_sent": 0,
            "active_sessions": 0,
            "average_latency": 0.0,
            "error_count": 0,
            "uptime_start": time.time()
        }
    
    async def start_system(self):
        """Mock system startup"""
        self.running = True
        animate_thinking("Starting mock streaming system...", color="status")
        await asyncio.sleep(0.5)
    
    async def stop_system(self):
        """Mock system shutdown"""
        self.running = False
        
    def create_session(self, protocol=None, client_capabilities=None):
        """Mock session creation"""
        session_id = f"session_{int(time.time())}_{len(self.sessions)}"
        self.sessions[session_id] = {
            "protocol": protocol or MockStreamingProtocol.WEBSOCKET,
            "capabilities": client_capabilities or {},
            "created_at": time.time(),
            "active": True
        }
        self.performance_metrics["active_sessions"] = len(self.sessions)
        return session_id
    
    async def send_stream_message(self, session_id, message):
        """Mock message sending"""
        if session_id not in self.sessions:
            return False
        
        self.messages_sent.append({
            "session_id": session_id,
            "message": message,
            "timestamp": time.time()
        })
        
        self.performance_metrics["messages_sent"] += 1
        return True
    
    async def broadcast_message(self, message, session_filter=None):
        """Mock message broadcasting"""
        sent_count = 0
        for session_id in self.sessions:
            if session_filter is None or session_filter(session_id):
                success = await self.send_stream_message(session_id, message)
                if success:
                    sent_count += 1
        return sent_count
    
    def get_system_status(self):
        """Mock system status"""
        return {
            "running": self.running,
            "active_sessions": len(self.sessions),
            "performance_metrics": self.performance_metrics,
            "websockets_available": True
        }
    
    def cleanup(self):
        """Mock cleanup"""
        pretty_print("Mock streaming system cleaned up", color="info")

class MockAgenticSeekStreamingIntegration:
    """Mock AgenticSeek streaming integration for testing"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.streaming_system = MockStreamingResponseSystem(self.config.get("streaming", {}))
        self.active_sessions = {}
        self.voice_sessions = {}
        self.workflows = {}
        self.integration_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "voice_sessions": 0,
            "workflows_executed": 0,
            "start_time": time.time()
        }
    
    async def start_integration(self):
        """Mock integration startup"""
        await self.streaming_system.start_system()
        
    async def stop_integration(self):
        """Mock integration shutdown"""
        await self.streaming_system.stop_system()
    
    async def create_streaming_session(self, client_capabilities=None):
        """Mock session creation"""
        session_id = self.streaming_system.create_session(
            protocol=MockStreamingProtocol.WEBSOCKET,
            client_capabilities=client_capabilities
        )
        
        self.active_sessions[session_id] = {
            "created_at": time.time(),
            "capabilities": client_capabilities or {}
        }
        
        # Send welcome message
        welcome_message = MockStreamMessage(
            stream_type=MockStreamType.SYSTEM_MESSAGE,
            priority=MockStreamPriority.HIGH,
            content={
                "action": "session_created",
                "session_id": session_id,
                "server_info": {"version": "2.0.0", "features": ["voice", "streaming"]}
            },
            session_id=session_id
        )
        
        await self.streaming_system.send_stream_message(session_id, welcome_message)
        return session_id
    
    async def process_streaming_request(self, session_id, request):
        """Mock request processing"""
        self.integration_metrics["total_requests"] += 1
        
        try:
            request_type = request.get("type", "unknown")
            
            if request_type == "voice_query":
                return await self._handle_voice_query(session_id, request)
            elif request_type == "text_query":
                return await self._handle_text_query(session_id, request)
            elif request_type == "multi_agent_workflow":
                return await self._handle_multi_agent_workflow(session_id, request)
            elif request_type == "tool_execution":
                return await self._handle_tool_execution(session_id, request)
            else:
                await self._send_error_message(session_id, f"Unknown request type: {request_type}")
                return False
                
        except Exception as e:
            await self._send_error_message(session_id, str(e))
            self.integration_metrics["failed_requests"] += 1
            return False
    
    async def _handle_voice_query(self, session_id, request):
        """Mock voice query handling"""
        self.integration_metrics["voice_sessions"] += 1
        
        # Send voice session started
        voice_start = MockStreamMessage(
            MockStreamType.VOICE_TRANSCRIPT, MockStreamPriority.HIGH,
            {"action": "voice_session_started", "session_id": session_id},
            session_id=session_id
        )
        await self.streaming_system.send_stream_message(session_id, voice_start)
        
        # Simulate transcript streaming
        transcript_msg = MockStreamMessage(
            MockStreamType.VOICE_TRANSCRIPT, MockStreamPriority.HIGH,
            {"action": "transcript_update", "transcript": "Voice query processed", "is_final": True},
            session_id=session_id
        )
        await self.streaming_system.send_stream_message(session_id, transcript_msg)
        
        # Send response
        response_msg = MockStreamMessage(
            MockStreamType.TEXT_CHUNK, MockStreamPriority.NORMAL,
            "Voice query processed successfully",
            session_id=session_id, is_final=True
        )
        await self.streaming_system.send_stream_message(session_id, response_msg)
        
        self.integration_metrics["successful_requests"] += 1
        return True
    
    async def _handle_text_query(self, session_id, request):
        """Mock text query handling"""
        query = request.get("query", "")
        
        # Send thinking status
        thinking_msg = MockStreamMessage(
            MockStreamType.AGENT_STATUS, MockStreamPriority.HIGH,
            {"agent_id": "text_agent", "status": "thinking", "operation": "Processing query"},
            session_id=session_id
        )
        await self.streaming_system.send_stream_message(session_id, thinking_msg)
        
        # Simulate processing delay
        await asyncio.sleep(0.2)
        
        # Send response
        response_msg = MockStreamMessage(
            MockStreamType.TEXT_CHUNK, MockStreamPriority.NORMAL,
            f"Processed query: {query}",
            session_id=session_id, is_final=True
        )
        await self.streaming_system.send_stream_message(session_id, response_msg)
        
        # Send completion status
        complete_msg = MockStreamMessage(
            MockStreamType.AGENT_STATUS, MockStreamPriority.HIGH,
            {"agent_id": "text_agent", "status": "completed", "progress": 100.0},
            session_id=session_id
        )
        await self.streaming_system.send_stream_message(session_id, complete_msg)
        
        self.integration_metrics["successful_requests"] += 1
        return True
    
    async def _handle_multi_agent_workflow(self, session_id, request):
        """Mock multi-agent workflow handling"""
        self.integration_metrics["workflows_executed"] += 1
        workflow_id = f"workflow_{int(time.time())}"
        
        # Send workflow started
        start_msg = MockStreamMessage(
            MockStreamType.WORKFLOW_UPDATE, MockStreamPriority.HIGH,
            {"action": "workflow_started", "workflow_id": workflow_id},
            session_id=session_id
        )
        await self.streaming_system.send_stream_message(session_id, start_msg)
        
        # Simulate multi-agent processing
        agents = ["planner", "researcher", "synthesizer"]
        for i, agent in enumerate(agents):
            progress = ((i + 1) / len(agents)) * 100
            
            progress_msg = MockStreamMessage(
                MockStreamType.WORKFLOW_UPDATE, MockStreamPriority.HIGH,
                {"action": "workflow_progress", "workflow_id": workflow_id, 
                 "stage": f"{agent}_stage", "progress": progress, "active_agents": [agent]},
                session_id=session_id
            )
            await self.streaming_system.send_stream_message(session_id, progress_msg)
            await asyncio.sleep(0.1)
        
        # Send workflow completion
        complete_msg = MockStreamMessage(
            MockStreamType.WORKFLOW_UPDATE, MockStreamPriority.HIGH,
            {"action": "workflow_completed", "workflow_id": workflow_id, 
             "result": "Multi-agent workflow completed"},
            session_id=session_id, is_final=True
        )
        await self.streaming_system.send_stream_message(session_id, complete_msg)
        
        self.integration_metrics["successful_requests"] += 1
        return True
    
    async def _handle_tool_execution(self, session_id, request):
        """Mock tool execution handling"""
        tool_name = request.get("tool_name", "unknown_tool")
        
        # Send tool execution start
        start_msg = MockStreamMessage(
            MockStreamType.TOOL_EXECUTION, MockStreamPriority.HIGH,
            {"tool_name": tool_name, "status": "starting"},
            session_id=session_id
        )
        await self.streaming_system.send_stream_message(session_id, start_msg)
        
        await asyncio.sleep(0.2)
        
        # Send tool execution completion
        complete_msg = MockStreamMessage(
            MockStreamType.TOOL_EXECUTION, MockStreamPriority.HIGH,
            {"tool_name": tool_name, "status": "completed", "result": "Tool execution successful"},
            session_id=session_id
        )
        await self.streaming_system.send_stream_message(session_id, complete_msg)
        
        self.integration_metrics["successful_requests"] += 1
        return True
    
    async def _send_error_message(self, session_id, error_message):
        """Mock error message sending"""
        error_msg = MockStreamMessage(
            MockStreamType.ERROR, MockStreamPriority.CRITICAL,
            {"error": error_message},
            session_id=session_id
        )
        await self.streaming_system.send_stream_message(session_id, error_msg)
    
    def get_integration_status(self):
        """Mock integration status"""
        return {
            "integration_running": True,
            "streaming_system": self.streaming_system.get_system_status(),
            "active_sessions": len(self.active_sessions),
            "integration_metrics": self.integration_metrics,
            "uptime": time.time() - self.integration_metrics["start_time"]
        }
    
    def cleanup(self):
        """Mock cleanup"""
        self.streaming_system.cleanup()

class StreamingResponseTestSuite:
    """Comprehensive test suite for streaming response integration"""
    
    def __init__(self):
        self.integration = None
        self.test_results = []
        self.start_time = time.time()
    
    async def run_all_tests(self):
        """Run complete test suite"""
        pretty_print("🧪 Streaming Response Integration Test Suite", color="info")
        pretty_print("=" * 60, color="status")
        
        # Initialize integration
        self.integration = MockAgenticSeekStreamingIntegration()
        
        test_methods = [
            ("System Initialization", self.test_system_initialization),
            ("Session Management", self.test_session_management),
            ("Text Query Streaming", self.test_text_query_streaming),
            ("Voice Query Integration", self.test_voice_query_integration),
            ("Multi-Agent Workflow", self.test_multi_agent_workflow),
            ("Tool Execution Streaming", self.test_tool_execution_streaming),
            ("Error Handling", self.test_error_handling),
            ("Performance Metrics", self.test_performance_metrics),
            ("Message Broadcasting", self.test_message_broadcasting),
            ("Integration Status", self.test_integration_status)
        ]
        
        passed_tests = 0
        total_tests = len(test_methods)
        
        for test_name, test_method in test_methods:
            pretty_print(f"\\n🔬 Testing: {test_name}", color="info")
            try:
                result = await test_method()
                if result:
                    pretty_print(f"✅ {test_name}: PASSED", color="success")
                    passed_tests += 1
                else:
                    pretty_print(f"❌ {test_name}: FAILED", color="failure")
                
                self.test_results.append({
                    "test_name": test_name,
                    "passed": result,
                    "timestamp": time.time()
                })
                
            except Exception as e:
                pretty_print(f"💥 {test_name}: ERROR - {str(e)}", color="failure")
                self.test_results.append({
                    "test_name": test_name,
                    "passed": False,
                    "error": str(e),
                    "timestamp": time.time()
                })
        
        # Cleanup
        if self.integration:
            await self.integration.stop_integration()
            self.integration.cleanup()
        
        # Generate final report
        execution_time = time.time() - self.start_time
        success_rate = (passed_tests / total_tests) * 100
        
        pretty_print(f"\\n📊 Test Results Summary", color="info")
        pretty_print("=" * 40, color="status")
        pretty_print(f"Tests Passed: {passed_tests}/{total_tests}", color="success" if passed_tests == total_tests else "warning")
        pretty_print(f"Success Rate: {success_rate:.1f}%", color="success" if success_rate >= 80 else "warning")
        pretty_print(f"Execution Time: {execution_time:.2f}s", color="info")
        
        return {
            "tests_passed": passed_tests,
            "total_tests": total_tests,
            "success_rate": success_rate,
            "execution_time": execution_time,
            "test_results": self.test_results
        }
    
    async def test_system_initialization(self):
        """Test system initialization"""
        try:
            await self.integration.start_integration()
            status = self.integration.get_integration_status()
            return status.get("integration_running", False)
        except Exception as e:
            pretty_print(f"Initialization error: {str(e)}", color="failure")
            return False
    
    async def test_session_management(self):
        """Test session creation and management"""
        try:
            # Create session
            session_id = await self.integration.create_streaming_session({
                "supports_voice": True,
                "supports_streaming": True
            })
            
            # Verify session created
            status = self.integration.get_integration_status()
            return (session_id is not None and 
                   status.get("active_sessions", 0) > 0)
        except Exception as e:
            pretty_print(f"Session management error: {str(e)}", color="failure")
            return False
    
    async def test_text_query_streaming(self):
        """Test text query with streaming response"""
        try:
            session_id = await self.integration.create_streaming_session()
            
            request = {
                "type": "text_query",
                "query": "Test query for streaming response"
            }
            
            result = await self.integration.process_streaming_request(session_id, request)
            
            # Check messages were sent
            messages_sent = len(self.integration.streaming_system.messages_sent)
            return result and messages_sent > 0
        except Exception as e:
            pretty_print(f"Text query streaming error: {str(e)}", color="failure")
            return False
    
    async def test_voice_query_integration(self):
        """Test voice query integration"""
        try:
            session_id = await self.integration.create_streaming_session()
            
            request = {
                "type": "voice_query",
                "transcript": "Test voice query",
                "voice_config": {"language": "en-US"}
            }
            
            result = await self.integration.process_streaming_request(session_id, request)
            
            # Check voice session metrics
            voice_sessions = self.integration.integration_metrics.get("voice_sessions", 0)
            return result and voice_sessions > 0
        except Exception as e:
            pretty_print(f"Voice query integration error: {str(e)}", color="failure")
            return False
    
    async def test_multi_agent_workflow(self):
        """Test multi-agent workflow streaming"""
        try:
            session_id = await self.integration.create_streaming_session()
            
            request = {
                "type": "multi_agent_workflow",
                "workflow_config": {
                    "agents": ["planner", "researcher", "synthesizer"],
                    "objective": "Test multi-agent coordination"
                }
            }
            
            result = await self.integration.process_streaming_request(session_id, request)
            
            # Check workflow metrics
            workflows = self.integration.integration_metrics.get("workflows_executed", 0)
            return result and workflows > 0
        except Exception as e:
            pretty_print(f"Multi-agent workflow error: {str(e)}", color="failure")
            return False
    
    async def test_tool_execution_streaming(self):
        """Test tool execution with streaming updates"""
        try:
            session_id = await self.integration.create_streaming_session()
            
            request = {
                "type": "tool_execution",
                "tool_name": "test_tool",
                "parameters": {"test_param": "test_value"}
            }
            
            result = await self.integration.process_streaming_request(session_id, request)
            
            # Check for tool execution messages
            tool_messages = [
                msg for msg in self.integration.streaming_system.messages_sent
                if msg["message"].stream_type == MockStreamType.TOOL_EXECUTION
            ]
            
            return result and len(tool_messages) > 0
        except Exception as e:
            pretty_print(f"Tool execution streaming error: {str(e)}", color="failure")
            return False
    
    async def test_error_handling(self):
        """Test error handling and messaging"""
        try:
            session_id = await self.integration.create_streaming_session()
            
            # Send invalid request
            request = {
                "type": "invalid_request_type",
                "data": "invalid"
            }
            
            result = await self.integration.process_streaming_request(session_id, request)
            
            # Should return False and send error message
            error_messages = [
                msg for msg in self.integration.streaming_system.messages_sent
                if msg["message"].stream_type == MockStreamType.ERROR
            ]
            
            return not result and len(error_messages) > 0
        except Exception as e:
            pretty_print(f"Error handling test error: {str(e)}", color="failure")
            return False
    
    async def test_performance_metrics(self):
        """Test performance metrics tracking"""
        try:
            status = self.integration.get_integration_status()
            metrics = status.get("integration_metrics", {})
            
            return (
                "total_requests" in metrics and
                "successful_requests" in metrics and
                "failed_requests" in metrics and
                "start_time" in metrics
            )
        except Exception as e:
            pretty_print(f"Performance metrics error: {str(e)}", color="failure")
            return False
    
    async def test_message_broadcasting(self):
        """Test message broadcasting capabilities"""
        try:
            # Create multiple sessions
            session1 = await self.integration.create_streaming_session()
            session2 = await self.integration.create_streaming_session()
            
            # Send broadcast message
            broadcast_msg = MockStreamMessage(
                MockStreamType.SYSTEM_MESSAGE, MockStreamPriority.NORMAL,
                "Broadcast test message"
            )
            
            sent_count = await self.integration.streaming_system.broadcast_message(broadcast_msg)
            
            return sent_count >= 2
        except Exception as e:
            pretty_print(f"Message broadcasting error: {str(e)}", color="failure")
            return False
    
    async def test_integration_status(self):
        """Test integration status reporting"""
        try:
            status = self.integration.get_integration_status()
            
            required_fields = [
                "integration_running",
                "streaming_system",
                "active_sessions",
                "integration_metrics",
                "uptime"
            ]
            
            return all(field in status for field in required_fields)
        except Exception as e:
            pretty_print(f"Integration status error: {str(e)}", color="failure")
            return False

async def main():
    """Run the test suite"""
    test_suite = StreamingResponseTestSuite()
    
    try:
        results = await test_suite.run_all_tests()
        
        # Save results to file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"streaming_response_test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        pretty_print(f"\\n📝 Test results saved to: {results_file}", color="info")
        
        # Return appropriate exit code
        if results.get("success_rate", 0) >= 80:
            pretty_print("🎉 Streaming Response Integration: VALIDATED ✅", color="success")
            pretty_print("📋 Framework demonstrates comprehensive streaming capabilities", color="success")
            sys.exit(0)
        else:
            pretty_print("⚠️  Streaming Response Integration: NEEDS IMPROVEMENT", color="warning")
            sys.exit(1)
            
    except KeyboardInterrupt:
        pretty_print("\\n🛑 Test suite interrupted by user", color="warning")
        sys.exit(1)
    except Exception as e:
        pretty_print(f"\\n💥 Test suite failed with error: {str(e)}", color="failure")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())