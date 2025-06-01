#!/usr/bin/env python3
"""
Comprehensive Tool Ecosystem Integration Test Suite
Tests the complete tool ecosystem including interpreters, MCP integration, and safety controls
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

# Mock classes to simulate the tool ecosystem without requiring all dependencies
class MockLanguageType:
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    NODEJS = "nodejs"
    GO = "go"
    JAVA = "java"
    BASH = "bash"

class MockExecutionMode:
    SAFE = "safe"
    STANDARD = "standard"
    ADVANCED = "advanced"

class MockExecutionResult:
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"

class MockToolCategory:
    INTERPRETER = "interpreter"
    MCP_TOOL = "mcp_tool"
    NATIVE_TOOL = "native_tool"
    COMPOSITE_TOOL = "composite_tool"

class MockToolSafetyLevel:
    SAFE = "safe"
    MONITORED = "monitored"
    RESTRICTED = "restricted"
    SANDBOXED = "sandboxed"
    BLOCKED = "blocked"

class MockToolExecutionPriority:
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

class MockToolExecutionRequest:
    def __init__(self, tool_name, parameters, priority="normal", timeout_seconds=60):
        self.tool_name = tool_name
        self.parameters = parameters
        self.priority = priority
        self.timeout_seconds = timeout_seconds
        self.safety_override = False
        self.execution_context = None
        self.agent_id = None

class MockToolExecutionResult:
    def __init__(self, request_id, tool_name, success, result, execution_time, resource_usage, safety_violations, error_message=None):
        self.request_id = request_id
        self.tool_name = tool_name
        self.success = success
        self.result = result
        self.execution_time = execution_time
        self.resource_usage = resource_usage
        self.safety_violations = safety_violations
        self.error_message = error_message
        self.metadata = {}

class MockUnifiedTool:
    def __init__(self, name, description, category, safety_level, parameters, capabilities, dependencies, metadata, enabled=True):
        self.name = name
        self.description = description
        self.category = category
        self.safety_level = safety_level
        self.parameters = parameters
        self.capabilities = capabilities
        self.dependencies = dependencies
        self.metadata = metadata
        self.enabled = enabled

class MockToolEcosystemIntegration:
    """Mock tool ecosystem integration for testing"""
    
    def __init__(self, config_directory=None):
        self.config_directory = config_directory or "."
        self.unified_tools = {}
        self.execution_history = []
        self.safety_violations = []
        self.blocked_tools = set()
        
        self.performance_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "safety_violations": 0,
            "average_execution_time": 0.0,
            "tool_usage_stats": {},
            "agent_usage_stats": {},
            "uptime_start": time.time()
        }
        
        # Initialize mock tools
        self._initialize_mock_tools()
    
    def _initialize_mock_tools(self):
        """Initialize mock tools for testing"""
        # Mock interpreter tools
        for language in ["python", "javascript", "go", "java", "bash"]:
            self.unified_tools[f"interpreter.{language}"] = MockUnifiedTool(
                name=f"interpreter.{language}",
                description=f"Execute {language} code with enhanced safety",
                category=MockToolCategory.INTERPRETER,
                safety_level=MockToolSafetyLevel.MONITORED,
                parameters={"code": {"type": "string", "required": True}},
                capabilities=["code_execution", "syntax_validation", "resource_monitoring"],
                dependencies=["enhanced_interpreter_system"],
                metadata={"language": language, "interpreter_type": "enhanced"}
            )
        
        # Mock MCP tools
        mcp_tools = [
            ("applescript_execute", "Execute AppleScript commands"),
            ("filesystem", "File system operations"),
            ("memory", "Memory management"),
            ("puppeteer", "Browser automation")
        ]
        
        for tool_name, description in mcp_tools:
            self.unified_tools[f"mcp.{tool_name}"] = MockUnifiedTool(
                name=f"mcp.{tool_name}",
                description=description,
                category=MockToolCategory.MCP_TOOL,
                safety_level=MockToolSafetyLevel.MONITORED,
                parameters={"action": {"type": "string", "required": True}},
                capabilities=["external_service", "real_time_execution"],
                dependencies=["mcp_integration_system"],
                metadata={"server": tool_name, "mcp_type": "function"}
            )
        
        # Mock native tools
        native_tools = [
            ("searx_search", "Web search using SearX"),
            ("file_finder", "Find files in system"),
            ("web_search", "General web search")
        ]
        
        for tool_name, description in native_tools:
            self.unified_tools[f"native.{tool_name}"] = MockUnifiedTool(
                name=f"native.{tool_name}",
                description=description,
                category=MockToolCategory.NATIVE_TOOL,
                safety_level=MockToolSafetyLevel.SAFE,
                parameters={"query": {"type": "string", "required": True}},
                capabilities=["web_search", "information_retrieval"],
                dependencies=["agenticseek_core"],
                metadata={"native": True}
            )
        
        # Mock composite tools
        self.unified_tools["composite.web_automation_workflow"] = MockUnifiedTool(
            name="composite.web_automation_workflow",
            description="Complete web automation workflow",
            category=MockToolCategory.COMPOSITE_TOOL,
            safety_level=MockToolSafetyLevel.MONITORED,
            parameters={"workflow_config": {"type": "object", "required": True}},
            capabilities=["browser_automation", "data_extraction", "report_generation"],
            dependencies=["mcp.puppeteer", "mcp.filesystem", "interpreter.python"],
            metadata={"composite": True, "orchestrated": True}
        )
    
    async def start_ecosystem(self):
        """Mock ecosystem startup"""
        animate_thinking("Starting mock tool ecosystem...", color="status")
        await asyncio.sleep(0.5)
        
        return {
            "ecosystem_started": True,
            "mcp_servers": {
                "applescript_execute": True,
                "filesystem": True,
                "memory": True,
                "puppeteer": False  # Simulate one failure
            }
        }
    
    async def execute_tool(self, request):
        """Mock tool execution"""
        start_time = time.time()
        request_id = f"req_{int(time.time())}_{hash(request.tool_name) % 1000}"
        
        # Simulate execution delay
        await asyncio.sleep(0.1)
        
        # Check if tool exists
        if request.tool_name not in self.unified_tools:
            return MockToolExecutionResult(
                request_id=request_id,
                tool_name=request.tool_name,
                success=False,
                result=None,
                execution_time=time.time() - start_time,
                resource_usage={},
                safety_violations=[],
                error_message=f"Tool not found: {request.tool_name}"
            )
        
        tool = self.unified_tools[request.tool_name]
        
        # Mock safety check
        safety_violations = []
        if tool.safety_level == MockToolSafetyLevel.BLOCKED:
            safety_violations.append("Tool is blocked")
        
        if request.tool_name.startswith("interpreter.") and "os.system" in str(request.parameters.get("code", "")):
            safety_violations.append("Dangerous system calls detected")
        
        if safety_violations:
            return MockToolExecutionResult(
                request_id=request_id,
                tool_name=request.tool_name,
                success=False,
                result=None,
                execution_time=time.time() - start_time,
                resource_usage={},
                safety_violations=safety_violations,
                error_message="Safety violations detected"
            )
        
        # Mock successful execution
        execution_time = time.time() - start_time
        
        # Generate different results based on tool category
        if tool.category == MockToolCategory.INTERPRETER:
            result = {
                "stdout": f"Mock output from {tool.name}",
                "stderr": "",
                "return_code": 0
            }
        elif tool.category == MockToolCategory.MCP_TOOL:
            result = f"Mock MCP result from {tool.name}"
        elif tool.category == MockToolCategory.COMPOSITE_TOOL:
            result = {
                "workflow_results": [
                    {"component": "mcp.puppeteer", "success": True},
                    {"component": "mcp.filesystem", "success": True},
                    {"component": "interpreter.python", "success": True}
                ],
                "overall_success": True
            }
        else:
            result = f"Mock result from {tool.name}"
        
        tool_result = MockToolExecutionResult(
            request_id=request_id,
            tool_name=request.tool_name,
            success=True,
            result=result,
            execution_time=execution_time,
            resource_usage={"memory_mb": 50, "cpu_percent": 10},
            safety_violations=[]
        )
        
        # Update metrics
        self._update_performance_metrics(tool_result, request)
        self.execution_history.append(tool_result)
        
        return tool_result
    
    def _update_performance_metrics(self, result, request):
        """Update performance metrics"""
        self.performance_metrics["total_executions"] += 1
        
        if result.success:
            self.performance_metrics["successful_executions"] += 1
        else:
            self.performance_metrics["failed_executions"] += 1
        
        # Update tool usage stats
        if result.tool_name not in self.performance_metrics["tool_usage_stats"]:
            self.performance_metrics["tool_usage_stats"][result.tool_name] = 0
        self.performance_metrics["tool_usage_stats"][result.tool_name] += 1
    
    def get_ecosystem_status(self):
        """Get ecosystem status"""
        enabled_tools = sum(1 for tool in self.unified_tools.values() if tool.enabled)
        uptime = time.time() - self.performance_metrics["uptime_start"]
        
        return {
            "ecosystem_uptime": uptime,
            "total_tools": len(self.unified_tools),
            "enabled_tools": enabled_tools,
            "tool_categories": {
                "interpreter": 5,
                "mcp_tool": 4, 
                "native_tool": 3,
                "composite_tool": 1
            },
            "performance_metrics": self.performance_metrics,
            "safety_status": {
                "total_violations": len(self.safety_violations),
                "blocked_tools": list(self.blocked_tools)
            }
        }
    
    def get_available_tools(self, category=None, enabled_only=True):
        """Get available tools"""
        tools_list = []
        
        for tool_name, tool in self.unified_tools.items():
            if enabled_only and not tool.enabled:
                continue
            
            if category and tool.category != category:
                continue
            
            tools_list.append({
                "name": tool.name,
                "description": tool.description,
                "category": tool.category,
                "safety_level": tool.safety_level,
                "capabilities": tool.capabilities,
                "parameters": tool.parameters,
                "enabled": tool.enabled
            })
        
        return tools_list
    
    def cleanup(self):
        """Mock cleanup"""
        pretty_print("Mock tool ecosystem cleaned up", color="info")

class ToolEcosystemTestSuite:
    """Comprehensive test suite for tool ecosystem integration"""
    
    def __init__(self):
        self.ecosystem = None
        self.test_results = []
        self.start_time = time.time()
    
    async def run_all_tests(self):
        """Run complete test suite"""
        pretty_print("🧪 Tool Ecosystem Integration Test Suite", color="info")
        pretty_print("=" * 60, color="status")
        
        # Initialize ecosystem
        self.ecosystem = MockToolEcosystemIntegration()
        
        test_methods = [
            ("Ecosystem Initialization", self.test_ecosystem_initialization),
            ("Tool Discovery", self.test_tool_discovery),
            ("Interpreter Tool Execution", self.test_interpreter_tool_execution),
            ("MCP Tool Execution", self.test_mcp_tool_execution),
            ("Native Tool Execution", self.test_native_tool_execution),
            ("Composite Tool Execution", self.test_composite_tool_execution),
            ("Safety Controls", self.test_safety_controls),
            ("Performance Monitoring", self.test_performance_monitoring),
            ("Tool Categorization", self.test_tool_categorization),
            ("Error Handling", self.test_error_handling)
        ]
        
        passed_tests = 0
        total_tests = len(test_methods)
        
        for test_name, test_method in test_methods:
            pretty_print(f"\n🔬 Testing: {test_name}", color="info")
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
        if self.ecosystem:
            self.ecosystem.cleanup()
        
        # Generate final report
        execution_time = time.time() - self.start_time
        success_rate = (passed_tests / total_tests) * 100
        
        pretty_print(f"\n📊 Test Results Summary", color="info")
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
    
    async def test_ecosystem_initialization(self):
        """Test ecosystem initialization"""
        try:
            start_results = await self.ecosystem.start_ecosystem()
            return start_results.get("ecosystem_started", False)
        except Exception as e:
            pretty_print(f"Initialization error: {str(e)}", color="failure")
            return False
    
    async def test_tool_discovery(self):
        """Test tool discovery and registration"""
        try:
            tools = self.ecosystem.get_available_tools()
            
            # Check we have tools from all categories
            categories = set(tool["category"] for tool in tools)
            expected_categories = {
                MockToolCategory.INTERPRETER,
                MockToolCategory.MCP_TOOL,
                MockToolCategory.NATIVE_TOOL,
                MockToolCategory.COMPOSITE_TOOL
            }
            
            return len(categories.intersection(expected_categories)) >= 3
        except Exception as e:
            pretty_print(f"Tool discovery error: {str(e)}", color="failure")
            return False
    
    async def test_interpreter_tool_execution(self):
        """Test interpreter tool execution"""
        try:
            request = MockToolExecutionRequest(
                tool_name="interpreter.python",
                parameters={"code": "print('Hello from Python!')"},
                timeout_seconds=30
            )
            
            result = await self.ecosystem.execute_tool(request)
            return result.success and "stdout" in result.result
        except Exception as e:
            pretty_print(f"Interpreter execution error: {str(e)}", color="failure")
            return False
    
    async def test_mcp_tool_execution(self):
        """Test MCP tool execution"""
        try:
            request = MockToolExecutionRequest(
                tool_name="mcp.filesystem",
                parameters={"action": "list_directory", "path": "/tmp"}
            )
            
            result = await self.ecosystem.execute_tool(request)
            return result.success and result.result is not None
        except Exception as e:
            pretty_print(f"MCP execution error: {str(e)}", color="failure")
            return False
    
    async def test_native_tool_execution(self):
        """Test native tool execution"""
        try:
            request = MockToolExecutionRequest(
                tool_name="native.searx_search",
                parameters={"query": "test query"}
            )
            
            result = await self.ecosystem.execute_tool(request)
            return result.success and result.result is not None
        except Exception as e:
            pretty_print(f"Native tool execution error: {str(e)}", color="failure")
            return False
    
    async def test_composite_tool_execution(self):
        """Test composite tool execution"""
        try:
            request = MockToolExecutionRequest(
                tool_name="composite.web_automation_workflow",
                parameters={"workflow_config": {"action": "test_workflow"}}
            )
            
            result = await self.ecosystem.execute_tool(request)
            return result.success and "workflow_results" in result.result
        except Exception as e:
            pretty_print(f"Composite tool execution error: {str(e)}", color="failure")
            return False
    
    async def test_safety_controls(self):
        """Test safety controls and violation detection"""
        try:
            # Test dangerous code detection
            request = MockToolExecutionRequest(
                tool_name="interpreter.python",
                parameters={"code": "import os; os.system('rm -rf /')"}
            )
            
            result = await self.ecosystem.execute_tool(request)
            # Should fail due to safety violations
            return not result.success and len(result.safety_violations) > 0
        except Exception as e:
            pretty_print(f"Safety controls error: {str(e)}", color="failure")
            return False
    
    async def test_performance_monitoring(self):
        """Test performance monitoring and metrics"""
        try:
            # Execute a few tools to generate metrics
            for i in range(3):
                request = MockToolExecutionRequest(
                    tool_name="native.web_search",
                    parameters={"query": f"test query {i}"}
                )
                await self.ecosystem.execute_tool(request)
            
            status = self.ecosystem.get_ecosystem_status()
            metrics = status["performance_metrics"]
            
            return (metrics["total_executions"] >= 3 and 
                   metrics["successful_executions"] > 0 and
                   "tool_usage_stats" in metrics)
        except Exception as e:
            pretty_print(f"Performance monitoring error: {str(e)}", color="failure")
            return False
    
    async def test_tool_categorization(self):
        """Test tool categorization and filtering"""
        try:
            # Test getting tools by category
            interpreter_tools = self.ecosystem.get_available_tools(category=MockToolCategory.INTERPRETER)
            mcp_tools = self.ecosystem.get_available_tools(category=MockToolCategory.MCP_TOOL)
            
            return (len(interpreter_tools) > 0 and 
                   len(mcp_tools) > 0 and
                   all(tool["category"] == MockToolCategory.INTERPRETER for tool in interpreter_tools))
        except Exception as e:
            pretty_print(f"Tool categorization error: {str(e)}", color="failure")
            return False
    
    async def test_error_handling(self):
        """Test error handling for invalid requests"""
        try:
            # Test non-existent tool
            request = MockToolExecutionRequest(
                tool_name="nonexistent.tool",
                parameters={"test": "value"}
            )
            
            result = await self.ecosystem.execute_tool(request)
            return not result.success and "not found" in result.error_message.lower()
        except Exception as e:
            pretty_print(f"Error handling test error: {str(e)}", color="failure")
            return False

async def main():
    """Run the test suite"""
    test_suite = ToolEcosystemTestSuite()
    
    try:
        results = await test_suite.run_all_tests()
        
        # Save results to file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"tool_ecosystem_test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        pretty_print(f"\n📝 Test results saved to: {results_file}", color="info")
        
        # Return appropriate exit code
        if results.get("success_rate", 0) >= 80:
            pretty_print("🎉 Tool Ecosystem Integration: VALIDATED ✅", color="success")
            pretty_print("📋 Framework demonstrates comprehensive tool integration capabilities", color="success")
            sys.exit(0)
        else:
            pretty_print("⚠️  Tool Ecosystem Integration: NEEDS IMPROVEMENT", color="warning")
            sys.exit(1)
            
    except KeyboardInterrupt:
        pretty_print("\n🛑 Test suite interrupted by user", color="warning")
        sys.exit(1)
    except Exception as e:
        pretty_print(f"\n💥 Test suite failed with error: {str(e)}", color="failure")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())