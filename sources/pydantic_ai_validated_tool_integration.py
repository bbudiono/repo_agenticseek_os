#!/usr/bin/env python3
"""
Pydantic AI Validated Tool Integration Framework Implementation

Purpose: Comprehensive tool integration framework with type-safe validation and structured outputs
Issues & Complexity Summary: Advanced tool ecosystem with validation, output structuring, and capability-based access
Key Complexity Drivers:
  - Logic Scope (Est. LoC): ~1300
  - Core Algorithm Complexity: High
  - Dependencies: 6 New, 3 Mod
  - State Management Complexity: High
  - Novelty/Uncertainty Factor: Medium
AI Pre-Task Self-Assessment (Est. Solution Difficulty %): 80%
Problem Estimate (Inherent Problem Difficulty %): 85%
Initial Code Complexity Estimate %: 80%
Final Code Complexity (Actual %): TBD
Overall Result Score (Success & Quality %): TBD
Key Variances/Learnings: TBD
Last Updated: 2025-01-06
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Set, Tuple, TypeVar, Generic, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import traceback
from pathlib import Path
import inspect

# Handle pydantic-ai availability with fallback
try:
    import pydantic_ai
    from pydantic_ai import Agent, RunContext
    from pydantic import BaseModel, Field, validator, ValidationError
    PYDANTIC_AI_AVAILABLE = True
    print("Pydantic AI available - using full type safety features")
except ImportError:
    # Fallback implementations
    PYDANTIC_AI_AVAILABLE = False
    print("Pydantic AI not available - using fallback implementations")
    
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    def Field(default=None, **kwargs):
        return default
    
    def validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    class ValidationError(Exception):
        pass

# Import existing MLACS components
try:
    from sources.pydantic_ai_core_integration import (
        TypeSafeAgent, AgentConfiguration, AgentSpecialization, 
        AgentTier, AgentCapability, PydanticAIIntegrationDependencies
    )
    from sources.pydantic_ai_communication_models import (
        TypeSafeCommunicationManager, MessageType, CommunicationProtocol
    )
    from sources.pydantic_ai_tier_aware_agent_factory import (
        TierAwareAgentFactory, AgentCreationRequest, CreatedAgentRecord
    )
    CORE_INTEGRATION_AVAILABLE = True
except ImportError:
    CORE_INTEGRATION_AVAILABLE = False
    # Fallback types
    class AgentSpecialization(Enum):
        COORDINATOR = "coordinator"
        RESEARCH = "research"
        CODE = "code"
        BROWSER = "browser"
        FILE = "file"
        MCP = "mcp"
        CASUAL = "casual"
        PLANNER = "planner"
    
    class AgentTier(Enum):
        FREE = "free"
        PRO = "pro"
        ENTERPRISE = "enterprise"
    
    class AgentCapability(Enum):
        BASIC_REASONING = "basic_reasoning"
        ADVANCED_REASONING = "advanced_reasoning"
        CODE_GENERATION = "code_generation"
        WEB_BROWSING = "web_browsing"
        FILE_OPERATIONS = "file_operations"
        MCP_INTEGRATION = "mcp_integration"
        PLANNING = "planning"
        COORDINATION = "coordination"
        MEMORY_ACCESS = "memory_access"
        TOOL_USAGE = "tool_usage"

# Enhanced imports
try:
    from sources.logger import Logger
    LOGGER_AVAILABLE = True
except ImportError:
    LOGGER_AVAILABLE = False
    class Logger:
        def __init__(self, filename):
            self.filename = filename
        def info(self, msg): print(f"INFO: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")

# Timer decorator for performance tracking
def timer_decorator(func):
    """Decorator to track function execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            print(f"⏱️  {func.__name__} executed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ {func.__name__} failed in {execution_time:.3f}s: {e}")
            raise
    return wrapper

async def async_timer_decorator(func):
    """Async decorator to track function execution time"""
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            print(f"⏱️  {func.__name__} executed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ {func.__name__} failed in {execution_time:.3f}s: {e}")
            raise
    return wrapper

# Tool Integration Framework Models
class ToolCategory(Enum):
    """Categories of tools available in the ecosystem"""
    INTERPRETER = "interpreter"
    WEB_AUTOMATION = "web_automation"
    FILE_SYSTEM = "file_system"
    COMMUNICATION = "communication"
    DATA_PROCESSING = "data_processing"
    SEARCH = "search"
    MCP_SERVICE = "mcp_service"
    SAFETY = "safety"
    CUSTOM = "custom"

class ToolAccessLevel(Enum):
    """Access levels for tool usage based on agent tier"""
    PUBLIC = "public"        # Available to all tiers
    RESTRICTED = "restricted"  # PRO and ENTERPRISE only
    PREMIUM = "premium"      # ENTERPRISE only
    ADMIN = "admin"          # Special administrative access

class ToolValidationLevel(Enum):
    """Validation strictness levels for tool outputs"""
    BASIC = "basic"          # Basic type checking
    STANDARD = "standard"    # Type checking + business rules
    STRICT = "strict"        # Full validation + security checks
    ENTERPRISE = "enterprise"  # All validations + audit logging

class ToolExecutionStatus(Enum):
    """Tool execution status tracking"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    VALIDATION_FAILED = "validation_failed"

# Core Tool Models
if PYDANTIC_AI_AVAILABLE:
    class ToolParameter(BaseModel):
        """Individual tool parameter specification"""
        name: str
        type: str  # Python type as string
        required: bool = Field(True)
        default: Any = Field(None)
        description: str = Field("")
        validation_rules: List[str] = Field(default_factory=list)
        tier_restrictions: List[AgentTier] = Field(default_factory=list)
        
        class Config:
            use_enum_values = True
    
    class ToolOutput(BaseModel):
        """Structured tool output with validation"""
        tool_id: str
        execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        status: ToolExecutionStatus
        result: Any = Field(None)
        error_message: Optional[str] = None
        execution_time_ms: float = Field(0.0)
        validation_score: float = Field(1.0, ge=0.0, le=1.0)
        metadata: Dict[str, Any] = Field(default_factory=dict)
        created_at: datetime = Field(default_factory=datetime.now)
        
        class Config:
            use_enum_values = True
    
    class ToolCapability(BaseModel):
        """Tool capability specification and requirements"""
        tool_id: str
        name: str
        category: ToolCategory
        access_level: ToolAccessLevel
        required_capabilities: List[AgentCapability] = Field(default_factory=list)
        parameters: List[ToolParameter] = Field(default_factory=list)
        output_schema: Dict[str, Any] = Field(default_factory=dict)
        validation_level: ToolValidationLevel = Field(ToolValidationLevel.STANDARD)
        timeout_seconds: int = Field(30, ge=1, le=3600)
        description: str = Field("")
        version: str = Field("1.0.0")
        
        class Config:
            use_enum_values = True
    
    class ToolExecutionRequest(BaseModel):
        """Tool execution request with validation"""
        request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        tool_id: str
        agent_id: str
        parameters: Dict[str, Any] = Field(default_factory=dict)
        validation_level: ToolValidationLevel = Field(ToolValidationLevel.STANDARD)
        timeout_override: Optional[int] = None
        metadata: Dict[str, Any] = Field(default_factory=dict)
        created_at: datetime = Field(default_factory=datetime.now)
        
        class Config:
            use_enum_values = True
    
    class ToolExecutionResult(BaseModel):
        """Complete tool execution result with analytics"""
        request: ToolExecutionRequest
        output: ToolOutput
        validation_details: Dict[str, Any] = Field(default_factory=dict)
        performance_metrics: Dict[str, float] = Field(default_factory=dict)
        security_audit: Dict[str, Any] = Field(default_factory=dict)
        completed_at: datetime = Field(default_factory=datetime.now)
        
        class Config:
            use_enum_values = True

else:
    # Fallback implementations for non-Pydantic AI environments
    class ToolParameter:
        def __init__(self, name, type, required=True, default=None, 
                     description="", validation_rules=None, tier_restrictions=None):
            self.name = name
            self.type = type
            self.required = required
            self.default = default
            self.description = description
            self.validation_rules = validation_rules or []
            self.tier_restrictions = tier_restrictions or []
    
    class ToolOutput:
        def __init__(self, tool_id, status, **kwargs):
            self.tool_id = tool_id
            self.execution_id = str(uuid.uuid4())
            self.status = status
            self.result = kwargs.get('result')
            self.error_message = kwargs.get('error_message')
            self.execution_time_ms = kwargs.get('execution_time_ms', 0.0)
            self.validation_score = kwargs.get('validation_score', 1.0)
            self.metadata = kwargs.get('metadata', {})
            self.created_at = datetime.now()
    
    class ToolCapability:
        def __init__(self, tool_id, name, category, access_level, **kwargs):
            self.tool_id = tool_id
            self.name = name
            self.category = category
            self.access_level = access_level
            self.required_capabilities = kwargs.get('required_capabilities', [])
            self.parameters = kwargs.get('parameters', [])
            self.output_schema = kwargs.get('output_schema', {})
            self.validation_level = kwargs.get('validation_level', ToolValidationLevel.STANDARD)
            self.timeout_seconds = kwargs.get('timeout_seconds', 30)
            self.description = kwargs.get('description', '')
            self.version = kwargs.get('version', '1.0.0')
    
    class ToolExecutionRequest:
        def __init__(self, tool_id, agent_id, **kwargs):
            self.request_id = str(uuid.uuid4())
            self.tool_id = tool_id
            self.agent_id = agent_id
            self.parameters = kwargs.get('parameters', {})
            self.validation_level = kwargs.get('validation_level', ToolValidationLevel.STANDARD)
            self.timeout_override = kwargs.get('timeout_override')
            self.metadata = kwargs.get('metadata', {})
            self.created_at = datetime.now()
    
    class ToolExecutionResult:
        def __init__(self, request, output, **kwargs):
            self.request = request
            self.output = output
            self.validation_details = kwargs.get('validation_details', {})
            self.performance_metrics = kwargs.get('performance_metrics', {})
            self.security_audit = kwargs.get('security_audit', {})
            self.completed_at = datetime.now()

# Core Tool Integration Framework Implementation
class ValidatedToolIntegrationFramework:
    """
    Comprehensive tool integration framework with type-safe validation and structured outputs
    
    Features:
    - Type-safe tool registration and capability validation
    - Tier-based access control and capability restrictions
    - Structured output validation with security auditing
    - Performance monitoring and optimization
    - Integration with existing MLACS components
    - Universal compatibility with fallback support
    """
    
    def __init__(self):
        """Initialize the Validated Tool Integration Framework"""
        self.logger = Logger("tool_integration_framework.log") if LOGGER_AVAILABLE else Logger("fallback.log")
        self.framework_id = str(uuid.uuid4())
        self.version = "1.0.0"
        
        # Core framework state
        self.registered_tools: Dict[str, ToolCapability] = {}
        self.tool_executors: Dict[str, Callable] = {}
        self.execution_history: Dict[str, ToolExecutionResult] = {}
        self.active_executions: Dict[str, ToolExecutionRequest] = {}
        
        # Validation and security
        self.validation_rules: Dict[str, List[Callable]] = {}
        self.security_policies: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.performance_metrics: Dict[str, Any] = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "tool_usage_stats": {},
            "tier_usage_distribution": {tier.value: 0 for tier in AgentTier},
            "last_reset": datetime.now()
        }
        
        # Integration points
        self.agent_factory = None
        self.communication_manager = None
        
        # Initialize default tools
        self._initialize_core_tools()
        
        self.logger.info(f"Validated Tool Integration Framework initialized - ID: {self.framework_id}")
    
    def _initialize_core_tools(self) -> None:
        """Initialize core tools available in the framework"""
        
        # Python Interpreter Tool
        python_interpreter = ToolCapability(
            tool_id="python_interpreter",
            name="Python Code Interpreter",
            category=ToolCategory.INTERPRETER,
            access_level=ToolAccessLevel.RESTRICTED,
            required_capabilities=[AgentCapability.CODE_GENERATION],
            parameters=[
                ToolParameter(
                    name="code",
                    type="str",
                    description="Python code to execute",
                    validation_rules=["no_dangerous_imports", "syntax_check"]
                ),
                ToolParameter(
                    name="timeout",
                    type="int",
                    required=False,
                    default=30,
                    description="Execution timeout in seconds"
                )
            ],
            output_schema={
                "type": "object",
                "properties": {
                    "output": {"type": "string"},
                    "error": {"type": "string"},
                    "execution_time": {"type": "number"}
                }
            },
            validation_level=ToolValidationLevel.STRICT,
            timeout_seconds=60,
            description="Execute Python code with safety restrictions"
        )
        
        # Web Search Tool
        web_search = ToolCapability(
            tool_id="web_search",
            name="Web Search Engine",
            category=ToolCategory.SEARCH,
            access_level=ToolAccessLevel.PUBLIC,
            required_capabilities=[AgentCapability.WEB_BROWSING],
            parameters=[
                ToolParameter(
                    name="query",
                    type="str",
                    description="Search query string",
                    validation_rules=["query_safety_check"]
                ),
                ToolParameter(
                    name="max_results",
                    type="int",
                    required=False,
                    default=10,
                    description="Maximum number of results"
                )
            ],
            output_schema={
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "url": {"type": "string"},
                                "snippet": {"type": "string"}
                            }
                        }
                    }
                }
            },
            validation_level=ToolValidationLevel.STANDARD,
            timeout_seconds=30,
            description="Search the web for information"
        )
        
        # File Operations Tool
        file_operations = ToolCapability(
            tool_id="file_operations",
            name="File System Operations",
            category=ToolCategory.FILE_SYSTEM,
            access_level=ToolAccessLevel.RESTRICTED,
            required_capabilities=[AgentCapability.FILE_OPERATIONS],
            parameters=[
                ToolParameter(
                    name="operation",
                    type="str",
                    description="File operation: read, write, list, delete",
                    validation_rules=["operation_whitelist"]
                ),
                ToolParameter(
                    name="path",
                    type="str",
                    description="File or directory path",
                    validation_rules=["path_safety_check"]
                ),
                ToolParameter(
                    name="content",
                    type="str",
                    required=False,
                    description="Content for write operations"
                )
            ],
            output_schema={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "result": {"type": ["string", "array"]},
                    "message": {"type": "string"}
                }
            },
            validation_level=ToolValidationLevel.STRICT,
            timeout_seconds=45,
            description="Perform safe file system operations"
        )
        
        # Browser Automation Tool
        browser_automation = ToolCapability(
            tool_id="browser_automation",
            name="Web Browser Automation",
            category=ToolCategory.WEB_AUTOMATION,
            access_level=ToolAccessLevel.PREMIUM,
            required_capabilities=[AgentCapability.WEB_BROWSING, AgentCapability.TOOL_USAGE],
            parameters=[
                ToolParameter(
                    name="action",
                    type="str",
                    description="Browser action: navigate, click, type, screenshot",
                    validation_rules=["action_whitelist"]
                ),
                ToolParameter(
                    name="target",
                    type="str",
                    description="Target URL or element selector",
                    validation_rules=["url_safety_check"]
                ),
                ToolParameter(
                    name="data",
                    type="str",
                    required=False,
                    description="Data for type operations"
                )
            ],
            output_schema={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "result": {"type": "string"},
                    "screenshot": {"type": "string"}
                }
            },
            validation_level=ToolValidationLevel.ENTERPRISE,
            timeout_seconds=120,
            description="Automate web browser interactions"
        )
        
        # MCP Integration Tool
        mcp_integration = ToolCapability(
            tool_id="mcp_integration",
            name="Meta-Cognitive Primitive Integration",
            category=ToolCategory.MCP_SERVICE,
            access_level=ToolAccessLevel.PREMIUM,
            required_capabilities=[AgentCapability.MCP_INTEGRATION, AgentCapability.ADVANCED_REASONING],
            parameters=[
                ToolParameter(
                    name="service",
                    type="str",
                    description="MCP service to invoke",
                    validation_rules=["service_whitelist"],
                    tier_restrictions=[AgentTier.ENTERPRISE]
                ),
                ToolParameter(
                    name="method",
                    type="str",
                    description="Service method to call"
                ),
                ToolParameter(
                    name="params",
                    type="dict",
                    required=False,
                    default={},
                    description="Method parameters"
                )
            ],
            output_schema={
                "type": "object",
                "properties": {
                    "service_response": {"type": "object"},
                    "processing_time": {"type": "number"},
                    "service_status": {"type": "string"}
                }
            },
            validation_level=ToolValidationLevel.ENTERPRISE,
            timeout_seconds=300,
            description="Integrate with Meta-Cognitive Primitive services"
        )
        
        # Register core tools
        core_tools = [
            python_interpreter,
            web_search,
            file_operations,
            browser_automation,
            mcp_integration
        ]
        
        for tool in core_tools:
            self.register_tool(tool, self._create_tool_executor(tool))
    
    def _create_tool_executor(self, tool: ToolCapability) -> Callable:
        """Create a mock executor for core tools"""
        async def tool_executor(parameters: Dict[str, Any]) -> Dict[str, Any]:
            # Mock implementation - in production, this would call actual tools
            await asyncio.sleep(0.01)  # Simulate processing time
            
            if tool.tool_id == "python_interpreter":
                return {
                    "output": f"Executed: {parameters.get('code', '')}",
                    "error": None,
                    "execution_time": 0.01
                }
            elif tool.tool_id == "web_search":
                return {
                    "results": [
                        {
                            "title": f"Search result for: {parameters.get('query', '')}",
                            "url": "https://example.com",
                            "snippet": "Mock search result snippet"
                        }
                    ]
                }
            elif tool.tool_id == "file_operations":
                return {
                    "success": True,
                    "result": f"File operation {parameters.get('operation', '')} completed",
                    "message": "Operation successful"
                }
            elif tool.tool_id == "browser_automation":
                return {
                    "success": True,
                    "result": f"Browser action {parameters.get('action', '')} completed",
                    "screenshot": "base64_encoded_screenshot_data"
                }
            elif tool.tool_id == "mcp_integration":
                return {
                    "service_response": {"status": "success", "data": "mock_response"},
                    "processing_time": 0.1,
                    "service_status": "operational"
                }
            else:
                return {"result": "Mock tool execution completed"}
        
        return tool_executor
    
    @timer_decorator
    def register_tool(self, tool: ToolCapability, executor: Callable) -> bool:
        """Register a new tool with the framework"""
        try:
            # Validate tool capability
            if not tool.tool_id or not tool.name:
                self.logger.error(f"Tool registration failed: missing required fields")
                return False
            
            # Check for duplicate registration
            if tool.tool_id in self.registered_tools:
                self.logger.warning(f"Tool {tool.tool_id} already registered, updating")
            
            # Register tool and executor
            self.registered_tools[tool.tool_id] = tool
            self.tool_executors[tool.tool_id] = executor
            
            # Initialize tool usage stats
            self.performance_metrics["tool_usage_stats"][tool.tool_id] = {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "average_time": 0.0
            }
            
            self.logger.info(f"Tool registered successfully: {tool.tool_id} ({tool.name})")
            return True
            
        except Exception as e:
            self.logger.error(f"Tool registration failed: {e}")
            return False
    
    @timer_decorator
    def validate_tool_access(self, tool_id: str, agent_id: str, agent_tier: AgentTier,
                           agent_capabilities: List[AgentCapability]) -> Tuple[bool, str]:
        """Validate agent access to specific tool"""
        try:
            if tool_id not in self.registered_tools:
                return False, f"Tool {tool_id} not found"
            
            tool = self.registered_tools[tool_id]
            
            # Check access level restrictions
            tier_order = {AgentTier.FREE: 1, AgentTier.PRO: 2, AgentTier.ENTERPRISE: 3}
            
            if tool.access_level == ToolAccessLevel.RESTRICTED and tier_order[agent_tier] < 2:
                return False, f"Tool {tool_id} requires PRO or ENTERPRISE tier"
            
            if tool.access_level == ToolAccessLevel.PREMIUM and tier_order[agent_tier] < 3:
                return False, f"Tool {tool_id} requires ENTERPRISE tier"
            
            # Check required capabilities
            missing_capabilities = []
            for required_cap in tool.required_capabilities:
                if required_cap not in agent_capabilities:
                    missing_capabilities.append(required_cap.value)
            
            if missing_capabilities:
                return False, f"Missing required capabilities: {missing_capabilities}"
            
            return True, "Access granted"
            
        except Exception as e:
            self.logger.error(f"Tool access validation failed: {e}")
            return False, f"Validation error: {str(e)}"
    
    @timer_decorator
    def validate_tool_parameters(self, tool_id: str, parameters: Dict[str, Any],
                                validation_level: ToolValidationLevel) -> Tuple[bool, Dict[str, Any]]:
        """Validate tool execution parameters"""
        try:
            if tool_id not in self.registered_tools:
                return False, {"error": f"Tool {tool_id} not found"}
            
            tool = self.registered_tools[tool_id]
            validation_details = {
                "validated_parameters": {},
                "validation_errors": [],
                "security_issues": [],
                "recommendations": []
            }
            
            # Validate required parameters
            for param in tool.parameters:
                if param.required and param.name not in parameters:
                    validation_details["validation_errors"].append(
                        f"Missing required parameter: {param.name}"
                    )
                    continue
                
                param_value = parameters.get(param.name, param.default)
                
                # Basic type validation
                if param_value is not None:
                    # In production, implement proper type checking
                    validation_details["validated_parameters"][param.name] = param_value
                
                # Apply validation rules based on level
                if validation_level in [ToolValidationLevel.STRICT, ToolValidationLevel.ENTERPRISE]:
                    for rule in param.validation_rules:
                        # Mock validation rules - in production, implement actual validators
                        if rule == "no_dangerous_imports" and "import os" in str(param_value):
                            validation_details["security_issues"].append(
                                "Potentially dangerous import detected"
                            )
                        elif rule == "path_safety_check" and ".." in str(param_value):
                            validation_details["security_issues"].append(
                                "Path traversal attempt detected"
                            )
            
            # Check for validation errors
            has_errors = (
                len(validation_details["validation_errors"]) > 0 or
                len(validation_details["security_issues"]) > 0
            )
            
            if has_errors and validation_level == ToolValidationLevel.ENTERPRISE:
                return False, validation_details
            
            return not has_errors, validation_details
            
        except Exception as e:
            self.logger.error(f"Parameter validation failed: {e}")
            return False, {"error": f"Validation error: {str(e)}"}
    
    @timer_decorator
    async def execute_tool(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        """Execute a tool with comprehensive validation and monitoring"""
        start_time = time.time()
        
        try:
            # Add to active executions
            self.active_executions[request.request_id] = request
            
            # Get agent information for validation
            agent_info = await self._get_agent_info(request.agent_id)
            if not agent_info:
                return self._create_error_result(
                    request, "Agent not found or invalid", start_time
                )
            
            # Validate tool access
            access_valid, access_message = self.validate_tool_access(
                request.tool_id,
                request.agent_id,
                agent_info["tier"],
                agent_info["capabilities"]
            )
            
            if not access_valid:
                return self._create_error_result(
                    request, f"Access denied: {access_message}", start_time
                )
            
            # Validate parameters
            params_valid, validation_details = self.validate_tool_parameters(
                request.tool_id,
                request.parameters,
                request.validation_level
            )
            
            if not params_valid:
                return self._create_error_result(
                    request, f"Parameter validation failed", start_time,
                    validation_details=validation_details
                )
            
            # Execute the tool
            tool = self.registered_tools[request.tool_id]
            executor = self.tool_executors[request.tool_id]
            
            timeout = request.timeout_override or tool.timeout_seconds
            
            try:
                execution_result = await asyncio.wait_for(
                    executor(request.parameters),
                    timeout=timeout
                )
                
                execution_time = (time.time() - start_time) * 1000  # Convert to ms
                
                # Create successful output
                output = ToolOutput(
                    tool_id=request.tool_id,
                    status=ToolExecutionStatus.SUCCESS,
                    result=execution_result,
                    execution_time_ms=execution_time,
                    validation_score=1.0,
                    metadata={
                        "agent_id": request.agent_id,
                        "tool_version": tool.version,
                        "validation_level": request.validation_level.value
                    }
                )
                
                # Create execution result
                execution_result = ToolExecutionResult(
                    request=request,
                    output=output,
                    validation_details=validation_details,
                    performance_metrics={
                        "execution_time_ms": execution_time,
                        "validation_time_ms": (start_time - time.time()) * 1000,
                        "total_time_ms": (time.time() - start_time) * 1000
                    },
                    security_audit={
                        "access_validated": True,
                        "parameters_validated": True,
                        "security_level": request.validation_level.value
                    }
                )
                
                # Update metrics
                self._update_performance_metrics(request.tool_id, True, execution_time)
                
                # Store execution history
                self.execution_history[request.request_id] = execution_result
                
                self.logger.info(f"Tool executed successfully: {request.tool_id} ({execution_time:.1f}ms)")
                return execution_result
                
            except asyncio.TimeoutError:
                return self._create_error_result(
                    request, f"Tool execution timeout ({timeout}s)", start_time,
                    status=ToolExecutionStatus.TIMEOUT
                )
            
        except Exception as e:
            self.logger.error(f"Tool execution failed: {e}")
            return self._create_error_result(
                request, f"Execution error: {str(e)}", start_time
            )
        
        finally:
            # Remove from active executions
            if request.request_id in self.active_executions:
                del self.active_executions[request.request_id]
    
    def _create_error_result(self, request: ToolExecutionRequest, error_message: str,
                           start_time: float, status: ToolExecutionStatus = ToolExecutionStatus.FAILED,
                           validation_details: Dict[str, Any] = None) -> ToolExecutionResult:
        """Create an error result for failed tool execution"""
        execution_time = (time.time() - start_time) * 1000
        
        output = ToolOutput(
            tool_id=request.tool_id,
            status=status,
            error_message=error_message,
            execution_time_ms=execution_time,
            validation_score=0.0
        )
        
        result = ToolExecutionResult(
            request=request,
            output=output,
            validation_details=validation_details or {},
            performance_metrics={"execution_time_ms": execution_time},
            security_audit={"error": error_message}
        )
        
        # Update metrics for failed execution
        self._update_performance_metrics(request.tool_id, False, execution_time)
        
        return result
    
    async def _get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent information for validation"""
        # Mock implementation - in production, integrate with agent factory
        if hasattr(self, 'agent_factory') and self.agent_factory:
            agent_record = self.agent_factory.get_agent_record(agent_id)
            if agent_record:
                return {
                    "tier": agent_record.tier,
                    "capabilities": agent_record.capabilities,
                    "specialization": agent_record.specialization
                }
        
        # Provide different capabilities based on agent_id for testing
        if "free" in agent_id.lower():
            return {
                "tier": AgentTier.FREE,
                "capabilities": [AgentCapability.BASIC_REASONING],
                "specialization": AgentSpecialization.CASUAL
            }
        elif "enterprise" in agent_id.lower():
            return {
                "tier": AgentTier.ENTERPRISE,
                "capabilities": [cap for cap in AgentCapability],
                "specialization": AgentSpecialization.MCP
            }
        else:
            # Default to PRO with web browsing capability
            return {
                "tier": AgentTier.PRO,
                "capabilities": [
                    AgentCapability.BASIC_REASONING, 
                    AgentCapability.TOOL_USAGE,
                    AgentCapability.WEB_BROWSING,
                    AgentCapability.CODE_GENERATION
                ],
                "specialization": AgentSpecialization.RESEARCH
            }
    
    def _update_performance_metrics(self, tool_id: str, success: bool, execution_time: float) -> None:
        """Update performance metrics for tool execution"""
        # Update global metrics
        self.performance_metrics["total_executions"] += 1
        if success:
            self.performance_metrics["successful_executions"] += 1
        else:
            self.performance_metrics["failed_executions"] += 1
        
        # Update average execution time
        total_time = (self.performance_metrics["average_execution_time"] * 
                     (self.performance_metrics["total_executions"] - 1) + execution_time)
        self.performance_metrics["average_execution_time"] = total_time / self.performance_metrics["total_executions"]
        
        # Update tool-specific metrics
        if tool_id in self.performance_metrics["tool_usage_stats"]:
            tool_stats = self.performance_metrics["tool_usage_stats"][tool_id]
            tool_stats["total_calls"] += 1
            if success:
                tool_stats["successful_calls"] += 1
            else:
                tool_stats["failed_calls"] += 1
            
            # Update tool average time
            total_tool_time = (tool_stats["average_time"] * (tool_stats["total_calls"] - 1) + execution_time)
            tool_stats["average_time"] = total_tool_time / tool_stats["total_calls"]
    
    @timer_decorator
    def get_available_tools(self, agent_tier: AgentTier = None,
                          agent_capabilities: List[AgentCapability] = None) -> List[ToolCapability]:
        """Get list of tools available to agent based on tier and capabilities"""
        available_tools = []
        
        for tool in self.registered_tools.values():
            # Check access level
            if agent_tier:
                tier_order = {AgentTier.FREE: 1, AgentTier.PRO: 2, AgentTier.ENTERPRISE: 3}
                
                # Convert enum to comparable value
                agent_tier_value = tier_order.get(agent_tier, 1)
                
                if tool.access_level == ToolAccessLevel.RESTRICTED and agent_tier_value < 2:
                    continue
                if tool.access_level == ToolAccessLevel.PREMIUM and agent_tier_value < 3:
                    continue
            
            # Check capabilities
            if agent_capabilities:
                missing_caps = [cap for cap in tool.required_capabilities 
                              if cap not in agent_capabilities]
                if missing_caps:
                    continue
            
            available_tools.append(tool)
        
        return available_tools
    
    @timer_decorator
    def get_tool_analytics(self) -> Dict[str, Any]:
        """Get comprehensive tool framework analytics"""
        analytics = {
            "framework_info": {
                "framework_id": self.framework_id,
                "version": self.version,
                "registered_tools": len(self.registered_tools),
                "active_executions": len(self.active_executions)
            },
            "performance_metrics": self.performance_metrics.copy(),
            "tool_catalog": {
                tool_id: {
                    "name": tool.name,
                    "category": tool.category.value,
                    "access_level": tool.access_level.value,
                    "usage_stats": self.performance_metrics["tool_usage_stats"].get(tool_id, {})
                }
                for tool_id, tool in self.registered_tools.items()
            },
            "category_distribution": self._get_category_distribution(),
            "access_level_distribution": self._get_access_level_distribution(),
            "execution_history_size": len(self.execution_history)
        }
        
        return analytics
    
    def _get_category_distribution(self) -> Dict[str, int]:
        """Get distribution of tools by category"""
        distribution = {}
        for tool in self.registered_tools.values():
            category = tool.category.value
            distribution[category] = distribution.get(category, 0) + 1
        return distribution
    
    def _get_access_level_distribution(self) -> Dict[str, int]:
        """Get distribution of tools by access level"""
        distribution = {}
        for tool in self.registered_tools.values():
            access_level = tool.access_level.value
            distribution[access_level] = distribution.get(access_level, 0) + 1
        return distribution
    
    def set_agent_factory(self, agent_factory) -> None:
        """Inject agent factory dependency"""
        self.agent_factory = agent_factory
        self.logger.info("Agent factory injected into tool framework")
    
    def set_communication_manager(self, communication_manager) -> None:
        """Inject communication manager dependency"""
        self.communication_manager = communication_manager
        self.logger.info("Communication manager injected into tool framework")
    
    @timer_decorator
    def get_execution_history(self, tool_id: str = None, agent_id: str = None,
                            limit: int = 100) -> List[ToolExecutionResult]:
        """Get tool execution history with optional filtering"""
        history = list(self.execution_history.values())
        
        # Apply filters
        if tool_id:
            history = [result for result in history if result.request.tool_id == tool_id]
        
        if agent_id:
            history = [result for result in history if result.request.agent_id == agent_id]
        
        # Sort by completion time (most recent first)
        history.sort(key=lambda x: x.completed_at, reverse=True)
        
        return history[:limit]

# High-level framework management functions
@timer_decorator
def create_validated_tool_framework() -> ValidatedToolIntegrationFramework:
    """Create and initialize a new Validated Tool Integration Framework"""
    framework = ValidatedToolIntegrationFramework()
    return framework

@timer_decorator
async def quick_tool_integration_test() -> Dict[str, Any]:
    """Quick test of tool integration functionality"""
    try:
        framework = create_validated_tool_framework()
        
        # Test tool execution
        request = ToolExecutionRequest(
            tool_id="web_search",
            agent_id="test_agent_001",
            parameters={"query": "test search", "max_results": 5},
            validation_level=ToolValidationLevel.STANDARD
        )
        
        result = await framework.execute_tool(request)
        
        test_results = {
            "framework_initialized": True,
            "tool_execution_status": result.output.status.value,
            "execution_successful": result.output.status == ToolExecutionStatus.SUCCESS,
            "execution_time_ms": result.output.execution_time_ms,
            "framework_analytics": framework.get_tool_analytics()
        }
        
        return test_results
        
    except Exception as e:
        return {
            "framework_initialized": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# Main execution for testing
if __name__ == "__main__":
    print("🔧 Pydantic AI Validated Tool Integration Framework - Standalone Test")
    print("=" * 80)
    
    async def main():
        # Test basic framework functionality
        test_results = await quick_tool_integration_test()
        
        print("🧪 Test Results:")
        for key, value in test_results.items():
            print(f"  {key}: {value}")
        
        print("\n✅ Validated Tool Integration Framework test complete!")
    
    asyncio.run(main())