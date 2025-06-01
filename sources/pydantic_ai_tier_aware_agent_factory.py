#!/usr/bin/env python3
"""
Pydantic AI Tier-Aware Agent Factory Implementation

Purpose: Comprehensive agent factory with tier-based capability validation and intelligent specialization
Issues & Complexity Summary: Advanced factory pattern with capability validation, tier restrictions, and dynamic agent creation
Key Complexity Drivers:
  - Logic Scope (Est. LoC): ~1200
  - Core Algorithm Complexity: High
  - Dependencies: 5 New, 2 Mod
  - State Management Complexity: High
  - Novelty/Uncertainty Factor: Medium
AI Pre-Task Self-Assessment (Est. Solution Difficulty %): 75%
Problem Estimate (Inherent Problem Difficulty %): 80%
Initial Code Complexity Estimate %: 75%
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
from typing import Dict, List, Any, Optional, Union, Set, Tuple, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
import json
import traceback
from pathlib import Path

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
    CORE_INTEGRATION_AVAILABLE = True
except ImportError:
    CORE_INTEGRATION_AVAILABLE = False
    # Fallback agent types
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

# Enhanced Tier-Aware Agent Factory Models
class AgentFactoryStatus(Enum):
    """Agent factory operational status"""
    ACTIVE = "active"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"
    ERROR = "error"

class AgentCreationResult(Enum):
    """Agent creation result status"""
    SUCCESS = "success"
    TIER_RESTRICTION = "tier_restriction"
    CAPABILITY_MISMATCH = "capability_mismatch"
    QUOTA_EXCEEDED = "quota_exceeded"
    VALIDATION_FAILED = "validation_failed"
    SYSTEM_ERROR = "system_error"

class AgentValidationLevel(Enum):
    """Agent validation strictness levels"""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    ENTERPRISE = "enterprise"

# Capability Models
if PYDANTIC_AI_AVAILABLE:
    class CapabilityRequirement(BaseModel):
        """Individual capability requirement specification"""
        capability: AgentCapability
        required: bool = Field(True)
        minimum_level: int = Field(1, ge=1, le=10)
        tier_restrictions: List[AgentTier] = Field(default_factory=list)
        dependencies: List[AgentCapability] = Field(default_factory=list)
        mutually_exclusive: List[AgentCapability] = Field(default_factory=list)
        
        class Config:
            use_enum_values = True
    
    class CapabilityValidationResult(BaseModel):
        """Result of capability validation process"""
        validation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        is_valid: bool
        required_capabilities: List[AgentCapability] = Field(default_factory=list)
        missing_capabilities: List[AgentCapability] = Field(default_factory=list)
        conflicting_capabilities: List[Tuple[AgentCapability, AgentCapability]] = Field(default_factory=list)
        tier_violations: List[str] = Field(default_factory=list)
        dependency_issues: Dict[str, List[str]] = Field(default_factory=dict)
        validation_score: float = Field(0.0, ge=0.0, le=1.0)
        recommendations: List[str] = Field(default_factory=list)
        created_at: datetime = Field(default_factory=datetime.now)
        
        class Config:
            use_enum_values = True
    
    class TierQuotaLimits(BaseModel):
        """Tier-based quota and capability limits"""
        tier: AgentTier
        max_agents: int = Field(10, ge=1)
        max_concurrent_agents: int = Field(3, ge=1)
        allowed_capabilities: List[AgentCapability] = Field(default_factory=list)
        restricted_capabilities: List[AgentCapability] = Field(default_factory=list)
        max_memory_mb: int = Field(512, ge=128)
        max_processing_time_seconds: int = Field(300, ge=60)
        priority_boost: float = Field(1.0, ge=0.1, le=5.0)
        cost_multiplier: float = Field(1.0, ge=0.1, le=10.0)
        
        class Config:
            use_enum_values = True
    
    class AgentSpecializationTemplate(BaseModel):
        """Template for specialized agent configuration"""
        specialization: AgentSpecialization
        default_capabilities: List[AgentCapability] = Field(default_factory=list)
        required_capabilities: List[CapabilityRequirement] = Field(default_factory=list)
        optional_capabilities: List[CapabilityRequirement] = Field(default_factory=list)
        minimum_tier: AgentTier = Field(AgentTier.FREE)
        recommended_tier: AgentTier = Field(AgentTier.PRO)
        validation_level: AgentValidationLevel = Field(AgentValidationLevel.STANDARD)
        performance_expectations: Dict[str, Any] = Field(default_factory=dict)
        resource_requirements: Dict[str, Any] = Field(default_factory=dict)
        integration_points: List[str] = Field(default_factory=list)
        
        class Config:
            use_enum_values = True
    
    class AgentCreationRequest(BaseModel):
        """Comprehensive agent creation request"""
        request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        agent_id: str = Field(default_factory=lambda: f"agent_{uuid.uuid4().hex[:8]}")
        specialization: AgentSpecialization
        tier: AgentTier
        requested_capabilities: List[AgentCapability] = Field(default_factory=list)
        custom_configuration: Dict[str, Any] = Field(default_factory=dict)
        validation_level: AgentValidationLevel = Field(AgentValidationLevel.STANDARD)
        priority: int = Field(5, ge=1, le=10)
        owner_id: str = Field(...)
        metadata: Dict[str, Any] = Field(default_factory=dict)
        communication_enabled: bool = Field(True)
        memory_persistence: bool = Field(False)
        created_at: datetime = Field(default_factory=datetime.now)
        
        class Config:
            use_enum_values = True
    
    class CreatedAgentRecord(BaseModel):
        """Record of successfully created agent"""
        record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        agent_id: str
        specialization: AgentSpecialization
        tier: AgentTier
        capabilities: List[AgentCapability] = Field(default_factory=list)
        validation_result: CapabilityValidationResult
        creation_request: AgentCreationRequest
        factory_version: str = Field("1.0.0")
        performance_baseline: Dict[str, Any] = Field(default_factory=dict)
        resource_allocation: Dict[str, Any] = Field(default_factory=dict)
        communication_manager_id: Optional[str] = None
        status: str = Field("active")
        created_at: datetime = Field(default_factory=datetime.now)
        last_updated: datetime = Field(default_factory=datetime.now)
        
        class Config:
            use_enum_values = True

else:
    # Fallback implementations for non-Pydantic AI environments
    class CapabilityRequirement:
        def __init__(self, capability, required=True, minimum_level=1, 
                     tier_restrictions=None, dependencies=None, mutually_exclusive=None):
            self.capability = capability
            self.required = required
            self.minimum_level = minimum_level
            self.tier_restrictions = tier_restrictions or []
            self.dependencies = dependencies or []
            self.mutually_exclusive = mutually_exclusive or []
    
    class CapabilityValidationResult:
        def __init__(self, is_valid, **kwargs):
            self.validation_id = str(uuid.uuid4())
            self.is_valid = is_valid
            self.required_capabilities = kwargs.get('required_capabilities', [])
            self.missing_capabilities = kwargs.get('missing_capabilities', [])
            self.conflicting_capabilities = kwargs.get('conflicting_capabilities', [])
            self.tier_violations = kwargs.get('tier_violations', [])
            self.dependency_issues = kwargs.get('dependency_issues', {})
            self.validation_score = kwargs.get('validation_score', 0.0)
            self.recommendations = kwargs.get('recommendations', [])
            self.created_at = datetime.now()
    
    class TierQuotaLimits:
        def __init__(self, tier, **kwargs):
            self.tier = tier
            self.max_agents = kwargs.get('max_agents', 10)
            self.max_concurrent_agents = kwargs.get('max_concurrent_agents', 3)
            self.allowed_capabilities = kwargs.get('allowed_capabilities', [])
            self.restricted_capabilities = kwargs.get('restricted_capabilities', [])
            self.max_memory_mb = kwargs.get('max_memory_mb', 512)
            self.max_processing_time_seconds = kwargs.get('max_processing_time_seconds', 300)
            self.priority_boost = kwargs.get('priority_boost', 1.0)
            self.cost_multiplier = kwargs.get('cost_multiplier', 1.0)
    
    class AgentSpecializationTemplate:
        def __init__(self, specialization, **kwargs):
            self.specialization = specialization
            self.default_capabilities = kwargs.get('default_capabilities', [])
            self.required_capabilities = kwargs.get('required_capabilities', [])
            self.optional_capabilities = kwargs.get('optional_capabilities', [])
            self.minimum_tier = kwargs.get('minimum_tier', AgentTier.FREE)
            self.recommended_tier = kwargs.get('recommended_tier', AgentTier.PRO)
            self.validation_level = kwargs.get('validation_level', AgentValidationLevel.STANDARD)
            self.performance_expectations = kwargs.get('performance_expectations', {})
            self.resource_requirements = kwargs.get('resource_requirements', {})
            self.integration_points = kwargs.get('integration_points', [])
    
    class AgentCreationRequest:
        def __init__(self, specialization, tier, owner_id, **kwargs):
            self.request_id = str(uuid.uuid4())
            self.agent_id = kwargs.get('agent_id', f"agent_{uuid.uuid4().hex[:8]}")
            self.specialization = specialization
            self.tier = tier
            self.requested_capabilities = kwargs.get('requested_capabilities', [])
            self.custom_configuration = kwargs.get('custom_configuration', {})
            self.validation_level = kwargs.get('validation_level', AgentValidationLevel.STANDARD)
            self.priority = kwargs.get('priority', 5)
            self.owner_id = owner_id
            self.metadata = kwargs.get('metadata', {})
            self.communication_enabled = kwargs.get('communication_enabled', True)
            self.memory_persistence = kwargs.get('memory_persistence', False)
            self.created_at = datetime.now()
    
    class CreatedAgentRecord:
        def __init__(self, agent_id, specialization, tier, capabilities, 
                     validation_result, creation_request, **kwargs):
            self.record_id = str(uuid.uuid4())
            self.agent_id = agent_id
            self.specialization = specialization
            self.tier = tier
            self.capabilities = capabilities
            self.validation_result = validation_result
            self.creation_request = creation_request
            self.factory_version = "1.0.0"
            self.performance_baseline = kwargs.get('performance_baseline', {})
            self.resource_allocation = kwargs.get('resource_allocation', {})
            self.communication_manager_id = kwargs.get('communication_manager_id')
            self.status = kwargs.get('status', 'active')
            self.created_at = datetime.now()
            self.last_updated = datetime.now()

# Core Tier-Aware Agent Factory Implementation
class TierAwareAgentFactory:
    """
    Comprehensive agent factory with tier-based capability validation and intelligent specialization
    
    Features:
    - Tier-aware agent creation with capability validation
    - Intelligent specialization templates and configuration
    - Quota management and resource allocation
    - Communication integration and lifecycle management
    - Performance monitoring and optimization
    """
    
    def __init__(self):
        """Initialize the Tier-Aware Agent Factory"""
        self.logger = Logger("tier_aware_agent_factory.log") if LOGGER_AVAILABLE else Logger("fallback.log")
        self.factory_id = str(uuid.uuid4())
        self.status = AgentFactoryStatus.ACTIVE
        self.version = "1.0.0"
        
        # Factory state management
        self.created_agents: Dict[str, CreatedAgentRecord] = {}
        self.active_requests: Dict[str, AgentCreationRequest] = {}
        self.tier_quotas: Dict[AgentTier, TierQuotaLimits] = {}
        self.specialization_templates: Dict[AgentSpecialization, AgentSpecializationTemplate] = {}
        
        # Performance and analytics
        self.creation_metrics: Dict[str, Any] = {
            "total_created": 0,
            "successful_creations": 0,
            "failed_creations": 0,
            "tier_distribution": {tier.value: 0 for tier in AgentTier},
            "specialization_distribution": {spec.value: 0 for spec in AgentSpecialization},
            "average_creation_time": 0.0,
            "last_reset": datetime.now()
        }
        
        # Communication integration
        if CORE_INTEGRATION_AVAILABLE:
            self.communication_manager = None  # Will be injected
        
        # Initialize default configurations
        self._initialize_tier_quotas()
        self._initialize_specialization_templates()
        
        self.logger.info(f"Tier-Aware Agent Factory initialized - ID: {self.factory_id}")
    
    def _initialize_tier_quotas(self) -> None:
        """Initialize default tier quota configurations"""
        # FREE Tier - Basic capabilities
        self.tier_quotas[AgentTier.FREE] = TierQuotaLimits(
            tier=AgentTier.FREE,
            max_agents=5,
            max_concurrent_agents=2,
            allowed_capabilities=[
                AgentCapability.BASIC_REASONING,
                AgentCapability.FILE_OPERATIONS
            ],
            restricted_capabilities=[
                AgentCapability.ADVANCED_REASONING,
                AgentCapability.MCP_INTEGRATION,
                AgentCapability.MEMORY_ACCESS
            ],
            max_memory_mb=256,
            max_processing_time_seconds=180,
            priority_boost=1.0,
            cost_multiplier=1.0
        )
        
        # PRO Tier - Advanced capabilities
        self.tier_quotas[AgentTier.PRO] = TierQuotaLimits(
            tier=AgentTier.PRO,
            max_agents=15,
            max_concurrent_agents=5,
            allowed_capabilities=[
                AgentCapability.BASIC_REASONING,
                AgentCapability.ADVANCED_REASONING,
                AgentCapability.CODE_GENERATION,
                AgentCapability.WEB_BROWSING,
                AgentCapability.FILE_OPERATIONS,
                AgentCapability.PLANNING,
                AgentCapability.TOOL_USAGE
            ],
            restricted_capabilities=[
                AgentCapability.MCP_INTEGRATION,
                AgentCapability.MEMORY_ACCESS
            ],
            max_memory_mb=1024,
            max_processing_time_seconds=600,
            priority_boost=1.5,
            cost_multiplier=2.0
        )
        
        # ENTERPRISE Tier - Full capabilities
        self.tier_quotas[AgentTier.ENTERPRISE] = TierQuotaLimits(
            tier=AgentTier.ENTERPRISE,
            max_agents=50,
            max_concurrent_agents=15,
            allowed_capabilities=list(AgentCapability),
            restricted_capabilities=[],
            max_memory_mb=4096,
            max_processing_time_seconds=3600,
            priority_boost=2.0,
            cost_multiplier=5.0
        )
    
    def _initialize_specialization_templates(self) -> None:
        """Initialize default specialization templates"""
        # Coordinator Agent Template
        self.specialization_templates[AgentSpecialization.COORDINATOR] = AgentSpecializationTemplate(
            specialization=AgentSpecialization.COORDINATOR,
            default_capabilities=[
                AgentCapability.ADVANCED_REASONING,
                AgentCapability.COORDINATION,
                AgentCapability.PLANNING
            ],
            required_capabilities=[
                CapabilityRequirement(
                    capability=AgentCapability.COORDINATION,
                    required=True,
                    minimum_level=7,
                    tier_restrictions=[AgentTier.PRO, AgentTier.ENTERPRISE]
                ),
                CapabilityRequirement(
                    capability=AgentCapability.PLANNING,
                    required=True,
                    minimum_level=6,
                    dependencies=[AgentCapability.ADVANCED_REASONING]
                )
            ],
            minimum_tier=AgentTier.PRO,
            recommended_tier=AgentTier.ENTERPRISE,
            validation_level=AgentValidationLevel.STRICT,
            performance_expectations={
                "coordination_efficiency": 0.85,
                "decision_accuracy": 0.90,
                "response_time_ms": 500
            },
            resource_requirements={
                "memory_mb": 1024,
                "cpu_priority": "high",
                "network_access": True
            },
            integration_points=["communication_manager", "memory_system", "analytics"]
        )
        
        # Research Agent Template
        self.specialization_templates[AgentSpecialization.RESEARCH] = AgentSpecializationTemplate(
            specialization=AgentSpecialization.RESEARCH,
            default_capabilities=[
                AgentCapability.BASIC_REASONING,
                AgentCapability.WEB_BROWSING,
                AgentCapability.FILE_OPERATIONS
            ],
            required_capabilities=[
                CapabilityRequirement(
                    capability=AgentCapability.WEB_BROWSING,
                    required=True,
                    minimum_level=5,
                    tier_restrictions=[AgentTier.PRO, AgentTier.ENTERPRISE]
                )
            ],
            minimum_tier=AgentTier.FREE,
            recommended_tier=AgentTier.PRO,
            validation_level=AgentValidationLevel.STANDARD,
            performance_expectations={
                "research_accuracy": 0.80,
                "information_retrieval": 0.85,
                "source_validation": 0.75
            },
            resource_requirements={
                "memory_mb": 512,
                "network_access": True,
                "storage_access": True
            },
            integration_points=["web_browser", "file_system", "knowledge_base"]
        )
        
        # Code Agent Template
        self.specialization_templates[AgentSpecialization.CODE] = AgentSpecializationTemplate(
            specialization=AgentSpecialization.CODE,
            default_capabilities=[
                AgentCapability.ADVANCED_REASONING,
                AgentCapability.CODE_GENERATION,
                AgentCapability.FILE_OPERATIONS
            ],
            required_capabilities=[
                CapabilityRequirement(
                    capability=AgentCapability.CODE_GENERATION,
                    required=True,
                    minimum_level=8,
                    tier_restrictions=[AgentTier.PRO, AgentTier.ENTERPRISE],
                    dependencies=[AgentCapability.ADVANCED_REASONING]
                )
            ],
            minimum_tier=AgentTier.PRO,
            recommended_tier=AgentTier.ENTERPRISE,
            validation_level=AgentValidationLevel.STRICT,
            performance_expectations={
                "code_quality": 0.90,
                "compilation_success": 0.95,
                "performance_optimization": 0.80
            },
            resource_requirements={
                "memory_mb": 2048,
                "cpu_priority": "high",
                "storage_access": True
            },
            integration_points=["ide_tools", "compiler", "testing_framework"]
        )
        
        # Browser Agent Template
        self.specialization_templates[AgentSpecialization.BROWSER] = AgentSpecializationTemplate(
            specialization=AgentSpecialization.BROWSER,
            default_capabilities=[
                AgentCapability.BASIC_REASONING,
                AgentCapability.WEB_BROWSING,
                AgentCapability.TOOL_USAGE
            ],
            required_capabilities=[
                CapabilityRequirement(
                    capability=AgentCapability.WEB_BROWSING,
                    required=True,
                    minimum_level=6,
                    tier_restrictions=[AgentTier.PRO, AgentTier.ENTERPRISE]
                )
            ],
            minimum_tier=AgentTier.PRO,
            recommended_tier=AgentTier.PRO,
            validation_level=AgentValidationLevel.STANDARD,
            performance_expectations={
                "navigation_success": 0.90,
                "data_extraction": 0.85,
                "interaction_accuracy": 0.80
            },
            resource_requirements={
                "memory_mb": 1024,
                "network_access": True,
                "browser_access": True
            },
            integration_points=["web_driver", "dom_parser", "screenshot_tools"]
        )
        
        # File Agent Template
        self.specialization_templates[AgentSpecialization.FILE] = AgentSpecializationTemplate(
            specialization=AgentSpecialization.FILE,
            default_capabilities=[
                AgentCapability.BASIC_REASONING,
                AgentCapability.FILE_OPERATIONS
            ],
            required_capabilities=[
                CapabilityRequirement(
                    capability=AgentCapability.FILE_OPERATIONS,
                    required=True,
                    minimum_level=7
                )
            ],
            minimum_tier=AgentTier.FREE,
            recommended_tier=AgentTier.PRO,
            validation_level=AgentValidationLevel.STANDARD,
            performance_expectations={
                "file_operation_success": 0.95,
                "data_integrity": 0.98,
                "processing_speed": 0.85
            },
            resource_requirements={
                "memory_mb": 512,
                "storage_access": True,
                "backup_access": True
            },
            integration_points=["file_system", "backup_manager", "compression_tools"]
        )
        
        # MCP Agent Template
        self.specialization_templates[AgentSpecialization.MCP] = AgentSpecializationTemplate(
            specialization=AgentSpecialization.MCP,
            default_capabilities=[
                AgentCapability.ADVANCED_REASONING,
                AgentCapability.MCP_INTEGRATION,
                AgentCapability.TOOL_USAGE
            ],
            required_capabilities=[
                CapabilityRequirement(
                    capability=AgentCapability.MCP_INTEGRATION,
                    required=True,
                    minimum_level=8,
                    tier_restrictions=[AgentTier.ENTERPRISE],
                    dependencies=[AgentCapability.ADVANCED_REASONING, AgentCapability.TOOL_USAGE]
                )
            ],
            minimum_tier=AgentTier.ENTERPRISE,
            recommended_tier=AgentTier.ENTERPRISE,
            validation_level=AgentValidationLevel.ENTERPRISE,
            performance_expectations={
                "mcp_integration": 0.95,
                "tool_coordination": 0.90,
                "service_reliability": 0.98
            },
            resource_requirements={
                "memory_mb": 2048,
                "network_access": True,
                "service_access": True
            },
            integration_points=["mcp_services", "tool_ecosystem", "service_registry"]
        )
        
        # Casual Agent Template
        self.specialization_templates[AgentSpecialization.CASUAL] = AgentSpecializationTemplate(
            specialization=AgentSpecialization.CASUAL,
            default_capabilities=[
                AgentCapability.BASIC_REASONING
            ],
            required_capabilities=[
                CapabilityRequirement(
                    capability=AgentCapability.BASIC_REASONING,
                    required=True,
                    minimum_level=3
                )
            ],
            minimum_tier=AgentTier.FREE,
            recommended_tier=AgentTier.FREE,
            validation_level=AgentValidationLevel.BASIC,
            performance_expectations={
                "conversation_quality": 0.70,
                "response_time": 0.80,
                "user_satisfaction": 0.75
            },
            resource_requirements={
                "memory_mb": 256,
                "network_access": False
            },
            integration_points=["conversation_manager"]
        )
        
        # Planner Agent Template
        self.specialization_templates[AgentSpecialization.PLANNER] = AgentSpecializationTemplate(
            specialization=AgentSpecialization.PLANNER,
            default_capabilities=[
                AgentCapability.ADVANCED_REASONING,
                AgentCapability.PLANNING,
                AgentCapability.COORDINATION
            ],
            required_capabilities=[
                CapabilityRequirement(
                    capability=AgentCapability.PLANNING,
                    required=True,
                    minimum_level=7,
                    tier_restrictions=[AgentTier.PRO, AgentTier.ENTERPRISE],
                    dependencies=[AgentCapability.ADVANCED_REASONING]
                )
            ],
            minimum_tier=AgentTier.PRO,
            recommended_tier=AgentTier.ENTERPRISE,
            validation_level=AgentValidationLevel.STRICT,
            performance_expectations={
                "planning_accuracy": 0.85,
                "strategy_effectiveness": 0.80,
                "execution_coordination": 0.90
            },
            resource_requirements={
                "memory_mb": 1024,
                "processing_priority": "high"
            },
            integration_points=["planning_engine", "coordination_system", "analytics"]
        )
    
    @timer_decorator
    def validate_capabilities(self, request: AgentCreationRequest) -> CapabilityValidationResult:
        """
        Comprehensive capability validation for agent creation request
        
        Args:
            request: Agent creation request to validate
            
        Returns:
            CapabilityValidationResult with detailed validation outcome
        """
        try:
            # Get specialization template
            template = self.specialization_templates.get(request.specialization)
            if not template:
                return CapabilityValidationResult(
                    is_valid=False,
                    tier_violations=[f"Unknown specialization: {request.specialization}"],
                    validation_score=0.0,
                    recommendations=["Use a supported agent specialization"]
                )
            
            # Get tier limits
            tier_limits = self.tier_quotas.get(request.tier)
            if not tier_limits:
                return CapabilityValidationResult(
                    is_valid=False,
                    tier_violations=[f"Unknown tier: {request.tier}"],
                    validation_score=0.0,
                    recommendations=["Use a valid tier (FREE/PRO/ENTERPRISE)"]
                )
            
            # Initialize validation tracking
            missing_capabilities = []
            conflicting_capabilities = []
            tier_violations = []
            dependency_issues = {}
            recommendations = []
            validation_score = 1.0
            
            # Validate minimum tier requirement
            tier_order = {AgentTier.FREE: 1, AgentTier.PRO: 2, AgentTier.ENTERPRISE: 3}
            if tier_order[request.tier] < tier_order[template.minimum_tier]:
                tier_violations.append(
                    f"Specialization {request.specialization.value} requires minimum tier {template.minimum_tier.value}, "
                    f"but {request.tier.value} was requested"
                )
                validation_score -= 0.3
            
            # Combine all required capabilities
            all_required_capabilities = set()
            
            # Add template default capabilities
            all_required_capabilities.update(template.default_capabilities)
            
            # Add template required capabilities
            for req_cap in template.required_capabilities:
                all_required_capabilities.add(req_cap.capability)
                
                # Check tier restrictions
                if req_cap.tier_restrictions and request.tier not in req_cap.tier_restrictions:
                    tier_violations.append(
                        f"Capability {req_cap.capability.value} is restricted to tiers: "
                        f"{[t.value for t in req_cap.tier_restrictions]}, but {request.tier.value} was requested"
                    )
                    validation_score -= 0.2
                
                # Check dependencies
                for dep_cap in req_cap.dependencies:
                    if dep_cap not in request.requested_capabilities and dep_cap not in template.default_capabilities:
                        dep_issues = dependency_issues.setdefault(req_cap.capability.value, [])
                        dep_issues.append(f"Missing dependency: {dep_cap.value}")
                        validation_score -= 0.1
                
                # Check mutually exclusive capabilities
                for mutex_cap in req_cap.mutually_exclusive:
                    if mutex_cap in request.requested_capabilities:
                        conflicting_capabilities.append((req_cap.capability, mutex_cap))
                        validation_score -= 0.15
            
            # Add explicitly requested capabilities
            all_required_capabilities.update(request.requested_capabilities)
            
            # Validate against tier restrictions
            for capability in all_required_capabilities:
                if capability in tier_limits.restricted_capabilities:
                    tier_violations.append(
                        f"Capability {capability.value} is not allowed for tier {request.tier.value}"
                    )
                    validation_score -= 0.2
                
                if capability not in tier_limits.allowed_capabilities and tier_limits.allowed_capabilities:
                    # If allowed_capabilities is specified and capability is not in it
                    missing_capabilities.append(capability)
                    validation_score -= 0.1
            
            # Check quota limits
            current_agent_count = len([a for a in self.created_agents.values() 
                                     if a.tier == request.tier and a.status == 'active'])
            if current_agent_count >= tier_limits.max_agents:
                tier_violations.append(
                    f"Agent quota exceeded for tier {request.tier.value}: "
                    f"{current_agent_count}/{tier_limits.max_agents}"
                )
                validation_score -= 0.3
            
            # Generate recommendations
            if tier_violations:
                recommendations.append("Consider upgrading to a higher tier for more capabilities")
            
            if missing_capabilities:
                recommendations.append("Review tier-allowed capabilities and adjust request")
            
            if conflicting_capabilities:
                recommendations.append("Remove conflicting capabilities from request")
            
            if dependency_issues:
                recommendations.append("Add missing capability dependencies")
            
            if validation_score < 0.7:
                recommendations.append("Consider reviewing specialization requirements")
            
            # Final validation result
            is_valid = (
                validation_score >= 0.7 and
                not tier_violations and
                not conflicting_capabilities and
                not dependency_issues
            )
            
            validation_result = CapabilityValidationResult(
                is_valid=is_valid,
                required_capabilities=list(all_required_capabilities),
                missing_capabilities=missing_capabilities,
                conflicting_capabilities=conflicting_capabilities,
                tier_violations=tier_violations,
                dependency_issues=dependency_issues,
                validation_score=max(0.0, validation_score),
                recommendations=recommendations
            )
            
            self.logger.info(f"Capability validation completed - Valid: {is_valid}, Score: {validation_score:.2f}")
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Capability validation failed: {e}")
            return CapabilityValidationResult(
                is_valid=False,
                validation_score=0.0,
                tier_violations=[f"Validation error: {str(e)}"],
                recommendations=["Retry validation with correct parameters"]
            )
    
    @timer_decorator
    async def create_agent(self, request: AgentCreationRequest) -> Tuple[AgentCreationResult, Optional[CreatedAgentRecord]]:
        """
        Create a new agent with tier-aware capability validation
        
        Args:
            request: Agent creation request with specifications
            
        Returns:
            Tuple of (creation_result, agent_record)
        """
        try:
            start_time = time.time()
            self.logger.info(f"Creating agent - ID: {request.agent_id}, Specialization: {request.specialization.value}")
            
            # Add to active requests
            self.active_requests[request.request_id] = request
            
            # Validate capabilities
            validation_result = self.validate_capabilities(request)
            if not validation_result.is_valid:
                self.logger.error(f"Agent creation failed - capability validation failed: {validation_result.tier_violations}")
                self.creation_metrics["failed_creations"] += 1
                del self.active_requests[request.request_id]
                return AgentCreationResult.CAPABILITY_MISMATCH, None
            
            # Check quota limits
            tier_limits = self.tier_quotas[request.tier]
            current_agents = [a for a in self.created_agents.values() 
                            if a.tier == request.tier and a.status == 'active']
            
            if len(current_agents) >= tier_limits.max_agents:
                self.logger.error(f"Agent creation failed - quota exceeded: {len(current_agents)}/{tier_limits.max_agents}")
                self.creation_metrics["failed_creations"] += 1
                del self.active_requests[request.request_id]
                return AgentCreationResult.QUOTA_EXCEEDED, None
            
            # Create agent configuration
            agent_config = self._create_agent_configuration(request, validation_result)
            
            # Create the actual agent instance
            if CORE_INTEGRATION_AVAILABLE:
                # Use full Pydantic AI integration
                agent_instance = await self._create_pydantic_agent(agent_config, request)
            else:
                # Use fallback agent creation
                agent_instance = self._create_fallback_agent(agent_config, request)
            
            # Set up communication if enabled
            communication_manager_id = None
            if request.communication_enabled and hasattr(self, 'communication_manager'):
                communication_manager_id = await self._setup_agent_communication(request.agent_id, request)
            
            # Create resource allocation profile
            resource_allocation = self._allocate_agent_resources(request, tier_limits)
            
            # Create performance baseline
            performance_baseline = self._create_performance_baseline(request)
            
            # Create agent record
            agent_record = CreatedAgentRecord(
                agent_id=request.agent_id,
                specialization=request.specialization,
                tier=request.tier,
                capabilities=validation_result.required_capabilities,
                validation_result=validation_result,
                creation_request=request,
                performance_baseline=performance_baseline,
                resource_allocation=resource_allocation,
                communication_manager_id=communication_manager_id,
                status="active"
            )
            
            # Store agent record
            self.created_agents[request.agent_id] = agent_record
            
            # Update metrics
            creation_time = time.time() - start_time
            self.creation_metrics["total_created"] += 1
            self.creation_metrics["successful_creations"] += 1
            self.creation_metrics["tier_distribution"][request.tier.value] += 1
            self.creation_metrics["specialization_distribution"][request.specialization.value] += 1
            
            # Update average creation time
            total_time = (self.creation_metrics["average_creation_time"] * 
                         (self.creation_metrics["total_created"] - 1) + creation_time)
            self.creation_metrics["average_creation_time"] = total_time / self.creation_metrics["total_created"]
            
            # Clean up active requests
            del self.active_requests[request.request_id]
            
            self.logger.info(f"Agent created successfully - ID: {request.agent_id}, Time: {creation_time:.3f}s")
            return AgentCreationResult.SUCCESS, agent_record
            
        except Exception as e:
            self.logger.error(f"Agent creation failed: {e}")
            self.creation_metrics["failed_creations"] += 1
            if request.request_id in self.active_requests:
                del self.active_requests[request.request_id]
            return AgentCreationResult.SYSTEM_ERROR, None
    
    def _create_agent_configuration(self, request: AgentCreationRequest, 
                                  validation_result: CapabilityValidationResult) -> Dict[str, Any]:
        """Create comprehensive agent configuration"""
        template = self.specialization_templates[request.specialization]
        tier_limits = self.tier_quotas[request.tier]
        
        config = {
            "agent_id": request.agent_id,
            "specialization": request.specialization,
            "tier": request.tier,
            "capabilities": validation_result.required_capabilities,
            "validation_level": request.validation_level,
            "memory_limit_mb": tier_limits.max_memory_mb,
            "processing_timeout": tier_limits.max_processing_time_seconds,
            "priority_boost": tier_limits.priority_boost,
            "performance_expectations": template.performance_expectations,
            "resource_requirements": template.resource_requirements,
            "integration_points": template.integration_points,
            "custom_configuration": request.custom_configuration,
            "metadata": request.metadata
        }
        
        return config
    
    async def _create_pydantic_agent(self, config: Dict[str, Any], 
                                   request: AgentCreationRequest) -> Any:
        """Create agent using full Pydantic AI integration"""
        # This would integrate with the actual Pydantic AI Core Integration
        # For now, return a placeholder
        return {
            "agent_type": "pydantic_ai",
            "config": config,
            "status": "created"
        }
    
    def _create_fallback_agent(self, config: Dict[str, Any], 
                             request: AgentCreationRequest) -> Any:
        """Create agent using fallback implementation"""
        return {
            "agent_type": "fallback",
            "config": config,
            "status": "created"
        }
    
    async def _setup_agent_communication(self, agent_id: str, 
                                       request: AgentCreationRequest) -> str:
        """Set up communication capabilities for agent"""
        if hasattr(self, 'communication_manager') and self.communication_manager:
            try:
                await self.communication_manager.register_agent(agent_id, {
                    'specialization': request.specialization.value,
                    'tier': request.tier.value,
                    'capabilities': [cap.value for cap in request.requested_capabilities]
                })
                return f"comm_mgr_{agent_id}"
            except Exception as e:
                self.logger.error(f"Failed to setup communication for agent {agent_id}: {e}")
                return None
        return None
    
    def _allocate_agent_resources(self, request: AgentCreationRequest, 
                                tier_limits: TierQuotaLimits) -> Dict[str, Any]:
        """Allocate resources for agent based on tier and requirements"""
        template = self.specialization_templates[request.specialization]
        
        allocated_resources = {
            "memory_mb": min(tier_limits.max_memory_mb, 
                           template.resource_requirements.get("memory_mb", 512)),
            "cpu_priority": template.resource_requirements.get("cpu_priority", "normal"),
            "network_access": template.resource_requirements.get("network_access", False),
            "storage_access": template.resource_requirements.get("storage_access", False),
            "processing_timeout": tier_limits.max_processing_time_seconds,
            "priority_multiplier": tier_limits.priority_boost,
            "cost_multiplier": tier_limits.cost_multiplier
        }
        
        return allocated_resources
    
    def _create_performance_baseline(self, request: AgentCreationRequest) -> Dict[str, Any]:
        """Create performance baseline expectations"""
        template = self.specialization_templates[request.specialization]
        
        baseline = {
            "expected_performance": template.performance_expectations,
            "baseline_timestamp": datetime.now().isoformat(),
            "validation_score": 0.0,  # Will be updated during operation
            "success_rate": 0.0,     # Will be tracked during operation
            "average_response_time": 0.0,  # Will be measured during operation
            "resource_efficiency": 0.0     # Will be calculated during operation
        }
        
        return baseline
    
    @timer_decorator
    def get_agent_record(self, agent_id: str) -> Optional[CreatedAgentRecord]:
        """Retrieve agent record by ID"""
        return self.created_agents.get(agent_id)
    
    @timer_decorator
    def list_agents(self, tier: Optional[AgentTier] = None, 
                   specialization: Optional[AgentSpecialization] = None,
                   status: Optional[str] = None) -> List[CreatedAgentRecord]:
        """List agents with optional filtering"""
        agents = list(self.created_agents.values())
        
        if tier:
            agents = [a for a in agents if a.tier == tier]
        
        if specialization:
            agents = [a for a in agents if a.specialization == specialization]
        
        if status:
            agents = [a for a in agents if a.status == status]
        
        return agents
    
    @timer_decorator
    def get_tier_utilization(self, tier: AgentTier) -> Dict[str, Any]:
        """Get current tier utilization statistics"""
        tier_limits = self.tier_quotas.get(tier)
        if not tier_limits:
            return {"error": f"Unknown tier: {tier}"}
        
        tier_agents = [a for a in self.created_agents.values() 
                      if a.tier == tier and a.status == 'active']
        
        utilization = {
            "tier": tier.value,
            "current_agents": len(tier_agents),
            "max_agents": tier_limits.max_agents,
            "utilization_percentage": (len(tier_agents) / tier_limits.max_agents) * 100,
            "available_slots": tier_limits.max_agents - len(tier_agents),
            "allowed_capabilities": [cap.value for cap in tier_limits.allowed_capabilities],
            "restricted_capabilities": [cap.value for cap in tier_limits.restricted_capabilities],
            "resource_limits": {
                "memory_mb": tier_limits.max_memory_mb,
                "max_concurrent": tier_limits.max_concurrent_agents,
                "processing_timeout": tier_limits.max_processing_time_seconds
            }
        }
        
        return utilization
    
    @timer_decorator
    def get_factory_analytics(self) -> Dict[str, Any]:
        """Get comprehensive factory analytics and metrics"""
        analytics = {
            "factory_info": {
                "factory_id": self.factory_id,
                "version": self.version,
                "status": self.status.value,
                "uptime": str(datetime.now() - self.creation_metrics["last_reset"])
            },
            "creation_metrics": self.creation_metrics.copy(),
            "tier_utilization": {
                tier.value: self.get_tier_utilization(tier) 
                for tier in AgentTier
            },
            "specialization_capacity": {
                spec.value: len([a for a in self.created_agents.values() 
                               if a.specialization == spec and a.status == 'active'])
                for spec in AgentSpecialization
            },
            "active_requests": len(self.active_requests),
            "total_agents": len(self.created_agents),
            "performance_summary": {
                "average_creation_time": self.creation_metrics["average_creation_time"],
                "success_rate": (self.creation_metrics["successful_creations"] / 
                               max(1, self.creation_metrics["total_created"])) * 100,
                "failure_rate": (self.creation_metrics["failed_creations"] / 
                               max(1, self.creation_metrics["total_created"])) * 100
            }
        }
        
        return analytics
    
    def set_communication_manager(self, communication_manager) -> None:
        """Inject communication manager dependency"""
        self.communication_manager = communication_manager
        self.logger.info("Communication manager injected into factory")
    
    @timer_decorator
    def update_agent_status(self, agent_id: str, status: str) -> bool:
        """Update agent status"""
        if agent_id in self.created_agents:
            self.created_agents[agent_id].status = status
            self.created_agents[agent_id].last_updated = datetime.now()
            self.logger.info(f"Agent {agent_id} status updated to {status}")
            return True
        return False
    
    @timer_decorator
    async def deactivate_agent(self, agent_id: str) -> bool:
        """Deactivate an agent and clean up resources"""
        try:
            if agent_id not in self.created_agents:
                return False
            
            agent_record = self.created_agents[agent_id]
            
            # Update status
            agent_record.status = "deactivated"
            agent_record.last_updated = datetime.now()
            
            # Clean up communication if enabled
            if agent_record.communication_manager_id and hasattr(self, 'communication_manager'):
                try:
                    # Assuming the communication manager has a deregister method
                    await self.communication_manager.deregister_agent(agent_id)
                except Exception as e:
                    self.logger.error(f"Failed to deregister agent {agent_id} from communication: {e}")
            
            self.logger.info(f"Agent {agent_id} deactivated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to deactivate agent {agent_id}: {e}")
            return False

# High-level factory management functions
@timer_decorator
def create_tier_aware_factory() -> TierAwareAgentFactory:
    """Create and initialize a new Tier-Aware Agent Factory"""
    factory = TierAwareAgentFactory()
    return factory

@timer_decorator
async def quick_agent_creation_test() -> Dict[str, Any]:
    """Quick test of agent creation functionality"""
    try:
        factory = create_tier_aware_factory()
        
        # Test FREE tier agent creation
        free_request = AgentCreationRequest(
            specialization=AgentSpecialization.CASUAL,
            tier=AgentTier.FREE,
            owner_id="test_user_001",
            requested_capabilities=[AgentCapability.BASIC_REASONING]
        )
        
        result, record = await factory.create_agent(free_request)
        
        test_results = {
            "factory_initialized": True,
            "agent_creation_result": result.value,
            "agent_created": record is not None,
            "agent_id": record.agent_id if record else None,
            "factory_analytics": factory.get_factory_analytics()
        }
        
        return test_results
        
    except Exception as e:
        return {
            "factory_initialized": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# Main execution for testing
if __name__ == "__main__":
    print("🏭 Pydantic AI Tier-Aware Agent Factory - Standalone Test")
    print("=" * 70)
    
    async def main():
        # Test basic factory functionality
        test_results = await quick_agent_creation_test()
        
        print("🧪 Test Results:")
        for key, value in test_results.items():
            print(f"  {key}: {value}")
        
        print("\n✅ Tier-Aware Agent Factory test complete!")
    
    asyncio.run(main())