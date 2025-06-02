#!/usr/bin/env python3
"""
SANDBOX FILE: For testing/development. See .cursorrules.

Pydantic AI Production Communication Workflows System
===================================================

* Purpose: Deploy production-ready communication workflows with live multi-agent coordination,
  real-time message routing, and intelligent workflow orchestration for MLACS
* Issues & Complexity Summary: Complex workflow state management, real-time communication,
  multi-agent coordination, and production deployment challenges
* Key Complexity Drivers:
  - Logic Scope (Est. LoC): ~1,600
  - Core Algorithm Complexity: Very High
  - Dependencies: 6 New, 4 Mod
  - State Management Complexity: Very High
  - Novelty/Uncertainty Factor: High
* AI Pre-Task Self-Assessment (Est. Solution Difficulty %): 90%
* Problem Estimate (Inherent Problem Difficulty %): 92%
* Initial Code Complexity Estimate %: 90%
* Justification for Estimates: Production workflows with real-time coordination and live deployment
* Final Code Complexity (Actual %): TBD
* Overall Result Score (Success & Quality %): TBD
* Key Variances/Learnings: TBD
* Last Updated: 2025-01-06

Provides:
- Production-ready communication workflows
- Live multi-agent coordination
- Real-time message routing and orchestration
- Workflow state management and persistence
- Performance monitoring and optimization
"""

import asyncio
import json
import logging
import sqlite3
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set, Tuple, Callable, AsyncGenerator
from dataclasses import dataclass, field
import threading
import hashlib
from collections import defaultdict, deque
import pickle
import websockets
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import signal
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Timer decorator for performance monitoring
def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} execution time: {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# Async timer decorator
def async_timer_decorator(func):
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} async execution time: {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# Try to import Pydantic AI, fall back to basic implementations
try:
    from pydantic import BaseModel, Field, ValidationError, validator
    from pydantic_ai import Agent
    PYDANTIC_AI_AVAILABLE = True
    logger.info("Pydantic AI successfully imported")
except ImportError:
    logger.warning("Pydantic AI not available, using fallback implementations")
    PYDANTIC_AI_AVAILABLE = False
    
    # Fallback BaseModel
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        def dict(self):
            return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        
        def json(self):
            return json.dumps(self.dict())
    
    # Fallback Field
    def Field(**kwargs):
        return kwargs.get('default', None)
    
    # Fallback ValidationError
    class ValidationError(Exception):
        pass
    
    # Fallback validator
    def validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    # Fallback Agent
    class Agent:
        def __init__(self, model=None, **kwargs):
            self.model = model
            for key, value in kwargs.items():
                setattr(self, key, value)

# Try to import advanced memory integration
try:
    from sources.pydantic_ai_advanced_memory_integration_production import (
        AdvancedMemoryIntegrationSystem,
        MemoryType,
        MemoryPriority
    )
    MEMORY_INTEGRATION_AVAILABLE = True
    logger.info("Advanced Memory Integration System available")
except ImportError:
    logger.warning("Memory integration not available, using fallback")
    MEMORY_INTEGRATION_AVAILABLE = False
    
    # Fallback memory classes
    class MemoryType(Enum):
        EPISODIC = "episodic"
        PROCEDURAL = "procedural"
    
    class MemoryPriority(Enum):
        HIGH = 2
        MEDIUM = 3
    
    class AdvancedMemoryIntegrationSystem:
        def __init__(self, **kwargs):
            self.memories = {}
        
        def store_memory(self, **kwargs):
            return str(uuid.uuid4())
        
        def retrieve_memory(self, memory_id):
            return None

# ================================
# Core Communication Workflow Enums and Models
# ================================

class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class MessageType(Enum):
    """Types of messages in the communication system"""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    STATUS_UPDATE = "status_update"
    COORDINATION = "coordination"
    BROADCAST = "broadcast"
    HEARTBEAT = "heartbeat"
    ERROR = "error"

class AgentRole(Enum):
    """Agent roles in the communication workflow"""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    MONITOR = "monitor"
    PROXY = "proxy"
    SPECIALIST = "specialist"

class CommunicationProtocol(Enum):
    """Communication protocols supported"""
    WEBSOCKET = "websocket"
    HTTP = "http"
    DIRECT = "direct"
    QUEUE = "queue"

class WorkflowPriority(Enum):
    """Workflow execution priority levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5

# ================================
# Communication Data Models
# ================================

class CommunicationMessage(BaseModel):
    """Individual communication message"""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = Field(default=MessageType.TASK_REQUEST)
    sender_id: str = ""
    receiver_id: str = ""
    content: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    priority: WorkflowPriority = Field(default=WorkflowPriority.MEDIUM)
    correlation_id: str = ""
    response_required: bool = False

    def __init__(self, **kwargs):
        if 'timestamp' not in kwargs:
            kwargs['timestamp'] = datetime.now()
        if 'correlation_id' not in kwargs:
            kwargs['correlation_id'] = str(uuid.uuid4())
        super().__init__(**kwargs)

class WorkflowDefinition(BaseModel):
    """Definition of a communication workflow"""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    participants: List[str] = Field(default_factory=list)
    triggers: List[Dict[str, Any]] = Field(default_factory=list)
    conditions: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = 300
    retry_policy: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def __init__(self, **kwargs):
        # Ensure datetime fields are set
        now = datetime.now()
        if 'created_at' not in kwargs:
            kwargs['created_at'] = now
        if 'updated_at' not in kwargs:
            kwargs['updated_at'] = now
        if 'steps' not in kwargs:
            kwargs['steps'] = []
        if 'participants' not in kwargs:
            kwargs['participants'] = []
        if 'triggers' not in kwargs:
            kwargs['triggers'] = []
        if 'conditions' not in kwargs:
            kwargs['conditions'] = {}
        if 'retry_policy' not in kwargs:
            kwargs['retry_policy'] = {}
        
        super().__init__(**kwargs)

class WorkflowExecution(BaseModel):
    """Runtime execution of a workflow"""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = ""
    status: WorkflowStatus = Field(default=WorkflowStatus.PENDING)
    current_step: int = 0
    context: Dict[str, Any] = Field(default_factory=dict)
    variables: Dict[str, Any] = Field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_details: Optional[str] = None
    participants: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    message_history: List[str] = Field(default_factory=list)

class AgentRegistration(BaseModel):
    """Agent registration in the communication system"""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    role: AgentRole = Field(default=AgentRole.WORKER)
    capabilities: List[str] = Field(default_factory=list)
    protocols: List[CommunicationProtocol] = Field(default_factory=list)
    endpoint: str = ""
    status: str = "active"
    last_heartbeat: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **kwargs):
        # Handle protocol conversion from strings
        if 'protocols' in kwargs and isinstance(kwargs['protocols'], list):
            protocols = []
            for protocol in kwargs['protocols']:
                if isinstance(protocol, str):
                    try:
                        # Map common protocol names
                        protocol_map = {
                            'websocket': CommunicationProtocol.WEBSOCKET,
                            'http': CommunicationProtocol.HTTP,
                            'direct': CommunicationProtocol.DIRECT,
                            'queue': CommunicationProtocol.QUEUE
                        }
                        if protocol in protocol_map:
                            protocols.append(protocol_map[protocol])
                        else:
                            protocols.append(CommunicationProtocol(protocol))
                    except ValueError:
                        # Default to DIRECT if invalid protocol
                        protocols.append(CommunicationProtocol.DIRECT)
                else:
                    protocols.append(protocol)
            kwargs['protocols'] = protocols
        elif 'protocols' in kwargs and isinstance(kwargs['protocols'], str):
            # Handle single protocol as string
            try:
                kwargs['protocols'] = [CommunicationProtocol(kwargs['protocols'])]
            except ValueError:
                kwargs['protocols'] = [CommunicationProtocol.DIRECT]
        
        # Handle role conversion from string
        if 'role' in kwargs and isinstance(kwargs['role'], str):
            try:
                kwargs['role'] = AgentRole(kwargs['role'])
            except ValueError:
                kwargs['role'] = AgentRole.WORKER
        
        # Ensure datetime fields are set
        if 'last_heartbeat' not in kwargs:
            kwargs['last_heartbeat'] = datetime.now()
        if 'capabilities' not in kwargs:
            kwargs['capabilities'] = []
        if 'protocols' not in kwargs:
            kwargs['protocols'] = [CommunicationProtocol.DIRECT]
        if 'metadata' not in kwargs:
            kwargs['metadata'] = {}
        
        super().__init__(**kwargs)

class CommunicationMetrics(BaseModel):
    """Communication system performance metrics"""
    
    total_messages: int = 0
    active_workflows: int = 0
    completed_workflows: int = 0
    failed_workflows: int = 0
    average_response_time_ms: float = 0.0
    message_throughput_per_second: float = 0.0
    active_agents: int = 0
    error_rate: float = 0.0

# ================================
# Production Communication Workflows System
# ================================

class ProductionCommunicationWorkflowsSystem:
    """
    Production-ready communication workflows system with live multi-agent 
    coordination and real-time message routing
    """
    
    def __init__(
        self,
        db_path: str = "communication_workflows.db",
        websocket_port: int = 8765,
        http_port: int = 8080,
        enable_persistence: bool = True,
        enable_monitoring: bool = True
    ):
        self.db_path = db_path
        self.websocket_port = websocket_port
        self.http_port = http_port
        self.enable_persistence = enable_persistence
        self.enable_monitoring = enable_monitoring
        
        # Core system components
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.agents: Dict[str, AgentRegistration] = {}
        self.message_queue: deque = deque()
        self.message_history: Dict[str, List[CommunicationMessage]] = defaultdict(list)
        
        # Communication infrastructure
        self.websocket_server = None
        self.http_server = None
        self.connected_clients: Set[websockets.WebSocketServerProtocol] = set()
        self.message_handlers: Dict[MessageType, Callable] = {}
        
        # Performance tracking
        self.metrics = CommunicationMetrics()
        self.response_times: deque = deque(maxlen=1000)
        self.message_counts: defaultdict = defaultdict(int)
        self.error_counts: defaultdict = defaultdict(int)
        
        # Threading and async management
        self.loop = None
        self.executor = ThreadPoolExecutor(max_workers=10)
        self._shutdown_event = threading.Event()
        self._lock = threading.RLock()
        
        # Memory integration
        self.memory_system = None
        if MEMORY_INTEGRATION_AVAILABLE:
            try:
                self.memory_system = AdvancedMemoryIntegrationSystem(
                    db_path="workflow_memory.db"
                )
                logger.info("Memory integration system initialized")
            except Exception as e:
                logger.warning(f"Memory integration initialization failed: {e}")
        
        # Initialize system
        self._initialize_system()

    def _initialize_system(self):
        """Initialize the communication workflows system"""
        try:
            # Initialize database
            if self.enable_persistence:
                self._initialize_database()
            
            # Register default message handlers
            self._register_default_handlers()
            
            # Load existing workflows and executions
            self._load_existing_data()
            
            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            logger.info("Production Communication Workflows System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize communication system: {e}")
            raise

    def _initialize_database(self):
        """Initialize SQLite database for persistence"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS workflows (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    description TEXT,
                    version TEXT,
                    steps TEXT,
                    participants TEXT,
                    triggers TEXT,
                    conditions TEXT,
                    timeout_seconds INTEGER,
                    retry_policy TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS executions (
                    id TEXT PRIMARY KEY,
                    workflow_id TEXT,
                    status TEXT,
                    current_step INTEGER,
                    context TEXT,
                    variables TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    error_details TEXT,
                    participants TEXT,
                    message_history TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS agents (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT,
                    role TEXT,
                    capabilities TEXT,
                    protocols TEXT,
                    endpoint TEXT,
                    status TEXT,
                    last_heartbeat TEXT,
                    metadata TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    type TEXT,
                    sender_id TEXT,
                    receiver_id TEXT,
                    content TEXT,
                    metadata TEXT,
                    timestamp TEXT,
                    expires_at TEXT,
                    priority INTEGER,
                    correlation_id TEXT,
                    response_required BOOLEAN
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_workflow_name ON workflows(name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_execution_status ON executions(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_status ON agents(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_message_timestamp ON messages(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_message_correlation ON messages(correlation_id)')
            
            conn.commit()
            conn.close()
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    def _register_default_handlers(self):
        """Register default message handlers"""
        self.message_handlers = {
            MessageType.TASK_REQUEST: self._handle_task_request,
            MessageType.TASK_RESPONSE: self._handle_task_response,
            MessageType.STATUS_UPDATE: self._handle_status_update,
            MessageType.COORDINATION: self._handle_coordination,
            MessageType.BROADCAST: self._handle_broadcast,
            MessageType.HEARTBEAT: self._handle_heartbeat,
            MessageType.ERROR: self._handle_error
        }

    @timer_decorator
    def create_workflow(self, workflow_def: Dict[str, Any]) -> str:
        """Create a new workflow definition"""
        try:
            workflow = WorkflowDefinition(**workflow_def)
            workflow_id = workflow.id
            
            self.workflows[workflow_id] = workflow
            
            if self.enable_persistence:
                self._persist_workflow(workflow)
            
            # Store in memory system
            if self.memory_system:
                try:
                    # Convert datetime objects to strings for JSON serialization
                    workflow_dict = workflow.dict()
                    if 'created_at' in workflow_dict and workflow_dict['created_at']:
                        workflow_dict['created_at'] = workflow_dict['created_at'].isoformat()
                    if 'updated_at' in workflow_dict and workflow_dict['updated_at']:
                        workflow_dict['updated_at'] = workflow_dict['updated_at'].isoformat()
                    
                    self.memory_system.store_memory(
                        content={
                            'workflow_definition': workflow_dict,
                            'operation': 'create_workflow'
                        },
                        memory_type=MemoryType.PROCEDURAL,
                        priority=MemoryPriority.HIGH,
                        tags={'workflow', 'creation', workflow.name}
                    )
                except Exception as e:
                    logger.warning(f"Failed to store workflow in memory system: {e}")
            
            logger.info(f"Created workflow {workflow_id}: {workflow.name}")
            return workflow_id
            
        except Exception as e:
            logger.error(f"Failed to create workflow: {e}")
            raise

    @timer_decorator
    def start_workflow(self, workflow_id: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Start execution of a workflow"""
        try:
            if workflow_id not in self.workflows:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            workflow = self.workflows[workflow_id]
            execution = WorkflowExecution(
                workflow_id=workflow_id,
                context=context or {},
                started_at=datetime.now(),
                status=WorkflowStatus.RUNNING
            )
            
            execution_id = execution.id
            self.executions[execution_id] = execution
            
            if self.enable_persistence:
                self._persist_execution(execution)
            
            # Store in memory system
            if self.memory_system:
                self.memory_system.store_memory(
                    content={
                        'workflow_execution': execution.dict(),
                        'operation': 'start_workflow'
                    },
                    memory_type=MemoryType.EPISODIC,
                    priority=MemoryPriority.HIGH,
                    tags={'workflow', 'execution', 'start'}
                )
            
            # Begin workflow execution
            asyncio.create_task(self._execute_workflow(execution_id))
            
            logger.info(f"Started workflow execution {execution_id}")
            return execution_id
            
        except Exception as e:
            logger.error(f"Failed to start workflow: {e}")
            raise

    async def _execute_workflow(self, execution_id: str):
        """Execute a workflow asynchronously"""
        try:
            execution = self.executions[execution_id]
            workflow = self.workflows[execution.workflow_id]
            
            for step_index, step in enumerate(workflow.steps):
                if execution.status != WorkflowStatus.RUNNING:
                    break
                
                execution.current_step = step_index
                
                # Execute workflow step
                success = await self._execute_workflow_step(execution, step)
                
                if not success:
                    execution.status = WorkflowStatus.FAILED
                    execution.error_details = f"Step {step_index} failed"
                    break
            
            if execution.status == WorkflowStatus.RUNNING:
                execution.status = WorkflowStatus.COMPLETED
                execution.completed_at = datetime.now()
            
            # Update persistence
            if self.enable_persistence:
                self._persist_execution(execution)
            
            # Store completion in memory
            if self.memory_system:
                self.memory_system.store_memory(
                    content={
                        'workflow_execution': execution.dict(),
                        'operation': 'complete_workflow',
                        'final_status': execution.status.value
                    },
                    memory_type=MemoryType.EPISODIC,
                    priority=MemoryPriority.MEDIUM,
                    tags={'workflow', 'completion', execution.status.value}
                )
            
            logger.info(f"Workflow execution {execution_id} completed with status: {execution.status.value}")
            
        except Exception as e:
            logger.error(f"Workflow execution {execution_id} failed: {e}")
            if execution_id in self.executions:
                self.executions[execution_id].status = WorkflowStatus.FAILED
                self.executions[execution_id].error_details = str(e)

    async def _execute_workflow_step(self, execution: WorkflowExecution, step: Dict[str, Any]) -> bool:
        """Execute a single workflow step"""
        try:
            step_type = step.get('type', 'message')
            
            if step_type == 'message':
                return await self._execute_message_step(execution, step)
            elif step_type == 'coordination':
                return await self._execute_coordination_step(execution, step)
            elif step_type == 'wait':
                return await self._execute_wait_step(execution, step)
            elif step_type == 'condition':
                return await self._execute_condition_step(execution, step)
            else:
                logger.warning(f"Unknown step type: {step_type}")
                return True
            
        except Exception as e:
            logger.error(f"Step execution failed: {e}")
            return False

    async def _execute_message_step(self, execution: WorkflowExecution, step: Dict[str, Any]) -> bool:
        """Execute a message step"""
        try:
            message_config = step.get('message', {})
            
            message = CommunicationMessage(
                type=MessageType(message_config.get('type', 'task_request')),
                sender_id=message_config.get('sender_id', 'workflow'),
                receiver_id=message_config.get('receiver_id', ''),
                content=message_config.get('content', {}),
                metadata={
                    'execution_id': execution.id,
                    'workflow_id': execution.workflow_id,
                    'step_index': execution.current_step
                }
            )
            
            # Send message
            await self.send_message(message)
            
            # If response required, wait for it
            if message.response_required:
                timeout = step.get('timeout', 30)
                response = await self._wait_for_response(message.correlation_id, timeout)
                if response:
                    execution.context[f'step_{execution.current_step}_response'] = response.dict()
                else:
                    logger.warning(f"No response received for message {message.id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Message step execution failed: {e}")
            return False

    async def _execute_coordination_step(self, execution: WorkflowExecution, step: Dict[str, Any]) -> bool:
        """Execute a coordination step"""
        try:
            coordination_config = step.get('coordination', {})
            participants = coordination_config.get('participants', [])
            
            # Send coordination messages to all participants
            coordination_tasks = []
            for participant in participants:
                message = CommunicationMessage(
                    type=MessageType.COORDINATION,
                    sender_id='workflow_coordinator',
                    receiver_id=participant,
                    content=coordination_config.get('content', {}),
                    metadata={
                        'execution_id': execution.id,
                        'coordination_type': coordination_config.get('type', 'sync')
                    }
                )
                coordination_tasks.append(self.send_message(message))
            
            # Wait for all coordination messages to be sent
            await asyncio.gather(*coordination_tasks)
            
            return True
            
        except Exception as e:
            logger.error(f"Coordination step execution failed: {e}")
            return False

    async def _execute_wait_step(self, execution: WorkflowExecution, step: Dict[str, Any]) -> bool:
        """Execute a wait step"""
        try:
            wait_seconds = step.get('wait_seconds', 1)
            await asyncio.sleep(wait_seconds)
            return True
            
        except Exception as e:
            logger.error(f"Wait step execution failed: {e}")
            return False

    async def _execute_condition_step(self, execution: WorkflowExecution, step: Dict[str, Any]) -> bool:
        """Execute a condition step"""
        try:
            condition = step.get('condition', {})
            variable = condition.get('variable', '')
            operator = condition.get('operator', '==')
            value = condition.get('value', '')
            
            current_value = execution.variables.get(variable)
            
            if operator == '==':
                result = current_value == value
            elif operator == '!=':
                result = current_value != value
            elif operator == '>':
                result = current_value > value
            elif operator == '<':
                result = current_value < value
            else:
                result = True
            
            return result
            
        except Exception as e:
            logger.error(f"Condition step execution failed: {e}")
            return False

    @async_timer_decorator
    async def send_message(self, message: CommunicationMessage):
        """Send a message through the communication system"""
        try:
            # Add to message history
            self.message_history[message.correlation_id].append(message)
            
            # Store in persistent storage
            if self.enable_persistence:
                self._persist_message(message)
            
            # Route message based on receiver
            if message.receiver_id in self.agents:
                agent = self.agents[message.receiver_id]
                await self._route_message_to_agent(message, agent)
            else:
                # Broadcast to all connected clients
                await self._broadcast_message(message)
            
            # Update metrics
            self.metrics.total_messages += 1
            self.message_counts[message.type.value] += 1
            
            logger.debug(f"Sent message {message.id} from {message.sender_id} to {message.receiver_id}")
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            self.error_counts['send_message'] += 1

    async def _route_message_to_agent(self, message: CommunicationMessage, agent: AgentRegistration):
        """Route message to a specific agent"""
        try:
            if CommunicationProtocol.WEBSOCKET in agent.protocols:
                await self._send_websocket_message(message, agent.endpoint)
            elif CommunicationProtocol.HTTP in agent.protocols:
                await self._send_http_message(message, agent.endpoint)
            else:
                # Store for direct retrieval
                self.message_queue.append(message)
            
        except Exception as e:
            logger.error(f"Failed to route message to agent {agent.agent_id}: {e}")

    async def _send_websocket_message(self, message: CommunicationMessage, endpoint: str):
        """Send message via WebSocket"""
        try:
            # Send to all connected WebSocket clients
            if self.connected_clients:
                message_data = json.dumps(message.dict(), default=str)
                await asyncio.gather(
                    *[client.send(message_data) for client in self.connected_clients],
                    return_exceptions=True
                )
        except Exception as e:
            logger.error(f"WebSocket message send failed: {e}")

    async def _send_http_message(self, message: CommunicationMessage, endpoint: str):
        """Send message via HTTP"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    json=message.dict(),
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status != 200:
                        logger.warning(f"HTTP message send failed with status {response.status}")
        except Exception as e:
            logger.error(f"HTTP message send failed: {e}")

    async def _broadcast_message(self, message: CommunicationMessage):
        """Broadcast message to all connected clients"""
        try:
            if self.connected_clients:
                message_data = json.dumps(message.dict(), default=str)
                await asyncio.gather(
                    *[client.send(message_data) for client in self.connected_clients],
                    return_exceptions=True
                )
        except Exception as e:
            logger.error(f"Broadcast message failed: {e}")

    async def _wait_for_response(self, correlation_id: str, timeout: int) -> Optional[CommunicationMessage]:
        """Wait for a response message with specific correlation ID"""
        try:
            start_time = time.time()
            while time.time() - start_time < timeout:
                # Check message history for response
                for message in self.message_history[correlation_id]:
                    if message.type == MessageType.TASK_RESPONSE:
                        return message
                
                await asyncio.sleep(0.1)
            
            return None
            
        except Exception as e:
            logger.error(f"Wait for response failed: {e}")
            return None

    # Message handlers
    async def _handle_task_request(self, message: CommunicationMessage):
        """Handle task request message"""
        logger.info(f"Handling task request: {message.id}")
        # Implementation depends on specific task requirements

    async def _handle_task_response(self, message: CommunicationMessage):
        """Handle task response message"""
        logger.info(f"Handling task response: {message.id}")
        # Update execution context or workflow state

    async def _handle_status_update(self, message: CommunicationMessage):
        """Handle status update message"""
        logger.info(f"Handling status update: {message.id}")
        # Update agent or workflow status

    async def _handle_coordination(self, message: CommunicationMessage):
        """Handle coordination message"""
        logger.info(f"Handling coordination: {message.id}")
        # Coordinate between multiple agents

    async def _handle_broadcast(self, message: CommunicationMessage):
        """Handle broadcast message"""
        logger.info(f"Handling broadcast: {message.id}")
        await self._broadcast_message(message)

    async def _handle_heartbeat(self, message: CommunicationMessage):
        """Handle heartbeat message"""
        if message.sender_id in self.agents:
            self.agents[message.sender_id].last_heartbeat = datetime.now()

    async def _handle_error(self, message: CommunicationMessage):
        """Handle error message"""
        logger.error(f"Received error message: {message.content}")
        self.error_counts['received_errors'] += 1

    @timer_decorator
    def register_agent(self, agent_config: Dict[str, Any]) -> str:
        """Register an agent in the communication system"""
        try:
            agent = AgentRegistration(**agent_config)
            agent_id = agent.id
            
            self.agents[agent_id] = agent
            
            if self.enable_persistence:
                self._persist_agent(agent)
            
            logger.info(f"Registered agent {agent_id}: {agent.agent_id}")
            return agent_id
            
        except Exception as e:
            logger.error(f"Failed to register agent: {e}")
            raise

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            self._update_metrics()
            
            return {
                'system_status': 'operational',
                'workflows': {
                    'total_definitions': len(self.workflows),
                    'active_executions': len([e for e in self.executions.values() if e.status == WorkflowStatus.RUNNING]),
                    'completed_executions': len([e for e in self.executions.values() if e.status == WorkflowStatus.COMPLETED]),
                    'failed_executions': len([e for e in self.executions.values() if e.status == WorkflowStatus.FAILED])
                },
                'communication': {
                    'registered_agents': len(self.agents),
                    'connected_clients': len(self.connected_clients),
                    'total_messages': self.metrics.total_messages,
                    'message_queue_size': len(self.message_queue)
                },
                'performance': {
                    'average_response_time_ms': self.metrics.average_response_time_ms,
                    'message_throughput_per_second': self.metrics.message_throughput_per_second,
                    'error_rate': self.metrics.error_rate
                },
                'integrations': {
                    'memory_system': self.memory_system is not None,
                    'persistence': self.enable_persistence,
                    'monitoring': self.enable_monitoring
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {'error': str(e)}

    def _update_metrics(self):
        """Update system performance metrics"""
        try:
            # Calculate average response time
            if self.response_times:
                self.metrics.average_response_time_ms = sum(self.response_times) / len(self.response_times)
            
            # Calculate error rate
            total_operations = sum(self.message_counts.values())
            total_errors = sum(self.error_counts.values())
            if total_operations > 0:
                self.metrics.error_rate = total_errors / total_operations
            
            # Update active counts
            self.metrics.active_workflows = len([e for e in self.executions.values() if e.status == WorkflowStatus.RUNNING])
            self.metrics.completed_workflows = len([e for e in self.executions.values() if e.status == WorkflowStatus.COMPLETED])
            self.metrics.failed_workflows = len([e for e in self.executions.values() if e.status == WorkflowStatus.FAILED])
            self.metrics.active_agents = len([a for a in self.agents.values() if a.status == 'active'])
            
        except Exception as e:
            logger.error(f"Metrics update failed: {e}")

    # Persistence methods
    def _persist_workflow(self, workflow: WorkflowDefinition):
        """Persist workflow to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Ensure datetime fields are valid
            created_at = workflow.created_at.isoformat() if workflow.created_at else datetime.now().isoformat()
            updated_at = workflow.updated_at.isoformat() if workflow.updated_at else datetime.now().isoformat()
            
            cursor.execute('''
                INSERT OR REPLACE INTO workflows 
                (id, name, description, version, steps, participants, triggers, 
                 conditions, timeout_seconds, retry_policy, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                workflow.id, workflow.name, workflow.description, workflow.version,
                json.dumps(workflow.steps if workflow.steps else []), 
                json.dumps(workflow.participants if workflow.participants else []),
                json.dumps(workflow.triggers if workflow.triggers else []), 
                json.dumps(workflow.conditions if workflow.conditions else {}),
                workflow.timeout_seconds, 
                json.dumps(workflow.retry_policy if workflow.retry_policy else {}),
                created_at, updated_at
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to persist workflow: {e}")

    def _persist_execution(self, execution: WorkflowExecution):
        """Persist execution to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO executions 
                (id, workflow_id, status, current_step, context, variables, 
                 started_at, completed_at, error_details, participants, message_history)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                execution.id, execution.workflow_id, execution.status.value,
                execution.current_step, json.dumps(execution.context),
                json.dumps(execution.variables),
                execution.started_at.isoformat() if execution.started_at else None,
                execution.completed_at.isoformat() if execution.completed_at else None,
                execution.error_details, json.dumps(execution.participants),
                json.dumps(execution.message_history)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to persist execution: {e}")

    def _persist_agent(self, agent: AgentRegistration):
        """Persist agent to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Ensure datetime field is valid
            last_heartbeat = agent.last_heartbeat.isoformat() if agent.last_heartbeat else datetime.now().isoformat()
            
            cursor.execute('''
                INSERT OR REPLACE INTO agents 
                (id, agent_id, role, capabilities, protocols, endpoint, 
                 status, last_heartbeat, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                agent.id, agent.agent_id, agent.role.value,
                json.dumps(agent.capabilities if agent.capabilities else []), 
                json.dumps([p.value for p in agent.protocols] if agent.protocols else ['direct']),
                agent.endpoint, agent.status, last_heartbeat,
                json.dumps(agent.metadata if agent.metadata else {})
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to persist agent: {e}")

    def _persist_message(self, message: CommunicationMessage):
        """Persist message to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO messages 
                (id, type, sender_id, receiver_id, content, metadata, 
                 timestamp, expires_at, priority, correlation_id, response_required)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                message.id, message.type.value, message.sender_id, message.receiver_id,
                json.dumps(message.content), json.dumps(message.metadata),
                message.timestamp.isoformat(),
                message.expires_at.isoformat() if message.expires_at else None,
                message.priority.value, message.correlation_id, message.response_required
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to persist message: {e}")

    def _load_existing_data(self):
        """Load existing data from database"""
        if not self.enable_persistence:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Load workflows
            cursor.execute('SELECT * FROM workflows')
            for row in cursor.fetchall():
                workflow = WorkflowDefinition(
                    id=row[0], name=row[1], description=row[2], version=row[3],
                    steps=json.loads(row[4]), participants=json.loads(row[5]),
                    triggers=json.loads(row[6]), conditions=json.loads(row[7]),
                    timeout_seconds=row[8], retry_policy=json.loads(row[9]),
                    created_at=datetime.fromisoformat(row[10]),
                    updated_at=datetime.fromisoformat(row[11])
                )
                self.workflows[workflow.id] = workflow
            
            # Load agents
            cursor.execute('SELECT * FROM agents WHERE status = "active"')
            for row in cursor.fetchall():
                agent = AgentRegistration(
                    id=row[0], agent_id=row[1], role=AgentRole(row[2]),
                    capabilities=json.loads(row[3]),
                    protocols=[CommunicationProtocol(p) for p in json.loads(row[4])],
                    endpoint=row[5], status=row[6],
                    last_heartbeat=datetime.fromisoformat(row[7]),
                    metadata=json.loads(row[8])
                )
                self.agents[agent.id] = agent
            
            conn.close()
            logger.info(f"Loaded {len(self.workflows)} workflows and {len(self.agents)} agents")
            
        except Exception as e:
            logger.warning(f"Failed to load existing data: {e}")

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self._shutdown_event.set()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def start_servers(self):
        """Start WebSocket and HTTP servers"""
        try:
            # Start WebSocket server
            if self.websocket_port:
                self.websocket_server = await websockets.serve(
                    self._websocket_handler,
                    "localhost",
                    self.websocket_port
                )
                logger.info(f"WebSocket server started on port {self.websocket_port}")
            
            # HTTP server would be started here if needed
            logger.info("Communication servers started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start servers: {e}")
            raise

    async def _websocket_handler(self, websocket, path):
        """Handle WebSocket connections"""
        try:
            self.connected_clients.add(websocket)
            logger.info(f"WebSocket client connected: {websocket.remote_address}")
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    comm_message = CommunicationMessage(**data)
                    
                    # Handle message based on type
                    handler = self.message_handlers.get(comm_message.type)
                    if handler:
                        await handler(comm_message)
                    
                except Exception as e:
                    logger.error(f"WebSocket message handling failed: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket client disconnected")
        except Exception as e:
            logger.error(f"WebSocket handler error: {e}")
        finally:
            self.connected_clients.discard(websocket)

    async def shutdown(self):
        """Gracefully shutdown the communication system"""
        try:
            logger.info("Shutting down communication system...")
            
            # Close WebSocket server
            if self.websocket_server:
                self.websocket_server.close()
                await self.websocket_server.wait_closed()
            
            # Close HTTP server
            if self.http_server:
                await self.http_server.cleanup()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            logger.info("Communication system shutdown complete")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")

# ================================
# Communication Workflow Factory
# ================================

class CommunicationWorkflowFactory:
    """Factory for creating communication workflow instances"""
    
    @staticmethod
    def create_production_system(
        config: Optional[Dict[str, Any]] = None
    ) -> ProductionCommunicationWorkflowsSystem:
        """Create a configured production communication system"""
        
        default_config = {
            'db_path': 'communication_workflows.db',
            'websocket_port': 8765,
            'http_port': 8080,
            'enable_persistence': True,
            'enable_monitoring': True
        }
        
        if config:
            default_config.update(config)
        
        return ProductionCommunicationWorkflowsSystem(**default_config)

# ================================
# Export Classes
# ================================

__all__ = [
    'ProductionCommunicationWorkflowsSystem',
    'CommunicationWorkflowFactory',
    'CommunicationMessage',
    'WorkflowDefinition',
    'WorkflowExecution',
    'AgentRegistration',
    'CommunicationMetrics',
    'WorkflowStatus',
    'MessageType',
    'AgentRole',
    'CommunicationProtocol',
    'WorkflowPriority',
    'timer_decorator',
    'async_timer_decorator'
]

# ================================
# Demo Functions
# ================================

async def demo_production_communication_workflows():
    """Demonstrate production communication workflows capabilities"""
    
    print("🔄 Production Communication Workflows Demo")
    print("=" * 50)
    
    # Create communication system
    comm_system = CommunicationWorkflowFactory.create_production_system({
        'websocket_port': 8766,  # Different port for demo
        'enable_persistence': True
    })
    
    print("\n1. Creating workflow definition...")
    
    # Create a sample workflow
    workflow_def = {
        'name': 'Multi-Agent Coordination Workflow',
        'description': 'Demonstrates multi-agent coordination with real-time communication',
        'steps': [
            {
                'type': 'message',
                'message': {
                    'type': 'task_request',
                    'sender_id': 'coordinator',
                    'receiver_id': 'agent_1',
                    'content': {'task': 'analyze_data', 'priority': 'high'},
                    'response_required': True
                },
                'timeout': 30
            },
            {
                'type': 'coordination',
                'coordination': {
                    'type': 'sync',
                    'participants': ['agent_1', 'agent_2', 'agent_3'],
                    'content': {'action': 'synchronize_state'}
                }
            },
            {
                'type': 'wait',
                'wait_seconds': 2
            },
            {
                'type': 'message',
                'message': {
                    'type': 'broadcast',
                    'sender_id': 'coordinator',
                    'content': {'message': 'Workflow completed successfully'}
                }
            }
        ],
        'participants': ['coordinator', 'agent_1', 'agent_2', 'agent_3'],
        'timeout_seconds': 300
    }
    
    workflow_id = comm_system.create_workflow(workflow_def)
    print(f"✅ Created workflow: {workflow_id}")
    
    print("\n2. Registering agents...")
    
    # Register agents
    agents = []
    for i in range(1, 4):
        agent_config = {
            'agent_id': f'agent_{i}',
            'role': 'worker',
            'capabilities': ['data_analysis', 'communication'],
            'protocols': ['websocket', 'direct'],
            'endpoint': f'ws://localhost:8766/agent_{i}',
            'status': 'active'
        }
        agent_id = comm_system.register_agent(agent_config)
        agents.append(agent_id)
        print(f"✅ Registered agent_{i}: {agent_id[:8]}...")
    
    print("\n3. Starting workflow execution...")
    
    # Start workflow
    execution_id = comm_system.start_workflow(
        workflow_id,
        context={'demo_mode': True, 'start_time': datetime.now().isoformat()}
    )
    print(f"✅ Started workflow execution: {execution_id[:8]}...")
    
    print("\n4. Simulating message communication...")
    
    # Send some test messages
    test_messages = [
        {
            'type': 'heartbeat',
            'sender_id': 'agent_1',
            'receiver_id': 'coordinator',
            'content': {'status': 'active', 'timestamp': datetime.now().isoformat()}
        },
        {
            'type': 'status_update',
            'sender_id': 'agent_2',
            'receiver_id': 'coordinator',
            'content': {'progress': 50, 'task': 'data_processing'}
        },
        {
            'type': 'task_response',
            'sender_id': 'agent_3',
            'receiver_id': 'coordinator',
            'content': {'result': 'analysis_complete', 'data': {'accuracy': 95.5}}
        }
    ]
    
    for msg_data in test_messages:
        message = CommunicationMessage(**msg_data)
        await comm_system.send_message(message)
        print(f"✅ Sent {message.type.value} message: {message.id[:8]}...")
    
    # Wait a bit for workflow processing
    await asyncio.sleep(3)
    
    print("\n5. System status and metrics...")
    
    status = comm_system.get_system_status()
    print(f"✅ System status: {status['system_status']}")
    print(f"✅ Total workflows: {status['workflows']['total_definitions']}")
    print(f"✅ Active executions: {status['workflows']['active_executions']}")
    print(f"✅ Registered agents: {status['communication']['registered_agents']}")
    print(f"✅ Total messages: {status['communication']['total_messages']}")
    
    if 'performance' in status:
        perf = status['performance']
        print(f"✅ Avg response time: {perf['average_response_time_ms']:.2f}ms")
        print(f"✅ Error rate: {perf['error_rate']:.2%}")
    
    print(f"✅ Memory integration: {status['integrations']['memory_system']}")
    print(f"✅ Persistence enabled: {status['integrations']['persistence']}")
    
    print("\n🎉 Production Communication Workflows Demo Complete!")
    return True

if __name__ == "__main__":
    # Run demo
    async def main():
        success = await demo_production_communication_workflows()
        print(f"\nDemo completed: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    asyncio.run(main())