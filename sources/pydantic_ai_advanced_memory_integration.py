#!/usr/bin/env python3
"""
SANDBOX FILE: For testing/development. See .cursorrules.

Pydantic AI Advanced Memory Integration System
====================

* Purpose: Comprehensive memory integration system for MLACS with cross-framework memory bridging,
  knowledge persistence, and intelligent memory management for enhanced agent coordination
* Issues & Complexity Summary: Complex memory state management, cross-framework compatibility,
  knowledge graph integration, and real-time memory synchronization challenges
* Key Complexity Drivers:
  - Logic Scope (Est. LoC): ~1,500
  - Core Algorithm Complexity: High
  - Dependencies: 5 New, 3 Mod
  - State Management Complexity: Very High
  - Novelty/Uncertainty Factor: High
* AI Pre-Task Self-Assessment (Est. Solution Difficulty %): 85%
* Problem Estimate (Inherent Problem Difficulty %): 88%
* Initial Code Complexity Estimate %: 85%
* Justification for Estimates: Advanced memory management with cross-framework integration
* Final Code Complexity (Actual %): TBD
* Overall Result Score (Success & Quality %): TBD
* Key Variances/Learnings: TBD
* Last Updated: 2025-01-06

Provides:
- Cross-framework memory bridging
- Knowledge persistence and retrieval
- Memory-aware agent coordination
- Intelligent caching and optimization
- Real-time memory synchronization
"""

import asyncio
import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set, Tuple, Callable
from uuid import uuid4
import threading
import hashlib
from collections import defaultdict, deque
from dataclasses import dataclass, field
import pickle
import zlib

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

# Try to import LangChain components, fall back if not available
try:
    from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
    from langchain.schema import BaseMemory
    LANGCHAIN_AVAILABLE = True
    logger.info("LangChain successfully imported")
except ImportError:
    logger.warning("LangChain not available, using fallback implementations")
    LANGCHAIN_AVAILABLE = False
    
    # Fallback LangChain components
    class BaseMemory:
        def __init__(self, **kwargs):
            self.memory = {}
    
    class ConversationBufferMemory(BaseMemory):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.chat_memory = []
    
    class ConversationSummaryMemory(BaseMemory):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.summary = ""

# Try to import LangGraph components, fall back if not available
try:
    from langgraph import StateGraph
    from langgraph.graph import MessagesState
    LANGGRAPH_AVAILABLE = True
    logger.info("LangGraph successfully imported")
except ImportError:
    logger.warning("LangGraph not available, using fallback implementations")
    LANGGRAPH_AVAILABLE = False
    
    # Fallback LangGraph components
    class MessagesState:
        def __init__(self, **kwargs):
            self.messages = []
    
    class StateGraph:
        def __init__(self, state_schema=None):
            self.nodes = {}
            self.edges = {}
            self.state_schema = state_schema

# ================================
# Core Memory Management Enums and Models
# ================================

class MemoryType(Enum):
    """Types of memory storage available in the system"""
    SHORT_TERM = "short_term"          # Active working memory
    LONG_TERM = "long_term"            # Persistent storage
    EPISODIC = "episodic"              # Event-based memory
    SEMANTIC = "semantic"              # Knowledge-based memory
    PROCEDURAL = "procedural"          # Process and skill memory
    CACHE = "cache"                    # Temporary fast access

class MemoryPriority(Enum):
    """Memory priority levels for retention and access"""
    CRITICAL = 1                       # Never delete, highest priority
    HIGH = 2                           # Important, rarely delete
    MEDIUM = 3                         # Standard priority
    LOW = 4                            # Can be deleted when space needed
    TEMP = 5                           # Temporary, delete frequently

class MemoryStatus(Enum):
    """Memory entry status"""
    ACTIVE = "active"                  # Currently active
    CACHED = "cached"                  # Stored in cache
    ARCHIVED = "archived"              # Long-term storage
    EXPIRED = "expired"                # Ready for deletion
    CORRUPTED = "corrupted"            # Data integrity issues

class CrossFrameworkBridge(Enum):
    """Cross-framework memory bridge types"""
    PYDANTIC_AI = "pydantic_ai"
    LANGCHAIN = "langchain"
    LANGGRAPH = "langgraph"
    NATIVE = "native"
    HYBRID = "hybrid"

# ================================
# Memory Data Models
# ================================

class MemoryEntry(BaseModel):
    """Individual memory entry with metadata"""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: Dict[str, Any] = Field(default_factory=dict)
    memory_type: MemoryType = Field(default=MemoryType.SHORT_TERM)
    priority: MemoryPriority = Field(default=MemoryPriority.MEDIUM)
    status: MemoryStatus = Field(default=MemoryStatus.ACTIVE)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    accessed_at: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    tags: Set[str] = Field(default_factory=set)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    compression_ratio: float = 1.0
    checksum: str = ""

    def __init__(self, **kwargs):
        # Ensure datetime fields are set
        now = datetime.now()
        if 'created_at' not in kwargs:
            kwargs['created_at'] = now
        if 'updated_at' not in kwargs:
            kwargs['updated_at'] = now
        if 'accessed_at' not in kwargs:
            kwargs['accessed_at'] = now
        if 'tags' not in kwargs:
            kwargs['tags'] = set()
        if 'metadata' not in kwargs:
            kwargs['metadata'] = {}
        
        super().__init__(**kwargs)
        if not self.checksum:
            self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Calculate content checksum for integrity verification"""
        content_str = json.dumps(self.content, sort_keys=True)
        return hashlib.md5(content_str.encode()).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify memory entry integrity"""
        return self.checksum == self._calculate_checksum()

class MemoryCluster(BaseModel):
    """Group of related memory entries"""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = ""
    entries: List[str] = Field(default_factory=list)  # Memory entry IDs
    cluster_type: str = "general"
    priority: MemoryPriority = Field(default=MemoryPriority.MEDIUM)
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **kwargs):
        # Ensure datetime and collection fields are set
        if 'created_at' not in kwargs:
            kwargs['created_at'] = datetime.now()
        if 'entries' not in kwargs:
            kwargs['entries'] = []
        if 'metadata' not in kwargs:
            kwargs['metadata'] = {}
        
        super().__init__(**kwargs)

class MemoryGraph(BaseModel):
    """Knowledge graph for memory relationships"""
    
    nodes: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    edges: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MemoryMetrics(BaseModel):
    """Memory system performance metrics"""
    
    total_entries: int = 0
    active_entries: int = 0
    cached_entries: int = 0
    archived_entries: int = 0
    memory_usage_mb: float = 0.0
    cache_hit_ratio: float = 0.0
    avg_access_time_ms: float = 0.0
    compression_ratio: float = 1.0
    integrity_violations: int = 0

# ================================
# Advanced Memory Integration System
# ================================

class AdvancedMemoryIntegrationSystem:
    """
    Comprehensive memory integration system for MLACS with cross-framework 
    memory bridging and intelligent knowledge management
    """
    
    def __init__(
        self,
        db_path: str = "memory_integration.db",
        cache_size_mb: int = 100,
        auto_cleanup: bool = True,
        compression_enabled: bool = True
    ):
        self.db_path = db_path
        self.cache_size_mb = cache_size_mb
        self.auto_cleanup = auto_cleanup
        self.compression_enabled = compression_enabled
        
        # Core storage systems
        self.memory_store: Dict[str, MemoryEntry] = {}
        self.memory_cache: Dict[str, Any] = {}
        self.memory_clusters: Dict[str, MemoryCluster] = {}
        self.knowledge_graph = MemoryGraph()
        
        # Performance tracking
        self.access_counts: defaultdict = defaultdict(int)
        self.access_times: defaultdict = defaultdict(list)
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Cross-framework bridges
        self.framework_bridges: Dict[CrossFrameworkBridge, Any] = {}
        
        # Synchronization
        self._lock = threading.RLock()
        self._initialized = False
        
        # Initialize system
        self._initialize_system()

    @timer_decorator
    def _initialize_system(self):
        """Initialize the memory integration system"""
        try:
            # Initialize database
            self._initialize_database()
            
            # Initialize framework bridges
            self._initialize_framework_bridges()
            
            # Load existing data
            self._load_existing_data()
            
            # Start background maintenance
            if self.auto_cleanup:
                self._start_maintenance_thread()
            
            self._initialized = True
            logger.info("Advanced Memory Integration System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory system: {e}")
            raise

    def _initialize_database(self):
        """Initialize SQLite database for persistent storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    memory_type TEXT,
                    priority INTEGER,
                    status TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    accessed_at TEXT,
                    expires_at TEXT,
                    tags TEXT,
                    metadata TEXT,
                    compression_ratio REAL,
                    checksum TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_clusters (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    entries TEXT,
                    cluster_type TEXT,
                    priority INTEGER,
                    created_at TEXT,
                    metadata TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_graph (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nodes TEXT,
                    edges TEXT,
                    metadata TEXT,
                    updated_at TEXT
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_entries(memory_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_priority ON memory_entries(priority)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON memory_entries(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON memory_entries(created_at)')
            
            conn.commit()
            conn.close()
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    def _initialize_framework_bridges(self):
        """Initialize cross-framework memory bridges"""
        try:
            # Pydantic AI Bridge
            if PYDANTIC_AI_AVAILABLE:
                self.framework_bridges[CrossFrameworkBridge.PYDANTIC_AI] = self._create_pydantic_ai_bridge()
            
            # LangChain Bridge
            if LANGCHAIN_AVAILABLE:
                self.framework_bridges[CrossFrameworkBridge.LANGCHAIN] = self._create_langchain_bridge()
            
            # LangGraph Bridge
            if LANGGRAPH_AVAILABLE:
                self.framework_bridges[CrossFrameworkBridge.LANGGRAPH] = self._create_langgraph_bridge()
            
            # Native Bridge (always available)
            self.framework_bridges[CrossFrameworkBridge.NATIVE] = self._create_native_bridge()
            
            logger.info(f"Initialized {len(self.framework_bridges)} framework bridges")
            
        except Exception as e:
            logger.error(f"Framework bridge initialization failed: {e}")
            raise

    def _create_pydantic_ai_bridge(self) -> Dict[str, Any]:
        """Create Pydantic AI memory bridge"""
        return {
            'type': 'pydantic_ai',
            'memory_interface': self._pydantic_ai_memory_interface,
            'state_converter': self._convert_to_pydantic_ai_state,
            'memory_retriever': self._retrieve_for_pydantic_ai
        }

    def _create_langchain_bridge(self) -> Dict[str, Any]:
        """Create LangChain memory bridge"""
        return {
            'type': 'langchain',
            'memory_interface': ConversationBufferMemory(),
            'state_converter': self._convert_to_langchain_state,
            'memory_retriever': self._retrieve_for_langchain
        }

    def _create_langgraph_bridge(self) -> Dict[str, Any]:
        """Create LangGraph memory bridge"""
        return {
            'type': 'langgraph',
            'memory_interface': MessagesState(),
            'state_converter': self._convert_to_langgraph_state,
            'memory_retriever': self._retrieve_for_langgraph
        }

    def _create_native_bridge(self) -> Dict[str, Any]:
        """Create native memory bridge"""
        return {
            'type': 'native',
            'memory_interface': self,
            'state_converter': self._convert_to_native_state,
            'memory_retriever': self._retrieve_for_native
        }

    @timer_decorator
    def store_memory(
        self,
        content: Dict[str, Any],
        memory_type: MemoryType = MemoryType.SHORT_TERM,
        priority: MemoryPriority = MemoryPriority.MEDIUM,
        tags: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        expires_in: Optional[timedelta] = None
    ) -> str:
        """Store a new memory entry"""
        with self._lock:
            try:
                # Create memory entry
                memory_id = str(uuid4())
                
                # Compress content if enabled
                if self.compression_enabled and len(json.dumps(content)) > 1024:
                    compressed_content = self._compress_content(content)
                    compression_ratio = len(json.dumps(content)) / len(json.dumps(compressed_content))
                else:
                    compressed_content = content
                    compression_ratio = 1.0
                
                # Set expiration
                expires_at = None
                if expires_in:
                    expires_at = datetime.now() + expires_in
                
                # Create memory entry
                memory_entry = MemoryEntry(
                    id=memory_id,
                    content=compressed_content,
                    memory_type=memory_type,
                    priority=priority,
                    tags=tags or set(),
                    metadata=metadata or {},
                    expires_at=expires_at,
                    compression_ratio=compression_ratio
                )
                
                # Store in memory
                self.memory_store[memory_id] = memory_entry
                
                # Update cache if high priority
                if priority.value <= MemoryPriority.HIGH.value:
                    self.memory_cache[memory_id] = compressed_content
                
                # Persist to database
                self._persist_memory_entry(memory_entry)
                
                # Update knowledge graph
                self._update_knowledge_graph(memory_entry)
                
                logger.info(f"Stored memory entry {memory_id} with type {memory_type.value}")
                return memory_id
                
            except Exception as e:
                logger.error(f"Failed to store memory: {e}")
                raise

    @timer_decorator
    def retrieve_memory(
        self,
        memory_id: str,
        update_access: bool = True
    ) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by ID"""
        with self._lock:
            try:
                start_time = time.time()
                
                # Check cache first
                if memory_id in self.memory_cache:
                    self.cache_hits += 1
                    memory_entry = self.memory_store.get(memory_id)
                    if memory_entry:
                        if update_access:
                            memory_entry.accessed_at = datetime.now()
                            self.access_counts[memory_id] += 1
                        
                        # Record access time
                        access_time = (time.time() - start_time) * 1000
                        self.access_times[memory_id].append(access_time)
                        
                        return memory_entry
                
                # Check memory store
                if memory_id in self.memory_store:
                    self.cache_misses += 1
                    memory_entry = self.memory_store[memory_id]
                    
                    # Decompress if needed
                    if memory_entry.compression_ratio < 1.0:
                        memory_entry.content = self._decompress_content(memory_entry.content)
                    
                    if update_access:
                        memory_entry.accessed_at = datetime.now()
                        self.access_counts[memory_id] += 1
                    
                    # Add to cache if frequently accessed
                    if self.access_counts[memory_id] > 5:
                        self.memory_cache[memory_id] = memory_entry.content
                    
                    # Record access time
                    access_time = (time.time() - start_time) * 1000
                    self.access_times[memory_id].append(access_time)
                    
                    return memory_entry
                
                # Load from database
                memory_entry = self._load_memory_from_db(memory_id)
                if memory_entry:
                    self.memory_store[memory_id] = memory_entry
                    
                    if update_access:
                        memory_entry.accessed_at = datetime.now()
                        self.access_counts[memory_id] += 1
                    
                    # Record access time
                    access_time = (time.time() - start_time) * 1000
                    self.access_times[memory_id].append(access_time)
                    
                    return memory_entry
                
                return None
                
            except Exception as e:
                logger.error(f"Failed to retrieve memory {memory_id}: {e}")
                return None

    @timer_decorator
    def search_memories(
        self,
        query: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        tags: Optional[Set[str]] = None,
        priority: Optional[MemoryPriority] = None,
        limit: int = 100
    ) -> List[MemoryEntry]:
        """Search for memory entries based on criteria"""
        with self._lock:
            try:
                results = []
                
                for memory_entry in self.memory_store.values():
                    # Check memory type filter
                    if memory_type and memory_entry.memory_type != memory_type:
                        continue
                    
                    # Check priority filter
                    if priority and memory_entry.priority != priority:
                        continue
                    
                    # Check tags filter
                    if tags and not tags.intersection(memory_entry.tags):
                        continue
                    
                    # Check query filter (simple text search)
                    if query:
                        content_str = json.dumps(memory_entry.content).lower()
                        if query.lower() not in content_str:
                            continue
                    
                    results.append(memory_entry)
                    
                    if len(results) >= limit:
                        break
                
                # Sort by priority and recency
                results.sort(
                    key=lambda x: (x.priority.value, -x.accessed_at.timestamp())
                )
                
                logger.info(f"Found {len(results)} matching memories")
                return results
                
            except Exception as e:
                logger.error(f"Memory search failed: {e}")
                return []

    @timer_decorator
    def create_memory_cluster(
        self,
        name: str,
        memory_ids: List[str],
        cluster_type: str = "general",
        priority: MemoryPriority = MemoryPriority.MEDIUM
    ) -> str:
        """Create a cluster of related memories"""
        with self._lock:
            try:
                cluster_id = str(uuid4())
                
                cluster = MemoryCluster(
                    id=cluster_id,
                    name=name,
                    entries=memory_ids,
                    cluster_type=cluster_type,
                    priority=priority
                )
                
                self.memory_clusters[cluster_id] = cluster
                self._persist_memory_cluster(cluster)
                
                logger.info(f"Created memory cluster {cluster_id} with {len(memory_ids)} entries")
                return cluster_id
                
            except Exception as e:
                logger.error(f"Failed to create memory cluster: {e}")
                raise

    @timer_decorator
    def get_cross_framework_memory(
        self,
        framework: CrossFrameworkBridge,
        memory_context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Get memory interface for specific framework"""
        try:
            if framework not in self.framework_bridges:
                logger.warning(f"Framework {framework.value} not available")
                return None
            
            bridge = self.framework_bridges[framework]
            retriever = bridge['memory_retriever']
            
            return retriever(memory_context or {})
            
        except Exception as e:
            logger.error(f"Cross-framework memory retrieval failed: {e}")
            return None

    def _pydantic_ai_memory_interface(self) -> Dict[str, Any]:
        """Pydantic AI memory interface"""
        return {
            'get_memory': self.retrieve_memory,
            'store_memory': self.store_memory,
            'search_memory': self.search_memories
        }

    def _convert_to_pydantic_ai_state(self, memory_entry: MemoryEntry) -> Dict[str, Any]:
        """Convert memory entry to Pydantic AI state format"""
        return {
            'memory_id': memory_entry.id,
            'content': memory_entry.content,
            'type': memory_entry.memory_type.value,
            'metadata': memory_entry.metadata,
            'timestamp': memory_entry.created_at.isoformat()
        }

    def _retrieve_for_pydantic_ai(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve memories formatted for Pydantic AI"""
        memory_type = context.get('memory_type')
        tags = context.get('tags')
        limit = context.get('limit', 10)
        
        memories = self.search_memories(
            memory_type=MemoryType(memory_type) if memory_type else None,
            tags=set(tags) if tags else None,
            limit=limit
        )
        
        return {
            'memories': [self._convert_to_pydantic_ai_state(m) for m in memories],
            'count': len(memories)
        }

    def _convert_to_langchain_state(self, memory_entry: MemoryEntry) -> Dict[str, Any]:
        """Convert memory entry to LangChain state format"""
        return {
            'input': memory_entry.content.get('input', ''),
            'output': memory_entry.content.get('output', ''),
            'history': memory_entry.content.get('history', [])
        }

    def _retrieve_for_langchain(self, context: Dict[str, Any]) -> Any:
        """Retrieve memories formatted for LangChain"""
        if LANGCHAIN_AVAILABLE:
            memory = ConversationBufferMemory()
            
            # Load recent conversations
            recent_memories = self.search_memories(
                memory_type=MemoryType.EPISODIC,
                limit=context.get('limit', 20)
            )
            
            for mem in recent_memories:
                if 'input' in mem.content and 'output' in mem.content:
                    memory.chat_memory.add_user_message(mem.content['input'])
                    memory.chat_memory.add_ai_message(mem.content['output'])
            
            return memory
        
        return None

    def _convert_to_langgraph_state(self, memory_entry: MemoryEntry) -> Dict[str, Any]:
        """Convert memory entry to LangGraph state format"""
        return {
            'messages': memory_entry.content.get('messages', []),
            'state': memory_entry.content.get('state', {}),
            'metadata': memory_entry.metadata
        }

    def _retrieve_for_langgraph(self, context: Dict[str, Any]) -> Any:
        """Retrieve memories formatted for LangGraph"""
        if LANGGRAPH_AVAILABLE:
            state = MessagesState()
            
            # Load workflow states
            workflow_memories = self.search_memories(
                memory_type=MemoryType.PROCEDURAL,
                limit=context.get('limit', 10)
            )
            
            for mem in workflow_memories:
                if 'messages' in mem.content:
                    state.messages.extend(mem.content['messages'])
            
            return state
        
        return None

    def _convert_to_native_state(self, memory_entry: MemoryEntry) -> Dict[str, Any]:
        """Convert memory entry to native state format"""
        return memory_entry.dict()

    def _retrieve_for_native(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve memories in native format"""
        return {
            'total_memories': len(self.memory_store),
            'cache_size': len(self.memory_cache),
            'clusters': len(self.memory_clusters),
            'framework_bridges': list(self.framework_bridges.keys())
        }

    def _compress_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Compress memory content"""
        try:
            json_str = json.dumps(content)
            compressed = zlib.compress(json_str.encode())
            return {'_compressed': True, '_data': compressed.hex()}
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return content

    def _decompress_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Decompress memory content"""
        try:
            if content.get('_compressed'):
                compressed_data = bytes.fromhex(content['_data'])
                decompressed = zlib.decompress(compressed_data)
                return json.loads(decompressed.decode())
            return content
        except Exception as e:
            logger.warning(f"Decompression failed: {e}")
            return content

    def _update_knowledge_graph(self, memory_entry: MemoryEntry):
        """Update knowledge graph with new memory entry"""
        try:
            # Ensure knowledge graph is initialized
            if self.knowledge_graph.nodes is None:
                self.knowledge_graph.nodes = {}
            if self.knowledge_graph.edges is None:
                self.knowledge_graph.edges = []
            
            # Add node for memory entry
            self.knowledge_graph.nodes[memory_entry.id] = {
                'type': memory_entry.memory_type.value,
                'priority': memory_entry.priority.value,
                'tags': list(memory_entry.tags) if memory_entry.tags else [],
                'created_at': memory_entry.created_at.isoformat() if memory_entry.created_at else datetime.now().isoformat()
            }
            
            # Create edges based on tags and content similarity
            for existing_id, existing_node in self.knowledge_graph.nodes.items():
                if existing_id != memory_entry.id:
                    # Check tag similarity
                    existing_tags = set(existing_node.get('tags', []))
                    memory_tags = memory_entry.tags if memory_entry.tags else set()
                    similarity = len(memory_tags.intersection(existing_tags))
                    
                    if similarity > 0:
                        self.knowledge_graph.edges.append({
                            'source': existing_id,
                            'target': memory_entry.id,
                            'type': 'tag_similarity',
                            'weight': similarity
                        })
            
        except Exception as e:
            logger.warning(f"Knowledge graph update failed: {e}")

    def _persist_memory_entry(self, memory_entry: MemoryEntry):
        """Persist memory entry to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Ensure all datetime fields are valid
            created_at = memory_entry.created_at.isoformat() if memory_entry.created_at else datetime.now().isoformat()
            updated_at = memory_entry.updated_at.isoformat() if memory_entry.updated_at else datetime.now().isoformat()
            accessed_at = memory_entry.accessed_at.isoformat() if memory_entry.accessed_at else datetime.now().isoformat()
            
            cursor.execute('''
                INSERT OR REPLACE INTO memory_entries 
                (id, content, memory_type, priority, status, created_at, updated_at, 
                 accessed_at, expires_at, tags, metadata, compression_ratio, checksum)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                memory_entry.id,
                json.dumps(memory_entry.content),
                memory_entry.memory_type.value,
                memory_entry.priority.value,
                memory_entry.status.value,
                created_at,
                updated_at,
                accessed_at,
                memory_entry.expires_at.isoformat() if memory_entry.expires_at else None,
                json.dumps(list(memory_entry.tags) if memory_entry.tags else []),
                json.dumps(memory_entry.metadata if memory_entry.metadata else {}),
                memory_entry.compression_ratio,
                memory_entry.checksum
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to persist memory entry: {e}")

    def _persist_memory_cluster(self, cluster: MemoryCluster):
        """Persist memory cluster to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Ensure datetime field is valid
            created_at = cluster.created_at.isoformat() if cluster.created_at else datetime.now().isoformat()
            
            cursor.execute('''
                INSERT OR REPLACE INTO memory_clusters 
                (id, name, entries, cluster_type, priority, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                cluster.id,
                cluster.name,
                json.dumps(cluster.entries if cluster.entries else []),
                cluster.cluster_type,
                cluster.priority.value,
                created_at,
                json.dumps(cluster.metadata if cluster.metadata else {})
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to persist memory cluster: {e}")

    def _load_memory_from_db(self, memory_id: str) -> Optional[MemoryEntry]:
        """Load memory entry from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT * FROM memory_entries WHERE id = ?',
                (memory_id,)
            )
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return MemoryEntry(
                    id=row[0],
                    content=json.loads(row[1]),
                    memory_type=MemoryType(row[2]),
                    priority=MemoryPriority(row[3]),
                    status=MemoryStatus(row[4]),
                    created_at=datetime.fromisoformat(row[5]),
                    updated_at=datetime.fromisoformat(row[6]),
                    accessed_at=datetime.fromisoformat(row[7]),
                    expires_at=datetime.fromisoformat(row[8]) if row[8] else None,
                    tags=set(json.loads(row[9])),
                    metadata=json.loads(row[10]),
                    compression_ratio=row[11],
                    checksum=row[12]
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load memory from database: {e}")
            return None

    def _load_existing_data(self):
        """Load existing data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Load memory entries
            cursor.execute('SELECT id FROM memory_entries LIMIT 1000')
            memory_ids = [row[0] for row in cursor.fetchall()]
            
            for memory_id in memory_ids:
                memory_entry = self._load_memory_from_db(memory_id)
                if memory_entry:
                    self.memory_store[memory_id] = memory_entry
            
            # Load memory clusters
            cursor.execute('SELECT * FROM memory_clusters')
            cluster_rows = cursor.fetchall()
            
            for row in cluster_rows:
                cluster = MemoryCluster(
                    id=row[0],
                    name=row[1],
                    entries=json.loads(row[2]),
                    cluster_type=row[3],
                    priority=MemoryPriority(row[4]),
                    created_at=datetime.fromisoformat(row[5]),
                    metadata=json.loads(row[6])
                )
                self.memory_clusters[cluster.id] = cluster
            
            conn.close()
            logger.info(f"Loaded {len(self.memory_store)} memories and {len(self.memory_clusters)} clusters")
            
        except Exception as e:
            logger.warning(f"Failed to load existing data: {e}")

    def _start_maintenance_thread(self):
        """Start background maintenance thread"""
        def maintenance_loop():
            while True:
                try:
                    time.sleep(60)  # Run every minute
                    self._cleanup_expired_memories()
                    self._optimize_cache()
                    self._update_metrics()
                except Exception as e:
                    logger.error(f"Maintenance error: {e}")
        
        thread = threading.Thread(target=maintenance_loop, daemon=True)
        thread.start()
        logger.info("Maintenance thread started")

    def _cleanup_expired_memories(self):
        """Clean up expired memory entries"""
        with self._lock:
            try:
                current_time = datetime.now()
                expired_ids = []
                
                for memory_id, memory_entry in self.memory_store.items():
                    if (memory_entry.expires_at and 
                        memory_entry.expires_at < current_time):
                        expired_ids.append(memory_id)
                
                for memory_id in expired_ids:
                    del self.memory_store[memory_id]
                    if memory_id in self.memory_cache:
                        del self.memory_cache[memory_id]
                
                if expired_ids:
                    logger.info(f"Cleaned up {len(expired_ids)} expired memories")
                    
            except Exception as e:
                logger.error(f"Cleanup failed: {e}")

    def _optimize_cache(self):
        """Optimize memory cache"""
        with self._lock:
            try:
                # Calculate current cache size
                cache_size = sum(
                    len(json.dumps(content)) 
                    for content in self.memory_cache.values()
                )
                
                max_size = self.cache_size_mb * 1024 * 1024  # Convert to bytes
                
                if cache_size > max_size:
                    # Remove least accessed items
                    access_counts = {
                        memory_id: self.access_counts[memory_id]
                        for memory_id in self.memory_cache.keys()
                    }
                    
                    sorted_items = sorted(
                        access_counts.items(),
                        key=lambda x: x[1]
                    )
                    
                    # Remove 25% of least accessed items
                    remove_count = len(sorted_items) // 4
                    for memory_id, _ in sorted_items[:remove_count]:
                        if memory_id in self.memory_cache:
                            del self.memory_cache[memory_id]
                    
                    logger.info(f"Optimized cache, removed {remove_count} items")
                    
            except Exception as e:
                logger.error(f"Cache optimization failed: {e}")

    def _update_metrics(self):
        """Update system metrics"""
        try:
            total_entries = len(self.memory_store) if self.memory_store else 0
            active_entries = 0
            if self.memory_store:
                active_entries = sum(
                    1 for entry in self.memory_store.values()
                    if entry and entry.status == MemoryStatus.ACTIVE
                )
            cached_entries = len(self.memory_cache) if self.memory_cache else 0
            
            cache_hit_ratio = 0.0
            if self.cache_hits + self.cache_misses > 0:
                cache_hit_ratio = self.cache_hits / (self.cache_hits + self.cache_misses)
            
            # Calculate average access time
            all_times = []
            if self.access_times:
                for times in self.access_times.values():
                    if times:
                        all_times.extend(times)
            
            avg_access_time = sum(all_times) / len(all_times) if all_times else 0.0
            
            # Update metrics
            self.current_metrics = MemoryMetrics(
                total_entries=total_entries,
                active_entries=active_entries,
                cached_entries=cached_entries,
                cache_hit_ratio=cache_hit_ratio,
                avg_access_time_ms=avg_access_time
            )
            
        except Exception as e:
            logger.error(f"Metrics update failed: {e}")
            # Create fallback metrics
            self.current_metrics = MemoryMetrics(
                total_entries=0,
                active_entries=0,
                cached_entries=0,
                cache_hit_ratio=0.0,
                avg_access_time_ms=0.0
            )

    @timer_decorator
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        with self._lock:
            try:
                self._update_metrics()
                
                # Safe access to collections
                memory_store_count = len(self.memory_store) if self.memory_store else 0
                memory_cache_count = len(self.memory_cache) if self.memory_cache else 0
                memory_clusters_count = len(self.memory_clusters) if self.memory_clusters else 0
                
                # Safe access to knowledge graph
                kg_nodes_count = 0
                kg_edges_count = 0
                if self.knowledge_graph and self.knowledge_graph.nodes:
                    kg_nodes_count = len(self.knowledge_graph.nodes)
                if self.knowledge_graph and self.knowledge_graph.edges:
                    kg_edges_count = len(self.knowledge_graph.edges)
                
                return {
                    'initialized': self._initialized,
                    'memory_stores': {
                        'total_entries': memory_store_count,
                        'cached_entries': memory_cache_count,
                        'clusters': memory_clusters_count
                    },
                    'performance': {
                        'cache_hits': self.cache_hits,
                        'cache_misses': self.cache_misses,
                        'cache_hit_ratio': self.current_metrics.cache_hit_ratio if hasattr(self, 'current_metrics') else 0.0,
                        'avg_access_time_ms': self.current_metrics.avg_access_time_ms if hasattr(self, 'current_metrics') else 0.0
                    },
                    'framework_bridges': {
                        bridge.value: 'available' 
                        for bridge in self.framework_bridges.keys()
                    } if self.framework_bridges else {},
                    'knowledge_graph': {
                        'nodes': kg_nodes_count,
                        'edges': kg_edges_count
                    }
                }
                
            except Exception as e:
                logger.error(f"Status retrieval failed: {e}")
                return {'error': str(e)}

# ================================
# Memory Integration Factory
# ================================

class MemoryIntegrationFactory:
    """Factory for creating memory integration instances"""
    
    @staticmethod
    def create_memory_system(
        config: Optional[Dict[str, Any]] = None
    ) -> AdvancedMemoryIntegrationSystem:
        """Create a configured memory integration system"""
        
        default_config = {
            'db_path': 'memory_integration.db',
            'cache_size_mb': 100,
            'auto_cleanup': True,
            'compression_enabled': True
        }
        
        if config:
            default_config.update(config)
        
        return AdvancedMemoryIntegrationSystem(**default_config)

# ================================
# Export Classes
# ================================

__all__ = [
    'AdvancedMemoryIntegrationSystem',
    'MemoryIntegrationFactory',
    'MemoryEntry',
    'MemoryCluster',
    'MemoryGraph',
    'MemoryMetrics',
    'MemoryType',
    'MemoryPriority',
    'MemoryStatus',
    'CrossFrameworkBridge',
    'timer_decorator'
]

# ================================
# Demo Functions
# ================================

def demo_advanced_memory_integration():
    """Demonstrate advanced memory integration capabilities"""
    
    print("🧠 Advanced Memory Integration System Demo")
    print("=" * 50)
    
    # Create memory system
    memory_system = MemoryIntegrationFactory.create_memory_system({
        'cache_size_mb': 50,
        'compression_enabled': True
    })
    
    # Store different types of memories
    print("\n1. Storing diverse memory types...")
    
    # Store episodic memory
    episodic_id = memory_system.store_memory(
        content={
            'event': 'user_interaction',
            'input': 'How do I implement memory integration?',
            'output': 'Use the AdvancedMemoryIntegrationSystem class...',
            'context': 'technical_discussion'
        },
        memory_type=MemoryType.EPISODIC,
        priority=MemoryPriority.HIGH,
        tags={'conversation', 'technical', 'memory'}
    )
    
    # Store semantic memory
    semantic_id = memory_system.store_memory(
        content={
            'concept': 'memory_integration',
            'definition': 'Cross-framework memory bridging system',
            'properties': ['persistent', 'searchable', 'intelligent'],
            'relationships': ['pydantic_ai', 'langchain', 'langgraph']
        },
        memory_type=MemoryType.SEMANTIC,
        priority=MemoryPriority.CRITICAL,
        tags={'knowledge', 'definitions', 'concepts'}
    )
    
    # Store procedural memory
    procedural_id = memory_system.store_memory(
        content={
            'process': 'memory_storage_workflow',
            'steps': [
                'validate_input',
                'compress_content',
                'calculate_checksum',
                'store_in_database',
                'update_cache'
            ],
            'conditions': {'compression_threshold': 1024}
        },
        memory_type=MemoryType.PROCEDURAL,
        priority=MemoryPriority.MEDIUM,
        tags={'workflow', 'process', 'storage'}
    )
    
    print(f"✅ Stored episodic memory: {episodic_id[:8]}...")
    print(f"✅ Stored semantic memory: {semantic_id[:8]}...")
    print(f"✅ Stored procedural memory: {procedural_id[:8]}...")
    
    # Demonstrate memory retrieval
    print("\n2. Retrieving and searching memories...")
    
    # Retrieve specific memory
    retrieved_memory = memory_system.retrieve_memory(episodic_id)
    if retrieved_memory:
        print(f"✅ Retrieved memory type: {retrieved_memory.memory_type.value}")
        print(f"   Priority: {retrieved_memory.priority.value}")
        print(f"   Tags: {retrieved_memory.tags}")
    
    # Search memories by criteria
    technical_memories = memory_system.search_memories(
        tags={'technical'},
        limit=5
    )
    print(f"✅ Found {len(technical_memories)} technical memories")
    
    # Search by memory type
    semantic_memories = memory_system.search_memories(
        memory_type=MemoryType.SEMANTIC,
        limit=10
    )
    print(f"✅ Found {len(semantic_memories)} semantic memories")
    
    # Create memory cluster
    print("\n3. Creating memory clusters...")
    
    cluster_id = memory_system.create_memory_cluster(
        name="Technical Discussion Cluster",
        memory_ids=[episodic_id, semantic_id, procedural_id],
        cluster_type="conversation",
        priority=MemoryPriority.HIGH
    )
    print(f"✅ Created memory cluster: {cluster_id[:8]}...")
    
    # Demonstrate cross-framework bridges
    print("\n4. Testing cross-framework bridges...")
    
    for framework in CrossFrameworkBridge:
        memory_interface = memory_system.get_cross_framework_memory(
            framework=framework,
            memory_context={'limit': 5, 'memory_type': 'episodic'}
        )
        
        if memory_interface:
            print(f"✅ {framework.value} bridge: Available")
        else:
            print(f"⚠️  {framework.value} bridge: Not available")
    
    # Show system status
    print("\n5. System status...")
    
    status = memory_system.get_system_status()
    print(f"✅ Total memories: {status['memory_stores']['total_entries']}")
    print(f"✅ Cached memories: {status['memory_stores']['cached_entries']}")
    print(f"✅ Memory clusters: {status['memory_stores']['clusters']}")
    print(f"✅ Available bridges: {len(status['framework_bridges'])}")
    print(f"✅ Knowledge graph nodes: {status['knowledge_graph']['nodes']}")
    print(f"✅ Knowledge graph edges: {status['knowledge_graph']['edges']}")
    
    if 'performance' in status:
        perf = status['performance']
        print(f"✅ Cache hit ratio: {perf['cache_hit_ratio']:.2%}")
        print(f"✅ Avg access time: {perf['avg_access_time_ms']:.2f}ms")
    
    print("\n🎉 Advanced Memory Integration Demo Complete!")
    return True

if __name__ == "__main__":
    # Run demo
    success = demo_advanced_memory_integration()
    print(f"\nDemo completed: {'✅ SUCCESS' if success else '❌ FAILED'}")