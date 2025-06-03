#!/usr/bin/env python3
"""
// SANDBOX FILE: For testing/development. See .cursorrules.

LANGGRAPH WORKFLOW STATE MANAGEMENT SYSTEM
==========================================

Purpose: Advanced workflow state management with checkpointing, recovery, versioning and distributed consistency
Issues & Complexity Summary: Complex state versioning with distributed coordination and optimized compression
Key Complexity Drivers:
  - Logic Scope (Est. LoC): ~2800
  - Core Algorithm Complexity: Very High
  - Dependencies: 9 New (LangGraph, State Versioning, Distributed Locks, Compression, Recovery, Rollback)
  - State Management Complexity: Very High
  - Novelty/Uncertainty Factor: High
* AI Pre-Task Self-Assessment (Est. Solution Difficulty %): 88%
* Problem Estimate (Inherent Problem Difficulty %): 90%
* Initial Code Complexity Estimate %: 87%
* Justification for Estimates: Complex distributed state management with versioning and recovery
* Final Code Complexity (Actual %): TBD
* Overall Result Score (Success & Quality %): TBD
* Key Variances/Learnings: TBD
* Last Updated: 2025-06-04

TASK-LANGGRAPH-005.2: Workflow State Management
Target: <200ms checkpoint creation, >99% recovery success, >40% compression ratio
"""

import asyncio
import sqlite3
import json
import time
import threading
import logging
import pickle
import gzip
import hashlib
import uuid
import zlib
import lz4.frame
from typing import Dict, List, Optional, Any, Tuple, Union, Callable, AsyncGenerator, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from collections import defaultdict, deque, OrderedDict
import weakref
import psutil
import platform
import copy
import threading
from threading import RLock
import heapq

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StateCompressionType(Enum):
    """State compression algorithms"""
    NONE = "none"
    GZIP = "gzip"
    ZLIB = "zlib"
    LZ4 = "lz4"
    HYBRID = "hybrid"

class StateVersionType(Enum):
    """State version types"""
    SNAPSHOT = "snapshot"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    COMPACT = "compact"

class CheckpointStrategy(Enum):
    """Checkpoint creation strategies"""
    TIME_BASED = "time_based"
    OPERATION_BASED = "operation_based"
    MEMORY_BASED = "memory_based"
    ADAPTIVE = "adaptive"
    MANUAL = "manual"

class RecoveryStrategy(Enum):
    """Recovery strategies"""
    IMMEDIATE = "immediate"
    LAZY = "lazy"
    PARALLEL = "parallel"
    SELECTIVE = "selective"

class ConsistencyLevel(Enum):
    """Distributed consistency levels"""
    EVENTUAL = "eventual"
    STRONG = "strong"
    CAUSAL = "causal"
    SEQUENTIAL = "sequential"

@dataclass
class StateVersion:
    """State version representation"""
    version_id: str
    workflow_id: str
    version_number: int
    version_type: StateVersionType
    parent_version_id: Optional[str]
    state_data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    size_bytes: int = 0
    compressed: bool = False
    compression_type: StateCompressionType = StateCompressionType.NONE
    checksum: str = ""
    
@dataclass
class CheckpointMetadata:
    """Checkpoint metadata"""
    checkpoint_id: str
    workflow_id: str
    strategy: CheckpointStrategy
    created_at: datetime
    size_bytes: int
    compression_ratio: float
    versions_included: List[str]
    recovery_time_estimate_ms: float
    consistency_level: ConsistencyLevel
    
@dataclass
class RecoveryPlan:
    """State recovery execution plan"""
    recovery_id: str
    workflow_id: str
    target_checkpoint_id: str
    strategy: RecoveryStrategy
    estimated_time_ms: float
    required_resources: Dict[str, Any]
    dependencies: List[str]
    rollback_points: List[str]

@dataclass
class DistributedLockInfo:
    """Distributed lock information"""
    lock_id: str
    workflow_id: str
    holder_id: str
    acquired_at: datetime
    expires_at: datetime
    lock_type: str
    recursive_count: int = 0

class StateCompressionEngine:
    """Advanced state compression with multiple algorithms"""
    
    def __init__(self):
        self.compression_stats = {
            "objects_compressed": 0,
            "total_size_before": 0,
            "total_size_after": 0,
            "compression_time_ms": 0
        }
        self.algorithm_performance = defaultdict(lambda: {"count": 0, "ratio": 0.0, "time_ms": 0.0})
    
    async def compress_state(self, state_data: Any, compression_type: StateCompressionType = StateCompressionType.HYBRID) -> Tuple[bytes, StateCompressionType, float]:
        """Compress state data with specified algorithm"""
        start_time = time.time()
        
        try:
            serialized_data = pickle.dumps(state_data)
            original_size = len(serialized_data)
            
            if compression_type == StateCompressionType.NONE:
                compressed_data = serialized_data
                actual_type = StateCompressionType.NONE
            elif compression_type == StateCompressionType.GZIP:
                compressed_data = gzip.compress(serialized_data, compresslevel=6)
                actual_type = StateCompressionType.GZIP
            elif compression_type == StateCompressionType.ZLIB:
                compressed_data = zlib.compress(serialized_data, level=6)
                actual_type = StateCompressionType.ZLIB
            elif compression_type == StateCompressionType.LZ4:
                compressed_data = lz4.frame.compress(serialized_data)
                actual_type = StateCompressionType.LZ4
            elif compression_type == StateCompressionType.HYBRID:
                # Test different algorithms and choose best
                compressed_data, actual_type = await self._hybrid_compression(serialized_data)
            else:
                compressed_data = serialized_data
                actual_type = StateCompressionType.NONE
            
            compressed_size = len(compressed_data)
            compression_ratio = 1.0 - (compressed_size / original_size) if original_size > 0 else 0.0
            compression_time = (time.time() - start_time) * 1000
            
            # Update statistics
            self.compression_stats["objects_compressed"] += 1
            self.compression_stats["total_size_before"] += original_size
            self.compression_stats["total_size_after"] += compressed_size
            self.compression_stats["compression_time_ms"] += compression_time
            
            # Update algorithm performance
            self.algorithm_performance[actual_type.value]["count"] += 1
            self.algorithm_performance[actual_type.value]["ratio"] += compression_ratio
            self.algorithm_performance[actual_type.value]["time_ms"] += compression_time
            
            logger.debug(f"Compressed state: {original_size} -> {compressed_size} bytes ({compression_ratio:.1%} reduction) using {actual_type.value}")
            
            return compressed_data, actual_type, compression_ratio
            
        except Exception as e:
            logger.error(f"State compression failed: {e}")
            # Fallback to uncompressed
            serialized_data = pickle.dumps(state_data)
            return serialized_data, StateCompressionType.NONE, 0.0
    
    async def decompress_state(self, compressed_data: bytes, compression_type: StateCompressionType) -> Any:
        """Decompress state data"""
        try:
            if compression_type == StateCompressionType.NONE:
                decompressed_data = compressed_data
            elif compression_type == StateCompressionType.GZIP:
                decompressed_data = gzip.decompress(compressed_data)
            elif compression_type == StateCompressionType.ZLIB:
                decompressed_data = zlib.decompress(compressed_data)
            elif compression_type == StateCompressionType.LZ4:
                decompressed_data = lz4.frame.decompress(compressed_data)
            else:
                decompressed_data = compressed_data
            
            return pickle.loads(decompressed_data)
            
        except Exception as e:
            logger.error(f"State decompression failed: {e}")
            # Try to recover with pickle directly
            return pickle.loads(compressed_data)
    
    async def _hybrid_compression(self, data: bytes) -> Tuple[bytes, StateCompressionType]:
        """Choose best compression algorithm for given data"""
        algorithms = [
            (StateCompressionType.LZ4, lambda d: lz4.frame.compress(d)),
            (StateCompressionType.ZLIB, lambda d: zlib.compress(d, level=3)),
            (StateCompressionType.GZIP, lambda d: gzip.compress(d, compresslevel=3))
        ]
        
        best_result = (data, StateCompressionType.NONE)
        best_ratio = 0.0
        
        for comp_type, compress_func in algorithms:
            try:
                compressed = compress_func(data)
                ratio = 1.0 - (len(compressed) / len(data))
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_result = (compressed, comp_type)
            except Exception:
                continue
        
        return best_result

class StateVersionManager:
    """Advanced state versioning with rollback capabilities"""
    
    def __init__(self, db_path: str = "state_versions.db"):
        self.db_path = db_path
        self.compression_engine = StateCompressionEngine()
        self.version_cache: OrderedDict = OrderedDict()
        self.cache_max_size = 100
        self.active_versions: Dict[str, StateVersion] = {}
        self.version_tree: Dict[str, List[str]] = defaultdict(list)  # Parent -> Children
        self._initialize_database()
        
    def _initialize_database(self):
        """Initialize state versioning database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS state_versions (
                        version_id TEXT PRIMARY KEY,
                        workflow_id TEXT NOT NULL,
                        version_number INTEGER NOT NULL,
                        version_type TEXT NOT NULL,
                        parent_version_id TEXT,
                        state_data BLOB NOT NULL,
                        metadata TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        size_bytes INTEGER NOT NULL,
                        compressed INTEGER NOT NULL,
                        compression_type TEXT NOT NULL,
                        checksum TEXT NOT NULL
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_state_versions_workflow 
                    ON state_versions(workflow_id)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_state_versions_parent 
                    ON state_versions(parent_version_id)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_state_versions_created 
                    ON state_versions(created_at)
                """)
                
        except Exception as e:
            logger.error(f"State version database initialization failed: {e}")
    
    async def create_version(self, workflow_id: str, state_data: Dict[str, Any], 
                           version_type: StateVersionType = StateVersionType.SNAPSHOT,
                           parent_version_id: Optional[str] = None) -> StateVersion:
        """Create new state version"""
        try:
            version_id = f"version_{workflow_id}_{uuid.uuid4().hex[:8]}"
            
            # Get next version number
            version_number = await self._get_next_version_number(workflow_id)
            
            # Compress state data
            compressed_data, compression_type, compression_ratio = await self.compression_engine.compress_state(
                state_data, StateCompressionType.HYBRID
            )
            
            # Create checksum
            checksum = hashlib.sha256(compressed_data).hexdigest()
            
            # Create version object
            version = StateVersion(
                version_id=version_id,
                workflow_id=workflow_id,
                version_number=version_number,
                version_type=version_type,
                parent_version_id=parent_version_id,
                state_data=state_data,
                size_bytes=len(compressed_data),
                compressed=compression_type != StateCompressionType.NONE,
                compression_type=compression_type,
                checksum=checksum
            )
            
            # Store in database
            await self._store_version_in_db(version, compressed_data)
            
            # Update caches and trees
            self.active_versions[version_id] = version
            if parent_version_id:
                self.version_tree[parent_version_id].append(version_id)
            
            # Add to cache
            self._add_to_cache(version_id, version)
            
            logger.info(f"Created state version {version_id} for workflow {workflow_id} (compression: {compression_ratio:.1%})")
            return version
            
        except Exception as e:
            logger.error(f"State version creation failed: {e}")
            raise
    
    async def get_version(self, version_id: str) -> Optional[StateVersion]:
        """Get state version by ID"""
        # Check cache first
        if version_id in self.version_cache:
            version = self.version_cache[version_id]
            # Move to end (LRU)
            self.version_cache.move_to_end(version_id)
            return version
        
        # Check active versions
        if version_id in self.active_versions:
            return self.active_versions[version_id]
        
        # Load from database
        return await self._load_version_from_db(version_id)
    
    async def rollback_to_version(self, workflow_id: str, target_version_id: str) -> bool:
        """Rollback workflow to specific version"""
        try:
            target_version = await self.get_version(target_version_id)
            if not target_version or target_version.workflow_id != workflow_id:
                logger.error(f"Target version {target_version_id} not found for workflow {workflow_id}")
                return False
            
            # Create rollback version
            rollback_version = await self.create_version(
                workflow_id=workflow_id,
                state_data=target_version.state_data,
                version_type=StateVersionType.SNAPSHOT,
                parent_version_id=target_version_id
            )
            
            logger.info(f"Rolled back workflow {workflow_id} to version {target_version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed for workflow {workflow_id}: {e}")
            return False
    
    async def get_version_history(self, workflow_id: str, limit: int = 50) -> List[StateVersion]:
        """Get version history for workflow"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT version_id FROM state_versions 
                    WHERE workflow_id = ? 
                    ORDER BY version_number DESC 
                    LIMIT ?
                """, (workflow_id, limit))
                
                version_ids = [row[0] for row in cursor.fetchall()]
                
                versions = []
                for version_id in version_ids:
                    version = await self.get_version(version_id)
                    if version:
                        versions.append(version)
                
                return versions
                
        except Exception as e:
            logger.error(f"Failed to get version history for {workflow_id}: {e}")
            return []
    
    async def _get_next_version_number(self, workflow_id: str) -> int:
        """Get next version number for workflow"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT MAX(version_number) FROM state_versions WHERE workflow_id = ?
                """, (workflow_id,))
                result = cursor.fetchone()
                return (result[0] or 0) + 1
        except Exception:
            return 1
    
    async def _store_version_in_db(self, version: StateVersion, compressed_data: bytes):
        """Store version in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO state_versions (
                    version_id, workflow_id, version_number, version_type, parent_version_id,
                    state_data, metadata, created_at, size_bytes, compressed, compression_type, checksum
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                version.version_id, version.workflow_id, version.version_number, 
                version.version_type.value, version.parent_version_id,
                compressed_data, json.dumps(version.metadata),
                version.created_at.isoformat(), version.size_bytes,
                1 if version.compressed else 0, version.compression_type.value, version.checksum
            ))
    
    async def _load_version_from_db(self, version_id: str) -> Optional[StateVersion]:
        """Load version from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM state_versions WHERE version_id = ?
                """, (version_id,))
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                # Decompress state data
                compressed_data = row[5]
                compression_type = StateCompressionType(row[10])
                state_data = await self.compression_engine.decompress_state(compressed_data, compression_type)
                
                version = StateVersion(
                    version_id=row[0],
                    workflow_id=row[1],
                    version_number=row[2],
                    version_type=StateVersionType(row[3]),
                    parent_version_id=row[4],
                    state_data=state_data,
                    metadata=json.loads(row[6]),
                    created_at=datetime.fromisoformat(row[7]),
                    size_bytes=row[8],
                    compressed=bool(row[9]),
                    compression_type=compression_type,
                    checksum=row[11]
                )
                
                # Add to cache
                self._add_to_cache(version_id, version)
                
                return version
                
        except Exception as e:
            logger.error(f"Failed to load version {version_id}: {e}")
            return None
    
    def _add_to_cache(self, version_id: str, version: StateVersion):
        """Add version to LRU cache"""
        if len(self.version_cache) >= self.cache_max_size:
            # Remove oldest
            self.version_cache.popitem(last=False)
        
        self.version_cache[version_id] = version

class DistributedLockManager:
    """Distributed locking for state consistency"""
    
    def __init__(self, db_path: str = "distributed_locks.db"):
        self.db_path = db_path
        self.local_locks: Dict[str, RLock] = {}
        self.lock_holders: Dict[str, DistributedLockInfo] = {}
        self.lock_waiters: Dict[str, List[asyncio.Future]] = defaultdict(list)
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize distributed locks database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS distributed_locks (
                        lock_id TEXT PRIMARY KEY,
                        workflow_id TEXT NOT NULL,
                        holder_id TEXT NOT NULL,
                        acquired_at TEXT NOT NULL,
                        expires_at TEXT NOT NULL,
                        lock_type TEXT NOT NULL,
                        recursive_count INTEGER NOT NULL DEFAULT 0
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_distributed_locks_workflow 
                    ON distributed_locks(workflow_id)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_distributed_locks_expires 
                    ON distributed_locks(expires_at)
                """)
                
        except Exception as e:
            logger.error(f"Distributed locks database initialization failed: {e}")
    
    async def acquire_lock(self, workflow_id: str, lock_type: str = "state_write", 
                          timeout_ms: float = 5000.0) -> Optional[str]:
        """Acquire distributed lock"""
        lock_id = f"lock_{workflow_id}_{lock_type}"
        holder_id = f"{platform.node()}_{threading.get_ident()}"
        
        start_time = time.time()
        timeout_seconds = timeout_ms / 1000.0
        
        while (time.time() - start_time) < timeout_seconds:
            try:
                # Clean expired locks first
                await self._clean_expired_locks()
                
                # Try to acquire lock
                acquired = await self._try_acquire_lock(lock_id, workflow_id, holder_id, lock_type)
                if acquired:
                    return lock_id
                
                # Wait a bit before retrying
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Lock acquisition failed: {e}")
                break
        
        logger.warning(f"Failed to acquire lock {lock_id} within {timeout_ms}ms")
        return None
    
    async def release_lock(self, lock_id: str) -> bool:
        """Release distributed lock"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("DELETE FROM distributed_locks WHERE lock_id = ?", (lock_id,))
                deleted = cursor.rowcount > 0
            
            # Remove from local tracking
            if lock_id in self.lock_holders:
                del self.lock_holders[lock_id]
            
            if lock_id in self.local_locks:
                del self.local_locks[lock_id]
            
            # Notify waiters
            if lock_id in self.lock_waiters:
                for future in self.lock_waiters[lock_id]:
                    if not future.done():
                        future.set_result(True)
                del self.lock_waiters[lock_id]
            
            if deleted:
                logger.debug(f"Released lock {lock_id}")
            
            return deleted
            
        except Exception as e:
            logger.error(f"Lock release failed for {lock_id}: {e}")
            return False
    
    async def _try_acquire_lock(self, lock_id: str, workflow_id: str, holder_id: str, lock_type: str) -> bool:
        """Try to acquire lock atomically"""
        expires_at = datetime.now() + timedelta(minutes=5)  # 5 minute expiry
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if lock already exists
                cursor = conn.execute("SELECT holder_id FROM distributed_locks WHERE lock_id = ?", (lock_id,))
                existing = cursor.fetchone()
                
                if existing:
                    # Check if same holder (recursive lock)
                    if existing[0] == holder_id:
                        conn.execute("""
                            UPDATE distributed_locks 
                            SET recursive_count = recursive_count + 1, expires_at = ?
                            WHERE lock_id = ?
                        """, (expires_at.isoformat(), lock_id))
                        return True
                    else:
                        return False
                
                # Try to acquire new lock
                conn.execute("""
                    INSERT INTO distributed_locks 
                    (lock_id, workflow_id, holder_id, acquired_at, expires_at, lock_type, recursive_count)
                    VALUES (?, ?, ?, ?, ?, ?, 0)
                """, (lock_id, workflow_id, holder_id, datetime.now().isoformat(), 
                      expires_at.isoformat(), lock_type))
                
                # Track locally
                lock_info = DistributedLockInfo(
                    lock_id=lock_id,
                    workflow_id=workflow_id,
                    holder_id=holder_id,
                    acquired_at=datetime.now(),
                    expires_at=expires_at,
                    lock_type=lock_type
                )
                self.lock_holders[lock_id] = lock_info
                
                return True
                
        except sqlite3.IntegrityError:
            # Lock already exists
            return False
        except Exception as e:
            logger.error(f"Failed to acquire lock {lock_id}: {e}")
            return False
    
    async def _clean_expired_locks(self):
        """Clean up expired locks"""
        try:
            now = datetime.now().isoformat()
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("DELETE FROM distributed_locks WHERE expires_at < ?", (now,))
                if cursor.rowcount > 0:
                    logger.debug(f"Cleaned {cursor.rowcount} expired locks")
        except Exception as e:
            logger.error(f"Failed to clean expired locks: {e}")

class AdvancedCheckpointManager:
    """Advanced checkpoint management with multiple strategies"""
    
    def __init__(self, version_manager: StateVersionManager, lock_manager: DistributedLockManager):
        self.version_manager = version_manager
        self.lock_manager = lock_manager
        self.checkpoints: Dict[str, CheckpointMetadata] = {}
        self.checkpoint_schedule: List[Tuple[float, str]] = []  # (timestamp, workflow_id) heap
        self.strategy_config = {
            CheckpointStrategy.TIME_BASED: {"interval_ms": 30000},  # 30 seconds
            CheckpointStrategy.OPERATION_BASED: {"operations_count": 10},
            CheckpointStrategy.MEMORY_BASED: {"memory_threshold_mb": 100},
            CheckpointStrategy.ADAPTIVE: {"auto_adjust": True}
        }
    
    async def create_checkpoint(self, workflow_id: str, strategy: CheckpointStrategy = CheckpointStrategy.ADAPTIVE) -> str:
        """Create checkpoint with specified strategy"""
        start_time = time.time()
        checkpoint_id = f"checkpoint_{workflow_id}_{uuid.uuid4().hex[:8]}"
        
        try:
            # Acquire distributed lock
            lock_id = await self.lock_manager.acquire_lock(workflow_id, "checkpoint_create")
            if not lock_id:
                raise Exception("Failed to acquire checkpoint lock")
            
            try:
                # Get current state versions
                versions = await self.version_manager.get_version_history(workflow_id, limit=1)
                if not versions:
                    raise Exception(f"No versions found for workflow {workflow_id}")
                
                current_version = versions[0]
                
                # Create checkpoint version
                checkpoint_version = await self.version_manager.create_version(
                    workflow_id=workflow_id,
                    state_data=current_version.state_data,
                    version_type=StateVersionType.SNAPSHOT,
                    parent_version_id=current_version.version_id
                )
                
                # Calculate compression ratio
                original_size = len(pickle.dumps(current_version.state_data))
                compression_ratio = 1.0 - (checkpoint_version.size_bytes / original_size) if original_size > 0 else 0.0
                
                # Create checkpoint metadata
                metadata = CheckpointMetadata(
                    checkpoint_id=checkpoint_id,
                    workflow_id=workflow_id,
                    strategy=strategy,
                    created_at=datetime.now(),
                    size_bytes=checkpoint_version.size_bytes,
                    compression_ratio=compression_ratio,
                    versions_included=[checkpoint_version.version_id],
                    recovery_time_estimate_ms=self._estimate_recovery_time(checkpoint_version.size_bytes),
                    consistency_level=ConsistencyLevel.STRONG
                )
                
                self.checkpoints[checkpoint_id] = metadata
                
                creation_time = (time.time() - start_time) * 1000
                logger.info(f"Created checkpoint {checkpoint_id} in {creation_time:.1f}ms (compression: {compression_ratio:.1%})")
                
                return checkpoint_id
                
            finally:
                await self.lock_manager.release_lock(lock_id)
                
        except Exception as e:
            logger.error(f"Checkpoint creation failed for {workflow_id}: {e}")
            raise
    
    async def recover_from_checkpoint(self, workflow_id: str, checkpoint_id: str, 
                                    strategy: RecoveryStrategy = RecoveryStrategy.IMMEDIATE) -> bool:
        """Recover workflow from checkpoint"""
        start_time = time.time()
        
        try:
            if checkpoint_id not in self.checkpoints:
                logger.error(f"Checkpoint {checkpoint_id} not found")
                return False
            
            checkpoint_metadata = self.checkpoints[checkpoint_id]
            
            # Acquire distributed lock
            lock_id = await self.lock_manager.acquire_lock(workflow_id, "recovery")
            if not lock_id:
                raise Exception("Failed to acquire recovery lock")
            
            try:
                # Create recovery plan
                recovery_plan = await self._create_recovery_plan(checkpoint_metadata, strategy)
                
                # Execute recovery based on strategy
                if strategy == RecoveryStrategy.IMMEDIATE:
                    success = await self._immediate_recovery(recovery_plan)
                elif strategy == RecoveryStrategy.LAZY:
                    success = await self._lazy_recovery(recovery_plan)
                elif strategy == RecoveryStrategy.PARALLEL:
                    success = await self._parallel_recovery(recovery_plan)
                else:
                    success = await self._selective_recovery(recovery_plan)
                
                recovery_time = (time.time() - start_time) * 1000
                
                if success:
                    logger.info(f"Recovered workflow {workflow_id} from checkpoint {checkpoint_id} in {recovery_time:.1f}ms")
                else:
                    logger.error(f"Recovery failed for workflow {workflow_id}")
                
                return success
                
            finally:
                await self.lock_manager.release_lock(lock_id)
                
        except Exception as e:
            logger.error(f"Recovery failed for {workflow_id}: {e}")
            return False
    
    async def _create_recovery_plan(self, checkpoint_metadata: CheckpointMetadata, 
                                  strategy: RecoveryStrategy) -> RecoveryPlan:
        """Create recovery execution plan"""
        return RecoveryPlan(
            recovery_id=f"recovery_{uuid.uuid4().hex[:8]}",
            workflow_id=checkpoint_metadata.workflow_id,
            target_checkpoint_id=checkpoint_metadata.checkpoint_id,
            strategy=strategy,
            estimated_time_ms=checkpoint_metadata.recovery_time_estimate_ms,
            required_resources={"memory_mb": checkpoint_metadata.size_bytes / (1024 * 1024)},
            dependencies=checkpoint_metadata.versions_included,
            rollback_points=[]
        )
    
    async def _immediate_recovery(self, plan: RecoveryPlan) -> bool:
        """Immediate recovery strategy"""
        try:
            checkpoint_metadata = self.checkpoints[plan.target_checkpoint_id]
            
            # Get the checkpoint version
            for version_id in checkpoint_metadata.versions_included:
                version = await self.version_manager.get_version(version_id)
                if version:
                    # Create new current version from checkpoint
                    await self.version_manager.create_version(
                        workflow_id=plan.workflow_id,
                        state_data=version.state_data,
                        version_type=StateVersionType.SNAPSHOT,
                        parent_version_id=version.version_id
                    )
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Immediate recovery failed: {e}")
            return False
    
    async def _lazy_recovery(self, plan: RecoveryPlan) -> bool:
        """Lazy recovery strategy - mark for recovery, execute later"""
        # For demo, just do immediate recovery
        return await self._immediate_recovery(plan)
    
    async def _parallel_recovery(self, plan: RecoveryPlan) -> bool:
        """Parallel recovery strategy"""
        # For demo, just do immediate recovery
        return await self._immediate_recovery(plan)
    
    async def _selective_recovery(self, plan: RecoveryPlan) -> bool:
        """Selective recovery strategy"""
        # For demo, just do immediate recovery
        return await self._immediate_recovery(plan)
    
    def _estimate_recovery_time(self, size_bytes: int) -> float:
        """Estimate recovery time based on data size"""
        # Rough estimate: 1MB/ms for decompression + overhead
        base_time_ms = (size_bytes / (1024 * 1024)) * 10  # 10ms per MB
        overhead_ms = 50  # Base overhead
        return base_time_ms + overhead_ms

class WorkflowStateOrchestrator:
    """Main orchestrator for workflow state management"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.version_manager = StateVersionManager(
            db_path=self.config.get("version_db_path", "state_versions.db")
        )
        self.lock_manager = DistributedLockManager(
            db_path=self.config.get("locks_db_path", "distributed_locks.db")
        )
        self.checkpoint_manager = AdvancedCheckpointManager(
            self.version_manager, self.lock_manager
        )
        
        # State tracking
        self.active_workflows: Dict[str, StateVersion] = {}
        self.operation_counts: Dict[str, int] = defaultdict(int)
        
        # Performance metrics
        self.metrics = {
            "checkpoints_created": 0,
            "recoveries_performed": 0,
            "versions_created": 0,
            "total_checkpoint_time_ms": 0.0,
            "total_recovery_time_ms": 0.0,
            "compression_ratio_avg": 0.0,
            "recovery_success_rate": 1.0
        }
        
        logger.info("Workflow State Orchestrator initialized")
    
    async def create_workflow(self, workflow_id: str, initial_state: Dict[str, Any]) -> bool:
        """Create new workflow with initial state"""
        try:
            # Create initial version
            version = await self.version_manager.create_version(
                workflow_id=workflow_id,
                state_data=initial_state,
                version_type=StateVersionType.SNAPSHOT
            )
            
            self.active_workflows[workflow_id] = version
            self.metrics["versions_created"] += 1
            
            logger.info(f"Created workflow {workflow_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create workflow {workflow_id}: {e}")
            return False
    
    async def update_workflow_state(self, workflow_id: str, state_updates: Dict[str, Any]) -> bool:
        """Update workflow state"""
        try:
            # Get current state
            current_version = self.active_workflows.get(workflow_id)
            if not current_version:
                # Try to load from database
                versions = await self.version_manager.get_version_history(workflow_id, limit=1)
                if not versions:
                    logger.error(f"Workflow {workflow_id} not found")
                    return False
                current_version = versions[0]
            
            # Create new state
            new_state_data = {**current_version.state_data, **state_updates}
            
            # Create incremental version
            new_version = await self.version_manager.create_version(
                workflow_id=workflow_id,
                state_data=new_state_data,
                version_type=StateVersionType.INCREMENTAL,
                parent_version_id=current_version.version_id
            )
            
            self.active_workflows[workflow_id] = new_version
            self.operation_counts[workflow_id] += 1
            self.metrics["versions_created"] += 1
            
            # Check if checkpoint should be created
            await self._check_checkpoint_conditions(workflow_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update workflow {workflow_id}: {e}")
            return False
    
    async def create_checkpoint(self, workflow_id: str, strategy: CheckpointStrategy = CheckpointStrategy.ADAPTIVE) -> str:
        """Create checkpoint for workflow"""
        start_time = time.time()
        
        try:
            checkpoint_id = await self.checkpoint_manager.create_checkpoint(workflow_id, strategy)
            
            creation_time = (time.time() - start_time) * 1000
            self.metrics["checkpoints_created"] += 1
            self.metrics["total_checkpoint_time_ms"] += creation_time
            
            # Update compression ratio average
            if checkpoint_id in self.checkpoint_manager.checkpoints:
                checkpoint_metadata = self.checkpoint_manager.checkpoints[checkpoint_id]
                current_avg = self.metrics["compression_ratio_avg"]
                count = self.metrics["checkpoints_created"]
                self.metrics["compression_ratio_avg"] = (
                    (current_avg * (count - 1) + checkpoint_metadata.compression_ratio) / count
                )
            
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint for {workflow_id}: {e}")
            return ""
    
    async def recover_workflow(self, workflow_id: str, checkpoint_id: str, 
                             strategy: RecoveryStrategy = RecoveryStrategy.IMMEDIATE) -> bool:
        """Recover workflow from checkpoint"""
        start_time = time.time()
        
        try:
            success = await self.checkpoint_manager.recover_from_checkpoint(workflow_id, checkpoint_id, strategy)
            
            recovery_time = (time.time() - start_time) * 1000
            self.metrics["recoveries_performed"] += 1
            self.metrics["total_recovery_time_ms"] += recovery_time
            
            # Update recovery success rate
            current_rate = self.metrics["recovery_success_rate"]
            count = self.metrics["recoveries_performed"]
            success_value = 1.0 if success else 0.0
            self.metrics["recovery_success_rate"] = (
                (current_rate * (count - 1) + success_value) / count
            )
            
            if success:
                # Reload active workflow state
                versions = await self.version_manager.get_version_history(workflow_id, limit=1)
                if versions:
                    self.active_workflows[workflow_id] = versions[0]
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to recover workflow {workflow_id}: {e}")
            return False
    
    async def rollback_workflow(self, workflow_id: str, target_version_id: str) -> bool:
        """Rollback workflow to specific version"""
        try:
            success = await self.version_manager.rollback_to_version(workflow_id, target_version_id)
            
            if success:
                # Update active workflow
                versions = await self.version_manager.get_version_history(workflow_id, limit=1)
                if versions:
                    self.active_workflows[workflow_id] = versions[0]
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to rollback workflow {workflow_id}: {e}")
            return False
    
    async def get_workflow_state(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current workflow state"""
        try:
            current_version = self.active_workflows.get(workflow_id)
            if not current_version:
                versions = await self.version_manager.get_version_history(workflow_id, limit=1)
                if versions:
                    current_version = versions[0]
                    self.active_workflows[workflow_id] = current_version
            
            return current_version.state_data if current_version else None
            
        except Exception as e:
            logger.error(f"Failed to get workflow state for {workflow_id}: {e}")
            return None
    
    async def _check_checkpoint_conditions(self, workflow_id: str):
        """Check if checkpoint should be created based on conditions"""
        operation_count = self.operation_counts[workflow_id]
        
        # Operation-based checkpointing
        if operation_count >= 10:  # Every 10 operations
            await self.create_checkpoint(workflow_id, CheckpointStrategy.OPERATION_BASED)
            self.operation_counts[workflow_id] = 0
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            **self.metrics,
            "average_checkpoint_time_ms": (
                self.metrics["total_checkpoint_time_ms"] / max(1, self.metrics["checkpoints_created"])
            ),
            "average_recovery_time_ms": (
                self.metrics["total_recovery_time_ms"] / max(1, self.metrics["recoveries_performed"])
            ),
            "active_workflows": len(self.active_workflows)
        }

# Demo and testing function
async def run_workflow_state_management_demo():
    """Demonstrate workflow state management capabilities"""
    
    print("🚀 Running Workflow State Management Demo")
    print("=" * 50)
    
    # Initialize orchestrator
    orchestrator = WorkflowStateOrchestrator()
    
    # Test data
    workflows = [
        ("workflow_001", {"step": "start", "data": {"value": 0}}),
        ("workflow_002", {"step": "init", "data": {"count": 1}}),
        ("workflow_003", {"step": "begin", "data": {"items": []}})
    ]
    
    results = {
        "workflows_created": 0,
        "checkpoints_created": 0,
        "recoveries_successful": 0,
        "rollbacks_successful": 0,
        "state_updates": 0,
        "performance_metrics": {}
    }
    
    try:
        # Test 1: Create workflows
        print("\n📋 Creating workflows...")
        for workflow_id, initial_state in workflows:
            success = await orchestrator.create_workflow(workflow_id, initial_state)
            if success:
                results["workflows_created"] += 1
                print(f"  ✅ Created {workflow_id}")
            else:
                print(f"  ❌ Failed to create {workflow_id}")
        
        # Test 2: Update workflow states
        print("\n📋 Updating workflow states...")
        for workflow_id, _ in workflows:
            for i in range(5):
                success = await orchestrator.update_workflow_state(
                    workflow_id, 
                    {"step": f"step_{i+1}", "data": {"value": i+1, "timestamp": time.time()}}
                )
                if success:
                    results["state_updates"] += 1
        
        print(f"  ✅ Completed {results['state_updates']} state updates")
        
        # Test 3: Create checkpoints
        print("\n📋 Creating checkpoints...")
        checkpoint_ids = []
        for workflow_id, _ in workflows:
            checkpoint_id = await orchestrator.create_checkpoint(workflow_id, CheckpointStrategy.MANUAL)
            if checkpoint_id:
                checkpoint_ids.append((workflow_id, checkpoint_id))
                results["checkpoints_created"] += 1
                print(f"  ✅ Created checkpoint for {workflow_id}: {checkpoint_id}")
        
        # Test 4: Continue updating states
        print("\n📋 Continuing state updates...")
        for workflow_id, _ in workflows:
            for i in range(3):
                await orchestrator.update_workflow_state(
                    workflow_id,
                    {"step": f"final_step_{i}", "data": {"final_value": i+10}}
                )
                results["state_updates"] += 1
        
        # Test 5: Test recovery
        print("\n📋 Testing checkpoint recovery...")
        for workflow_id, checkpoint_id in checkpoint_ids:
            success = await orchestrator.recover_workflow(workflow_id, checkpoint_id, RecoveryStrategy.IMMEDIATE)
            if success:
                results["recoveries_successful"] += 1
                print(f"  ✅ Recovered {workflow_id} from {checkpoint_id}")
            else:
                print(f"  ❌ Failed to recover {workflow_id}")
        
        # Test 6: Test rollback
        print("\n📋 Testing version rollback...")
        for workflow_id, _ in workflows[:1]:  # Test with first workflow
            # Get version history
            versions = await orchestrator.version_manager.get_version_history(workflow_id, limit=5)
            if len(versions) >= 3:
                target_version = versions[2]  # Rollback to 3rd most recent
                success = await orchestrator.rollback_workflow(workflow_id, target_version.version_id)
                if success:
                    results["rollbacks_successful"] += 1
                    print(f"  ✅ Rolled back {workflow_id} to version {target_version.version_id}")
        
        # Test 7: Get final states
        print("\n📋 Getting final workflow states...")
        for workflow_id, _ in workflows:
            state = await orchestrator.get_workflow_state(workflow_id)
            if state:
                print(f"  ✅ {workflow_id}: {state.get('step', 'unknown')} - {state.get('data', {}).get('value', 'N/A')}")
        
        # Get performance metrics
        results["performance_metrics"] = await orchestrator.get_performance_metrics()
        
        # Display results
        print("\n" + "=" * 50)
        print("🎯 Demo Results Summary")
        print("=" * 50)
        print(f"Workflows Created: {results['workflows_created']}")
        print(f"State Updates: {results['state_updates']}")
        print(f"Checkpoints Created: {results['checkpoints_created']}")
        print(f"Recoveries Successful: {results['recoveries_successful']}")
        print(f"Rollbacks Successful: {results['rollbacks_successful']}")
        
        metrics = results["performance_metrics"]
        print(f"\n📊 Performance Metrics:")
        print(f"  Average Checkpoint Time: {metrics['average_checkpoint_time_ms']:.1f}ms")
        print(f"  Average Recovery Time: {metrics['average_recovery_time_ms']:.1f}ms")
        print(f"  Compression Ratio: {metrics['compression_ratio_avg']:.1%}")
        print(f"  Recovery Success Rate: {metrics['recovery_success_rate']:.1%}")
        print(f"  Versions Created: {metrics['versions_created']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return results

if __name__ == "__main__":
    asyncio.run(run_workflow_state_management_demo())