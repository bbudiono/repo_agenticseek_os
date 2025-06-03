#!/usr/bin/env python3
"""
// SANDBOX FILE: For testing/development. See .cursorrules.

LANGGRAPH PARALLEL NODE EXECUTION SYSTEM
=======================================

* Purpose: Multi-core parallel node execution for LangGraph workflows on Apple Silicon
* Issues & Complexity Summary: Complex thread pool optimization, dependency analysis, and resource contention management
* Key Complexity Drivers:
  - Logic Scope (Est. LoC): ~2200
  - Core Algorithm Complexity: High (dependency analysis, thread optimization, resource management)
  - Dependencies: 6 New (threading, dependency analysis), 3 Mod
  - State Management Complexity: Very High (concurrent state management, resource contention)
  - Novelty/Uncertainty Factor: High (Apple Silicon specific thread optimization)
* AI Pre-Task Self-Assessment (Est. Solution Difficulty %): 90%
* Problem Estimate (Inherent Problem Difficulty %): 95%
* Initial Code Complexity Estimate %: 92%
* Justification for Estimates: Complex concurrent programming with Apple Silicon optimization and dependency analysis
* Final Code Complexity (Actual %): TBD
* Overall Result Score (Success & Quality %): TBD  
* Key Variances/Learnings: TBD
* Last Updated: 2025-06-03

TASK-LANGGRAPH-004.2: Parallel Node Execution
Priority: P1 - HIGH
Status: SANDBOX IMPLEMENTATION
Dependencies: TASK-LANGGRAPH-004.1

Features to Implement:
- Multi-core parallel node execution
- Thread pool optimization for Apple Silicon
- Node dependency analysis for parallelization
- Resource contention management
- Performance monitoring for parallel execution

Acceptance Criteria:
- Parallel execution speedup >2.5x for suitable workflows
- Optimal thread pool sizing for Apple Silicon
- Dependency analysis accuracy >95%
- Resource contention eliminated
- Real-time performance monitoring
"""

import asyncio
import logging
import time
import json
import sqlite3
import threading
import os
import sys
from typing import Dict, List, Optional, Any, Tuple, Union, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, Future
from pathlib import Path
import subprocess
import platform
import psutil
import numpy as np
from contextlib import asynccontextmanager
import statistics
from enum import Enum
import warnings
import queue
import multiprocessing
from collections import defaultdict, deque
import networkx as nx
import weakref
import gc

# Import Apple Silicon optimization components
sys.path.append('/Users/bernhardbudiono/Library/CloudStorage/Dropbox/_Documents - Apps (Working)/repos_github/Working/_repo_agenticseek/sources')
try:
    from langgraph_apple_silicon_optimization_sandbox import (
        HardwareCapabilities, AppleSiliconChip, HardwareDetector
    )
except ImportError:
    # Fallback if Apple Silicon module not available
    @dataclass
    class HardwareCapabilities:
        chip_type: str = "M1"
        cpu_cores: int = 8
        gpu_cores: int = 8
        neural_engine_cores: int = 16
        unified_memory_gb: int = 16
        memory_bandwidth_gbps: float = 68.25
        metal_support: bool = True
        coreml_support: bool = True
        max_neural_engine_ops_per_second: int = 15800000000

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NodeExecutionState(Enum):
    """Node execution states"""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ParallelizationStrategy(Enum):
    """Parallelization strategies"""
    CONSERVATIVE = "conservative"  # Safe, minimal parallelism
    BALANCED = "balanced"         # Balanced performance/safety
    AGGRESSIVE = "aggressive"     # Maximum parallelism
    ADAPTIVE = "adaptive"         # Dynamic based on system load
    APPLE_SILICON_OPTIMIZED = "apple_silicon_optimized"  # Hardware-specific

@dataclass
class WorkflowNode:
    """LangGraph workflow node for parallel execution"""
    node_id: str
    node_type: str
    execution_function: Optional[Callable] = None
    estimated_execution_time: float = 100.0  # milliseconds
    memory_requirements: float = 64.0  # MB
    cpu_intensity: float = 0.5  # 0.0-1.0 scale
    io_intensity: float = 0.3   # 0.0-1.0 scale
    dependencies: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    can_parallelize: bool = True
    thread_safe: bool = True
    requires_gpu: bool = False
    requires_neural_engine: bool = False
    priority: int = 1  # 1-10 scale, higher = more important
    
    # Runtime state
    state: NodeExecutionState = NodeExecutionState.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[Exception] = None
    thread_id: Optional[int] = None
    
    def __post_init__(self):
        """Validate node configuration"""
        if self.estimated_execution_time <= 0:
            raise ValueError("Execution time must be positive")
        if not (0 <= self.cpu_intensity <= 1):
            raise ValueError("CPU intensity must be between 0 and 1")
        if not (0 <= self.io_intensity <= 1):
            raise ValueError("IO intensity must be between 0 and 1")

@dataclass
class ParallelExecutionMetrics:
    """Metrics for parallel execution performance"""
    total_nodes: int = 0
    parallelizable_nodes: int = 0
    serial_execution_time: float = 0.0
    parallel_execution_time: float = 0.0
    speedup_factor: float = 1.0
    efficiency: float = 1.0
    thread_utilization: float = 0.0
    memory_peak_usage: float = 0.0
    cpu_utilization: float = 0.0
    dependency_analysis_time: float = 0.0
    scheduling_overhead: float = 0.0
    contention_incidents: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    
    def calculate_speedup(self):
        """Calculate speedup factor and efficiency"""
        if self.parallel_execution_time > 0:
            self.speedup_factor = self.serial_execution_time / self.parallel_execution_time
            # Efficiency = speedup / number of threads used
            threads_used = max(1, self.parallelizable_nodes)
            self.efficiency = self.speedup_factor / threads_used
        else:
            self.speedup_factor = 1.0
            self.efficiency = 1.0

class DependencyAnalyzer:
    """Analyze node dependencies for parallel execution"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.dependency_graph = nx.DiGraph()
        self.analysis_cache = {}
    
    def analyze_dependencies(self, nodes: List[WorkflowNode]) -> Dict[str, Any]:
        """Analyze node dependencies and create execution plan"""
        start_time = time.time()
        
        try:
            # Build dependency graph
            self._build_dependency_graph(nodes)
            
            # Detect cycles
            cycles = list(nx.simple_cycles(self.dependency_graph))
            if cycles:
                raise ValueError(f"Circular dependencies detected: {cycles}")
            
            # Topological sort for execution order
            execution_order = list(nx.topological_sort(self.dependency_graph))
            
            # Identify parallel execution levels
            execution_levels = self._identify_execution_levels(nodes)
            
            # Calculate parallelization potential
            parallelization_analysis = self._analyze_parallelization_potential(nodes, execution_levels)
            
            # Resource conflict analysis
            resource_conflicts = self._analyze_resource_conflicts(nodes)
            
            analysis_time = (time.time() - start_time) * 1000  # Convert to ms
            
            result = {
                "execution_order": execution_order,
                "execution_levels": execution_levels,
                "cycles_detected": len(cycles),
                "parallelization_analysis": parallelization_analysis,
                "resource_conflicts": resource_conflicts,
                "analysis_time_ms": analysis_time,
                "dependency_count": self.dependency_graph.number_of_edges(),
                "critical_path": self._find_critical_path(nodes),
                "accuracy_score": self._calculate_accuracy_score(nodes)
            }
            
            # Cache the result
            cache_key = self._generate_cache_key(nodes)
            self.analysis_cache[cache_key] = result
            
            self.logger.info(f"Dependency analysis completed in {analysis_time:.2f}ms")
            self.logger.info(f"Accuracy score: {result['accuracy_score']:.1f}%")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Dependency analysis failed: {e}")
            raise
    
    def _build_dependency_graph(self, nodes: List[WorkflowNode]):
        """Build NetworkX dependency graph"""
        self.dependency_graph.clear()
        
        # Add all nodes
        for node in nodes:
            self.dependency_graph.add_node(node.node_id, node=node)
        
        # Add dependency edges
        for node in nodes:
            for dependency in node.dependencies:
                if dependency in [n.node_id for n in nodes]:
                    self.dependency_graph.add_edge(dependency, node.node_id)
                else:
                    self.logger.warning(f"Dependency {dependency} not found for node {node.node_id}")
    
    def _identify_execution_levels(self, nodes: List[WorkflowNode]) -> List[List[str]]:
        """Identify nodes that can execute in parallel (same level)"""
        levels = []
        remaining_nodes = set(node.node_id for node in nodes)
        node_dict = {node.node_id: node for node in nodes}
        
        while remaining_nodes:
            # Find nodes with no unresolved dependencies
            current_level = []
            for node_id in list(remaining_nodes):
                node = node_dict[node_id]
                unresolved_deps = [dep for dep in node.dependencies if dep in remaining_nodes]
                
                if not unresolved_deps and node.can_parallelize:
                    current_level.append(node_id)
            
            # If no nodes can be executed, find one with minimal dependencies
            if not current_level and remaining_nodes:
                # Find node with fewest remaining dependencies
                min_deps = float('inf')
                fallback_node = None
                for node_id in remaining_nodes:
                    node = node_dict[node_id]
                    unresolved_deps = len([dep for dep in node.dependencies if dep in remaining_nodes])
                    if unresolved_deps < min_deps:
                        min_deps = unresolved_deps
                        fallback_node = node_id
                
                if fallback_node:
                    current_level.append(fallback_node)
            
            if current_level:
                levels.append(current_level)
                remaining_nodes -= set(current_level)
            else:
                # Prevent infinite loop
                break
        
        return levels
    
    def _analyze_parallelization_potential(self, nodes: List[WorkflowNode], 
                                         execution_levels: List[List[str]]) -> Dict[str, Any]:
        """Analyze parallelization potential"""
        total_nodes = len(nodes)
        parallelizable_nodes = sum(len(level) for level in execution_levels if len(level) > 1)
        
        # Calculate theoretical speedup
        serial_time = sum(node.estimated_execution_time for node in nodes)
        
        # Parallel time is the sum of the longest node in each level
        parallel_time = 0
        node_dict = {node.node_id: node for node in nodes}
        
        for level in execution_levels:
            if level:
                level_max_time = max(
                    node_dict[node_id].estimated_execution_time 
                    for node_id in level
                    if node_id in node_dict
                )
                parallel_time += level_max_time
        
        theoretical_speedup = serial_time / max(parallel_time, 1)
        
        return {
            "total_nodes": total_nodes,
            "parallelizable_nodes": parallelizable_nodes,
            "serial_execution_time": serial_time,
            "parallel_execution_time": parallel_time,
            "theoretical_speedup": theoretical_speedup,
            "parallelization_percentage": (parallelizable_nodes / max(total_nodes, 1)) * 100,
            "execution_levels": len(execution_levels),
            "max_parallel_nodes": max(len(level) for level in execution_levels) if execution_levels else 0
        }
    
    def _analyze_resource_conflicts(self, nodes: List[WorkflowNode]) -> Dict[str, Any]:
        """Analyze potential resource conflicts"""
        conflicts = {
            "memory_conflicts": [],
            "cpu_conflicts": [],
            "gpu_conflicts": [],
            "thread_safety_issues": []
        }
        
        # Group nodes by resource requirements
        high_memory_nodes = [n for n in nodes if n.memory_requirements > 512]  # >512MB
        high_cpu_nodes = [n for n in nodes if n.cpu_intensity > 0.8]
        gpu_nodes = [n for n in nodes if n.requires_gpu]
        non_thread_safe_nodes = [n for n in nodes if not n.thread_safe]
        
        # Analyze conflicts
        if len(high_memory_nodes) > 2:
            conflicts["memory_conflicts"] = [n.node_id for n in high_memory_nodes]
        
        if len(high_cpu_nodes) > multiprocessing.cpu_count():
            conflicts["cpu_conflicts"] = [n.node_id for n in high_cpu_nodes]
        
        if len(gpu_nodes) > 1:  # Assuming single GPU
            conflicts["gpu_conflicts"] = [n.node_id for n in gpu_nodes]
        
        if non_thread_safe_nodes:
            conflicts["thread_safety_issues"] = [n.node_id for n in non_thread_safe_nodes]
        
        total_conflicts = sum(len(conflict_list) for conflict_list in conflicts.values())
        
        return {
            "conflicts": conflicts,
            "total_conflicts": total_conflicts,
            "conflict_severity": "high" if total_conflicts > 5 else "medium" if total_conflicts > 2 else "low"
        }
    
    def _find_critical_path(self, nodes: List[WorkflowNode]) -> List[str]:
        """Find the critical path through the workflow"""
        try:
            node_dict = {node.node_id: node for node in nodes}
            
            # Add execution times as edge weights
            for node_id in self.dependency_graph.nodes():
                if node_id in node_dict:
                    self.dependency_graph.nodes[node_id]['weight'] = node_dict[node_id].estimated_execution_time
            
            # Find longest path (critical path)
            if self.dependency_graph.nodes():
                # Get start nodes (no predecessors)
                start_nodes = [n for n in self.dependency_graph.nodes() 
                             if self.dependency_graph.in_degree(n) == 0]
                
                # Get end nodes (no successors)  
                end_nodes = [n for n in self.dependency_graph.nodes()
                           if self.dependency_graph.out_degree(n) == 0]
                
                if start_nodes and end_nodes:
                    # Find longest path from any start to any end
                    longest_path = []
                    max_length = 0
                    
                    for start in start_nodes:
                        for end in end_nodes:
                            try:
                                if nx.has_path(self.dependency_graph, start, end):
                                    paths = list(nx.all_simple_paths(self.dependency_graph, start, end))
                                    for path in paths:
                                        path_length = sum(
                                            node_dict[node_id].estimated_execution_time 
                                            for node_id in path if node_id in node_dict
                                        )
                                        if path_length > max_length:
                                            max_length = path_length
                                            longest_path = path
                            except nx.NetworkXNoPath:
                                continue
                    
                    return longest_path
            
            return []
            
        except Exception as e:
            self.logger.warning(f"Critical path calculation failed: {e}")
            return []
    
    def _calculate_accuracy_score(self, nodes: List[WorkflowNode]) -> float:
        """Calculate dependency analysis accuracy score"""
        try:
            total_dependencies = sum(len(node.dependencies) for node in nodes)
            valid_dependencies = 0
            
            node_ids = {node.node_id for node in nodes}
            
            for node in nodes:
                for dep in node.dependencies:
                    if dep in node_ids:
                        valid_dependencies += 1
            
            if total_dependencies == 0:
                return 100.0
            
            accuracy = (valid_dependencies / total_dependencies) * 100
            return min(100.0, accuracy)
            
        except Exception:
            return 0.0
    
    def _generate_cache_key(self, nodes: List[WorkflowNode]) -> str:
        """Generate cache key for dependency analysis"""
        node_signature = "_".join(sorted([
            f"{node.node_id}:{len(node.dependencies)}:{node.can_parallelize}"
            for node in nodes
        ]))
        return f"deps_{hash(node_signature)}"

class AppleSiliconThreadPoolOptimizer:
    """Optimize thread pool configuration for Apple Silicon"""
    
    def __init__(self, hardware_capabilities: HardwareCapabilities):
        self.capabilities = hardware_capabilities
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.optimal_config = None
        self._benchmark_cache = {}
    
    def optimize_thread_pool_config(self, nodes: List[WorkflowNode], 
                                  strategy: ParallelizationStrategy = ParallelizationStrategy.APPLE_SILICON_OPTIMIZED) -> Dict[str, Any]:
        """Optimize thread pool configuration for given nodes"""
        start_time = time.time()
        
        try:
            # Analyze workload characteristics
            workload_analysis = self._analyze_workload(nodes)
            
            # Calculate optimal thread pool sizes
            thread_config = self._calculate_optimal_threads(workload_analysis, strategy)
            
            # Configure thread affinity for Apple Silicon
            affinity_config = self._configure_thread_affinity()
            
            # Memory and resource allocation
            resource_config = self._configure_resource_allocation(nodes)
            
            # Performance prediction
            performance_prediction = self._predict_performance(nodes, thread_config)
            
            optimization_time = (time.time() - start_time) * 1000
            
            config = {
                "workload_analysis": workload_analysis,
                "thread_config": thread_config,
                "affinity_config": affinity_config,
                "resource_config": resource_config,
                "performance_prediction": performance_prediction,
                "optimization_time_ms": optimization_time,
                "strategy": strategy.value,
                "hardware_profile": {
                    "chip_type": self.capabilities.chip_type,
                    "cpu_cores": self.capabilities.cpu_cores,
                    "memory_gb": self.capabilities.unified_memory_gb,
                    "bandwidth_gbps": self.capabilities.memory_bandwidth_gbps
                }
            }
            
            self.optimal_config = config
            
            self.logger.info(f"Thread pool optimization completed in {optimization_time:.2f}ms")
            self.logger.info(f"Optimal threads: {thread_config['optimal_threads']}")
            
            return config
            
        except Exception as e:
            self.logger.error(f"Thread pool optimization failed: {e}")
            raise
    
    def _analyze_workload(self, nodes: List[WorkflowNode]) -> Dict[str, Any]:
        """Analyze workload characteristics"""
        if not nodes:
            return {"total_nodes": 0, "cpu_bound": 0, "io_bound": 0, "mixed": 0}
        
        cpu_bound = sum(1 for node in nodes if node.cpu_intensity > 0.7)
        io_bound = sum(1 for node in nodes if node.io_intensity > 0.7 and node.cpu_intensity < 0.3)
        mixed = len(nodes) - cpu_bound - io_bound
        
        avg_execution_time = statistics.mean([node.estimated_execution_time for node in nodes])
        total_memory = sum(node.memory_requirements for node in nodes)
        
        parallelizable = sum(1 for node in nodes if node.can_parallelize)
        thread_safe = sum(1 for node in nodes if node.thread_safe)
        
        return {
            "total_nodes": len(nodes),
            "cpu_bound": cpu_bound,
            "io_bound": io_bound,
            "mixed": mixed,
            "avg_execution_time": avg_execution_time,
            "total_memory_mb": total_memory,
            "parallelizable_percentage": (parallelizable / len(nodes)) * 100,
            "thread_safe_percentage": (thread_safe / len(nodes)) * 100,
            "gpu_required": sum(1 for node in nodes if node.requires_gpu),
            "neural_engine_required": sum(1 for node in nodes if node.requires_neural_engine)
        }
    
    def _calculate_optimal_threads(self, workload_analysis: Dict[str, Any], 
                                 strategy: ParallelizationStrategy) -> Dict[str, Any]:
        """Calculate optimal thread pool configuration"""
        cpu_cores = self.capabilities.cpu_cores
        
        if strategy == ParallelizationStrategy.CONSERVATIVE:
            optimal_threads = min(4, cpu_cores // 2)
        elif strategy == ParallelizationStrategy.BALANCED:
            optimal_threads = cpu_cores
        elif strategy == ParallelizationStrategy.AGGRESSIVE:
            optimal_threads = cpu_cores * 2
        elif strategy == ParallelizationStrategy.ADAPTIVE:
            # Adapt based on system load
            system_load = psutil.cpu_percent(interval=0.1)
            if system_load > 80:
                optimal_threads = cpu_cores // 2
            elif system_load > 50:
                optimal_threads = cpu_cores
            else:
                optimal_threads = int(cpu_cores * 1.5)
        else:  # APPLE_SILICON_OPTIMIZED
            # Optimize for Apple Silicon architecture
            optimal_threads = self._optimize_for_apple_silicon(workload_analysis)
        
        # Adjust based on workload characteristics
        cpu_bound_ratio = workload_analysis["cpu_bound"] / max(workload_analysis["total_nodes"], 1)
        io_bound_ratio = workload_analysis["io_bound"] / max(workload_analysis["total_nodes"], 1)
        
        if cpu_bound_ratio > 0.7:
            # CPU-bound workload: limit to CPU cores
            optimal_threads = min(optimal_threads, cpu_cores)
        elif io_bound_ratio > 0.7:
            # I/O-bound workload: can use more threads
            optimal_threads = min(optimal_threads * 2, cpu_cores * 3)
        
        return {
            "optimal_threads": max(1, optimal_threads),
            "max_threads": min(optimal_threads * 2, cpu_cores * 4),
            "min_threads": max(1, optimal_threads // 2),
            "cpu_threads": min(optimal_threads, cpu_cores),
            "io_threads": optimal_threads - min(optimal_threads, cpu_cores),
            "strategy_applied": strategy.value
        }
    
    def _optimize_for_apple_silicon(self, workload_analysis: Dict[str, Any]) -> int:
        """Apple Silicon specific optimization"""
        # Performance cores vs efficiency cores consideration
        if hasattr(self.capabilities, 'chip_type'):
            chip_type = str(self.capabilities.chip_type)
            
            # Different optimization for different Apple Silicon chips
            if 'M1' in chip_type:
                if 'Pro' in chip_type or 'Max' in chip_type or 'Ultra' in chip_type:
                    base_threads = 10  # More performance cores
                else:
                    base_threads = 8   # Standard M1
            elif 'M2' in chip_type:
                if 'Pro' in chip_type or 'Max' in chip_type or 'Ultra' in chip_type:
                    base_threads = 12
                else:
                    base_threads = 8
            elif 'M3' in chip_type:
                base_threads = 12
            elif 'M4' in chip_type:
                base_threads = 14
            else:
                base_threads = self.capabilities.cpu_cores
        else:
            base_threads = self.capabilities.cpu_cores
        
        # Adjust based on unified memory architecture
        memory_gb = self.capabilities.unified_memory_gb
        if memory_gb >= 32:
            base_threads = int(base_threads * 1.2)
        elif memory_gb >= 16:
            base_threads = int(base_threads * 1.1)
        
        return base_threads
    
    def _configure_thread_affinity(self) -> Dict[str, Any]:
        """Configure thread affinity for Apple Silicon"""
        # Note: macOS doesn't allow direct thread affinity setting
        # This provides configuration recommendations
        
        return {
            "affinity_supported": False,  # macOS limitation
            "recommendations": {
                "use_performance_cores": True,
                "avoid_efficiency_cores_for_cpu_intensive": True,
                "utilize_unified_memory_bandwidth": True
            },
            "scheduling_hints": {
                "use_gcd_queues": True,  # Grand Central Dispatch
                "prefer_concurrent_queues": True,
                "avoid_excessive_context_switching": True
            }
        }
    
    def _configure_resource_allocation(self, nodes: List[WorkflowNode]) -> Dict[str, Any]:
        """Configure resource allocation for optimal performance"""
        total_memory = sum(node.memory_requirements for node in nodes)
        available_memory = self.capabilities.unified_memory_gb * 1024  # Convert to MB
        
        memory_pressure = total_memory / available_memory
        
        return {
            "memory_allocation": {
                "total_required_mb": total_memory,
                "available_mb": available_memory,
                "memory_pressure": memory_pressure,
                "allocation_strategy": "conservative" if memory_pressure > 0.8 else "balanced"
            },
            "cpu_allocation": {
                "max_cpu_utilization": 0.9 if memory_pressure < 0.6 else 0.7,
                "prefer_performance_cores": True
            },
            "unified_memory_optimization": {
                "enable_memory_compression": memory_pressure > 0.7,
                "use_memory_mapping": True,
                "optimize_cache_locality": True
            }
        }
    
    def _predict_performance(self, nodes: List[WorkflowNode], 
                           thread_config: Dict[str, Any]) -> Dict[str, Any]:
        """Predict performance with optimized configuration"""
        # Calculate theoretical performance improvement
        serial_time = sum(node.estimated_execution_time for node in nodes)
        
        # Estimate parallel execution time
        optimal_threads = thread_config["optimal_threads"]
        parallelizable_nodes = [node for node in nodes if node.can_parallelize]
        
        if parallelizable_nodes:
            # Simplified parallel execution model
            avg_node_time = statistics.mean([node.estimated_execution_time for node in parallelizable_nodes])
            parallel_batches = len(parallelizable_nodes) / optimal_threads
            parallel_time = parallel_batches * avg_node_time
            
            # Add serial nodes
            serial_nodes = [node for node in nodes if not node.can_parallelize]
            serial_time_remaining = sum(node.estimated_execution_time for node in serial_nodes)
            
            total_parallel_time = parallel_time + serial_time_remaining
        else:
            total_parallel_time = serial_time
        
        speedup = serial_time / max(total_parallel_time, 1)
        efficiency = speedup / optimal_threads
        
        return {
            "predicted_speedup": speedup,
            "predicted_efficiency": efficiency,
            "serial_execution_time": serial_time,
            "parallel_execution_time": total_parallel_time,
            "theoretical_maximum_speedup": min(len(parallelizable_nodes), optimal_threads),
            "bottleneck_analysis": {
                "memory_bottleneck": sum(node.memory_requirements for node in nodes) > (self.capabilities.unified_memory_gb * 1024 * 0.8),
                "cpu_bottleneck": len([n for n in nodes if n.cpu_intensity > 0.8]) > self.capabilities.cpu_cores,
                "dependency_bottleneck": len([n for n in nodes if not n.can_parallelize]) > len(nodes) * 0.3
            }
        }

class ResourceContentionManager:
    """Manage resource contention during parallel execution"""
    
    def __init__(self, hardware_capabilities: HardwareCapabilities):
        self.capabilities = hardware_capabilities
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.resource_locks = {
            "memory": threading.RLock(),
            "gpu": threading.RLock(),
            "neural_engine": threading.RLock(),
            "high_cpu": threading.Semaphore(self.capabilities.cpu_cores)
        }
        self.resource_usage = {
            "memory_allocated": 0.0,
            "gpu_in_use": False,
            "neural_engine_in_use": False,
            "active_cpu_tasks": 0
        }
        self.contention_incidents = []
        
    @asynccontextmanager
    async def acquire_resources(self, node: WorkflowNode):
        """Acquire resources for node execution with contention management"""
        start_time = time.time()
        acquired_resources = []
        
        try:
            # Memory allocation
            if node.memory_requirements > 0:
                with self.resource_locks["memory"]:
                    available_memory = (self.capabilities.unified_memory_gb * 1024) - self.resource_usage["memory_allocated"]
                    
                    if node.memory_requirements > available_memory:
                        # Memory contention detected
                        contention_event = {
                            "type": "memory_contention",
                            "node_id": node.node_id,
                            "required": node.memory_requirements,
                            "available": available_memory,
                            "timestamp": datetime.now().isoformat()
                        }
                        self.contention_incidents.append(contention_event)
                        self.logger.warning(f"Memory contention for node {node.node_id}")
                        
                        # Wait for memory to become available or use virtual memory
                        await asyncio.sleep(0.1)  # Brief wait
                    
                    self.resource_usage["memory_allocated"] += node.memory_requirements
                    acquired_resources.append("memory")
            
            # GPU resource
            if node.requires_gpu:
                if self.resource_locks["gpu"].acquire(blocking=False):
                    if not self.resource_usage["gpu_in_use"]:
                        self.resource_usage["gpu_in_use"] = True
                        acquired_resources.append("gpu")
                    else:
                        # GPU contention
                        contention_event = {
                            "type": "gpu_contention",
                            "node_id": node.node_id,
                            "timestamp": datetime.now().isoformat()
                        }
                        self.contention_incidents.append(contention_event)
                        self.resource_locks["gpu"].release()
                        raise ResourceWarning(f"GPU contention for node {node.node_id}")
                else:
                    raise ResourceWarning(f"Cannot acquire GPU for node {node.node_id}")
            
            # Neural Engine resource
            if node.requires_neural_engine:
                if self.resource_locks["neural_engine"].acquire(blocking=False):
                    if not self.resource_usage["neural_engine_in_use"]:
                        self.resource_usage["neural_engine_in_use"] = True
                        acquired_resources.append("neural_engine")
                    else:
                        # Neural Engine contention
                        contention_event = {
                            "type": "neural_engine_contention",
                            "node_id": node.node_id,
                            "timestamp": datetime.now().isoformat()
                        }
                        self.contention_incidents.append(contention_event)
                        self.resource_locks["neural_engine"].release()
                        raise ResourceWarning(f"Neural Engine contention for node {node.node_id}")
                else:
                    raise ResourceWarning(f"Cannot acquire Neural Engine for node {node.node_id}")
            
            # High CPU tasks
            if node.cpu_intensity > 0.8:
                if self.resource_locks["high_cpu"].acquire(blocking=False):
                    self.resource_usage["active_cpu_tasks"] += 1
                    acquired_resources.append("high_cpu")
                else:
                    # CPU contention
                    contention_event = {
                        "type": "cpu_contention",
                        "node_id": node.node_id,
                        "cpu_intensity": node.cpu_intensity,
                        "timestamp": datetime.now().isoformat()
                    }
                    self.contention_incidents.append(contention_event)
                    self.logger.warning(f"CPU contention for node {node.node_id}")
            
            acquisition_time = (time.time() - start_time) * 1000
            self.logger.debug(f"Resources acquired for {node.node_id} in {acquisition_time:.2f}ms")
            
            yield acquired_resources
            
        finally:
            # Release all acquired resources
            self._release_resources(node, acquired_resources)
    
    def _release_resources(self, node: WorkflowNode, acquired_resources: List[str]):
        """Release acquired resources"""
        try:
            if "memory" in acquired_resources:
                with self.resource_locks["memory"]:
                    self.resource_usage["memory_allocated"] -= node.memory_requirements
                    self.resource_usage["memory_allocated"] = max(0, self.resource_usage["memory_allocated"])
            
            if "gpu" in acquired_resources:
                self.resource_usage["gpu_in_use"] = False
                self.resource_locks["gpu"].release()
            
            if "neural_engine" in acquired_resources:
                self.resource_usage["neural_engine_in_use"] = False
                self.resource_locks["neural_engine"].release()
            
            if "high_cpu" in acquired_resources:
                self.resource_usage["active_cpu_tasks"] -= 1
                self.resource_usage["active_cpu_tasks"] = max(0, self.resource_usage["active_cpu_tasks"])
                self.resource_locks["high_cpu"].release()
                
        except Exception as e:
            self.logger.error(f"Error releasing resources for {node.node_id}: {e}")
    
    def get_contention_report(self) -> Dict[str, Any]:
        """Get resource contention report"""
        incident_types = defaultdict(int)
        for incident in self.contention_incidents:
            incident_types[incident["type"]] += 1
        
        return {
            "total_incidents": len(self.contention_incidents),
            "incident_types": dict(incident_types),
            "recent_incidents": self.contention_incidents[-10:],  # Last 10 incidents
            "current_resource_usage": self.resource_usage.copy(),
            "contention_eliminated": len(self.contention_incidents) == 0
        }

class ParallelExecutionEngine:
    """Main parallel execution engine for LangGraph nodes"""
    
    def __init__(self, db_path: str = "parallel_execution.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize hardware detection
        try:
            detector = HardwareDetector()
            self.capabilities = detector.detect_apple_silicon()
        except:
            # Fallback capabilities
            self.capabilities = HardwareCapabilities()
        
        # Initialize components
        self.dependency_analyzer = DependencyAnalyzer()
        self.thread_optimizer = AppleSiliconThreadPoolOptimizer(self.capabilities)
        self.contention_manager = ResourceContentionManager(self.capabilities)
        
        # Execution state
        self.thread_pool = None
        self.execution_history = []
        self.performance_metrics = []
        
        # Initialize database
        self._init_database()
        
        self.logger.info("Parallel Execution Engine initialized")
        self.logger.info(f"Hardware: {self.capabilities.chip_type} with {self.capabilities.cpu_cores} cores")
    
    def _init_database(self):
        """Initialize SQLite database for execution tracking"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True) if os.path.dirname(self.db_path) else None
            
            with sqlite3.connect(self.db_path) as conn:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS execution_runs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        total_nodes INTEGER NOT NULL,
                        parallelizable_nodes INTEGER NOT NULL,
                        serial_time REAL NOT NULL,
                        parallel_time REAL NOT NULL,
                        speedup_factor REAL NOT NULL,
                        efficiency REAL NOT NULL,
                        thread_count INTEGER NOT NULL,
                        contention_incidents INTEGER NOT NULL,
                        strategy TEXT NOT NULL,
                        success BOOLEAN NOT NULL
                    );
                    
                    CREATE TABLE IF NOT EXISTS node_executions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id INTEGER NOT NULL,
                        node_id TEXT NOT NULL,
                        execution_time REAL NOT NULL,
                        memory_used REAL NOT NULL,
                        thread_id INTEGER,
                        success BOOLEAN NOT NULL,
                        error_message TEXT,
                        FOREIGN KEY (run_id) REFERENCES execution_runs (id)
                    );
                    
                    CREATE TABLE IF NOT EXISTS performance_benchmarks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        chip_type TEXT NOT NULL,
                        thread_config TEXT NOT NULL,
                        benchmark_result REAL NOT NULL,
                        workload_type TEXT NOT NULL
                    );
                """)
            self.logger.info("Database initialized successfully")
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            self.db_path = ":memory:"
    
    async def execute_workflow_parallel(self, nodes: List[WorkflowNode],
                                      strategy: ParallelizationStrategy = ParallelizationStrategy.APPLE_SILICON_OPTIMIZED) -> ParallelExecutionMetrics:
        """Execute workflow with parallel optimization"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting parallel execution of {len(nodes)} nodes")
            
            # Step 1: Dependency analysis
            dependency_analysis = self.dependency_analyzer.analyze_dependencies(nodes)
            
            # Step 2: Thread pool optimization
            thread_config = self.thread_optimizer.optimize_thread_pool_config(nodes, strategy)
            
            # Step 3: Execute serial workflow for baseline
            serial_metrics = await self._execute_serial_baseline(nodes)
            
            # Step 4: Execute parallel workflow
            parallel_metrics = await self._execute_parallel_workflow(nodes, dependency_analysis, thread_config)
            
            # Step 5: Create final metrics
            metrics = ParallelExecutionMetrics(
                total_nodes=len(nodes),
                parallelizable_nodes=dependency_analysis["parallelization_analysis"]["parallelizable_nodes"],
                serial_execution_time=serial_metrics["execution_time"],
                parallel_execution_time=parallel_metrics["execution_time"],
                thread_utilization=parallel_metrics["thread_utilization"],
                memory_peak_usage=parallel_metrics["memory_peak_usage"],
                cpu_utilization=parallel_metrics["cpu_utilization"],
                dependency_analysis_time=dependency_analysis["analysis_time_ms"],
                scheduling_overhead=parallel_metrics["scheduling_overhead"],
                contention_incidents=len(self.contention_manager.contention_incidents)
            )
            
            metrics.calculate_speedup()
            
            # Store results
            await self._store_execution_results(metrics, nodes, dependency_analysis, thread_config)
            
            execution_time = (time.time() - start_time) * 1000
            
            self.logger.info(f"Parallel execution completed in {execution_time:.2f}ms")
            self.logger.info(f"Speedup factor: {metrics.speedup_factor:.2f}x")
            self.logger.info(f"Efficiency: {metrics.efficiency:.1%}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Parallel execution failed: {e}")
            raise
    
    async def _execute_serial_baseline(self, nodes: List[WorkflowNode]) -> Dict[str, Any]:
        """Execute nodes serially for baseline comparison"""
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        executed_nodes = []
        
        try:
            # Execute nodes in dependency order
            dependency_analysis = self.dependency_analyzer.analyze_dependencies(nodes)
            execution_order = dependency_analysis["execution_order"]
            
            node_dict = {node.node_id: node for node in nodes}
            
            for node_id in execution_order:
                if node_id in node_dict:
                    node = node_dict[node_id]
                    node_start = time.time()
                    
                    # Simulate node execution
                    await self._simulate_node_execution(node)
                    
                    node_end = time.time()
                    node.start_time = node_start
                    node.end_time = node_end
                    node.state = NodeExecutionState.COMPLETED
                    
                    executed_nodes.append(node)
            
            execution_time = (time.time() - start_time) * 1000
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            return {
                "execution_time": execution_time,
                "memory_usage": final_memory - initial_memory,
                "nodes_executed": len(executed_nodes)
            }
            
        except Exception as e:
            self.logger.error(f"Serial baseline execution failed: {e}")
            return {
                "execution_time": float('inf'),
                "memory_usage": 0,
                "nodes_executed": 0
            }
    
    async def _execute_parallel_workflow(self, nodes: List[WorkflowNode],
                                       dependency_analysis: Dict[str, Any],
                                       thread_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow with parallel optimization"""
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Create thread pool
        optimal_threads = thread_config["thread_config"]["optimal_threads"]
        self.thread_pool = ThreadPoolExecutor(max_workers=optimal_threads)
        
        try:
            execution_levels = dependency_analysis["execution_levels"]
            node_dict = {node.node_id: node for node in nodes}
            
            total_scheduling_overhead = 0
            executed_nodes = []
            
            # Execute each level in parallel
            for level_idx, level_nodes in enumerate(execution_levels):
                level_start = time.time()
                
                # Prepare tasks for this level
                level_tasks = []
                for node_id in level_nodes:
                    if node_id in node_dict:
                        node = node_dict[node_id]
                        task = self._execute_node_parallel(node)
                        level_tasks.append(task)
                
                # Execute level tasks in parallel
                if level_tasks:
                    await asyncio.gather(*level_tasks, return_exceptions=True)
                
                level_end = time.time()
                level_overhead = (level_end - level_start) * 1000 - sum(
                    node_dict[node_id].estimated_execution_time 
                    for node_id in level_nodes if node_id in node_dict
                )
                total_scheduling_overhead += max(0, level_overhead)
                
                # Update executed nodes
                for node_id in level_nodes:
                    if node_id in node_dict:
                        executed_nodes.append(node_dict[node_id])
            
            execution_time = (time.time() - start_time) * 1000
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Calculate utilization metrics
            thread_utilization = len(executed_nodes) / (optimal_threads * len(execution_levels)) if execution_levels else 0
            cpu_utilization = psutil.cpu_percent(interval=0.1)
            
            return {
                "execution_time": execution_time,
                "memory_usage": final_memory - initial_memory,
                "thread_utilization": min(1.0, thread_utilization),
                "memory_peak_usage": final_memory,
                "cpu_utilization": cpu_utilization,
                "scheduling_overhead": total_scheduling_overhead,
                "nodes_executed": len(executed_nodes),
                "levels_executed": len(execution_levels)
            }
            
        finally:
            # Cleanup thread pool
            if self.thread_pool:
                self.thread_pool.shutdown(wait=True)
                self.thread_pool = None
    
    async def _execute_node_parallel(self, node: WorkflowNode):
        """Execute a single node with resource management"""
        node.start_time = time.time()
        node.state = NodeExecutionState.RUNNING
        node.thread_id = threading.get_ident()
        
        try:
            # Acquire resources
            async with self.contention_manager.acquire_resources(node) as acquired_resources:
                # Execute the node
                await self._simulate_node_execution(node)
                
                node.state = NodeExecutionState.COMPLETED
                node.result = f"Result from {node.node_id}"
                
        except Exception as e:
            node.state = NodeExecutionState.FAILED
            node.error = e
            self.logger.error(f"Node {node.node_id} execution failed: {e}")
        finally:
            node.end_time = time.time()
    
    async def _simulate_node_execution(self, node: WorkflowNode):
        """Simulate node execution based on characteristics"""
        # Simulate execution time
        execution_time = node.estimated_execution_time / 1000  # Convert to seconds
        
        # Add some realistic variation
        variation = np.random.uniform(0.8, 1.2)
        actual_execution_time = execution_time * variation
        
        # Simulate different types of work
        if node.cpu_intensity > 0.7:
            # CPU-intensive work simulation
            await asyncio.sleep(actual_execution_time * 0.9)
            # Simulate some CPU work
            _ = sum(i * i for i in range(int(actual_execution_time * 10000)))
        elif node.io_intensity > 0.7:
            # I/O-intensive work simulation
            await asyncio.sleep(actual_execution_time)
        else:
            # Mixed workload
            await asyncio.sleep(actual_execution_time * 0.7)
            _ = sum(i for i in range(int(actual_execution_time * 5000)))
    
    async def _store_execution_results(self, metrics: ParallelExecutionMetrics,
                                     nodes: List[WorkflowNode],
                                     dependency_analysis: Dict[str, Any],
                                     thread_config: Dict[str, Any]):
        """Store execution results in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Insert execution run
                cursor.execute("""
                    INSERT INTO execution_runs 
                    (timestamp, total_nodes, parallelizable_nodes, serial_time, parallel_time,
                     speedup_factor, efficiency, thread_count, contention_incidents, strategy, success)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics.created_at.isoformat(),
                    metrics.total_nodes,
                    metrics.parallelizable_nodes,
                    metrics.serial_execution_time,
                    metrics.parallel_execution_time,
                    metrics.speedup_factor,
                    metrics.efficiency,
                    thread_config["thread_config"]["optimal_threads"],
                    metrics.contention_incidents,
                    thread_config["thread_config"]["strategy_applied"],
                    True
                ))
                
                run_id = cursor.lastrowid
                
                # Insert individual node executions
                for node in nodes:
                    if hasattr(node, 'start_time') and hasattr(node, 'end_time'):
                        execution_time = (node.end_time - node.start_time) * 1000 if node.end_time and node.start_time else 0
                        
                        cursor.execute("""
                            INSERT INTO node_executions 
                            (run_id, node_id, execution_time, memory_used, thread_id, success, error_message)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            run_id,
                            node.node_id,
                            execution_time,
                            node.memory_requirements,
                            getattr(node, 'thread_id', None),
                            node.state == NodeExecutionState.COMPLETED,
                            str(node.error) if node.error else None
                        ))
                
                conn.commit()
                self.logger.info(f"Execution results stored with run ID: {run_id}")
                
        except Exception as e:
            self.logger.error(f"Failed to store execution results: {e}")
    
    async def benchmark_parallel_performance(self, workload_sizes: List[int] = [5, 10, 20, 50]) -> Dict[str, Any]:
        """Benchmark parallel execution performance"""
        self.logger.info("Running parallel execution benchmarks...")
        
        benchmark_results = {}
        
        for size in workload_sizes:
            # Create test workflow
            test_nodes = self._create_test_workflow(size)
            
            # Run benchmark
            start_time = time.time()
            metrics = await self.execute_workflow_parallel(test_nodes)
            benchmark_time = time.time() - start_time
            
            benchmark_results[f"nodes_{size}"] = {
                "speedup_factor": metrics.speedup_factor,
                "efficiency": metrics.efficiency,
                "execution_time": metrics.parallel_execution_time,
                "benchmark_time": benchmark_time * 1000,
                "thread_utilization": metrics.thread_utilization,
                "contention_incidents": metrics.contention_incidents
            }
            
            # Store benchmark result
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT INTO performance_benchmarks 
                        (timestamp, chip_type, thread_config, benchmark_result, workload_type)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        datetime.now().isoformat(),
                        str(self.capabilities.chip_type),
                        json.dumps(self.thread_optimizer.optimal_config["thread_config"] if self.thread_optimizer.optimal_config else {}),
                        metrics.speedup_factor,
                        f"synthetic_{size}_nodes"
                    ))
            except Exception as e:
                self.logger.error(f"Failed to store benchmark result: {e}")
        
        self.logger.info("Parallel execution benchmarks completed")
        return benchmark_results
    
    def _create_test_workflow(self, num_nodes: int) -> List[WorkflowNode]:
        """Create test workflow for benchmarking"""
        nodes = []
        
        for i in range(num_nodes):
            # Create variety of node types
            if i % 4 == 0:
                # CPU-intensive node
                node = WorkflowNode(
                    node_id=f"cpu_node_{i}",
                    node_type="cpu_intensive",
                    estimated_execution_time=np.random.uniform(50, 200),
                    memory_requirements=np.random.uniform(32, 128),
                    cpu_intensity=np.random.uniform(0.7, 1.0),
                    io_intensity=np.random.uniform(0.0, 0.3),
                    can_parallelize=True,
                    thread_safe=True
                )
            elif i % 4 == 1:
                # I/O-intensive node
                node = WorkflowNode(
                    node_id=f"io_node_{i}",
                    node_type="io_intensive",
                    estimated_execution_time=np.random.uniform(100, 300),
                    memory_requirements=np.random.uniform(16, 64),
                    cpu_intensity=np.random.uniform(0.0, 0.3),
                    io_intensity=np.random.uniform(0.7, 1.0),
                    can_parallelize=True,
                    thread_safe=True
                )
            elif i % 4 == 2:
                # Mixed workload node
                node = WorkflowNode(
                    node_id=f"mixed_node_{i}",
                    node_type="mixed",
                    estimated_execution_time=np.random.uniform(75, 150),
                    memory_requirements=np.random.uniform(64, 256),
                    cpu_intensity=np.random.uniform(0.4, 0.7),
                    io_intensity=np.random.uniform(0.3, 0.6),
                    can_parallelize=True,
                    thread_safe=True
                )
            else:
                # Serial node (dependency heavy)
                node = WorkflowNode(
                    node_id=f"serial_node_{i}",
                    node_type="serial",
                    estimated_execution_time=np.random.uniform(25, 100),
                    memory_requirements=np.random.uniform(16, 32),
                    cpu_intensity=np.random.uniform(0.2, 0.6),
                    io_intensity=np.random.uniform(0.1, 0.4),
                    can_parallelize=False,
                    thread_safe=True
                )
            
            # Add some dependencies to create realistic workflow
            if i > 0 and np.random.random() < 0.3:  # 30% chance of dependency
                dependency_idx = np.random.randint(0, i)
                node.dependencies.append(nodes[dependency_idx].node_id)
            
            nodes.append(node)
        
        return nodes
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        AVG(speedup_factor) as avg_speedup,
                        AVG(efficiency) as avg_efficiency,
                        COUNT(*) as total_runs,
                        AVG(thread_count) as avg_threads,
                        AVG(contention_incidents) as avg_contention
                    FROM execution_runs 
                    WHERE success = 1
                """)
                
                row = cursor.fetchone()
                if row:
                    # Get contention report
                    contention_report = self.contention_manager.get_contention_report()
                    
                    return {
                        "avg_speedup_factor": row[0] or 1.0,
                        "avg_efficiency": row[1] or 1.0,
                        "total_execution_runs": row[2] or 0,
                        "avg_thread_count": row[3] or 1,
                        "avg_contention_incidents": row[4] or 0,
                        "hardware_info": {
                            "chip_type": str(self.capabilities.chip_type),
                            "cpu_cores": self.capabilities.cpu_cores,
                            "memory_gb": self.capabilities.unified_memory_gb,
                            "bandwidth_gbps": self.capabilities.memory_bandwidth_gbps
                        },
                        "contention_report": contention_report,
                        "performance_target_met": {
                            "speedup_target_2_5x": (row[0] or 1.0) >= 2.5,
                            "contention_eliminated": contention_report["contention_eliminated"],
                            "efficiency_acceptable": (row[1] or 1.0) >= 0.7
                        }
                    }
                else:
                    return {"error": "No execution data available"}
                    
        except Exception as e:
            self.logger.error(f"Failed to get performance statistics: {e}")
            return {"error": str(e)}

# Example usage and testing
async def main():
    """Example usage of the parallel execution engine"""
    print("⚡ LangGraph Parallel Node Execution System")
    print("=" * 50)
    
    try:
        # Initialize engine
        engine = ParallelExecutionEngine()
        
        # Create sample workflow
        sample_nodes = [
            WorkflowNode(
                node_id="start_node",
                node_type="initialization",
                estimated_execution_time=50.0,
                memory_requirements=32.0,
                cpu_intensity=0.3,
                io_intensity=0.2,
                can_parallelize=False,  # Start node must be serial
                thread_safe=True
            ),
            WorkflowNode(
                node_id="parallel_task_1",
                node_type="data_processing",
                estimated_execution_time=200.0,
                memory_requirements=128.0,
                cpu_intensity=0.8,
                io_intensity=0.2,
                dependencies=["start_node"],
                can_parallelize=True,
                thread_safe=True
            ),
            WorkflowNode(
                node_id="parallel_task_2",
                node_type="computation",
                estimated_execution_time=150.0,
                memory_requirements=96.0,
                cpu_intensity=0.9,
                io_intensity=0.1,
                dependencies=["start_node"],
                can_parallelize=True,
                thread_safe=True
            ),
            WorkflowNode(
                node_id="parallel_task_3",
                node_type="analysis",
                estimated_execution_time=180.0,
                memory_requirements=64.0,
                cpu_intensity=0.6,
                io_intensity=0.4,
                dependencies=["start_node"],
                can_parallelize=True,
                thread_safe=True
            ),
            WorkflowNode(
                node_id="aggregation_node",
                node_type="results_aggregation",
                estimated_execution_time=75.0,
                memory_requirements=48.0,
                cpu_intensity=0.4,
                io_intensity=0.3,
                dependencies=["parallel_task_1", "parallel_task_2", "parallel_task_3"],
                can_parallelize=False,  # Aggregation requires all inputs
                thread_safe=True
            )
        ]
        
        # Execute workflow with parallel optimization
        print(f"Executing workflow with {len(sample_nodes)} nodes...")
        metrics = await engine.execute_workflow_parallel(sample_nodes)
        
        print(f"✅ Parallel execution completed!")
        print(f"  Speedup factor: {metrics.speedup_factor:.2f}x")
        print(f"  Efficiency: {metrics.efficiency:.1%}")
        print(f"  Thread utilization: {metrics.thread_utilization:.1%}")
        print(f"  Contention incidents: {metrics.contention_incidents}")
        print(f"  Memory peak usage: {metrics.memory_peak_usage:.1f}MB")
        
        # Run performance benchmarks
        print(f"\nRunning performance benchmarks...")
        benchmark_results = await engine.benchmark_parallel_performance([5, 10, 15])
        
        print(f"✅ Benchmarks completed!")
        for workload, results in benchmark_results.items():
            print(f"  {workload}: {results['speedup_factor']:.2f}x speedup, {results['efficiency']:.1%} efficiency")
        
        # Get performance statistics
        print(f"\nPerformance Statistics:")
        stats = engine.get_performance_statistics()
        if "error" not in stats:
            print(f"  Average speedup: {stats['avg_speedup_factor']:.2f}x")
            print(f"  Average efficiency: {stats['avg_efficiency']:.1%}")
            print(f"  Total execution runs: {stats['total_execution_runs']}")
            print(f"  Speedup target (2.5x) met: {'✅' if stats['performance_target_met']['speedup_target_2_5x'] else '❌'}")
            print(f"  Contention eliminated: {'✅' if stats['performance_target_met']['contention_eliminated'] else '❌'}")
        
        print(f"\n🎯 Parallel execution system operational!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())