#!/usr/bin/env python3
"""
Comprehensive Task Analysis and Routing System Testing Suite
Testing TASK-LANGGRAPH-001.2: Task Analysis and Routing System
Following Sandbox TDD methodology with comprehensive headless testing and crash analysis
"""

import asyncio
import json
import time
import logging
import traceback
import sqlite3
import os
import sys
import psutil
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from dataclasses import asdict
import statistics
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import gc
import signal

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langgraph_task_analysis_routing_sandbox import (
    IntelligentTaskRouter, RoutingStrategy, TaskPriority, RoutingDecision,
    TaskMetrics, ResourceType, AnalysisStatus, FrameworkType
)

# Configure logging for crash detection
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('task_routing_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SystemMonitor:
    """Real-time system resource monitoring for crash detection"""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = []
        self.alerts = []
        self.thresholds = {
            "cpu_percent": 90,
            "memory_percent": 85,
            "disk_io_percent": 80
        }
    
    def start_monitoring(self):
        """Start continuous system monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1)
        logger.info("System monitoring stopped")
    
    def _monitor_loop(self):
        """Continuous monitoring loop"""
        while self.monitoring:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                metric = {
                    "timestamp": time.time(),
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_percent": (disk.used / disk.total) * 100,
                    "process_count": len(psutil.pids())
                }
                
                self.metrics.append(metric)
                
                # Check thresholds and generate alerts
                if cpu_percent > self.thresholds["cpu_percent"]:
                    self.alerts.append({
                        "timestamp": time.time(),
                        "type": "HIGH_CPU",
                        "value": cpu_percent,
                        "threshold": self.thresholds["cpu_percent"]
                    })
                
                if memory.percent > self.thresholds["memory_percent"]:
                    self.alerts.append({
                        "timestamp": time.time(),
                        "type": "HIGH_MEMORY",
                        "value": memory.percent,
                        "threshold": self.thresholds["memory_percent"]
                    })
                
                # Keep only recent metrics (last 1000 points)
                if len(self.metrics) > 1000:
                    self.metrics = self.metrics[-1000:]
                
                time.sleep(0.5)  # Monitor every 500ms
                
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                time.sleep(1)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        if not self.metrics:
            return {}
        
        recent_metrics = self.metrics[-10:]  # Last 10 readings
        
        return {
            "avg_cpu_percent": np.mean([m["cpu_percent"] for m in recent_metrics]),
            "max_cpu_percent": max([m["cpu_percent"] for m in recent_metrics]),
            "avg_memory_percent": np.mean([m["memory_percent"] for m in recent_metrics]),
            "max_memory_percent": max([m["memory_percent"] for m in recent_metrics]),
            "memory_available_gb": recent_metrics[-1]["memory_available_gb"],
            "alert_count": len(self.alerts),
            "monitoring_duration": len(self.metrics) * 0.5  # seconds
        }

class CrashDetector:
    """Advanced crash detection and analysis system"""
    
    def __init__(self):
        self.crash_logs = []
        self.exception_counts = {}
        self.memory_leaks = []
        self.timeout_tracking = []
        self.signal_handlers_installed = False
        
    def install_signal_handlers(self):
        """Install signal handlers for crash detection"""
        if not self.signal_handlers_installed:
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
            if hasattr(signal, 'SIGUSR1'):
                signal.signal(signal.SIGUSR1, self._signal_handler)
            self.signal_handlers_installed = True
            logger.info("Signal handlers installed for crash detection")
    
    def _signal_handler(self, signum, frame):
        """Handle system signals"""
        crash_info = {
            "timestamp": time.time(),
            "type": "SIGNAL_RECEIVED",
            "signal": signum,
            "frame_info": str(frame),
            "stack_trace": traceback.format_stack()
        }
        self.crash_logs.append(crash_info)
        logger.warning(f"Signal {signum} received - potential crash scenario")
    
    def log_exception(self, exception: Exception, context: str = ""):
        """Log exception with detailed context"""
        exception_type = type(exception).__name__
        
        crash_info = {
            "timestamp": time.time(),
            "type": "EXCEPTION",
            "exception_type": exception_type,
            "exception_message": str(exception),
            "context": context,
            "stack_trace": traceback.format_exc(),
            "memory_usage": self._get_memory_usage()
        }
        
        self.crash_logs.append(crash_info)
        
        # Track exception frequency
        self.exception_counts[exception_type] = self.exception_counts.get(exception_type, 0) + 1
        
        logger.error(f"Exception logged: {exception_type} in {context}")
    
    def detect_memory_leak(self, operation_name: str, initial_memory: float, final_memory: float):
        """Detect potential memory leaks"""
        memory_increase = final_memory - initial_memory
        
        if memory_increase > 100:  # More than 100MB increase
            leak_info = {
                "timestamp": time.time(),
                "operation": operation_name,
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory,
                "memory_increase_mb": memory_increase,
                "gc_count": len(gc.get_objects())
            }
            
            self.memory_leaks.append(leak_info)
            logger.warning(f"Potential memory leak detected in {operation_name}: +{memory_increase:.1f}MB")
    
    def log_timeout(self, operation_name: str, timeout_seconds: float, actual_duration: float):
        """Log operation timeouts"""
        timeout_info = {
            "timestamp": time.time(),
            "operation": operation_name,
            "timeout_threshold": timeout_seconds,
            "actual_duration": actual_duration,
            "exceeded_by": actual_duration - timeout_seconds
        }
        
        self.timeout_tracking.append(timeout_info)
        logger.warning(f"Timeout detected in {operation_name}: {actual_duration:.1f}s > {timeout_seconds:.1f}s")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    def get_crash_summary(self) -> Dict[str, Any]:
        """Get comprehensive crash analysis summary"""
        return {
            "total_crashes": len(self.crash_logs),
            "exception_types": dict(self.exception_counts),
            "memory_leaks_detected": len(self.memory_leaks),
            "timeouts_detected": len(self.timeout_tracking),
            "most_common_exception": max(self.exception_counts.items(), key=lambda x: x[1])[0] if self.exception_counts else None,
            "total_memory_leaked_mb": sum(leak["memory_increase_mb"] for leak in self.memory_leaks),
            "crash_logs": self.crash_logs[-10:]  # Last 10 crashes
        }

class ComprehensiveTaskRoutingTest:
    """Comprehensive testing suite for Task Analysis and Routing System"""
    
    def __init__(self):
        self.test_session_id = f"routing_test_{int(time.time())}"
        self.start_time = time.time()
        self.system_monitor = SystemMonitor()
        self.crash_detector = CrashDetector()
        self.test_results = []
        self.performance_metrics = []
        self.stress_test_results = []
        self.edge_case_results = []
        
        # Install crash detection
        self.crash_detector.install_signal_handlers()
        
        # Enhanced test scenarios covering all complexity levels and strategies
        self.test_scenarios = [
            # Simple Tasks
            {
                "description": "Extract contact information from a text file",
                "strategy": RoutingStrategy.SPEED_FIRST,
                "priority": TaskPriority.LOW,
                "expected_framework": FrameworkType.LANGCHAIN,
                "category": "simple_extraction",
                "timeout": 5.0
            },
            {
                "description": "Translate a short paragraph to French",
                "strategy": RoutingStrategy.RESOURCE_EFFICIENT,
                "priority": TaskPriority.LOW,
                "expected_framework": FrameworkType.LANGCHAIN,
                "category": "simple_language",
                "timeout": 5.0
            },
            
            # Moderate Tasks
            {
                "description": "Analyze sentiment in customer reviews with categorization and reporting",
                "strategy": RoutingStrategy.BALANCED,
                "priority": TaskPriority.MEDIUM,
                "expected_framework": FrameworkType.LANGCHAIN,
                "category": "moderate_analysis",
                "timeout": 10.0
            },
            {
                "description": "Sequential pipeline for document processing with memory persistence and state tracking",
                "strategy": RoutingStrategy.OPTIMAL,
                "priority": TaskPriority.MEDIUM,
                "expected_framework": FrameworkType.LANGGRAPH,
                "category": "moderate_stateful",
                "timeout": 15.0
            },
            
            # Complex Tasks
            {
                "description": "Multi-agent coordination for real-time financial analysis with parallel processing, state management, and conditional decision making",
                "strategy": RoutingStrategy.QUALITY_FIRST,
                "priority": TaskPriority.HIGH,
                "expected_framework": FrameworkType.LANGGRAPH,
                "category": "complex_multiagent",
                "timeout": 20.0
            },
            {
                "description": "Graph-based knowledge extraction workflow with conditional branching, state transitions, and iterative refinement using specialized agents",
                "strategy": RoutingStrategy.OPTIMAL,
                "priority": TaskPriority.CRITICAL,
                "expected_framework": FrameworkType.LANGGRAPH,
                "category": "complex_graph",
                "timeout": 25.0
            },
            
            # Extreme Complexity
            {
                "description": "Distributed multi-agent system for comprehensive market analysis with real-time coordination, complex state machines, parallel execution, iterative optimization, and cross-agent memory sharing",
                "strategy": RoutingStrategy.QUALITY_FIRST,
                "priority": TaskPriority.EMERGENCY,
                "expected_framework": FrameworkType.LANGGRAPH,
                "category": "extreme_coordination",
                "timeout": 30.0
            },
            
            # Strategy-Specific Tests
            {
                "description": "Resource-intensive data processing requiring efficient resource management",
                "strategy": RoutingStrategy.RESOURCE_EFFICIENT,
                "priority": TaskPriority.MEDIUM,
                "expected_framework": FrameworkType.LANGCHAIN,
                "category": "resource_focused",
                "timeout": 12.0
            },
            {
                "description": "Speed-critical real-time processing with minimal latency requirements",
                "strategy": RoutingStrategy.SPEED_FIRST,
                "priority": TaskPriority.HIGH,
                "expected_framework": FrameworkType.LANGCHAIN,
                "category": "speed_focused",
                "timeout": 8.0
            },
            
            # Edge Cases
            {
                "description": "Complex task with mixed requirements needing balanced approach",
                "strategy": RoutingStrategy.BALANCED,
                "priority": TaskPriority.HIGH,
                "expected_framework": FrameworkType.LANGGRAPH,
                "category": "edge_balanced",
                "timeout": 15.0
            }
        ]
        
        logger.info(f"Initialized comprehensive testing suite with {len(self.test_scenarios)} scenarios")
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Execute comprehensive test suite with crash monitoring"""
        
        logger.info("🧪 Starting Comprehensive Task Analysis and Routing Testing")
        logger.info("=" * 80)
        
        # Start system monitoring
        self.system_monitor.start_monitoring()
        
        try:
            # Test basic functionality
            await self._test_basic_routing_functionality()
            
            # Test routing accuracy
            await self._test_routing_accuracy()
            
            # Test performance under load
            await self._test_performance_load()
            
            # Test edge cases and error handling
            await self._test_edge_cases()
            
            # Test memory management
            await self._test_memory_management()
            
            # Test timeout handling
            await self._test_timeout_handling()
            
            # Test concurrent routing
            await self._test_concurrent_routing()
            
            # Generate comprehensive report
            report = await self._generate_comprehensive_report()
            
            logger.info("✅ Comprehensive testing completed successfully")
            return report
            
        except Exception as e:
            self.crash_detector.log_exception(e, "comprehensive_testing")
            logger.error(f"❌ Comprehensive testing crashed: {e}")
            raise
        finally:
            # Stop monitoring
            self.system_monitor.stop_monitoring()
    
    async def _test_basic_routing_functionality(self):
        """Test basic routing functionality"""
        
        logger.info("🔧 Testing Basic Routing Functionality")
        logger.info("-" * 40)
        
        router = IntelligentTaskRouter()
        
        try:
            for i, scenario in enumerate(self.test_scenarios[:3], 1):  # Test first 3 scenarios
                initial_memory = self.crash_detector._get_memory_usage()
                start_time = time.time()
                
                logger.info(f"📋 Basic Test {i}: {scenario['category']}")
                
                try:
                    # Route the task with timeout monitoring
                    routing_decision = await asyncio.wait_for(
                        router.route_task(
                            scenario['description'],
                            scenario['strategy'],
                            scenario['priority']
                        ),
                        timeout=scenario['timeout']
                    )
                    
                    routing_time = (time.time() - start_time) * 1000
                    final_memory = self.crash_detector._get_memory_usage()
                    
                    # Check for memory leaks
                    self.crash_detector.detect_memory_leak(f"basic_test_{i}", initial_memory, final_memory)
                    
                    # Validate results
                    framework_correct = routing_decision.selected_framework == scenario['expected_framework']
                    
                    result = {
                        "test_id": f"basic_{i}",
                        "scenario": scenario,
                        "routing_decision": asdict(routing_decision),
                        "routing_time_ms": routing_time,
                        "framework_correct": framework_correct,
                        "memory_used_mb": final_memory - initial_memory,
                        "timestamp": time.time()
                    }
                    
                    self.test_results.append(result)
                    
                    logger.info(f"   🎯 Selected: {routing_decision.selected_framework.value}")
                    logger.info(f"   📊 Confidence: {routing_decision.confidence.value}")
                    logger.info(f"   ⚡ Complexity: {routing_decision.complexity_factor * 100:.1f}")
                    logger.info(f"   ⏱️  Time: {routing_time:.1f}ms")
                    logger.info(f"   {'✅ CORRECT' if framework_correct else '❌ INCORRECT'}")
                    
                except asyncio.TimeoutError:
                    self.crash_detector.log_timeout(f"basic_test_{i}", scenario['timeout'], time.time() - start_time)
                    logger.error(f"   ❌ TIMEOUT after {scenario['timeout']}s")
                except Exception as e:
                    self.crash_detector.log_exception(e, f"basic_test_{i}")
                    logger.error(f"   ❌ ERROR: {e}")
                    
        except Exception as e:
            self.crash_detector.log_exception(e, "basic_functionality")
            logger.error(f"❌ Basic functionality test crashed: {e}")
            raise
    
    async def _test_routing_accuracy(self):
        """Test routing accuracy across all scenarios"""
        
        logger.info("🎯 Testing Routing Accuracy")
        logger.info("-" * 40)
        
        router = IntelligentTaskRouter()
        correct_decisions = 0
        total_decisions = len(self.test_scenarios)
        category_accuracy = {}
        
        try:
            for i, scenario in enumerate(self.test_scenarios, 1):
                initial_memory = self.crash_detector._get_memory_usage()
                start_time = time.time()
                
                logger.info(f"📋 Accuracy Test {i}/{total_decisions}: {scenario['category']}")
                
                try:
                    # Route the task
                    routing_decision = await asyncio.wait_for(
                        router.route_task(
                            scenario['description'],
                            scenario['strategy'],
                            scenario['priority']
                        ),
                        timeout=scenario['timeout']
                    )
                    
                    routing_time = (time.time() - start_time) * 1000
                    final_memory = self.crash_detector._get_memory_usage()
                    
                    # Check correctness
                    framework_correct = routing_decision.selected_framework == scenario['expected_framework']
                    if framework_correct:
                        correct_decisions += 1
                    
                    # Track category accuracy
                    category = scenario['category']
                    if category not in category_accuracy:
                        category_accuracy[category] = {"correct": 0, "total": 0}
                    category_accuracy[category]["total"] += 1
                    if framework_correct:
                        category_accuracy[category]["correct"] += 1
                    
                    # Store accuracy data
                    accuracy_data = {
                        "test_id": f"accuracy_{i}",
                        "scenario": scenario,
                        "routing_decision": asdict(routing_decision),
                        "routing_time_ms": routing_time,
                        "framework_correct": framework_correct,
                        "expected_framework": scenario['expected_framework'].value,
                        "selected_framework": routing_decision.selected_framework.value,
                        "complexity_score": routing_decision.complexity_factor * 100,
                        "memory_used_mb": final_memory - initial_memory,
                        "timestamp": time.time()
                    }
                    
                    self.test_results.append(accuracy_data)
                    
                    logger.info(f"   Expected: {scenario['expected_framework'].value}")
                    logger.info(f"   Selected: {routing_decision.selected_framework.value}")
                    logger.info(f"   {'✅ CORRECT' if framework_correct else '❌ INCORRECT'}")
                    
                except asyncio.TimeoutError:
                    self.crash_detector.log_timeout(f"accuracy_test_{i}", scenario['timeout'], time.time() - start_time)
                    logger.error(f"   ❌ TIMEOUT after {scenario['timeout']}s")
                except Exception as e:
                    self.crash_detector.log_exception(e, f"accuracy_test_{i}")
                    logger.error(f"   ❌ ERROR: {e}")
            
            overall_accuracy = correct_decisions / total_decisions
            logger.info(f"🎯 Overall Accuracy: {overall_accuracy:.1%} ({correct_decisions}/{total_decisions})")
            
            # Log category accuracy
            for category, stats in category_accuracy.items():
                cat_accuracy = stats["correct"] / stats["total"]
                logger.info(f"   📊 {category}: {cat_accuracy:.1%} ({stats['correct']}/{stats['total']})")
            
        except Exception as e:
            self.crash_detector.log_exception(e, "routing_accuracy")
            logger.error(f"❌ Routing accuracy test crashed: {e}")
            raise
    
    async def _test_performance_load(self):
        """Test performance under concurrent load"""
        
        logger.info("⚡ Testing Performance Under Load")
        logger.info("-" * 40)
        
        router = IntelligentTaskRouter()
        
        try:
            # Test concurrent routing
            concurrent_tests = 20
            test_scenario = self.test_scenarios[0]  # Use simple scenario for load testing
            
            logger.info(f"🔄 Running {concurrent_tests} concurrent routing decisions")
            
            initial_memory = self.crash_detector._get_memory_usage()
            start_time = time.time()
            
            # Create concurrent tasks
            tasks = [
                router.route_task(
                    f"{test_scenario['description']} - Load Test {i}",
                    test_scenario['strategy'],
                    test_scenario['priority']
                )
                for i in range(concurrent_tests)
            ]
            
            # Execute concurrently with timeout
            try:
                routing_decisions = await asyncio.wait_for(
                    asyncio.gather(*tasks),
                    timeout=30.0  # 30 second timeout for load test
                )
                
                total_time = time.time() - start_time
                final_memory = self.crash_detector._get_memory_usage()
                avg_time_per_decision = (total_time / concurrent_tests) * 1000
                
                # Check for memory leaks
                self.crash_detector.detect_memory_leak("concurrent_load_test", initial_memory, final_memory)
                
                performance_metrics = {
                    "concurrent_decisions": concurrent_tests,
                    "total_time_seconds": total_time,
                    "avg_time_per_decision_ms": avg_time_per_decision,
                    "decisions_per_second": concurrent_tests / total_time,
                    "all_decisions_successful": len(routing_decisions) == concurrent_tests,
                    "memory_used_mb": final_memory - initial_memory,
                    "timestamp": time.time()
                }
                
                self.performance_metrics.append(performance_metrics)
                
                logger.info(f"⚡ Concurrent Performance Results:")
                logger.info(f"   Total Time: {total_time:.2f}s")
                logger.info(f"   Avg Time/Decision: {avg_time_per_decision:.1f}ms")
                logger.info(f"   Decisions/Second: {concurrent_tests / total_time:.1f}")
                logger.info(f"   Success Rate: {len(routing_decisions)}/{concurrent_tests}")
                logger.info(f"   Memory Used: {final_memory - initial_memory:.1f}MB")
                
            except asyncio.TimeoutError:
                self.crash_detector.log_timeout("concurrent_load_test", 30.0, time.time() - start_time)
                logger.error(f"   ❌ Load test timed out after 30 seconds")
                
        except Exception as e:
            self.crash_detector.log_exception(e, "performance_load")
            logger.error(f"❌ Performance load test crashed: {e}")
            raise
    
    async def _test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        
        logger.info("🧩 Testing Edge Cases")
        logger.info("-" * 40)
        
        router = IntelligentTaskRouter()
        
        edge_cases = [
            ("", "empty_string"),
            ("a", "single_character"),
            ("A" * 5000, "very_long_string"),
            ("Special chars: @#$%^&*()_+{}[]|\\:;\"'<>?,./", "special_characters"),
            ("Mixed language: Hello 世界 Bonjour мир", "mixed_languages"),
            ("Numbers only: 123456789", "numbers_only"),
            ("Newlines\nand\ttabs\rand\r\nspecial\x00chars", "control_characters")
        ]
        
        try:
            for i, (edge_case, description) in enumerate(edge_cases, 1):
                logger.info(f"🧩 Edge Case {i}: {description}")
                
                initial_memory = self.crash_detector._get_memory_usage()
                
                try:
                    start_time = time.time()
                    routing_decision = await asyncio.wait_for(
                        router.route_task(edge_case, RoutingStrategy.BALANCED, TaskPriority.MEDIUM),
                        timeout=10.0
                    )
                    routing_time = (time.time() - start_time) * 1000
                    final_memory = self.crash_detector._get_memory_usage()
                    
                    edge_case_result = {
                        "test_id": f"edge_case_{i}",
                        "description": description,
                        "input": edge_case[:100],  # Limit input length for logging
                        "routing_decision": asdict(routing_decision),
                        "routing_time_ms": routing_time,
                        "memory_used_mb": final_memory - initial_memory,
                        "success": True,
                        "timestamp": time.time()
                    }
                    
                    self.edge_case_results.append(edge_case_result)
                    
                    logger.info(f"   ✅ Handled successfully")
                    logger.info(f"   🎯 Selected: {routing_decision.selected_framework.value}")
                    logger.info(f"   ⏱️  Time: {routing_time:.1f}ms")
                    
                except asyncio.TimeoutError:
                    self.crash_detector.log_timeout(f"edge_case_{i}", 10.0, time.time() - start_time)
                    logger.error(f"   ❌ Timeout after 10 seconds")
                except Exception as edge_error:
                    self.crash_detector.log_exception(edge_error, f"edge_case_{i}")
                    logger.warning(f"   ⚠️  Edge case failed: {edge_error}")
                    
                    edge_case_result = {
                        "test_id": f"edge_case_{i}",
                        "description": description,
                        "input": edge_case[:100],
                        "error": str(edge_error),
                        "success": False,
                        "timestamp": time.time()
                    }
                    self.edge_case_results.append(edge_case_result)
                    
        except Exception as e:
            self.crash_detector.log_exception(e, "edge_cases")
            logger.error(f"❌ Edge cases test crashed: {e}")
            raise
    
    async def _test_memory_management(self):
        """Test memory management and leak detection"""
        
        logger.info("🧠 Testing Memory Management")
        logger.info("-" * 40)
        
        router = IntelligentTaskRouter()
        
        try:
            # Run multiple routing cycles to detect memory leaks
            cycles = 5
            items_per_cycle = 10
            
            for cycle in range(cycles):
                logger.info(f"🔄 Memory Test Cycle {cycle + 1}/{cycles}")
                
                initial_memory = self.crash_detector._get_memory_usage()
                
                # Process multiple tasks in this cycle
                for item in range(items_per_cycle):
                    try:
                        await router.route_task(
                            f"Memory test task {cycle}_{item} with varying complexity levels",
                            RoutingStrategy.OPTIMAL,
                            TaskPriority.MEDIUM
                        )
                        
                        # Force garbage collection periodically
                        if item % 3 == 0:
                            gc.collect()
                            
                    except Exception as e:
                        self.crash_detector.log_exception(e, f"memory_test_cycle_{cycle}_item_{item}")
                
                final_memory = self.crash_detector._get_memory_usage()
                memory_increase = final_memory - initial_memory
                
                logger.info(f"   📊 Cycle {cycle + 1}: {memory_increase:.1f}MB increase")
                
                # Check for significant memory increase
                if memory_increase > 50:  # More than 50MB per cycle is concerning
                    self.crash_detector.detect_memory_leak(f"memory_cycle_{cycle}", initial_memory, final_memory)
                
                # Brief pause between cycles
                await asyncio.sleep(0.1)
            
            # Final garbage collection and memory check
            gc.collect()
            logger.info("🧠 Memory management testing completed")
            
        except Exception as e:
            self.crash_detector.log_exception(e, "memory_management")
            logger.error(f"❌ Memory management test crashed: {e}")
            raise
    
    async def _test_timeout_handling(self):
        """Test timeout handling and recovery"""
        
        logger.info("⏰ Testing Timeout Handling")
        logger.info("-" * 40)
        
        router = IntelligentTaskRouter()
        
        try:
            # Test various timeout scenarios
            timeout_tests = [
                {"description": "Quick task", "timeout": 5.0, "should_timeout": False},
                {"description": "Very complex task with extremely detailed requirements and multiple coordination needs", "timeout": 0.001, "should_timeout": True},
                {"description": "Medium complexity task", "timeout": 2.0, "should_timeout": False}
            ]
            
            for i, test in enumerate(timeout_tests, 1):
                logger.info(f"⏰ Timeout Test {i}: {test['description'][:50]}...")
                
                start_time = time.time()
                
                try:
                    routing_decision = await asyncio.wait_for(
                        router.route_task(
                            test['description'],
                            RoutingStrategy.OPTIMAL,
                            TaskPriority.MEDIUM
                        ),
                        timeout=test['timeout']
                    )
                    
                    actual_duration = time.time() - start_time
                    
                    if test['should_timeout']:
                        logger.warning(f"   ⚠️  Expected timeout but completed in {actual_duration:.3f}s")
                    else:
                        logger.info(f"   ✅ Completed successfully in {actual_duration:.3f}s")
                        
                except asyncio.TimeoutError:
                    actual_duration = time.time() - start_time
                    
                    if test['should_timeout']:
                        logger.info(f"   ✅ Timed out as expected after {actual_duration:.3f}s")
                    else:
                        logger.error(f"   ❌ Unexpected timeout after {actual_duration:.3f}s")
                        self.crash_detector.log_timeout(f"timeout_test_{i}", test['timeout'], actual_duration)
                        
                except Exception as e:
                    self.crash_detector.log_exception(e, f"timeout_test_{i}")
                    logger.error(f"   ❌ Error during timeout test: {e}")
            
        except Exception as e:
            self.crash_detector.log_exception(e, "timeout_handling")
            logger.error(f"❌ Timeout handling test crashed: {e}")
            raise
    
    async def _test_concurrent_routing(self):
        """Test concurrent routing with different strategies"""
        
        logger.info("🔀 Testing Concurrent Routing")
        logger.info("-" * 40)
        
        router = IntelligentTaskRouter()
        
        try:
            # Create tasks with different strategies running concurrently
            concurrent_tasks = [
                (RoutingStrategy.SPEED_FIRST, "Speed-focused task requiring minimal latency"),
                (RoutingStrategy.QUALITY_FIRST, "Quality-focused complex analysis task"),
                (RoutingStrategy.RESOURCE_EFFICIENT, "Resource-efficient batch processing"),
                (RoutingStrategy.BALANCED, "Balanced approach multi-step workflow"),
                (RoutingStrategy.OPTIMAL, "Optimal strategy for complex coordination")
            ]
            
            logger.info(f"🔀 Running {len(concurrent_tasks)} different strategies concurrently")
            
            initial_memory = self.crash_detector._get_memory_usage()
            start_time = time.time()
            
            # Create and execute concurrent tasks
            tasks = [
                router.route_task(description, strategy, TaskPriority.MEDIUM)
                for strategy, description in concurrent_tasks
            ]
            
            try:
                routing_decisions = await asyncio.wait_for(
                    asyncio.gather(*tasks),
                    timeout=15.0
                )
                
                total_time = time.time() - start_time
                final_memory = self.crash_detector._get_memory_usage()
                
                logger.info(f"🔀 Concurrent Strategy Results:")
                logger.info(f"   Total Time: {total_time:.2f}s")
                logger.info(f"   Success Rate: {len(routing_decisions)}/{len(concurrent_tasks)}")
                logger.info(f"   Memory Used: {final_memory - initial_memory:.1f}MB")
                
                # Log individual results
                for i, (decision, (strategy, description)) in enumerate(zip(routing_decisions, concurrent_tasks)):
                    logger.info(f"   {strategy.value}: {decision.selected_framework.value} "
                               f"({decision.confidence.value})")
                
            except asyncio.TimeoutError:
                self.crash_detector.log_timeout("concurrent_routing", 15.0, time.time() - start_time)
                logger.error(f"   ❌ Concurrent routing timed out after 15 seconds")
                
        except Exception as e:
            self.crash_detector.log_exception(e, "concurrent_routing")
            logger.error(f"❌ Concurrent routing test crashed: {e}")
            raise
    
    async def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report with crash analysis"""
        
        logger.info("📊 Generating Comprehensive Report")
        logger.info("-" * 40)
        
        total_duration = time.time() - self.start_time
        
        # Calculate accuracy metrics
        accuracy_results = [r for r in self.test_results if r.get('framework_correct') is not None]
        overall_accuracy = sum(1 for r in accuracy_results if r['framework_correct']) / len(accuracy_results) if accuracy_results else 0
        
        # Calculate performance metrics
        routing_times = [r['routing_time_ms'] for r in self.test_results if 'routing_time_ms' in r]
        avg_routing_time = statistics.mean(routing_times) if routing_times else 0
        
        # Get system metrics
        system_metrics = self.system_monitor.get_current_metrics()
        
        # Get crash analysis
        crash_summary = self.crash_detector.get_crash_summary()
        
        # Calculate memory usage
        memory_usage = [r.get('memory_used_mb', 0) for r in self.test_results]
        total_memory_used = sum(memory_usage)
        
        # Edge case success rate
        edge_case_success_rate = sum(1 for r in self.edge_case_results if r.get('success', False)) / len(self.edge_case_results) if self.edge_case_results else 0
        
        report = {
            "test_session_id": self.test_session_id,
            "timestamp": time.time(),
            "test_duration_seconds": total_duration,
            "test_summary": {
                "total_scenarios": len(self.test_scenarios),
                "scenarios_tested": len(accuracy_results),
                "overall_accuracy": overall_accuracy,
                "avg_routing_time_ms": avg_routing_time,
                "edge_cases_tested": len(self.edge_case_results),
                "edge_case_success_rate": edge_case_success_rate,
                "crashes_detected": crash_summary['total_crashes'],
                "memory_leaks_detected": crash_summary['memory_leaks_detected'],
                "timeouts_detected": crash_summary['timeouts_detected']
            },
            "performance_analysis": {
                "routing_latency": {
                    "avg_ms": avg_routing_time,
                    "min_ms": min(routing_times) if routing_times else 0,
                    "max_ms": max(routing_times) if routing_times else 0,
                    "latency_target_met": avg_routing_time < 50.0
                },
                "concurrent_performance": self.performance_metrics,
                "memory_management": {
                    "total_memory_used_mb": total_memory_used,
                    "avg_memory_per_task_mb": total_memory_used / len(self.test_results) if self.test_results else 0,
                    "memory_leaks": crash_summary['total_memory_leaked_mb']
                }
            },
            "reliability_analysis": {
                "crash_count": crash_summary['total_crashes'],
                "exception_types": crash_summary['exception_types'],
                "timeout_count": crash_summary['timeouts_detected'],
                "stability_score": 1.0 - (crash_summary['total_crashes'] / max(len(self.test_results), 1)),
                "edge_case_handling": "excellent" if edge_case_success_rate > 0.9 else "good" if edge_case_success_rate > 0.7 else "needs_improvement"
            },
            "system_metrics": system_metrics,
            "crash_analysis": crash_summary,
            "detailed_results": {
                "test_results": self.test_results,
                "performance_metrics": self.performance_metrics,
                "edge_case_results": self.edge_case_results
            },
            "recommendations": self._generate_recommendations(overall_accuracy, avg_routing_time, crash_summary),
            "test_status": self._determine_test_status(overall_accuracy, crash_summary, edge_case_success_rate)
        }
        
        # Save comprehensive report with proper enum serialization
        report_filename = f"task_routing_comprehensive_test_report_{self.test_session_id}.json"
        
        def serialize_enums(obj):
            """Custom serializer for enum objects"""
            if hasattr(obj, 'value'):
                return obj.value
            if isinstance(obj, dict):
                return {str(k): serialize_enums(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [serialize_enums(item) for item in obj]
            return str(obj)
        
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2, default=serialize_enums)
        
        logger.info(f"📋 Comprehensive Test Report Summary:")
        logger.info(f"   Overall Accuracy: {overall_accuracy:.1%}")
        logger.info(f"   Average Routing Time: {avg_routing_time:.1f}ms")
        logger.info(f"   Crashes Detected: {crash_summary['total_crashes']}")
        logger.info(f"   Memory Leaks: {crash_summary['memory_leaks_detected']}")
        logger.info(f"   Edge Case Success: {edge_case_success_rate:.1%}")
        logger.info(f"   Test Status: {report['test_status']}")
        logger.info(f"   Report saved: {report_filename}")
        
        return report
    
    def _generate_recommendations(self, accuracy: float, avg_time: float, crash_summary: Dict) -> List[str]:
        """Generate recommendations based on comprehensive test results"""
        
        recommendations = []
        
        # Accuracy recommendations
        if accuracy < 0.8:
            recommendations.append(f"🔧 CRITICAL: Accuracy {accuracy:.1%} is below 80% target. Review routing algorithms.")
        elif accuracy < 0.9:
            recommendations.append("📊 Good accuracy but room for improvement in edge cases")
        
        # Performance recommendations
        if avg_time > 20:
            recommendations.append(f"⚡ PERFORMANCE: Average routing time {avg_time:.1f}ms could be optimized")
        elif avg_time > 10:
            recommendations.append("🚀 Good performance but consider caching for repeated patterns")
        
        # Reliability recommendations
        if crash_summary['total_crashes'] > 0:
            recommendations.append("🛡️  STABILITY: Address crash logs and improve error handling")
            if crash_summary['most_common_exception']:
                recommendations.append(f"🔍 Focus on {crash_summary['most_common_exception']} exception handling")
        
        # Memory recommendations
        if crash_summary['memory_leaks_detected'] > 0:
            recommendations.append(f"🧠 MEMORY: {crash_summary['memory_leaks_detected']} memory leaks detected - optimize resource cleanup")
        
        # Success recommendations
        if accuracy >= 0.85 and avg_time <= 10 and crash_summary['total_crashes'] == 0:
            recommendations.append("✅ READY FOR PRODUCTION: Excellent performance across all metrics")
            recommendations.append("🚀 Consider implementing advanced optimization strategies")
        
        return recommendations
    
    def _determine_test_status(self, accuracy: float, crash_summary: Dict, edge_case_success: float) -> str:
        """Determine overall test status"""
        
        if crash_summary['total_crashes'] > 5:
            return "FAILED - MULTIPLE_CRASHES"
        elif accuracy < 0.7:
            return "FAILED - LOW_ACCURACY"
        elif crash_summary['memory_leaks_detected'] > 3:
            return "FAILED - MEMORY_LEAKS"
        elif accuracy >= 0.85 and crash_summary['total_crashes'] == 0 and edge_case_success > 0.8:
            return "PASSED - EXCELLENT"
        elif accuracy >= 0.8 and crash_summary['total_crashes'] <= 1:
            return "PASSED - GOOD"
        else:
            return "NEEDS_IMPROVEMENT"

async def main():
    """Run comprehensive task analysis and routing system testing"""
    
    print("🧪 AgenticSeek Task Analysis and Routing System - Comprehensive Testing Suite")
    print("=" * 80)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Initialize and run comprehensive tests
        test_suite = ComprehensiveTaskRoutingTest()
        report = await test_suite.run_comprehensive_tests()
        
        print("\n" + "=" * 80)
        print("🎯 COMPREHENSIVE TESTING COMPLETED")
        print("=" * 80)
        print(f"📊 Overall Accuracy: {report['test_summary']['overall_accuracy']:.1%}")
        print(f"⚡ Average Routing Time: {report['test_summary']['avg_routing_time_ms']:.1f}ms")
        print(f"🛡️  Crashes Detected: {report['test_summary']['crashes_detected']}")
        print(f"🧠 Memory Leaks: {report['test_summary']['memory_leaks_detected']}")
        print(f"⏰ Timeouts: {report['test_summary']['timeouts_detected']}")
        print(f"🧩 Edge Case Success: {report['test_summary']['edge_case_success_rate']:.1%}")
        print(f"📈 Test Status: {report['test_status']}")
        print()
        
        print("🔥 RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"   {rec}")
        print()
        
        if "PASSED" in report['test_status']:
            print("✅ TASK ANALYSIS AND ROUTING SYSTEM READY FOR PRODUCTION DEPLOYMENT!")
        else:
            print("❌ TASK ANALYSIS AND ROUTING SYSTEM REQUIRES FIXES BEFORE PRODUCTION")
        
        return report
        
    except Exception as e:
        print(f"❌ CRITICAL FAILURE: Comprehensive testing crashed: {e}")
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    # Run comprehensive tests
    results = asyncio.run(main())
    
    # Exit with appropriate code
    if results and "PASSED" in results.get('test_status', ''):
        exit_code = 0
    else:
        exit_code = 1
    
    print(f"\n🏁 Testing completed with exit code: {exit_code}")
    exit(exit_code)