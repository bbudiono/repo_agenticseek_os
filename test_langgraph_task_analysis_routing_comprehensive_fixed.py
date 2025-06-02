#!/usr/bin/env python3
"""
Comprehensive Task Analysis and Routing System Testing Suite - Fixed Version
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

async def run_quick_comprehensive_test():
    """Run a simplified comprehensive test suite"""
    
    print("🧪 AgenticSeek Task Analysis and Routing System - Quick Comprehensive Testing")
    print("=" * 80)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize monitoring
    system_monitor = SystemMonitor()
    crash_detector = CrashDetector()
    crash_detector.install_signal_handlers()
    system_monitor.start_monitoring()
    
    test_results = []
    start_time = time.time()
    
    try:
        # Initialize router
        router = IntelligentTaskRouter()
        
        # Test scenarios
        test_scenarios = [
            {
                "description": "Extract contact information from a text file",
                "strategy": RoutingStrategy.SPEED_FIRST,
                "priority": TaskPriority.LOW,
                "expected_framework": FrameworkType.LANGCHAIN,
                "category": "simple_extraction"
            },
            {
                "description": "Multi-agent coordination for real-time financial analysis with parallel processing, state management, and conditional decision making",
                "strategy": RoutingStrategy.QUALITY_FIRST,
                "priority": TaskPriority.HIGH,
                "expected_framework": FrameworkType.LANGGRAPH,
                "category": "complex_multiagent"
            },
            {
                "description": "Graph-based knowledge extraction workflow with conditional branching, state transitions, and iterative refinement using specialized agents",
                "strategy": RoutingStrategy.OPTIMAL,
                "priority": TaskPriority.CRITICAL,
                "expected_framework": FrameworkType.LANGGRAPH,
                "category": "complex_graph"
            }
        ]
        
        print(f"Testing {len(test_scenarios)} routing scenarios")
        print()
        
        correct_decisions = 0
        routing_times = []
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"📋 Test Scenario {i}: {scenario['category']}")
            print(f"Description: {scenario['description'][:80]}...")
            
            initial_memory = crash_detector._get_memory_usage()
            start_test_time = time.time()
            
            try:
                # Route the task with timeout monitoring
                routing_decision = await asyncio.wait_for(
                    router.route_task(
                        scenario['description'],
                        scenario['strategy'],
                        scenario['priority']
                    ),
                    timeout=15.0
                )
                
                routing_time = (time.time() - start_test_time) * 1000
                final_memory = crash_detector._get_memory_usage()
                
                # Check for memory leaks
                crash_detector.detect_memory_leak(f"test_{i}", initial_memory, final_memory)
                
                # Validate results
                framework_correct = routing_decision.selected_framework == scenario['expected_framework']
                if framework_correct:
                    correct_decisions += 1
                
                routing_times.append(routing_time)
                
                # Convert enum data to strings for safe storage
                result = {
                    "test_id": f"test_{i}",
                    "scenario": {
                        "description": scenario["description"],
                        "strategy": scenario["strategy"].value,
                        "priority": scenario["priority"].value,
                        "expected_framework": scenario["expected_framework"].value,
                        "category": scenario["category"]
                    },
                    "routing_decision": {
                        "selected_framework": routing_decision.selected_framework.value,
                        "confidence": routing_decision.confidence.value,
                        "complexity_factor": routing_decision.complexity_factor,
                        "routing_latency": routing_decision.routing_latency
                    },
                    "routing_time_ms": routing_time,
                    "framework_correct": framework_correct,
                    "memory_used_mb": final_memory - initial_memory,
                    "timestamp": time.time()
                }
                
                test_results.append(result)
                
                print(f"   🎯 Selected: {routing_decision.selected_framework.value}")
                print(f"   📊 Confidence: {routing_decision.confidence.value}")
                print(f"   ⚡ Complexity: {routing_decision.complexity_factor * 100:.1f}")
                print(f"   ⏱️  Time: {routing_time:.1f}ms")
                print(f"   {'✅ CORRECT' if framework_correct else '❌ INCORRECT'}")
                print()
                
            except asyncio.TimeoutError:
                crash_detector.log_timeout(f"test_{i}", 15.0, time.time() - start_test_time)
                print(f"   ❌ TIMEOUT after 15s")
                print()
            except Exception as e:
                crash_detector.log_exception(e, f"test_{i}")
                print(f"   ❌ ERROR: {e}")
                print()
        
        # Calculate results
        total_duration = time.time() - start_time
        overall_accuracy = correct_decisions / len(test_scenarios) if test_scenarios else 0
        avg_routing_time = statistics.mean(routing_times) if routing_times else 0
        
        # Get system metrics
        system_metrics = system_monitor.get_current_metrics()
        crash_summary = crash_detector.get_crash_summary()
        
        # Generate summary report
        report = {
            "test_session_id": f"routing_test_{int(time.time())}",
            "timestamp": time.time(),
            "test_duration_seconds": total_duration,
            "test_summary": {
                "total_scenarios": len(test_scenarios),
                "scenarios_tested": len(test_results),
                "overall_accuracy": overall_accuracy,
                "avg_routing_time_ms": avg_routing_time,
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
                }
            },
            "reliability_analysis": {
                "crash_count": crash_summary['total_crashes'],
                "exception_types": crash_summary['exception_types'],
                "timeout_count": crash_summary['timeouts_detected'],
                "stability_score": 1.0 - (crash_summary['total_crashes'] / max(len(test_results), 1))
            },
            "system_metrics": system_metrics,
            "crash_analysis": crash_summary,
            "detailed_results": test_results
        }
        
        # Determine test status
        if crash_summary['total_crashes'] > 3:
            test_status = "FAILED - MULTIPLE_CRASHES"
        elif overall_accuracy < 0.7:
            test_status = "FAILED - LOW_ACCURACY"
        elif crash_summary['memory_leaks_detected'] > 2:
            test_status = "FAILED - MEMORY_LEAKS"
        elif overall_accuracy >= 0.85 and crash_summary['total_crashes'] == 0:
            test_status = "PASSED - EXCELLENT"
        elif overall_accuracy >= 0.8 and crash_summary['total_crashes'] <= 1:
            test_status = "PASSED - GOOD"
        else:
            test_status = "NEEDS_IMPROVEMENT"
        
        report["test_status"] = test_status
        
        # Generate recommendations
        recommendations = []
        if overall_accuracy < 0.8:
            recommendations.append(f"🔧 CRITICAL: Accuracy {overall_accuracy:.1%} is below 80% target. Review routing algorithms.")
        elif overall_accuracy < 0.9:
            recommendations.append("📊 Good accuracy but room for improvement in edge cases")
        
        if avg_routing_time > 20:
            recommendations.append(f"⚡ PERFORMANCE: Average routing time {avg_routing_time:.1f}ms could be optimized")
        elif avg_routing_time > 10:
            recommendations.append("🚀 Good performance but consider caching for repeated patterns")
        
        if crash_summary['total_crashes'] > 0:
            recommendations.append("🛡️  STABILITY: Address crash logs and improve error handling")
        
        if overall_accuracy >= 0.85 and avg_routing_time <= 10 and crash_summary['total_crashes'] == 0:
            recommendations.append("✅ READY FOR PRODUCTION: Excellent performance across all metrics")
        
        report["recommendations"] = recommendations
        
        # Save report
        report_filename = f"task_routing_comprehensive_test_report_{report['test_session_id']}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print("=" * 80)
        print("🎯 COMPREHENSIVE TESTING COMPLETED")
        print("=" * 80)
        print(f"📊 Overall Accuracy: {overall_accuracy:.1%}")
        print(f"⚡ Average Routing Time: {avg_routing_time:.1f}ms")
        print(f"🛡️  Crashes Detected: {crash_summary['total_crashes']}")
        print(f"🧠 Memory Leaks: {crash_summary['memory_leaks_detected']}")
        print(f"⏰ Timeouts: {crash_summary['timeouts_detected']}")
        print(f"📈 Test Status: {test_status}")
        print()
        
        print("🔥 RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"   {rec}")
        print()
        
        if "PASSED" in test_status:
            print("✅ TASK ANALYSIS AND ROUTING SYSTEM READY FOR PRODUCTION DEPLOYMENT!")
        else:
            print("❌ TASK ANALYSIS AND ROUTING SYSTEM REQUIRES FIXES BEFORE PRODUCTION")
        
        print(f"📋 Report saved: {report_filename}")
        
        return report
        
    except Exception as e:
        crash_detector.log_exception(e, "comprehensive_testing")
        print(f"❌ CRITICAL FAILURE: Comprehensive testing crashed: {e}")
        print(traceback.format_exc())
        return None
    finally:
        # Stop monitoring
        system_monitor.stop_monitoring()

if __name__ == "__main__":
    # Run comprehensive tests
    results = asyncio.run(run_quick_comprehensive_test())
    
    # Exit with appropriate code
    if results and "PASSED" in results.get('test_status', ''):
        exit_code = 0
    else:
        exit_code = 1
    
    print(f"\n🏁 Testing completed with exit code: {exit_code}")
    exit(exit_code)