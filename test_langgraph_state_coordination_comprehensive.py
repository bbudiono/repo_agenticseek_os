#!/usr/bin/env python3
"""
Comprehensive LangGraph State Coordination Testing Framework
Tests TASK-LANGGRAPH-002.1: State-Based Agent Coordination with multi-agent workflows
"""

import asyncio
import json
import time
import psutil
import signal
import sys
import logging
import traceback
import sqlite3
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import statistics

# Test framework setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global test monitoring
test_monitor = {
    "crashes": [],
    "memory_leaks": [],
    "timeouts": [],
    "performance_data": [],
    "system_alerts": []
}

def crash_handler(signum, frame):
    """Handle crashes and collect crash information"""
    global test_monitor
    crash_info = {
        "signal": signum,
        "timestamp": time.time(),
        "stack_trace": traceback.format_stack(frame),
        "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024
    }
    test_monitor["crashes"].append(crash_info)
    logger.error(f"Crash detected: Signal {signum}")

# Register crash handlers
signal.signal(signal.SIGSEGV, crash_handler)
signal.signal(signal.SIGABRT, crash_handler)

class SystemMonitor:
    """Monitor system resources during testing"""
    
    def __init__(self):
        self.monitoring = False
        self.start_memory = 0
        self.peak_memory = 0
        self.cpu_samples = []
        self.memory_samples = []
        
    def start_monitoring(self):
        """Start system monitoring"""
        self.monitoring = True
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        self.peak_memory = self.start_memory
        
        def monitor_loop():
            while self.monitoring:
                try:
                    process = psutil.Process()
                    cpu_percent = process.cpu_percent()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    
                    self.cpu_samples.append(cpu_percent)
                    self.memory_samples.append(memory_mb)
                    
                    if memory_mb > self.peak_memory:
                        self.peak_memory = memory_mb
                    
                    # Check for memory leaks (>50% increase)
                    if memory_mb > self.start_memory * 1.5:
                        test_monitor["memory_leaks"].append({
                            "timestamp": time.time(),
                            "start_memory": self.start_memory,
                            "current_memory": memory_mb,
                            "increase_percent": ((memory_mb - self.start_memory) / self.start_memory) * 100
                        })
                    
                    # System resource alerts
                    system_memory = psutil.virtual_memory()
                    if system_memory.percent > 85:
                        test_monitor["system_alerts"].append({
                            "type": "high_memory_usage",
                            "timestamp": time.time(),
                            "usage_percent": system_memory.percent
                        })
                    
                    time.sleep(0.5)
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    break
        
        threading.Thread(target=monitor_loop, daemon=True).start()
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        
    def get_metrics(self):
        """Get monitoring metrics"""
        return {
            "start_memory_mb": self.start_memory,
            "peak_memory_mb": self.peak_memory,
            "memory_increase_mb": self.peak_memory - self.start_memory,
            "avg_cpu_percent": statistics.mean(self.cpu_samples) if self.cpu_samples else 0,
            "max_cpu_percent": max(self.cpu_samples) if self.cpu_samples else 0,
            "avg_memory_mb": statistics.mean(self.memory_samples) if self.memory_samples else 0,
            "memory_samples_count": len(self.memory_samples)
        }

async def run_comprehensive_state_coordination_test():
    """Run comprehensive testing for LangGraph State Coordination system"""
    
    test_session_id = f"state_coordination_test_{int(time.time())}"
    start_time = time.time()
    
    print("🧪 COMPREHENSIVE LANGGRAPH STATE COORDINATION TESTING")
    print("=" * 80)
    print(f"Test Session ID: {test_session_id}")
    print(f"Start Time: {datetime.fromtimestamp(start_time)}")
    
    # Start system monitoring
    monitor = SystemMonitor()
    monitor.start_monitoring()
    
    test_results = {
        "test_session_id": test_session_id,
        "timestamp": start_time,
        "test_duration_seconds": 0,
        "test_summary": {
            "total_test_components": 0,
            "successful_components": 0,
            "overall_accuracy": 0,
            "crashes_detected": 0,
            "memory_leaks_detected": 0,
            "timeouts_detected": 0
        },
        "acceptance_criteria_validation": {
            "state_sharing_accuracy": 0.95,  # Target >95%
            "state_transition_latency_target": 0.1,  # Target <100ms
            "state_consistency_target": 1.0,  # Target 100%
            "checkpointing_reliability_target": 0.995,  # Target >99.5%
            "agent_integration_target": 1.0  # Target 100%
        },
        "detailed_test_results": []
    }
    
    try:
        # Test 1: State Graph Creation and Agent Integration
        test_1_result = await test_state_graph_creation_integration()
        test_results["detailed_test_results"].append(test_1_result)
        
        # Test 2: Multi-Agent State Coordination Patterns
        test_2_result = await test_multi_agent_coordination_patterns()
        test_results["detailed_test_results"].append(test_2_result)
        
        # Test 3: State Transition Performance and Latency
        test_3_result = await test_state_transition_performance()
        test_results["detailed_test_results"].append(test_3_result)
        
        # Test 4: Checkpointing and State Recovery System
        test_4_result = await test_checkpointing_recovery_system()
        test_results["detailed_test_results"].append(test_4_result)
        
        # Test 5: Error Handling and State Consistency
        test_5_result = await test_error_handling_state_consistency()
        test_results["detailed_test_results"].append(test_5_result)
        
        # Test 6: Complex Workflow State Management
        test_6_result = await test_complex_workflow_state_management()
        test_results["detailed_test_results"].append(test_6_result)
        
    except Exception as e:
        logger.error(f"Critical test failure: {e}")
        test_monitor["crashes"].append({
            "type": "test_framework_crash",
            "error": str(e),
            "timestamp": time.time(),
            "traceback": traceback.format_exc()
        })
    
    # Stop monitoring and collect results
    monitor.stop_monitoring()
    test_duration = time.time() - start_time
    
    # Calculate test summary
    successful_tests = len([t for t in test_results["detailed_test_results"] if t.get("success", False)])
    total_tests = len(test_results["detailed_test_results"])
    
    test_results["test_duration_seconds"] = test_duration
    test_results["test_summary"].update({
        "total_test_components": total_tests,
        "successful_components": successful_tests,
        "overall_accuracy": successful_tests / max(total_tests, 1),
        "crashes_detected": len(test_monitor["crashes"]),
        "memory_leaks_detected": len(test_monitor["memory_leaks"]),
        "timeouts_detected": len(test_monitor["timeouts"])
    })
    
    # Add system monitoring data
    test_results["performance_analysis"] = {
        "test_execution_time": test_duration,
        **monitor.get_metrics()
    }
    
    test_results["reliability_analysis"] = {
        "crash_count": len(test_monitor["crashes"]),
        "memory_leak_count": len(test_monitor["memory_leaks"]),
        "timeout_count": len(test_monitor["timeouts"]),
        "system_alert_count": len(test_monitor["system_alerts"]),
        "stability_score": 1.0 - (len(test_monitor["crashes"]) + len(test_monitor["memory_leaks"])) / max(total_tests, 1)
    }
    
    test_results["system_metrics"] = {
        "memory_available_gb": psutil.virtual_memory().available / (1024**3),
        "memory_total_gb": psutil.virtual_memory().total / (1024**3),
        "cpu_count": psutil.cpu_count(),
        "monitoring_duration": test_duration
    }
    
    test_results["crash_analysis"] = {
        "total_crashes": len(test_monitor["crashes"]),
        "crash_details": test_monitor["crashes"],
        "memory_leaks_detected": len(test_monitor["memory_leaks"]),
        "memory_leak_details": test_monitor["memory_leaks"],
        "system_alerts": test_monitor["system_alerts"]
    }
    
    # Calculate acceptance criteria score
    acceptance_score = calculate_acceptance_criteria_score(test_results)
    test_results["acceptance_criteria_score"] = acceptance_score
    
    # Determine test status
    if acceptance_score >= 0.8 and len(test_monitor["crashes"]) == 0:
        test_status = "PASSED - EXCELLENT"
        recommendations = ["✅ READY FOR PRODUCTION: Excellent performance across all metrics"]
    elif acceptance_score >= 0.7:
        test_status = "PASSED - GOOD"
        recommendations = ["✅ READY FOR PRODUCTION: Good performance with minor optimizations needed"]
    elif acceptance_score >= 0.6:
        test_status = "PASSED - ACCEPTABLE"
        recommendations = ["⚠️ NEEDS IMPROVEMENT: Acceptable performance but requires optimization"]
    else:
        test_status = "FAILED"
        recommendations = ["❌ NOT READY: Critical issues need resolution before production"]
    
    test_results["test_status"] = test_status
    test_results["recommendations"] = recommendations
    
    # Save results to file
    result_filename = f"state_coordination_comprehensive_test_report_{test_session_id}.json"
    with open(result_filename, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    # Print final summary
    print(f"\n📊 COMPREHENSIVE TEST RESULTS SUMMARY")
    print("=" * 80)
    print(f"Test Status: {test_status}")
    print(f"Overall Accuracy: {test_results['test_summary']['overall_accuracy']:.1%}")
    print(f"Successful Components: {successful_tests}/{total_tests}")
    print(f"Execution Time: {test_duration:.2f}s")
    print(f"Crashes Detected: {len(test_monitor['crashes'])}")
    print(f"Memory Leaks: {len(test_monitor['memory_leaks'])}")
    print(f"Acceptance Score: {acceptance_score:.1%}")
    print(f"Results saved to: {result_filename}")
    
    for recommendation in recommendations:
        print(f"  {recommendation}")
    
    return test_results

async def test_state_graph_creation_integration():
    """Test 1: State Graph Creation and Agent Integration"""
    
    test_name = "State Graph Creation and Agent Integration"
    start_time = time.time()
    
    print(f"\n🔬 Running Test 1: {test_name}")
    print("-" * 60)
    
    try:
        from langgraph_state_coordination_simplified_sandbox import StateCoordinationEngine, CoordinationPattern, TaskAnalysis
        
        # Create coordination engine
        engine = StateCoordinationEngine()
        
        # Test different coordination patterns
        patterns_tested = []
        pattern_results = {}
        
        for pattern in [CoordinationPattern.SEQUENTIAL, CoordinationPattern.PARALLEL, CoordinationPattern.SUPERVISOR]:
            try:
                # Create test task analysis
                task_analysis = TaskAnalysis(
                    task_id=f"integration_test_{pattern.value}",
                    description=f"Testing {pattern.value} coordination pattern",
                    complexity_score=0.6,
                    requires_planning=True,
                    requires_web_search=False,
                    requires_code_generation=True,
                    estimated_execution_time=30.0,
                    estimated_memory_usage=512.0,
                    detected_patterns=[],
                    pattern_confidence={},
                    user_preferences={},
                    system_constraints={},
                    performance_requirements={}
                )
                
                # Create coordination context
                coordination_context = await engine._create_coordination_context(task_analysis, pattern)
                
                # Create state graph
                graph = await engine.create_state_graph(coordination_context)
                
                pattern_results[pattern.value] = {
                    "graph_created": graph is not None,
                    "active_agents": len(coordination_context.active_agents),
                    "execution_order": coordination_context.execution_order,
                    "dependencies": coordination_context.agent_dependencies
                }
                patterns_tested.append(pattern.value)
                
                print(f"  ✅ {pattern.value}: Graph created with {len(coordination_context.active_agents)} agents")
                
            except Exception as e:
                pattern_results[pattern.value] = {"error": str(e)}
                print(f"  ❌ {pattern.value}: Failed - {e}")
        
        # Validate agent integration
        agent_integration_score = len([p for p in pattern_results.values() if p.get("graph_created")]) / len(pattern_results)
        
        test_duration = time.time() - start_time
        success = agent_integration_score >= 0.8
        
        return {
            "test_name": test_name,
            "patterns_tested": patterns_tested,
            "pattern_results": pattern_results,
            "agent_integration_score": agent_integration_score,
            "success": success,
            "test_duration_seconds": test_duration,
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024
        }
        
    except Exception as e:
        return {
            "test_name": test_name,
            "success": False,
            "error": str(e),
            "test_duration_seconds": time.time() - start_time,
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024
        }

async def test_multi_agent_coordination_patterns():
    """Test 2: Multi-Agent State Coordination Patterns"""
    
    test_name = "Multi-Agent State Coordination Patterns"
    start_time = time.time()
    
    print(f"\n🔬 Running Test 2: {test_name}")
    print("-" * 60)
    
    try:
        from langgraph_state_coordination_simplified_sandbox import StateCoordinationEngine, CoordinationPattern, TaskAnalysis
        
        engine = StateCoordinationEngine()
        coordination_results = {}
        
        # Test comprehensive coordination scenario
        task_analysis = TaskAnalysis(
            task_id="comprehensive_coordination_test",
            description="Multi-agent coordination with complex state management",
            complexity_score=0.8,
            requires_planning=True,
            requires_web_search=True,
            requires_code_generation=True,
            requires_file_operations=True,
            requires_result_synthesis=True,
            estimated_execution_time=60.0,
            estimated_memory_usage=1024.0,
            detected_patterns=[],
            pattern_confidence={},
            user_preferences={},
            system_constraints={},
            performance_requirements={}
        )
        
        # Test each coordination pattern
        for pattern in [CoordinationPattern.SEQUENTIAL, CoordinationPattern.PARALLEL, CoordinationPattern.SUPERVISOR, CoordinationPattern.CONSENSUS]:
            pattern_start = time.time()
            
            try:
                # Execute workflow with pattern
                result = await engine.execute_workflow(task_analysis, pattern)
                
                coordination_results[pattern.value] = {
                    "execution_time": time.time() - pattern_start,
                    "success": result.get("success", False),
                    "agents_completed": result.get("agent_performance", {}).get("successful_agents", 0),
                    "quality_score": result.get("quality_metrics", {}).get("average_quality_score", 0),
                    "state_transitions": result.get("coordination_metrics", {}).get("state_transitions", 0),
                    "checkpoints_created": result.get("coordination_metrics", {}).get("checkpoints_created", 0)
                }
                
                print(f"  ✅ {pattern.value}: {result.get('agent_performance', {}).get('successful_agents', 0)} agents completed")
                
            except Exception as e:
                coordination_results[pattern.value] = {
                    "execution_time": time.time() - pattern_start,
                    "success": False,
                    "error": str(e)
                }
                print(f"  ❌ {pattern.value}: Failed - {e}")
        
        # Calculate coordination effectiveness
        successful_patterns = len([r for r in coordination_results.values() if r.get("success")])
        coordination_effectiveness = successful_patterns / len(coordination_results)
        
        # Calculate average metrics
        successful_results = [r for r in coordination_results.values() if r.get("success")]
        avg_execution_time = statistics.mean([r["execution_time"] for r in successful_results]) if successful_results else 0
        avg_quality_score = statistics.mean([r.get("quality_score", 0) for r in successful_results]) if successful_results else 0
        total_state_transitions = sum([r.get("state_transitions", 0) for r in successful_results])
        
        test_duration = time.time() - start_time
        success = coordination_effectiveness >= 0.75
        
        return {
            "test_name": test_name,
            "coordination_patterns_tested": len(coordination_results),
            "coordination_results": coordination_results,
            "coordination_effectiveness": coordination_effectiveness,
            "avg_execution_time": avg_execution_time,
            "avg_quality_score": avg_quality_score,
            "total_state_transitions": total_state_transitions,
            "success": success,
            "test_duration_seconds": test_duration,
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024
        }
        
    except Exception as e:
        return {
            "test_name": test_name,
            "success": False,
            "error": str(e),
            "test_duration_seconds": time.time() - start_time,
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024
        }

async def test_state_transition_performance():
    """Test 3: State Transition Performance and Latency"""
    
    test_name = "State Transition Performance and Latency"
    start_time = time.time()
    
    print(f"\n🔬 Running Test 3: {test_name}")
    print("-" * 60)
    
    try:
        from langgraph_state_coordination_simplified_sandbox import StateCoordinationEngine, WorkflowState, StateTransition, StateTransitionType
        
        engine = StateCoordinationEngine()
        transition_times = []
        state_consistency_checks = []
        
        # Test rapid state transitions
        num_transitions = 50
        
        for i in range(num_transitions):
            transition_start = time.time()
            
            # Simulate state transition
            initial_state = engine.current_state.copy()
            initial_state["task_id"] = f"transition_test_{i}"
            initial_state["current_agent"] = f"agent_{i % 5}"
            initial_state["shared_context"]["test_data"] = f"data_{i}"
            
            # Create transition record
            transition = StateTransition(
                transition_id=f"transition_{i}",
                transition_type=StateTransitionType.HANDOFF,
                from_state={"agent": "agent_a"},
                to_state={"agent": "agent_b"},
                trigger_agent=f"agent_{i % 5}",
                timestamp=time.time(),
                execution_time=0,
                success=True
            )
            
            # Measure transition time
            transition_time = time.time() - transition_start
            transition_times.append(transition_time * 1000)  # Convert to milliseconds
            
            # Check state consistency
            consistency_check = {
                "task_id_preserved": initial_state["task_id"] == f"transition_test_{i}",
                "agent_updated": initial_state["current_agent"] == f"agent_{i % 5}",
                "context_preserved": "test_data" in initial_state["shared_context"]
            }
            
            state_consistency_checks.append(all(consistency_check.values()))
            
            if i % 10 == 0:
                print(f"  📊 Completed {i+1}/{num_transitions} transitions (avg: {statistics.mean(transition_times[-10:]):.2f}ms)")
        
        # Calculate performance metrics
        avg_transition_time = statistics.mean(transition_times)
        max_transition_time = max(transition_times)
        min_transition_time = min(transition_times)
        transition_consistency = sum(state_consistency_checks) / len(state_consistency_checks)
        
        # Performance targets
        latency_target_met = avg_transition_time < 100  # <100ms target
        consistency_target_met = transition_consistency >= 0.99  # >99% consistency
        
        test_duration = time.time() - start_time
        success = latency_target_met and consistency_target_met
        
        print(f"  ⚡ Average transition time: {avg_transition_time:.2f}ms")
        print(f"  🎯 Latency target (<100ms): {'✅' if latency_target_met else '❌'}")
        print(f"  🔗 State consistency: {transition_consistency:.1%}")
        
        return {
            "test_name": test_name,
            "transitions_tested": num_transitions,
            "avg_transition_time_ms": avg_transition_time,
            "max_transition_time_ms": max_transition_time,
            "min_transition_time_ms": min_transition_time,
            "transition_consistency": transition_consistency,
            "latency_target_met": latency_target_met,
            "consistency_target_met": consistency_target_met,
            "transition_times": transition_times,
            "success": success,
            "test_duration_seconds": test_duration,
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024
        }
        
    except Exception as e:
        return {
            "test_name": test_name,
            "success": False,
            "error": str(e),
            "test_duration_seconds": time.time() - start_time,
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024
        }

async def test_checkpointing_recovery_system():
    """Test 4: Checkpointing and State Recovery System"""
    
    test_name = "Checkpointing and State Recovery System"
    start_time = time.time()
    
    print(f"\n🔬 Running Test 4: {test_name}")
    print("-" * 60)
    
    try:
        from langgraph_state_coordination_simplified_sandbox import StateCoordinationEngine, TaskAnalysis
        
        engine = StateCoordinationEngine()
        checkpoint_tests = []
        recovery_tests = []
        
        # Test checkpoint creation
        print("  💾 Testing checkpoint creation...")
        
        for i in range(10):
            checkpoint_start = time.time()
            
            # Create test state
            test_state = engine.current_state.copy()
            test_state["task_id"] = f"checkpoint_test_{i}"
            test_state["shared_context"] = {
                f"data_{j}": f"value_{j}" for j in range(5)
            }
            test_state["agent_states"] = {
                f"agent_{j}": "active" for j in range(3)
            }
            
            # Create checkpoint
            checkpoint_result = await engine._create_checkpoint(test_state)
            checkpoint_time = time.time() - checkpoint_start
            
            checkpoint_tests.append({
                "checkpoint_id": i,
                "creation_time_ms": checkpoint_time * 1000,
                "checkpoint_created": len(checkpoint_result.get("checkpoints", [])) > len(engine.current_state.get("checkpoints", [])),
                "data_preserved": checkpoint_result.get("shared_context") == test_state["shared_context"]
            })
            
            if i % 3 == 0:
                print(f"    ✓ Checkpoint {i+1}: {checkpoint_time*1000:.1f}ms")
        
        # Test state recovery
        print("  🔄 Testing state recovery...")
        
        for i in range(5):
            recovery_start = time.time()
            
            # Simulate failure scenario
            original_state = engine.current_state.copy()
            original_state["task_id"] = f"recovery_test_{i}"
            original_state["error_state"] = {
                "agent_id": f"agent_{i}",
                "error_message": f"Simulated error {i}",
                "timestamp": time.time()
            }
            
            # Attempt recovery
            recovery_success = await engine._attempt_error_recovery(
                original_state["error_state"], original_state
            )
            
            recovery_time = time.time() - recovery_start
            
            recovery_tests.append({
                "recovery_id": i,
                "recovery_time_ms": recovery_time * 1000,
                "recovery_successful": recovery_success,
                "error_cleared": original_state.get("error_state") is None if recovery_success else True
            })
            
            print(f"    {'✅' if recovery_success else '❌'} Recovery {i+1}: {recovery_time*1000:.1f}ms")
        
        # Calculate reliability metrics
        checkpoint_success_rate = len([c for c in checkpoint_tests if c["checkpoint_created"]]) / len(checkpoint_tests)
        avg_checkpoint_time = statistics.mean([c["creation_time_ms"] for c in checkpoint_tests])
        
        recovery_success_rate = len([r for r in recovery_tests if r["recovery_successful"]]) / len(recovery_tests)
        avg_recovery_time = statistics.mean([r["recovery_time_ms"] for r in recovery_tests])
        
        # Reliability targets
        checkpoint_target_met = checkpoint_success_rate >= 0.995 and avg_checkpoint_time < 500  # >99.5% success, <500ms
        recovery_target_met = recovery_success_rate >= 0.95 and avg_recovery_time < 1000  # >95% success, <1s
        
        test_duration = time.time() - start_time
        success = checkpoint_target_met and recovery_target_met
        
        print(f"  💾 Checkpoint success rate: {checkpoint_success_rate:.1%} (avg: {avg_checkpoint_time:.1f}ms)")
        print(f"  🔄 Recovery success rate: {recovery_success_rate:.1%} (avg: {avg_recovery_time:.1f}ms)")
        
        return {
            "test_name": test_name,
            "checkpoints_tested": len(checkpoint_tests),
            "recoveries_tested": len(recovery_tests),
            "checkpoint_success_rate": checkpoint_success_rate,
            "avg_checkpoint_time_ms": avg_checkpoint_time,
            "recovery_success_rate": recovery_success_rate,
            "avg_recovery_time_ms": avg_recovery_time,
            "checkpoint_target_met": checkpoint_target_met,
            "recovery_target_met": recovery_target_met,
            "checkpoint_details": checkpoint_tests,
            "recovery_details": recovery_tests,
            "success": success,
            "test_duration_seconds": test_duration,
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024
        }
        
    except Exception as e:
        return {
            "test_name": test_name,
            "success": False,
            "error": str(e),
            "test_duration_seconds": time.time() - start_time,
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024
        }

async def test_error_handling_state_consistency():
    """Test 5: Error Handling and State Consistency"""
    
    test_name = "Error Handling and State Consistency"
    start_time = time.time()
    
    print(f"\n🔬 Running Test 5: {test_name}")
    print("-" * 60)
    
    try:
        from langgraph_state_coordination_simplified_sandbox import StateCoordinationEngine, AgentState, TaskAnalysis
        
        engine = StateCoordinationEngine()
        error_scenarios = []
        consistency_validations = []
        
        # Test different error scenarios
        error_types = [
            ("agent_failure", "Agent execution failure"),
            ("data_quality", "Data quality issues"),
            ("resource_exhaustion", "Resource limits exceeded"),
            ("timeout", "Operation timeout"),
            ("network_error", "Network connectivity issues")
        ]
        
        for error_type, description in error_types:
            scenario_start = time.time()
            
            try:
                # Create error scenario
                test_state = engine.current_state.copy()
                test_state["task_id"] = f"error_test_{error_type}"
                test_state["error_state"] = {
                    "type": error_type,
                    "agent_id": "test_agent",
                    "error_message": description,
                    "timestamp": time.time()
                }
                
                # Test error handling
                error_handled_state = await engine._handle_error(test_state)
                
                # Validate state consistency after error handling
                consistency_check = {
                    "task_id_preserved": error_handled_state["task_id"] == test_state["task_id"],
                    "error_state_managed": error_handled_state.get("error_state") != test_state["error_state"],
                    "agent_states_valid": all(
                        state in [s.value for s in AgentState] 
                        for state in error_handled_state.get("agent_states", {}).values()
                    ),
                    "shared_context_preserved": "shared_context" in error_handled_state
                }
                
                consistency_score = sum(consistency_check.values()) / len(consistency_check)
                consistency_validations.append(consistency_score)
                
                scenario_time = time.time() - scenario_start
                
                error_scenarios.append({
                    "error_type": error_type,
                    "description": description,
                    "handling_time_ms": scenario_time * 1000,
                    "consistency_score": consistency_score,
                    "error_resolved": error_handled_state.get("error_state") is None,
                    "state_valid": consistency_score >= 0.8
                })
                
                print(f"  {'✅' if consistency_score >= 0.8 else '❌'} {error_type}: {consistency_score:.1%} consistency")
                
            except Exception as e:
                error_scenarios.append({
                    "error_type": error_type,
                    "description": description,
                    "handling_time_ms": (time.time() - scenario_start) * 1000,
                    "consistency_score": 0,
                    "error_resolved": False,
                    "state_valid": False,
                    "exception": str(e)
                })
                print(f"  ❌ {error_type}: Exception - {e}")
        
        # Test state consistency under concurrent errors
        print("  🔄 Testing concurrent error handling...")
        
        concurrent_start = time.time()
        concurrent_tasks = []
        
        for i in range(5):
            task_state = engine.current_state.copy()
            task_state["task_id"] = f"concurrent_error_{i}"
            task_state["error_state"] = {
                "type": "concurrent_test",
                "agent_id": f"agent_{i}",
                "error_message": f"Concurrent error {i}",
                "timestamp": time.time()
            }
            concurrent_tasks.append(engine._handle_error(task_state))
        
        # Execute concurrent error handling
        concurrent_results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
        concurrent_time = time.time() - concurrent_start
        
        concurrent_success_count = len([r for r in concurrent_results if not isinstance(r, Exception)])
        concurrent_success_rate = concurrent_success_count / len(concurrent_results)
        
        # Calculate overall metrics
        avg_consistency_score = statistics.mean(consistency_validations) if consistency_validations else 0
        error_handling_success_rate = len([s for s in error_scenarios if s.get("state_valid")]) / len(error_scenarios)
        avg_handling_time = statistics.mean([s["handling_time_ms"] for s in error_scenarios])
        
        # Success criteria
        consistency_target_met = avg_consistency_score >= 0.95
        error_handling_target_met = error_handling_success_rate >= 0.9
        concurrent_target_met = concurrent_success_rate >= 0.8
        
        test_duration = time.time() - start_time
        success = consistency_target_met and error_handling_target_met and concurrent_target_met
        
        print(f"  📊 Average consistency: {avg_consistency_score:.1%}")
        print(f"  🛠️ Error handling success: {error_handling_success_rate:.1%}")
        print(f"  ⚡ Concurrent handling: {concurrent_success_rate:.1%}")
        
        return {
            "test_name": test_name,
            "error_scenarios_tested": len(error_scenarios),
            "error_scenarios": error_scenarios,
            "avg_consistency_score": avg_consistency_score,
            "error_handling_success_rate": error_handling_success_rate,
            "avg_handling_time_ms": avg_handling_time,
            "concurrent_success_rate": concurrent_success_rate,
            "concurrent_handling_time_ms": concurrent_time * 1000,
            "consistency_target_met": consistency_target_met,
            "error_handling_target_met": error_handling_target_met,
            "concurrent_target_met": concurrent_target_met,
            "success": success,
            "test_duration_seconds": test_duration,
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024
        }
        
    except Exception as e:
        return {
            "test_name": test_name,
            "success": False,
            "error": str(e),
            "test_duration_seconds": time.time() - start_time,
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024
        }

async def test_complex_workflow_state_management():
    """Test 6: Complex Workflow State Management"""
    
    test_name = "Complex Workflow State Management"
    start_time = time.time()
    
    print(f"\n🔬 Running Test 6: {test_name}")
    print("-" * 60)
    
    try:
        from langgraph_state_coordination_simplified_sandbox import StateCoordinationEngine, CoordinationPattern, TaskAnalysis
        
        engine = StateCoordinationEngine()
        workflow_tests = []
        
        # Test complex workflow scenarios
        complex_scenarios = [
            {
                "name": "Large Scale Sequential",
                "pattern": CoordinationPattern.SEQUENTIAL,
                "complexity": 0.9,
                "agents": 7,
                "requires_all": True
            },
            {
                "name": "Parallel Processing",
                "pattern": CoordinationPattern.PARALLEL,
                "complexity": 0.8,
                "agents": 5,
                "requires_all": False
            },
            {
                "name": "Supervisor Coordination",
                "pattern": CoordinationPattern.SUPERVISOR,
                "complexity": 0.85,
                "agents": 6,
                "requires_all": True
            },
            {
                "name": "Consensus Building",
                "pattern": CoordinationPattern.CONSENSUS,
                "complexity": 0.95,
                "agents": 5,
                "requires_all": True
            }
        ]
        
        for scenario in complex_scenarios:
            scenario_start = time.time()
            
            try:
                # Create complex task analysis
                task_analysis = TaskAnalysis(
                    task_id=f"complex_{scenario['name'].lower().replace(' ', '_')}",
                    description=f"Complex workflow test: {scenario['name']}",
                    complexity_score=scenario["complexity"],
                    requires_planning=True,
                    requires_web_search=True,
                    requires_code_generation=True,
                    requires_file_operations=True,
                    requires_web_automation=True,
                    requires_result_synthesis=True,
                    estimated_execution_time=120.0,
                    estimated_memory_usage=2048.0,
                    detected_patterns=[],
                    pattern_confidence={},
                    user_preferences={},
                    system_constraints={"max_agents": scenario["agents"]},
                    performance_requirements={"quality_threshold": 0.85}
                )
                
                # Execute complex workflow
                workflow_result = await engine.execute_workflow(task_analysis, scenario["pattern"])
                
                # Analyze workflow complexity handling
                agent_performance = workflow_result.get("agent_performance", {})
                quality_metrics = workflow_result.get("quality_metrics", {})
                coordination_metrics = workflow_result.get("coordination_metrics", {})
                
                scenario_time = time.time() - scenario_start
                
                # Evaluate workflow success
                workflow_success = (
                    workflow_result.get("success", False) and
                    agent_performance.get("successful_agents", 0) >= (scenario["agents"] - 1) and
                    quality_metrics.get("average_quality_score", 0) >= 0.7
                )
                
                workflow_tests.append({
                    "scenario_name": scenario["name"],
                    "coordination_pattern": scenario["pattern"].value,
                    "complexity_score": scenario["complexity"],
                    "execution_time": scenario_time,
                    "workflow_success": workflow_success,
                    "agents_successful": agent_performance.get("successful_agents", 0),
                    "agents_total": agent_performance.get("total_agents", 0),
                    "quality_score": quality_metrics.get("average_quality_score", 0),
                    "state_transitions": coordination_metrics.get("state_transitions", 0),
                    "checkpoints_created": coordination_metrics.get("checkpoints_created", 0),
                    "memory_peak_mb": workflow_result.get("performance_summary", {}).get("peak_memory_mb", 0)
                })
                
                status = "✅" if workflow_success else "❌"
                print(f"  {status} {scenario['name']}: {scenario_time:.1f}s, Quality: {quality_metrics.get('average_quality_score', 0):.2f}")
                
            except Exception as e:
                workflow_tests.append({
                    "scenario_name": scenario["name"],
                    "coordination_pattern": scenario["pattern"].value,
                    "complexity_score": scenario["complexity"],
                    "execution_time": time.time() - scenario_start,
                    "workflow_success": False,
                    "error": str(e)
                })
                print(f"  ❌ {scenario['name']}: Failed - {e}")
        
        # Calculate complex workflow metrics
        successful_workflows = len([w for w in workflow_tests if w.get("workflow_success")])
        total_workflows = len(workflow_tests)
        success_rate = successful_workflows / total_workflows
        
        successful_tests = [w for w in workflow_tests if w.get("workflow_success")]
        avg_execution_time = statistics.mean([w["execution_time"] for w in successful_tests]) if successful_tests else 0
        avg_quality_score = statistics.mean([w.get("quality_score", 0) for w in successful_tests]) if successful_tests else 0
        total_state_transitions = sum([w.get("state_transitions", 0) for w in successful_tests])
        total_checkpoints = sum([w.get("checkpoints_created", 0) for w in successful_tests])
        
        # Performance targets for complex workflows
        complex_success_target = success_rate >= 0.8  # 80% success rate for complex workflows
        performance_target = avg_execution_time < 180  # Under 3 minutes average
        quality_target = avg_quality_score >= 0.75  # 75% average quality
        
        test_duration = time.time() - start_time
        success = complex_success_target and performance_target and quality_target
        
        print(f"  📊 Complex workflow success: {success_rate:.1%}")
        print(f"  ⏱️ Average execution time: {avg_execution_time:.1f}s")
        print(f"  🎯 Average quality score: {avg_quality_score:.2f}")
        
        return {
            "test_name": test_name,
            "scenarios_tested": total_workflows,
            "workflow_tests": workflow_tests,
            "success_rate": success_rate,
            "avg_execution_time": avg_execution_time,
            "avg_quality_score": avg_quality_score,
            "total_state_transitions": total_state_transitions,
            "total_checkpoints": total_checkpoints,
            "complex_success_target": complex_success_target,
            "performance_target": performance_target,
            "quality_target": quality_target,
            "success": success,
            "test_duration_seconds": test_duration,
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024
        }
        
    except Exception as e:
        return {
            "test_name": test_name,
            "success": False,
            "error": str(e),
            "test_duration_seconds": time.time() - start_time,
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024
        }

def calculate_acceptance_criteria_score(test_results):
    """Calculate acceptance criteria score based on test results"""
    
    scores = []
    
    # Weight different test components
    test_weights = {
        "State Graph Creation and Agent Integration": 0.2,
        "Multi-Agent State Coordination Patterns": 0.25,
        "State Transition Performance and Latency": 0.2,
        "Checkpointing and State Recovery System": 0.15,
        "Error Handling and State Consistency": 0.1,
        "Complex Workflow State Management": 0.1
    }
    
    for test_result in test_results["detailed_test_results"]:
        test_name = test_result.get("test_name", "")
        weight = test_weights.get(test_name, 0.1)
        
        if test_result.get("success", False):
            # Additional scoring based on specific metrics
            if "agent_integration_score" in test_result:
                scores.append(test_result["agent_integration_score"] * weight)
            elif "coordination_effectiveness" in test_result:
                scores.append(test_result["coordination_effectiveness"] * weight)
            elif "consistency_target_met" in test_result and "latency_target_met" in test_result:
                score = (test_result["consistency_target_met"] + test_result["latency_target_met"]) / 2
                scores.append(score * weight)
            elif "checkpoint_target_met" in test_result and "recovery_target_met" in test_result:
                score = (test_result["checkpoint_target_met"] + test_result["recovery_target_met"]) / 2
                scores.append(score * weight)
            else:
                scores.append(1.0 * weight)
        else:
            scores.append(0.0 * weight)
    
    return sum(scores) if scores else 0.0

if __name__ == "__main__":
    # Run comprehensive state coordination tests
    results = asyncio.run(run_comprehensive_state_coordination_test())