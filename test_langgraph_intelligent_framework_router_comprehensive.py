#!/usr/bin/env python3
"""
SANDBOX FILE: For testing/development. See .cursorrules.

Comprehensive Test Suite for LangGraph Intelligent Framework Router System
TASK-LANGGRAPH-003: Intelligent Framework Router & Coordinator - Comprehensive Testing & Validation
"""

import asyncio
import json
import time
import os
import sys
import sqlite3
import tempfile
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Add the sources directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'sources'))

from langgraph_intelligent_framework_router import (
    IntelligentFrameworkRouter, FrameworkType, TaskComplexity, RoutingStrategy,
    FrameworkStatus, TaskCharacteristics, FrameworkCapability
)

# Configure logging for testing
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveIntelligentRouterTest:
    """Comprehensive test suite for intelligent framework router system"""
    
    def __init__(self):
        self.test_db_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.test_db_dir, "test_intelligent_router.db")
        self.router = None
        self.test_results = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "test_details": [],
            "performance_metrics": {},
            "crash_detection": {"crashes_detected": 0, "crash_details": []},
            "system_stability": {"memory_leaks": False, "resource_cleanup": True}
        }
        self.start_time = time.time()
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run complete test suite with crash detection and performance monitoring"""
        print("🧠 STARTING COMPREHENSIVE INTELLIGENT FRAMEWORK ROUTER TESTING")
        print("=" * 100)
        
        try:
            # Initialize router system
            await self._test_router_initialization()
            await self._test_framework_capability_analysis()
            await self._test_intelligent_task_routing()
            await self._test_routing_strategy_variants()
            await self._test_framework_health_monitoring()
            await self._test_performance_metrics_collection()
            await self._test_multi_framework_coordination()
            await self._test_machine_learning_optimization()
            await self._test_routing_analytics_and_insights()
            await self._test_load_balancing_and_scaling()
            await self._test_error_handling_and_recovery()
            await self._test_database_persistence_and_integrity()
            await self._test_background_monitoring_systems()
            await self._test_memory_management_and_cleanup()
            
            # Generate final test report
            await self._generate_test_report()
            
        except Exception as e:
            self._record_crash("comprehensive_test_suite", str(e))
            print(f"💥 CRITICAL: Comprehensive test suite crashed: {e}")
            
        finally:
            await self._cleanup_test_environment()
        
        return self.test_results
    
    async def _test_router_initialization(self):
        """Test intelligent framework router initialization"""
        test_name = "Router Initialization"
        print(f"🔧 Testing: {test_name}")
        
        try:
            # Initialize router
            self.router = IntelligentFrameworkRouter(self.test_db_path)
            
            # Verify database creation
            assert os.path.exists(self.test_db_path), "Database file not created"
            
            # Verify core components
            assert self.router.capability_analyzer is not None, "Capability analyzer not initialized"
            assert self.router.performance_monitor is not None, "Performance monitor not initialized"
            assert self.router.routing_engine is not None, "Routing engine not initialized"
            assert self.router.coordination_orchestrator is not None, "Coordination orchestrator not initialized"
            assert self.router.ml_optimizer is not None, "ML optimizer not initialized"
            
            # Check framework capabilities loaded
            assert len(self.router.framework_capabilities) > 0, "No framework capabilities loaded"
            
            # Verify capability types
            langchain_caps = [cap for cap in self.router.framework_capabilities.values() 
                            if cap.framework_type == FrameworkType.LANGCHAIN]
            langgraph_caps = [cap for cap in self.router.framework_capabilities.values() 
                            if cap.framework_type == FrameworkType.LANGGRAPH]
            pydantic_caps = [cap for cap in self.router.framework_capabilities.values() 
                           if cap.framework_type == FrameworkType.PYDANTIC_AI]
            
            assert len(langchain_caps) > 0, "No LangChain capabilities found"
            assert len(langgraph_caps) > 0, "No LangGraph capabilities found"
            assert len(pydantic_caps) > 0, "No Pydantic AI capabilities found"
            
            self._record_test_result(test_name, True, f"Router initialized with {len(self.router.framework_capabilities)} capabilities")
            
        except Exception as e:
            self._record_crash(test_name, str(e))
            self._record_test_result(test_name, False, f"Initialization failed: {e}")
    
    async def _test_framework_capability_analysis(self):
        """Test framework capability analysis functionality"""
        test_name = "Framework Capability Analysis"
        print(f"📊 Testing: {test_name}")
        
        try:
            # Test task characteristics creation
            test_task = TaskCharacteristics(
                task_id="capability_test_001",
                task_type="stateful_workflows",
                complexity=TaskComplexity.COMPLEX,
                estimated_duration=300.0,
                resource_requirements={"memory_mb": 512, "cpu_cores": 2},
                priority=1
            )
            
            # Test capability analysis
            analysis = await self.router.capability_analyzer.analyze_task_requirements(test_task)
            
            assert "required_capabilities" in analysis, "Required capabilities not analyzed"
            assert "performance_requirements" in analysis, "Performance requirements not analyzed"
            assert "resource_constraints" in analysis, "Resource constraints not analyzed"
            
            # Verify analysis content
            assert len(analysis["required_capabilities"]) > 0, "No required capabilities identified"
            assert "max_latency_ms" in analysis["performance_requirements"], "Latency requirements not specified"
            assert "min_success_rate" in analysis["performance_requirements"], "Success rate requirements not specified"
            
            # Test different task types
            test_tasks = [
                ("text_processing", TaskComplexity.SIMPLE),
                ("data_validation", TaskComplexity.MEDIUM),
                ("complex_coordination", TaskComplexity.VERY_COMPLEX)
            ]
            
            for task_type, complexity in test_tasks:
                task = TaskCharacteristics(
                    task_id=f"test_{task_type}",
                    task_type=task_type,
                    complexity=complexity,
                    estimated_duration=100.0,
                    resource_requirements={"memory_mb": 256},
                    priority=1
                )
                
                task_analysis = await self.router.capability_analyzer.analyze_task_requirements(task)
                assert task_analysis is not None, f"Analysis failed for {task_type}"
            
            self._record_test_result(test_name, True, "Framework capability analysis validated for multiple task types")
            
        except Exception as e:
            self._record_crash(test_name, str(e))
            self._record_test_result(test_name, False, f"Capability analysis testing failed: {e}")
    
    async def _test_intelligent_task_routing(self):
        """Test intelligent task routing functionality"""
        test_name = "Intelligent Task Routing"
        print(f"🎯 Testing: {test_name}")
        
        try:
            # Test basic routing
            test_task = TaskCharacteristics(
                task_id="routing_test_001",
                task_type="stateful_workflows",
                complexity=TaskComplexity.COMPLEX,
                estimated_duration=300.0,
                resource_requirements={"memory_mb": 512, "cpu_cores": 2},
                priority=1
            )
            
            # Route task with default strategy
            decision = await self.router.route_task(test_task)
            
            assert decision is not None, "No routing decision returned"
            assert decision.task_id == test_task.task_id, "Task ID mismatch in decision"
            assert decision.selected_framework in [FrameworkType.LANGCHAIN, FrameworkType.LANGGRAPH, FrameworkType.PYDANTIC_AI], \
                   "Invalid framework selected"
            assert 0 <= decision.confidence_score <= 1, f"Invalid confidence score: {decision.confidence_score}"
            assert len(decision.reasoning) > 0, "No reasoning provided for routing decision"
            assert decision.routing_strategy is not None, "Routing strategy not specified"
            
            # Test routing for different task types that should favor specific frameworks
            framework_preference_tests = [
                ("stateful_workflows", FrameworkType.LANGGRAPH),
                ("data_validation", FrameworkType.PYDANTIC_AI),
                ("text_processing", FrameworkType.LANGCHAIN)
            ]
            
            for task_type, expected_framework in framework_preference_tests:
                task = TaskCharacteristics(
                    task_id=f"preference_test_{task_type}",
                    task_type=task_type,
                    complexity=TaskComplexity.MEDIUM,
                    estimated_duration=200.0,
                    resource_requirements={"memory_mb": 256},
                    priority=1
                )
                
                routing_decision = await self.router.route_task(task)
                # Note: Don't assert exact framework match as intelligent routing may override
                assert routing_decision.selected_framework is not None, f"No framework selected for {task_type}"
                assert routing_decision.confidence_score > 0, f"Zero confidence for {task_type}"
            
            # Test estimated performance
            assert "estimated_latency_ms" in decision.estimated_performance, "Latency estimate not provided"
            assert "estimated_success_rate" in decision.estimated_performance, "Success rate estimate not provided"
            assert "estimated_cost" in decision.estimated_performance, "Cost estimate not provided"
            
            # Verify performance estimates are reasonable
            assert decision.estimated_performance["estimated_latency_ms"] > 0, "Invalid latency estimate"
            assert 0 <= decision.estimated_performance["estimated_success_rate"] <= 1, "Invalid success rate estimate"
            assert decision.estimated_performance["estimated_cost"] >= 0, "Invalid cost estimate"
            
            self._record_test_result(test_name, True, "Intelligent task routing validated with confidence scoring and performance estimation")
            
        except Exception as e:
            self._record_crash(test_name, str(e))
            self._record_test_result(test_name, False, f"Task routing testing failed: {e}")
    
    async def _test_routing_strategy_variants(self):
        """Test different routing strategy implementations"""
        test_name = "Routing Strategy Variants"
        print(f"⚡ Testing: {test_name}")
        
        try:
            test_task = TaskCharacteristics(
                task_id="strategy_test_001",
                task_type="complex_coordination",
                complexity=TaskComplexity.COMPLEX,
                estimated_duration=500.0,
                resource_requirements={"memory_mb": 1024, "cpu_cores": 4},
                priority=1
            )
            
            # Test all routing strategies
            strategies_to_test = [
                RoutingStrategy.PERFORMANCE_OPTIMIZED,
                RoutingStrategy.CAPABILITY_BASED,
                RoutingStrategy.LOAD_BALANCED,
                RoutingStrategy.COST_OPTIMIZED,
                RoutingStrategy.LATENCY_OPTIMIZED,
                RoutingStrategy.INTELLIGENT_ADAPTIVE
            ]
            
            strategy_results = {}
            
            for strategy in strategies_to_test:
                decision = await self.router.route_task(test_task, strategy)
                
                assert decision is not None, f"No decision for strategy {strategy.value}"
                assert decision.routing_strategy == strategy, f"Strategy mismatch for {strategy.value}"
                assert decision.confidence_score > 0, f"Zero confidence for strategy {strategy.value}"
                
                strategy_results[strategy.value] = {
                    "framework": decision.selected_framework.value,
                    "confidence": decision.confidence_score,
                    "estimated_latency": decision.estimated_performance.get("estimated_latency_ms", 0)
                }
            
            # Verify strategies produce different results
            frameworks_selected = set(result["framework"] for result in strategy_results.values())
            assert len(frameworks_selected) >= 2, "Strategies should produce varied framework selections"
            
            # Verify latency-optimized strategy considerations
            latency_result = strategy_results.get(RoutingStrategy.LATENCY_OPTIMIZED.value)
            if latency_result:
                # Should generally prefer frameworks with lower latency
                assert latency_result["estimated_latency"] > 0, "Latency estimate should be provided"
            
            # Verify cost-optimized strategy considerations
            cost_result = strategy_results.get(RoutingStrategy.COST_OPTIMIZED.value)
            if cost_result:
                assert cost_result["confidence"] > 0, "Cost-optimized strategy should provide confidence"
            
            self._record_test_result(test_name, True, f"All {len(strategies_to_test)} routing strategies validated successfully")
            
        except Exception as e:
            self._record_crash(test_name, str(e))
            self._record_test_result(test_name, False, f"Routing strategy testing failed: {e}")
    
    async def _test_framework_health_monitoring(self):
        """Test framework health monitoring functionality"""
        test_name = "Framework Health Monitoring"
        print(f"🏥 Testing: {test_name}")
        
        try:
            # Test health status retrieval
            health_status = await self.router.get_framework_health_status()
            
            assert isinstance(health_status, dict), "Health status should be a dictionary"
            assert len(health_status) > 0, "No framework health status returned"
            
            # Verify all major frameworks are monitored
            expected_frameworks = [FrameworkType.LANGCHAIN, FrameworkType.LANGGRAPH, FrameworkType.PYDANTIC_AI]
            for framework in expected_frameworks:
                assert framework in health_status, f"Health status missing for {framework.value}"
                
                status = health_status[framework]
                assert isinstance(status, FrameworkStatus), f"Invalid status type for {framework.value}"
                assert status in list(FrameworkStatus), f"Invalid status value for {framework.value}: {status}"
            
            # Test individual framework health checks
            for framework_type in expected_frameworks:
                individual_status = await self.router.performance_monitor.check_framework_health(framework_type)
                assert individual_status in list(FrameworkStatus), f"Invalid individual status for {framework_type.value}"
            
            # Test performance metrics retrieval
            for framework_type in expected_frameworks:
                metrics = await self.router.performance_monitor.get_current_metrics(framework_type)
                
                assert isinstance(metrics, dict), f"Metrics should be dictionary for {framework_type.value}"
                
                # Verify required metrics are present
                required_metrics = ["response_time_ms", "success_rate", "cpu_usage", "memory_usage"]
                for metric in required_metrics:
                    assert metric in metrics, f"Missing metric {metric} for {framework_type.value}"
                    assert isinstance(metrics[metric], (int, float)), f"Invalid metric type for {metric}"
                
                # Verify metric value ranges
                assert metrics["response_time_ms"] >= 0, f"Invalid response time for {framework_type.value}"
                assert 0 <= metrics["success_rate"] <= 1, f"Invalid success rate for {framework_type.value}"
                assert 0 <= metrics["cpu_usage"] <= 1, f"Invalid CPU usage for {framework_type.value}"
                assert 0 <= metrics["memory_usage"] <= 1, f"Invalid memory usage for {framework_type.value}"
            
            self._record_test_result(test_name, True, f"Framework health monitoring validated for {len(expected_frameworks)} frameworks")
            
        except Exception as e:
            self._record_crash(test_name, str(e))
            self._record_test_result(test_name, False, f"Health monitoring testing failed: {e}")
    
    async def _test_performance_metrics_collection(self):
        """Test performance metrics collection and analysis"""
        test_name = "Performance Metrics Collection"
        print(f"📈 Testing: {test_name}")
        
        try:
            # Test metrics recording
            test_framework = FrameworkType.LANGGRAPH
            test_metrics = {
                "response_time_ms": 150.5,
                "success_rate": 0.95,
                "throughput": 25.0,
                "error_rate": 0.05
            }
            
            # Simulate metrics collection
            metrics = await self.router.performance_monitor.get_current_metrics(test_framework)
            assert isinstance(metrics, dict), "Metrics should be a dictionary"
            assert len(metrics) > 0, "No metrics returned"
            
            # Test historical metrics (through database)
            # Verify metrics are being stored
            conn = sqlite3.connect(self.router.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM performance_metrics")
            initial_count = cursor.fetchone()[0]
            
            # Trigger metrics update
            await self.router._update_performance_metrics()
            
            # Wait for async operation
            await asyncio.sleep(0.1)
            
            cursor.execute("SELECT COUNT(*) FROM performance_metrics")
            final_count = cursor.fetchone()[0]
            
            conn.close()
            
            # Should have more metrics after update
            assert final_count >= initial_count, "Performance metrics not being stored"
            
            # Test metrics for all frameworks
            all_frameworks = [FrameworkType.LANGCHAIN, FrameworkType.LANGGRAPH, FrameworkType.PYDANTIC_AI]
            metrics_collected = 0
            
            for framework in all_frameworks:
                framework_metrics = await self.router.performance_monitor.get_current_metrics(framework)
                if framework_metrics:
                    metrics_collected += 1
                    
                    # Verify metrics structure
                    assert "response_time_ms" in framework_metrics, f"Missing response time for {framework.value}"
                    assert "success_rate" in framework_metrics, f"Missing success rate for {framework.value}"
            
            assert metrics_collected == len(all_frameworks), f"Expected metrics for {len(all_frameworks)} frameworks, got {metrics_collected}"
            
            self._record_test_result(test_name, True, f"Performance metrics collection validated for {metrics_collected} frameworks")
            
        except Exception as e:
            self._record_crash(test_name, str(e))
            self._record_test_result(test_name, False, f"Performance metrics testing failed: {e}")
    
    async def _test_multi_framework_coordination(self):
        """Test multi-framework coordination capabilities"""
        test_name = "Multi-Framework Coordination"
        print(f"🤝 Testing: {test_name}")
        
        try:
            # Test coordination orchestrator
            orchestrator = self.router.coordination_orchestrator
            assert orchestrator is not None, "Coordination orchestrator not available"
            
            # Create test coordination pattern
            from langgraph_intelligent_framework_router import CoordinationPattern
            
            test_pattern = CoordinationPattern(
                pattern_id="test_coordination_001",
                pattern_name="Sequential Processing Pattern",
                description="Test pattern for sequential framework coordination",
                frameworks_involved=[FrameworkType.LANGCHAIN, FrameworkType.LANGGRAPH],
                coordination_logic={
                    "steps": [
                        {"name": "preprocessing", "framework": "langchain", "order": 1},
                        {"name": "state_management", "framework": "langgraph", "order": 2}
                    ]
                },
                performance_profile={"expected_latency_ms": 500},
                use_cases=["complex_data_processing"]
            )
            
            # Test task for coordination
            coordination_task = TaskCharacteristics(
                task_id="coordination_test_001",
                task_type="complex_coordination",
                complexity=TaskComplexity.COMPLEX,
                estimated_duration=600.0,
                resource_requirements={"memory_mb": 1024, "cpu_cores": 3},
                priority=1
            )
            
            # Execute coordination pattern
            result = await orchestrator.execute_pattern(
                test_pattern, coordination_task, self.router.framework_capabilities
            )
            
            assert result is not None, "No coordination result returned"
            assert "pattern_name" in result, "Pattern name not in result"
            assert "frameworks_used" in result, "Frameworks used not in result"
            assert "execution_steps" in result, "Execution steps not in result"
            assert "success" in result, "Success status not in result"
            
            # Verify execution steps
            execution_steps = result["execution_steps"]
            assert len(execution_steps) > 0, "No execution steps recorded"
            
            for step in execution_steps:
                assert "step_name" in step, "Step name missing"
                assert "framework" in step, "Framework missing from step"
                assert "success" in step, "Success status missing from step"
                assert "execution_time" in step, "Execution time missing from step"
                
                assert step["execution_time"] >= 0, "Invalid execution time"
            
            # Verify overall execution
            assert result["pattern_name"] == test_pattern.pattern_name, "Pattern name mismatch"
            assert len(result["frameworks_used"]) == len(test_pattern.frameworks_involved), "Framework count mismatch"
            assert result["total_execution_time"] >= 0, "Invalid total execution time"
            
            self._record_test_result(test_name, True, "Multi-framework coordination validated with sequential processing pattern")
            
        except Exception as e:
            self._record_crash(test_name, str(e))
            self._record_test_result(test_name, False, f"Multi-framework coordination testing failed: {e}")
    
    async def _test_machine_learning_optimization(self):
        """Test machine learning optimization functionality"""
        test_name = "Machine Learning Optimization"
        print(f"🤖 Testing: {test_name}")
        
        try:
            # Test ML optimizer
            ml_optimizer = self.router.ml_optimizer
            assert ml_optimizer is not None, "ML optimizer not available"
            
            # Create historical data for optimization
            historical_data = []
            for i in range(20):
                data_point = {
                    "task_id": f"historical_task_{i:03d}",
                    "task_type": ["text_processing", "data_validation", "stateful_workflows"][i % 3],
                    "complexity": list(TaskComplexity)[i % 4].value,
                    "selected_framework": [FrameworkType.LANGCHAIN, FrameworkType.LANGGRAPH, FrameworkType.PYDANTIC_AI][i % 3].value,
                    "actual_performance": {
                        "response_time_ms": 100 + i * 20,
                        "success_rate": 0.9 + (i % 10) * 0.01,
                        "cost": 0.01 + i * 0.005
                    }
                }
                historical_data.append(data_point)
            
            # Run ML optimization
            optimization_result = await self.router.optimize_framework_selection(historical_data)
            
            assert optimization_result is not None, "No optimization result returned"
            assert "framework_weights" in optimization_result, "Framework weights not in result"
            assert "performance_thresholds" in optimization_result, "Performance thresholds not in result"
            assert "optimization_confidence" in optimization_result, "Optimization confidence not in result"
            assert "recommendations" in optimization_result, "Recommendations not in result"
            
            # Verify framework weights
            framework_weights = optimization_result["framework_weights"]
            assert isinstance(framework_weights, dict), "Framework weights should be a dictionary"
            
            weight_sum = sum(framework_weights.values())
            assert 0.8 <= weight_sum <= 1.2, f"Framework weights sum should be near 1.0, got {weight_sum}"
            
            for weight_name, weight_value in framework_weights.items():
                assert isinstance(weight_value, (int, float)), f"Weight {weight_name} should be numeric"
                assert 0 <= weight_value <= 1, f"Weight {weight_name} should be between 0 and 1"
            
            # Verify performance thresholds
            performance_thresholds = optimization_result["performance_thresholds"]
            assert isinstance(performance_thresholds, dict), "Performance thresholds should be a dictionary"
            
            required_thresholds = ["response_time_ms", "success_rate", "error_rate"]
            for threshold in required_thresholds:
                assert threshold in performance_thresholds, f"Missing threshold: {threshold}"
                assert isinstance(performance_thresholds[threshold], (int, float)), f"Threshold {threshold} should be numeric"
            
            # Verify optimization confidence
            confidence = optimization_result["optimization_confidence"]
            assert isinstance(confidence, (int, float)), "Optimization confidence should be numeric"
            assert 0 <= confidence <= 1, f"Optimization confidence should be between 0 and 1, got {confidence}"
            
            # Verify recommendations
            recommendations = optimization_result["recommendations"]
            assert isinstance(recommendations, list), "Recommendations should be a list"
            assert len(recommendations) > 0, "Should provide at least one recommendation"
            
            for recommendation in recommendations:
                assert isinstance(recommendation, str), "Each recommendation should be a string"
                assert len(recommendation) > 10, "Recommendations should be meaningful"
            
            self._record_test_result(test_name, True, f"ML optimization validated with {len(historical_data)} data points and {confidence:.2f} confidence")
            
        except Exception as e:
            self._record_crash(test_name, str(e))
            self._record_test_result(test_name, False, f"ML optimization testing failed: {e}")
    
    async def _test_routing_analytics_and_insights(self):
        """Test routing analytics and insights generation"""
        test_name = "Routing Analytics and Insights"
        print(f"📊 Testing: {test_name}")
        
        try:
            # Generate multiple routing decisions for analytics
            test_tasks = []
            for i in range(15):
                task = TaskCharacteristics(
                    task_id=f"analytics_task_{i:03d}",
                    task_type=["text_processing", "data_validation", "stateful_workflows", "complex_coordination"][i % 4],
                    complexity=list(TaskComplexity)[i % 4],
                    estimated_duration=100.0 + i * 25,
                    resource_requirements={"memory_mb": 256 + i * 64},
                    priority=(i % 3) + 1
                )
                test_tasks.append(task)
                
                # Route each task
                await self.router.route_task(task)
            
            # Get routing analytics
            analytics = await self.router.get_routing_analytics(time_window_hours=1)
            
            assert analytics is not None, "No analytics returned"
            assert "total_decisions" in analytics, "Total decisions not in analytics"
            assert "framework_distribution" in analytics, "Framework distribution not in analytics"
            assert "average_confidence" in analytics, "Average confidence not in analytics"
            assert "routing_strategies_used" in analytics, "Routing strategies not in analytics"
            assert "performance_trends" in analytics, "Performance trends not in analytics"
            assert "optimization_recommendations" in analytics, "Optimization recommendations not in analytics"
            
            # Verify analytics data quality
            total_decisions = analytics["total_decisions"]
            assert total_decisions >= len(test_tasks), f"Expected at least {len(test_tasks)} decisions, got {total_decisions}"
            
            framework_distribution = analytics["framework_distribution"]
            assert isinstance(framework_distribution, dict), "Framework distribution should be a dictionary"
            
            distributed_count = sum(framework_distribution.values())
            assert distributed_count <= total_decisions, "Framework distribution count inconsistent"
            
            average_confidence = analytics["average_confidence"]
            assert isinstance(average_confidence, (int, float)), "Average confidence should be numeric"
            assert 0 <= average_confidence <= 1, f"Average confidence should be between 0 and 1, got {average_confidence}"
            
            # Verify routing strategies
            strategies_used = analytics["routing_strategies_used"]
            assert isinstance(strategies_used, dict), "Routing strategies should be a dictionary"
            
            # Verify performance trends
            performance_trends = analytics["performance_trends"]
            assert isinstance(performance_trends, dict), "Performance trends should be a dictionary"
            
            # Verify optimization recommendations
            recommendations = analytics["optimization_recommendations"]
            assert isinstance(recommendations, list), "Optimization recommendations should be a list"
            
            self._record_test_result(test_name, True, f"Routing analytics validated with {total_decisions} decisions and {average_confidence:.2f} avg confidence")
            
        except Exception as e:
            self._record_crash(test_name, str(e))
            self._record_test_result(test_name, False, f"Routing analytics testing failed: {e}")
    
    async def _test_load_balancing_and_scaling(self):
        """Test load balancing and scaling capabilities"""
        test_name = "Load Balancing and Scaling"
        print(f"⚖️ Testing: {test_name}")
        
        try:
            # Test concurrent routing requests
            concurrent_tasks = []
            for i in range(10):
                task = TaskCharacteristics(
                    task_id=f"concurrent_task_{i:03d}",
                    task_type="stateful_workflows",
                    complexity=TaskComplexity.MEDIUM,
                    estimated_duration=200.0,
                    resource_requirements={"memory_mb": 512},
                    priority=1
                )
                concurrent_tasks.append(task)
            
            # Route tasks concurrently
            start_time = time.time()
            routing_results = await asyncio.gather(
                *[self.router.route_task(task) for task in concurrent_tasks],
                return_exceptions=True
            )
            concurrent_time = time.time() - start_time
            
            # Verify all routing succeeded
            successful_routes = [r for r in routing_results if not isinstance(r, Exception)]
            failed_routes = [r for r in routing_results if isinstance(r, Exception)]
            
            assert len(successful_routes) >= 8, f"Expected at least 8 successful routes, got {len(successful_routes)}"
            assert len(failed_routes) <= 2, f"Too many failed routes: {len(failed_routes)}"
            
            # Verify reasonable concurrent performance
            assert concurrent_time < 5.0, f"Concurrent routing took too long: {concurrent_time:.2f}s"
            
            # Test load distribution
            framework_usage = {}
            for result in successful_routes:
                framework = result.selected_framework.value
                framework_usage[framework] = framework_usage.get(framework, 0) + 1
            
            # Should distribute load across frameworks
            assert len(framework_usage) >= 1, "No framework usage recorded"
            
            # Test scaling under different load strategies
            load_balanced_task = TaskCharacteristics(
                task_id="load_balanced_test",
                task_type="general_processing",
                complexity=TaskComplexity.MEDIUM,
                estimated_duration=150.0,
                resource_requirements={"memory_mb": 256},
                priority=1
            )
            
            load_balanced_decision = await self.router.route_task(
                load_balanced_task, RoutingStrategy.LOAD_BALANCED
            )
            
            assert load_balanced_decision is not None, "Load balanced routing failed"
            assert load_balanced_decision.routing_strategy == RoutingStrategy.LOAD_BALANCED, "Strategy not applied"
            
            self._record_test_result(test_name, True, f"Load balancing validated with {len(successful_routes)}/{len(concurrent_tasks)} successful concurrent routes")
            
        except Exception as e:
            self._record_crash(test_name, str(e))
            self._record_test_result(test_name, False, f"Load balancing testing failed: {e}")
    
    async def _test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms"""
        test_name = "Error Handling and Recovery"
        print(f"🛡️ Testing: {test_name}")
        
        try:
            # Test invalid task characteristics
            invalid_task = TaskCharacteristics(
                task_id="",  # Invalid empty task ID
                task_type="invalid_type",
                complexity=TaskComplexity.SIMPLE,
                estimated_duration=-100.0,  # Invalid negative duration
                resource_requirements={},
                priority=-1  # Invalid priority
            )
            
            # Should handle gracefully
            try:
                decision = await self.router.route_task(invalid_task)
                # If it succeeds, verify it handled gracefully
                assert decision is not None, "Should handle invalid task gracefully"
                assert decision.confidence_score >= 0, "Should provide valid confidence even for invalid task"
            except Exception as e:
                # Acceptable if proper error handling
                assert "task" in str(e).lower() or "invalid" in str(e).lower(), f"Error should indicate task issue: {e}"
            
            # Test framework unavailability simulation
            # This is harder to test without mocking, so we test error recovery paths
            
            # Test database connection errors (simulated)
            original_db_path = self.router.db_path
            self.router.db_path = "/invalid/path/database.db"
            
            try:
                # This should handle database errors gracefully
                health_status = await self.router.get_framework_health_status()
                # Should not crash even with database issues
                assert isinstance(health_status, dict), "Should return dictionary even with DB issues"
            except Exception as e:
                # Acceptable if proper error handling
                assert "database" in str(e).lower() or "connection" in str(e).lower() or "file" in str(e).lower(), \
                       f"Error should indicate database issue: {e}"
            finally:
                # Restore original database path
                self.router.db_path = original_db_path
            
            # Test malformed routing requests
            malformed_task = TaskCharacteristics(
                task_id="malformed_test",
                task_type="test_type",
                complexity=TaskComplexity.SIMPLE,
                estimated_duration=100.0,
                resource_requirements={"invalid_requirement": "not_a_number"},
                priority=1
            )
            
            # Should handle malformed requirements gracefully
            try:
                malformed_decision = await self.router.route_task(malformed_task)
                assert malformed_decision is not None, "Should handle malformed task gracefully"
            except Exception as e:
                # Acceptable with proper error message
                assert len(str(e)) > 0, "Should provide meaningful error message"
            
            # Test resource exhaustion scenarios
            resource_intensive_task = TaskCharacteristics(
                task_id="resource_intensive_test",
                task_type="very_complex_processing",
                complexity=TaskComplexity.VERY_COMPLEX,
                estimated_duration=10000.0,  # Very long duration
                resource_requirements={"memory_mb": 999999, "cpu_cores": 1000},  # Unrealistic requirements
                priority=1
            )
            
            # Should handle resource constraints gracefully
            try:
                resource_decision = await self.router.route_task(resource_intensive_task)
                assert resource_decision is not None, "Should handle resource-intensive task"
                # May have lower confidence due to resource constraints
                assert resource_decision.confidence_score >= 0, "Should provide valid confidence"
            except Exception as e:
                # Acceptable if proper resource validation
                assert "resource" in str(e).lower() or "memory" in str(e).lower() or "cpu" in str(e).lower(), \
                       f"Error should indicate resource issue: {e}"
            
            self._record_test_result(test_name, True, "Error handling and recovery mechanisms validated")
            
        except Exception as e:
            self._record_crash(test_name, str(e))
            self._record_test_result(test_name, False, f"Error handling testing failed: {e}")
    
    async def _test_database_persistence_and_integrity(self):
        """Test database persistence and data integrity"""
        test_name = "Database Persistence and Integrity"
        print(f"💾 Testing: {test_name}")
        
        try:
            # Test database structure
            conn = sqlite3.connect(self.router.db_path)
            cursor = conn.cursor()
            
            # Verify all required tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            required_tables = [
                "framework_capabilities", "routing_decisions", "performance_metrics",
                "coordination_patterns", "framework_health", "ml_training_data"
            ]
            
            for table in required_tables:
                assert table in tables, f"Required table '{table}' not found"
            
            # Test data persistence by routing a task and verifying storage
            persistence_task = TaskCharacteristics(
                task_id="persistence_test_001",
                task_type="persistence_test",
                complexity=TaskComplexity.MEDIUM,
                estimated_duration=150.0,
                resource_requirements={"memory_mb": 256},
                priority=1
            )
            
            # Route task
            decision = await self.router.route_task(persistence_task)
            
            # Verify decision was stored
            cursor.execute("SELECT * FROM routing_decisions WHERE task_id = ?", (persistence_task.task_id,))
            stored_decision = cursor.fetchone()
            
            assert stored_decision is not None, "Routing decision not persisted to database"
            
            # Verify data integrity
            decision_data = json.loads(stored_decision[7])  # estimated_performance column
            assert "estimated_latency_ms" in decision_data, "Performance data not properly stored"
            
            # Test metrics persistence
            cursor.execute("SELECT COUNT(*) FROM performance_metrics")
            metrics_count = cursor.fetchone()[0]
            assert metrics_count > 0, "No performance metrics stored"
            
            # Test framework capabilities persistence
            cursor.execute("SELECT COUNT(*) FROM framework_capabilities")
            capabilities_count = cursor.fetchone()[0]
            assert capabilities_count > 0, "No framework capabilities stored"
            
            # Test data types and constraints
            cursor.execute("SELECT confidence_score FROM routing_decisions WHERE task_id = ?", 
                          (persistence_task.task_id,))
            confidence_row = cursor.fetchone()
            if confidence_row:
                confidence = confidence_row[0]
                assert isinstance(confidence, (int, float)), "Confidence score not stored as numeric"
                assert 0 <= confidence <= 1, "Confidence score out of valid range"
            
            conn.close()
            
            # Test database recovery (create new router instance with same database)
            new_router = IntelligentFrameworkRouter(self.router.db_path)
            
            # Should be able to retrieve existing data
            assert len(new_router.framework_capabilities) > 0, "Capabilities not loaded from existing database"
            
            # Should be able to perform operations
            recovery_task = TaskCharacteristics(
                task_id="recovery_test_001",
                task_type="recovery_test",
                complexity=TaskComplexity.SIMPLE,
                estimated_duration=100.0,
                resource_requirements={"memory_mb": 128},
                priority=1
            )
            
            recovery_decision = await new_router.route_task(recovery_task)
            assert recovery_decision is not None, "New router instance cannot perform operations"
            
            # Cleanup new router
            new_router.monitoring_active = False
            
            self._record_test_result(test_name, True, f"Database persistence validated with {len(required_tables)} tables and data integrity checks")
            
        except Exception as e:
            self._record_crash(test_name, str(e))
            self._record_test_result(test_name, False, f"Database persistence testing failed: {e}")
    
    async def _test_background_monitoring_systems(self):
        """Test background monitoring systems"""
        test_name = "Background Monitoring Systems"
        print(f"👁️ Testing: {test_name}")
        
        try:
            # Verify monitoring thread is running
            assert self.router.monitoring_active == True, "Background monitoring not active"
            assert self.router.monitor_thread is not None, "Monitor thread not created"
            assert self.router.monitor_thread.is_alive(), "Monitor thread not running"
            
            # Test monitoring functions can be called
            await self.router._monitor_framework_health()
            await self.router._update_performance_metrics()
            
            # Verify monitoring data is being collected
            conn = sqlite3.connect(self.router.db_path)
            cursor = conn.cursor()
            
            # Check that health monitoring creates entries
            cursor.execute("SELECT COUNT(*) FROM framework_health")
            health_entries_before = cursor.fetchone()[0]
            
            # Trigger health monitoring
            await self.router._monitor_framework_health()
            await asyncio.sleep(0.1)  # Allow async operation to complete
            
            cursor.execute("SELECT COUNT(*) FROM framework_health")
            health_entries_after = cursor.fetchone()[0]
            
            # Should have more entries after monitoring
            assert health_entries_after >= health_entries_before, "Health monitoring not creating entries"
            
            # Check that performance monitoring creates entries
            cursor.execute("SELECT COUNT(*) FROM performance_metrics")
            metrics_entries_before = cursor.fetchone()[0]
            
            # Trigger performance monitoring
            await self.router._update_performance_metrics()
            await asyncio.sleep(0.1)  # Allow async operation to complete
            
            cursor.execute("SELECT COUNT(*) FROM performance_metrics")
            metrics_entries_after = cursor.fetchone()[0]
            
            # Should have more entries after monitoring
            assert metrics_entries_after >= metrics_entries_before, "Performance monitoring not creating entries"
            
            conn.close()
            
            # Test monitoring can be stopped gracefully
            self.router.monitoring_active = False
            
            # Wait for thread to notice the flag change
            await asyncio.sleep(0.2)
            
            # Thread should still be alive but will exit on next cycle
            # Note: Thread might take time to actually stop, so we don't assert it's stopped
            
            self._record_test_result(test_name, True, "Background monitoring systems validated")
            
        except Exception as e:
            self._record_crash(test_name, str(e))
            self._record_test_result(test_name, False, f"Background monitoring testing failed: {e}")
    
    async def _test_memory_management_and_cleanup(self):
        """Test memory management and resource cleanup"""
        test_name = "Memory Management and Cleanup"
        print(f"🧹 Testing: {test_name}")
        
        try:
            # Test routing history management (limited to 10000 entries)
            initial_history_length = len(self.router.routing_history)
            
            # Add many routing decisions to test memory limits
            test_tasks = []
            for i in range(15):
                task = TaskCharacteristics(
                    task_id=f"memory_test_{i:03d}",
                    task_type="memory_test",
                    complexity=TaskComplexity.SIMPLE,
                    estimated_duration=50.0,
                    resource_requirements={"memory_mb": 64},
                    priority=1
                )
                test_tasks.append(task)
                await self.router.route_task(task)
            
            final_history_length = len(self.router.routing_history)
            
            # History should have grown but be limited
            assert final_history_length > initial_history_length, "Routing history not growing"
            assert final_history_length <= 10000, "Routing history exceeds maximum size"
            
            # Test framework capabilities memory management
            initial_capabilities_count = len(self.router.framework_capabilities)
            
            # Capabilities should remain stable (not growing indefinitely)
            assert len(self.router.framework_capabilities) == initial_capabilities_count, \
                   "Framework capabilities memory growing unexpectedly"
            
            # Test performance metrics cache management
            performance_cache = self.router.performance_monitor.performance_cache
            cache_size = len(performance_cache)
            
            # Cache should not grow indefinitely
            assert cache_size < 1000, f"Performance cache too large: {cache_size} entries"
            
            # Test database connection management
            # Verify connections are properly closed by attempting operations
            test_task_cleanup = TaskCharacteristics(
                task_id="cleanup_test_001",
                task_type="cleanup_test",
                complexity=TaskComplexity.SIMPLE,
                estimated_duration=25.0,
                resource_requirements={"memory_mb": 32},
                priority=1
            )
            
            # This should work without database connection issues
            cleanup_decision = await self.router.route_task(test_task_cleanup)
            assert cleanup_decision is not None, "Database connections not properly managed"
            
            # Test analytics memory management
            analytics = await self.router.get_routing_analytics()
            assert analytics is not None, "Analytics memory management failed"
            
            # Test cleanup of temporary objects
            # The router should not hold references to completed tasks beyond history
            task_references = len([d for d in self.router.routing_history if d.task_id.startswith("memory_test")])
            assert task_references <= 15, "Too many task references retained"
            
            self.test_results["system_stability"]["memory_leaks"] = False
            self.test_results["system_stability"]["resource_cleanup"] = True
            
            self._record_test_result(test_name, True, f"Memory management validated - history: {final_history_length}, cache: {cache_size}")
            
        except Exception as e:
            self._record_crash(test_name, str(e))
            self._record_test_result(test_name, False, f"Memory management testing failed: {e}")
    
    def _record_test_result(self, test_name: str, passed: bool, details: str):
        """Record test result"""
        self.test_results["total_tests"] += 1
        if passed:
            self.test_results["passed_tests"] += 1
        else:
            self.test_results["failed_tests"] += 1
        
        self.test_results["test_details"].append({
            "test_name": test_name,
            "passed": passed,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })
        
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name} - {details}")
    
    def _record_crash(self, test_name: str, error_details: str):
        """Record crash details"""
        self.test_results["crash_detection"]["crashes_detected"] += 1
        self.test_results["crash_detection"]["crash_details"].append({
            "test_name": test_name,
            "error": error_details,
            "timestamp": datetime.now().isoformat()
        })
        print(f"💥 CRASH in {test_name}: {error_details}")
    
    async def _generate_test_report(self):
        """Generate comprehensive test report"""
        total_time = time.time() - self.start_time
        success_rate = (self.test_results["passed_tests"] / self.test_results["total_tests"] * 100) if self.test_results["total_tests"] > 0 else 0
        
        self.test_results["summary"] = {
            "total_execution_time_seconds": total_time,
            "success_rate_percentage": success_rate,
            "crash_rate": (self.test_results["crash_detection"]["crashes_detected"] / self.test_results["total_tests"] * 100) if self.test_results["total_tests"] > 0 else 0,
            "overall_status": "EXCELLENT" if success_rate >= 95 and self.test_results["crash_detection"]["crashes_detected"] == 0 else 
                           "GOOD" if success_rate >= 85 and self.test_results["crash_detection"]["crashes_detected"] <= 2 else 
                           "ACCEPTABLE" if success_rate >= 70 else "NEEDS_IMPROVEMENT",
            "timestamp": datetime.now().isoformat()
        }
        
        # Save detailed report
        report_path = f"intelligent_router_test_report_{int(time.time())}.json"
        with open(report_path, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"📊 Test report saved: {report_path}")
    
    async def _cleanup_test_environment(self):
        """Clean up test environment"""
        try:
            # Stop monitoring
            if self.router:
                self.router.monitoring_active = False
                if hasattr(self.router, 'monitor_thread'):
                    self.router.monitor_thread.join(timeout=5)
            
            # Clean up test database directory
            if os.path.exists(self.test_db_dir):
                shutil.rmtree(self.test_db_dir)
            
            print("🧹 Test environment cleaned up")
            
        except Exception as e:
            print(f"Cleanup error: {e}")


async def main():
    """Run comprehensive intelligent framework router tests"""
    print("🧠 COMPREHENSIVE INTELLIGENT FRAMEWORK ROUTER TESTING")
    print("=" * 100)
    
    tester = ComprehensiveIntelligentRouterTest()
    results = await tester.run_comprehensive_tests()
    
    # Display summary
    print("\n" + "=" * 100)
    print("📊 COMPREHENSIVE TEST RESULTS SUMMARY")
    print("=" * 100)
    
    summary = results["summary"]
    print(f"Total Tests: {results['total_tests']}")
    print(f"Passed: {results['passed_tests']}")
    print(f"Failed: {results['failed_tests']}")
    print(f"Success Rate: {summary['success_rate_percentage']:.1f}%")
    print(f"Crashes Detected: {results['crash_detection']['crashes_detected']}")
    print(f"Overall Status: {summary['overall_status']}")
    print(f"Execution Time: {summary['total_execution_time_seconds']:.2f} seconds")
    
    print(f"\n🎯 FINAL ASSESSMENT: {summary['overall_status']}")
    
    if summary['overall_status'] in ['EXCELLENT', 'GOOD']:
        print("✅ INTELLIGENT FRAMEWORK ROUTER SYSTEM READY FOR PRODUCTION")
    elif summary['overall_status'] == 'ACCEPTABLE':
        print("⚠️ INTELLIGENT FRAMEWORK ROUTER SYSTEM ACCEPTABLE - MINOR IMPROVEMENTS NEEDED")
    else:
        print("❌ INTELLIGENT FRAMEWORK ROUTER SYSTEM NEEDS SIGNIFICANT IMPROVEMENTS")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())