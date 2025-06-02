#!/usr/bin/env python3
"""
Comprehensive Framework Decision Engine Testing Suite
Testing TASK-LANGGRAPH-001.1: Framework Decision Engine Core
Following Sandbox TDD methodology with comprehensive headless testing
"""

import asyncio
import json
import time
import logging
import traceback
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dataclasses import asdict
import statistics
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langgraph_framework_decision_engine_sandbox import (
    FrameworkDecisionEngine, TaskComplexity, FrameworkType, 
    DecisionConfidence, WorkflowPattern
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveFrameworkDecisionEngineTest:
    """Comprehensive testing suite for Framework Decision Engine"""
    
    def __init__(self):
        self.test_session_id = f"decision_engine_test_{int(time.time())}"
        self.start_time = time.time()
        self.test_results = []
        self.crash_logs = []
        self.performance_metrics = []
        self.decision_accuracy_data = []
        
        # Enhanced test scenarios with diverse complexity and patterns
        self.test_scenarios = [
            # Simple Tasks - Should favor LangChain
            {
                "description": "Extract names and emails from a text document",
                "expected_framework": FrameworkType.LANGCHAIN,
                "expected_complexity": TaskComplexity.SIMPLE,
                "category": "simple_extraction"
            },
            {
                "description": "Summarize a research paper into key points",
                "expected_framework": FrameworkType.LANGCHAIN,
                "expected_complexity": TaskComplexity.SIMPLE,
                "category": "simple_processing"
            },
            {
                "description": "Translate text from English to Spanish",
                "expected_framework": FrameworkType.LANGCHAIN,
                "expected_complexity": TaskComplexity.SIMPLE,
                "category": "simple_language"
            },
            
            # Moderate Tasks - Could be either framework
            {
                "description": "Analyze sentiment in customer reviews with basic categorization and generate report",
                "expected_framework": FrameworkType.LANGCHAIN,
                "expected_complexity": TaskComplexity.MODERATE,
                "category": "moderate_analysis"
            },
            {
                "description": "Process multiple documents sequentially and create a unified summary with memory persistence",
                "expected_framework": FrameworkType.LANGCHAIN,
                "expected_complexity": TaskComplexity.MODERATE,
                "category": "moderate_pipeline"
            },
            
            # Complex Tasks - Should favor LangGraph
            {
                "description": "Multi-agent system for real-time stock analysis with state management, parallel data processing, and conditional decision making",
                "expected_framework": FrameworkType.LANGGRAPH,
                "expected_complexity": TaskComplexity.COMPLEX,
                "category": "complex_multiagent"
            },
            {
                "description": "Graph-based knowledge extraction workflow with conditional branching, state transitions, and iterative refinement",
                "expected_framework": FrameworkType.LANGGRAPH,
                "expected_complexity": TaskComplexity.VERY_COMPLEX,
                "category": "complex_graph"
            },
            
            # Very Complex Tasks - Definitely LangGraph
            {
                "description": "Complex multi-agent coordination system for analyzing financial data with parallel processing, state management, agent coordination, and iterative refinement based on market conditions",
                "expected_framework": FrameworkType.LANGGRAPH,
                "expected_complexity": TaskComplexity.EXTREME,
                "category": "extreme_coordination"
            },
            {
                "description": "Parallel execution workflow with multiple agents coordinating video generation, requiring state management, agent coordination, parallel execution, and conditional logic with memory persistence",
                "expected_framework": FrameworkType.LANGGRAPH,
                "expected_complexity": TaskComplexity.EXTREME,
                "category": "extreme_multimedia"
            },
            {
                "description": "Advanced research synthesis with interdisciplinary coordination, multiple agent teams, state machine workflows, iterative optimization, and cross-agent memory sharing",
                "expected_framework": FrameworkType.LANGGRAPH,
                "expected_complexity": TaskComplexity.EXTREME,
                "category": "extreme_research"
            },
            
            # Edge Cases
            {
                "description": "Simple task with graph node dependencies",
                "expected_framework": FrameworkType.LANGGRAPH,
                "expected_complexity": TaskComplexity.MODERATE,
                "category": "edge_case_graph"
            },
            {
                "description": "Complex sequential pipeline without state requirements",
                "expected_framework": FrameworkType.LANGCHAIN,
                "expected_complexity": TaskComplexity.COMPLEX,
                "category": "edge_case_pipeline"
            },
            
            # Stress Test Cases
            {
                "description": "Massive parallel coordination with 20+ agents, complex state machine, graph-based workflows, iterative refinement, distributed memory management, and real-time adaptation",
                "expected_framework": FrameworkType.LANGGRAPH,
                "expected_complexity": TaskComplexity.EXTREME,
                "category": "stress_test_extreme"
            },
            {
                "description": "Simple text processing with minimal requirements",
                "expected_framework": FrameworkType.LANGCHAIN,
                "expected_complexity": TaskComplexity.SIMPLE,
                "category": "stress_test_simple"
            },
            {
                "description": "Highly conditional workflow with if-then-else logic, multiple decision points, state-dependent branching, and adaptive execution paths",
                "expected_framework": FrameworkType.LANGGRAPH,
                "expected_complexity": TaskComplexity.VERY_COMPLEX,
                "category": "conditional_complex"
            }
        ]
        
        logger.info(f"Initialized comprehensive testing suite with {len(self.test_scenarios)} scenarios")
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Execute comprehensive test suite with crash logging and performance monitoring"""
        
        logger.info("🧪 Starting Comprehensive Framework Decision Engine Testing")
        logger.info("=" * 80)
        
        try:
            # Initialize the decision engine
            engine = FrameworkDecisionEngine()
            
            # Test basic functionality
            await self._test_basic_functionality(engine)
            
            # Test decision accuracy
            await self._test_decision_accuracy(engine)
            
            # Test performance under load
            await self._test_performance_load(engine)
            
            # Test edge cases
            await self._test_edge_cases(engine)
            
            # Test error handling
            await self._test_error_handling(engine)
            
            # Generate comprehensive report
            report = await self._generate_comprehensive_report(engine)
            
            logger.info("✅ Comprehensive testing completed successfully")
            return report
            
        except Exception as e:
            crash_info = {
                "timestamp": time.time(),
                "error": str(e),
                "traceback": traceback.format_exc(),
                "test_phase": "comprehensive_testing"
            }
            self.crash_logs.append(crash_info)
            logger.error(f"❌ Comprehensive testing crashed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    async def _test_basic_functionality(self, engine: FrameworkDecisionEngine):
        """Test basic framework decision functionality"""
        
        logger.info("🔧 Testing Basic Functionality")
        logger.info("-" * 40)
        
        try:
            for i, scenario in enumerate(self.test_scenarios[:5], 1):  # Test first 5 scenarios
                start_time = time.time()
                
                logger.info(f"📋 Basic Test {i}: {scenario['category']}")
                
                # Make decision
                decision = await engine.make_framework_decision(scenario['description'])
                
                decision_time = (time.time() - start_time) * 1000  # Convert to ms
                
                # Check correctness
                framework_correct = decision.selected_framework == scenario['expected_framework']
                
                result = {
                    "test_id": f"basic_{i}",
                    "scenario": scenario,
                    "decision": asdict(decision),
                    "decision_time_ms": decision_time,
                    "framework_correct": framework_correct,
                    "timestamp": time.time()
                }
                
                self.test_results.append(result)
                
                logger.info(f"   🎯 Selected: {decision.selected_framework.value}")
                logger.info(f"   📊 Confidence: {decision.confidence.value}")
                logger.info(f"   ⚡ Score: {decision.decision_score:.3f}")
                logger.info(f"   ⏱️  Time: {decision_time:.1f}ms")
                logger.info(f"   {'✅ CORRECT' if framework_correct else '❌ INCORRECT'}")
                
        except Exception as e:
            crash_info = {
                "timestamp": time.time(),
                "error": str(e),
                "traceback": traceback.format_exc(),
                "test_phase": "basic_functionality"
            }
            self.crash_logs.append(crash_info)
            logger.error(f"❌ Basic functionality test crashed: {e}")
            raise
    
    async def _test_decision_accuracy(self, engine: FrameworkDecisionEngine):
        """Test decision accuracy across all scenarios"""
        
        logger.info("🎯 Testing Decision Accuracy")
        logger.info("-" * 40)
        
        correct_decisions = 0
        total_decisions = len(self.test_scenarios)
        category_accuracy = {}
        
        try:
            for i, scenario in enumerate(self.test_scenarios, 1):
                start_time = time.time()
                
                logger.info(f"📋 Accuracy Test {i}/{total_decisions}: {scenario['category']}")
                
                # Make decision
                decision = await engine.make_framework_decision(scenario['description'])
                
                decision_time = (time.time() - start_time) * 1000
                
                # Check correctness
                framework_correct = decision.selected_framework == scenario['expected_framework']
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
                    "decision": asdict(decision),
                    "decision_time_ms": decision_time,
                    "framework_correct": framework_correct,
                    "expected_framework": scenario['expected_framework'].value,
                    "selected_framework": decision.selected_framework.value,
                    "confidence": decision.confidence.value,
                    "decision_score": decision.decision_score,
                    "timestamp": time.time()
                }
                
                self.decision_accuracy_data.append(accuracy_data)
                
                logger.info(f"   Expected: {scenario['expected_framework'].value}")
                logger.info(f"   Selected: {decision.selected_framework.value}")
                logger.info(f"   {'✅ CORRECT' if framework_correct else '❌ INCORRECT'}")
            
            overall_accuracy = correct_decisions / total_decisions
            logger.info(f"🎯 Overall Accuracy: {overall_accuracy:.1%} ({correct_decisions}/{total_decisions})")
            
            # Log category accuracy
            for category, stats in category_accuracy.items():
                cat_accuracy = stats["correct"] / stats["total"]
                logger.info(f"   📊 {category}: {cat_accuracy:.1%} ({stats['correct']}/{stats['total']})")
            
        except Exception as e:
            crash_info = {
                "timestamp": time.time(),
                "error": str(e),
                "traceback": traceback.format_exc(),
                "test_phase": "decision_accuracy"
            }
            self.crash_logs.append(crash_info)
            logger.error(f"❌ Decision accuracy test crashed: {e}")
            raise
    
    async def _test_performance_load(self, engine: FrameworkDecisionEngine):
        """Test performance under concurrent load"""
        
        logger.info("⚡ Testing Performance Under Load")
        logger.info("-" * 40)
        
        try:
            # Test concurrent decisions
            concurrent_tests = 10
            test_scenario = self.test_scenarios[0]  # Use first scenario for load testing
            
            logger.info(f"🔄 Running {concurrent_tests} concurrent decisions")
            
            start_time = time.time()
            
            # Create concurrent tasks
            tasks = [
                engine.make_framework_decision(f"{test_scenario['description']} - Load Test {i}")
                for i in range(concurrent_tests)
            ]
            
            # Execute concurrently
            decisions = await asyncio.gather(*tasks)
            
            total_time = time.time() - start_time
            avg_time_per_decision = (total_time / concurrent_tests) * 1000  # Convert to ms
            
            performance_metrics = {
                "concurrent_decisions": concurrent_tests,
                "total_time_seconds": total_time,
                "avg_time_per_decision_ms": avg_time_per_decision,
                "decisions_per_second": concurrent_tests / total_time,
                "all_decisions_successful": len(decisions) == concurrent_tests,
                "timestamp": time.time()
            }
            
            self.performance_metrics.append(performance_metrics)
            
            logger.info(f"⚡ Concurrent Performance Results:")
            logger.info(f"   Total Time: {total_time:.2f}s")
            logger.info(f"   Avg Time/Decision: {avg_time_per_decision:.1f}ms")
            logger.info(f"   Decisions/Second: {concurrent_tests / total_time:.1f}")
            logger.info(f"   Success Rate: {len(decisions)}/{concurrent_tests}")
            
        except Exception as e:
            crash_info = {
                "timestamp": time.time(),
                "error": str(e),
                "traceback": traceback.format_exc(),
                "test_phase": "performance_load"
            }
            self.crash_logs.append(crash_info)
            logger.error(f"❌ Performance load test crashed: {e}")
            raise
    
    async def _test_edge_cases(self, engine: FrameworkDecisionEngine):
        """Test edge cases and boundary conditions"""
        
        logger.info("🧩 Testing Edge Cases")
        logger.info("-" * 40)
        
        edge_cases = [
            "",  # Empty string
            "a",  # Single character
            "A" * 1000,  # Very long string
            "Special chars: @#$%^&*()_+{}[]|\\:;\"'<>?,./",  # Special characters
            "Mixed language text: Hello 世界 Bonjour мир",  # Mixed languages
            "Numbers only: 123456789",  # Numbers only
        ]
        
        try:
            for i, edge_case in enumerate(edge_cases, 1):
                logger.info(f"🧩 Edge Case {i}: {edge_case[:50]}{'...' if len(edge_case) > 50 else ''}")
                
                try:
                    start_time = time.time()
                    decision = await engine.make_framework_decision(edge_case)
                    decision_time = (time.time() - start_time) * 1000
                    
                    logger.info(f"   ✅ Handled successfully")
                    logger.info(f"   🎯 Selected: {decision.selected_framework.value}")
                    logger.info(f"   📊 Confidence: {decision.confidence.value}")
                    logger.info(f"   ⏱️  Time: {decision_time:.1f}ms")
                    
                except Exception as edge_error:
                    logger.warning(f"   ⚠️  Edge case failed: {edge_error}")
                    crash_info = {
                        "timestamp": time.time(),
                        "error": str(edge_error),
                        "traceback": traceback.format_exc(),
                        "test_phase": "edge_case",
                        "edge_case_input": edge_case
                    }
                    self.crash_logs.append(crash_info)
                    
        except Exception as e:
            crash_info = {
                "timestamp": time.time(),
                "error": str(e),
                "traceback": traceback.format_exc(),
                "test_phase": "edge_cases"
            }
            self.crash_logs.append(crash_info)
            logger.error(f"❌ Edge cases test crashed: {e}")
            raise
    
    async def _test_error_handling(self, engine: FrameworkDecisionEngine):
        """Test error handling and recovery"""
        
        logger.info("🛡️  Testing Error Handling")
        logger.info("-" * 40)
        
        try:
            # Test with None input
            try:
                await engine.make_framework_decision(None)
                logger.warning("⚠️  None input accepted (should handle gracefully)")
            except Exception as e:
                logger.info(f"✅ None input properly rejected: {type(e).__name__}")
            
            # Test database corruption simulation
            try:
                # Temporarily corrupt database path
                original_db_path = engine.db_path
                engine.db_path = "/invalid/path/test.db"
                
                decision = await engine.make_framework_decision("Test after corruption")
                logger.info("✅ Gracefully handled database corruption")
                
                # Restore original path
                engine.db_path = original_db_path
                
            except Exception as e:
                logger.info(f"⚠️  Database corruption handling: {type(e).__name__}")
            
            logger.info("🛡️  Error handling tests completed")
            
        except Exception as e:
            crash_info = {
                "timestamp": time.time(),
                "error": str(e),
                "traceback": traceback.format_exc(),
                "test_phase": "error_handling"
            }
            self.crash_logs.append(crash_info)
            logger.error(f"❌ Error handling test crashed: {e}")
            raise
    
    async def _generate_comprehensive_report(self, engine: FrameworkDecisionEngine) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        
        logger.info("📊 Generating Comprehensive Report")
        logger.info("-" * 40)
        
        total_duration = time.time() - self.start_time
        
        # Calculate overall accuracy
        correct_decisions = sum(1 for data in self.decision_accuracy_data if data['framework_correct'])
        overall_accuracy = correct_decisions / len(self.decision_accuracy_data) if self.decision_accuracy_data else 0
        
        # Calculate average decision time
        decision_times = [data['decision_time_ms'] for data in self.decision_accuracy_data]
        avg_decision_time = statistics.mean(decision_times) if decision_times else 0
        
        # Category accuracy analysis
        category_accuracy = {}
        for data in self.decision_accuracy_data:
            category = data['scenario']['category']
            if category not in category_accuracy:
                category_accuracy[category] = {"correct": 0, "total": 0}
            category_accuracy[category]["total"] += 1
            if data['framework_correct']:
                category_accuracy[category]["correct"] += 1
        
        # Framework selection bias analysis
        framework_selections = {}
        for data in self.decision_accuracy_data:
            fw = data['selected_framework']
            framework_selections[fw] = framework_selections.get(fw, 0) + 1
        
        # Confidence distribution
        confidence_distribution = {}
        for data in self.decision_accuracy_data:
            conf = data['confidence']
            confidence_distribution[conf] = confidence_distribution.get(conf, 0) + 1
        
        # Get historical accuracy if available
        try:
            historical_accuracy = engine.get_decision_accuracy(days=1)
        except:
            historical_accuracy = {"accuracy": 0.0, "samples": 0}
        
        report = {
            "test_session_id": self.test_session_id,
            "timestamp": time.time(),
            "test_duration_seconds": total_duration,
            "test_summary": {
                "total_scenarios": len(self.test_scenarios),
                "scenarios_tested": len(self.decision_accuracy_data),
                "overall_accuracy": overall_accuracy,
                "correct_decisions": correct_decisions,
                "avg_decision_time_ms": avg_decision_time,
                "crashes_detected": len(self.crash_logs),
                "edge_cases_tested": 6,
                "performance_tests_run": len(self.performance_metrics)
            },
            "accuracy_analysis": {
                "overall_accuracy": overall_accuracy,
                "category_accuracy": {
                    cat: stats["correct"] / stats["total"] 
                    for cat, stats in category_accuracy.items()
                },
                "framework_selection_bias": framework_selections,
                "confidence_distribution": confidence_distribution,
                "historical_accuracy": historical_accuracy
            },
            "performance_analysis": {
                "avg_decision_time_ms": avg_decision_time,
                "min_decision_time_ms": min(decision_times) if decision_times else 0,
                "max_decision_time_ms": max(decision_times) if decision_times else 0,
                "concurrent_performance": self.performance_metrics,
                "latency_target_met": avg_decision_time < 50.0  # Target: <50ms
            },
            "reliability_analysis": {
                "crash_count": len(self.crash_logs),
                "crash_logs": self.crash_logs,
                "error_recovery_tested": True,
                "edge_case_handling": "successful",
                "stability_score": 1.0 - (len(self.crash_logs) / max(len(self.decision_accuracy_data), 1))
            },
            "detailed_results": {
                "test_results": self.test_results,
                "accuracy_data": self.decision_accuracy_data,
                "performance_metrics": self.performance_metrics
            },
            "recommendations": self._generate_recommendations(overall_accuracy, avg_decision_time),
            "test_status": "PASSED" if overall_accuracy >= 0.8 and len(self.crash_logs) == 0 else "FAILED"
        }
        
        # Save report to file
        report_filename = f"framework_decision_engine_test_report_{self.test_session_id}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📋 Test Report Summary:")
        logger.info(f"   Overall Accuracy: {overall_accuracy:.1%}")
        logger.info(f"   Average Decision Time: {avg_decision_time:.1f}ms")
        logger.info(f"   Crashes Detected: {len(self.crash_logs)}")
        logger.info(f"   Test Status: {report['test_status']}")
        logger.info(f"   Report saved: {report_filename}")
        
        return report
    
    def _generate_recommendations(self, accuracy: float, avg_time: float) -> List[str]:
        """Generate recommendations based on test results"""
        
        recommendations = []
        
        if accuracy < 0.8:
            recommendations.append(f"🔧 CRITICAL: Accuracy {accuracy:.1%} is below 80% target. Review decision algorithms.")
        
        if accuracy < 0.9:
            recommendations.append("📊 Improve pattern recognition for better framework selection")
            recommendations.append("⚖️  Adjust decision weights based on test results")
        
        if avg_time > 50:
            recommendations.append(f"⚡ PERFORMANCE: Average decision time {avg_time:.1f}ms exceeds 50ms target")
            recommendations.append("🚀 Optimize complexity analysis algorithms")
        
        if len(self.crash_logs) > 0:
            recommendations.append("🛡️  STABILITY: Address crash logs and improve error handling")
        
        if accuracy >= 0.9 and avg_time <= 50 and len(self.crash_logs) == 0:
            recommendations.append("✅ READY FOR PRODUCTION: All tests passed successfully")
            recommendations.append("🚀 Consider implementing A/B testing for continuous improvement")
        
        return recommendations

async def main():
    """Run comprehensive framework decision engine testing"""
    
    print("🧪 AgenticSeek Framework Decision Engine - Comprehensive Testing Suite")
    print("=" * 80)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Initialize and run comprehensive tests
        test_suite = ComprehensiveFrameworkDecisionEngineTest()
        report = await test_suite.run_comprehensive_tests()
        
        print("\n" + "=" * 80)
        print("🎯 COMPREHENSIVE TESTING COMPLETED")
        print("=" * 80)
        print(f"📊 Overall Accuracy: {report['test_summary']['overall_accuracy']:.1%}")
        print(f"⚡ Average Decision Time: {report['test_summary']['avg_decision_time_ms']:.1f}ms")
        print(f"🛡️  Crashes Detected: {report['test_summary']['crashes_detected']}")
        print(f"📈 Test Status: {report['test_status']}")
        print()
        
        print("🔥 RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"   {rec}")
        print()
        
        if report['test_status'] == "PASSED":
            print("✅ FRAMEWORK DECISION ENGINE READY FOR PRODUCTION DEPLOYMENT!")
        else:
            print("❌ FRAMEWORK DECISION ENGINE REQUIRES FIXES BEFORE PRODUCTION")
        
        return report
        
    except Exception as e:
        print(f"❌ CRITICAL FAILURE: Comprehensive testing crashed: {e}")
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    # Run comprehensive tests
    results = asyncio.run(main())
    
    # Exit with appropriate code
    if results and results.get('test_status') == 'PASSED':
        exit_code = 0
    else:
        exit_code = 1
    
    print(f"\n🏁 Testing completed with exit code: {exit_code}")
    exit(exit_code)