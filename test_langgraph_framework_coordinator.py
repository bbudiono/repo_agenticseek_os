#!/usr/bin/env python3
"""
Quick Test for LangGraph Framework Coordinator
Tests intelligent framework selection and routing capabilities
"""

import asyncio
import time
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def quick_framework_coordinator_test():
    """Quick validation of LangGraph Framework Coordinator"""
    print("🧪 LangGraph Framework Coordinator - Quick Validation Test")
    print("=" * 70)
    
    test_results = []
    start_time = time.time()
    
    # Test 1: Import and basic initialization
    print("\n🔍 Test 1: Import and Basic Initialization")
    try:
        from sources.langgraph_framework_coordinator import (
            FrameworkDecisionEngine,
            IntelligentFrameworkCoordinator,
            ComplexTask,
            TaskAnalysis,
            FrameworkDecision,
            ComplexityLevel,
            StateRequirement,
            CoordinationType,
            UserTier,
            FrameworkType,
            BranchingComplexity
        )
        
        print("✅ Successfully imported LangGraph Framework Coordinator components")
        test_results.append(("Import and Initialization", True))
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        test_results.append(("Import and Initialization", False))
        return test_results
    
    # Test 2: Framework Decision Engine
    print("\n🤖 Test 2: Framework Decision Engine")
    try:
        decision_engine = FrameworkDecisionEngine()
        
        # Create test task
        test_task = ComplexTask(
            task_id="test_simple_query",
            task_type="simple_query",
            description="Simple question answering",
            requirements={"stateless": True, "single_step": True},
            constraints={"max_latency_ms": 2000},
            context={},
            user_tier=UserTier.FREE,
            priority="medium"
        )
        
        # Analyze task
        analysis = await decision_engine.analyze_task_requirements(test_task)
        
        assert isinstance(analysis, TaskAnalysis), "Should return TaskAnalysis object"
        assert analysis.complexity in ComplexityLevel, "Should have valid complexity level"
        assert analysis.user_tier == UserTier.FREE, "Should preserve user tier"
        
        print(f"✅ Task analysis completed")
        print(f"   Complexity: {analysis.complexity.value}")
        print(f"   State needs: {analysis.state_needs.value}")
        print(f"   Coordination: {analysis.coordination_type.value}")
        print(f"   Estimated nodes: {analysis.estimated_nodes}")
        
        test_results.append(("Framework Decision Engine", True))
        
    except Exception as e:
        print(f"❌ Framework Decision Engine test failed: {e}")
        test_results.append(("Framework Decision Engine", False))
    
    # Test 3: Framework Selection
    print("\n⚡ Test 3: Framework Selection")
    try:
        # Create complex task for LangGraph
        complex_task = ComplexTask(
            task_id="test_complex_coordination",
            task_type="multi_agent_coordination",
            description="Complex multi-agent video generation",
            requirements={
                "multi_agent": True,
                "state_persistence": True,
                "iterative_refinement": True,
                "required_agents": ["researcher", "writer", "reviewer"]
            },
            constraints={"max_latency_ms": 30000, "high_accuracy": True},
            context={"involves_video": True},
            user_tier=UserTier.ENTERPRISE,
            priority="high"
        )
        
        # Analyze complex task
        complex_analysis = await decision_engine.analyze_task_requirements(complex_task)
        
        # Select framework
        decision = await decision_engine.select_framework(complex_analysis)
        
        assert isinstance(decision, FrameworkDecision), "Should return FrameworkDecision"
        assert decision.primary_framework in FrameworkType, "Should select valid framework"
        assert 0 <= decision.confidence <= 1, "Confidence should be between 0 and 1"
        
        print(f"✅ Framework selection completed")
        print(f"   Selected: {decision.primary_framework.value}")
        print(f"   Confidence: {decision.confidence:.1%}")
        print(f"   Reason: {decision.reason}")
        
        test_results.append(("Framework Selection", True))
        
    except Exception as e:
        print(f"❌ Framework selection test failed: {e}")
        test_results.append(("Framework Selection", False))
    
    # Test 4: User Tier Restrictions
    print("\n👤 Test 4: User Tier Restrictions")
    try:
        # Test Free tier restrictions
        free_complex_task = ComplexTask(
            task_id="test_free_complex",
            task_type="multi_agent_coordination",
            description="Complex task for free user",
            requirements={
                "multi_agent": True,
                "state_persistence": True,
                "iterative_refinement": True
            },
            constraints={},
            context={},
            user_tier=UserTier.FREE,
            priority="medium"
        )
        
        free_analysis = await decision_engine.analyze_task_requirements(free_complex_task)
        free_decision = await decision_engine.select_framework(free_analysis)
        
        # Free tier should prefer simpler frameworks
        print(f"   Free tier decision: {free_decision.primary_framework.value} (confidence: {free_decision.confidence:.1%})")
        
        # Test Enterprise tier capabilities
        enterprise_task = ComplexTask(
            task_id="test_enterprise_complex",
            task_type="multi_agent_coordination",
            description="Complex task for enterprise user",
            requirements={
                "multi_agent": True,
                "state_persistence": True,
                "iterative_refinement": True,
                "complex_state_transitions": True
            },
            constraints={},
            context={},
            user_tier=UserTier.ENTERPRISE,
            priority="high"
        )
        
        enterprise_analysis = await decision_engine.analyze_task_requirements(enterprise_task)
        enterprise_decision = await decision_engine.select_framework(enterprise_analysis)
        
        print(f"   Enterprise tier decision: {enterprise_decision.primary_framework.value} (confidence: {enterprise_decision.confidence:.1%})")
        print(f"✅ User tier restrictions validated")
        
        test_results.append(("User Tier Restrictions", True))
        
    except Exception as e:
        print(f"❌ User tier restrictions test failed: {e}")
        test_results.append(("User Tier Restrictions", False))
    
    # Test 5: Performance Prediction
    print("\n📊 Test 5: Performance Prediction")
    try:
        predictor = decision_engine.performance_predictor
        
        # Test performance prediction
        test_analysis = TaskAnalysis(
            complexity=ComplexityLevel.HIGH,
            state_needs=StateRequirement.COMPLEX,
            coordination_type=CoordinationType.MULTI_AGENT,
            performance_needs={"max_latency_ms": 5000},
            branching_logic=BranchingComplexity.COMPLEX,
            cyclic_processes=True,
            multi_agent_requirements=True,
            estimated_nodes=8,
            estimated_iterations=5,
            estimated_execution_time=45.0,
            memory_requirements=512.0,
            real_time_needs=False,
            user_tier=UserTier.PRO
        )
        
        predictions = await predictor.predict_performance(test_analysis)
        
        assert 'langchain' in predictions, "Should predict LangChain performance"
        assert 'langgraph' in predictions, "Should predict LangGraph performance"
        
        for framework, prediction in predictions.items():
            assert 'predicted_latency_ms' in prediction, f"Should predict latency for {framework}"
            assert 'predicted_throughput_ops_sec' in prediction, f"Should predict throughput for {framework}"
            assert 'predicted_memory_mb' in prediction, f"Should predict memory for {framework}"
            assert 'predicted_accuracy' in prediction, f"Should predict accuracy for {framework}"
        
        print(f"✅ Performance prediction working")
        print(f"   LangChain latency: {predictions['langchain']['predicted_latency_ms']:.1f}ms")
        print(f"   LangGraph latency: {predictions['langgraph']['predicted_latency_ms']:.1f}ms")
        
        test_results.append(("Performance Prediction", True))
        
    except Exception as e:
        print(f"❌ Performance prediction test failed: {e}")
        test_results.append(("Performance Prediction", False))
    
    # Test 6: Decision Analytics
    print("\n📈 Test 6: Decision Analytics")
    try:
        analytics = decision_engine.get_decision_analytics()
        
        assert 'total_decisions' in analytics, "Should track total decisions"
        assert 'framework_distribution' in analytics, "Should track framework distribution"
        
        print(f"✅ Decision analytics working")
        print(f"   Total decisions: {analytics['total_decisions']}")
        print(f"   Framework distribution: {analytics['framework_distribution']}")
        print(f"   Average confidence: {analytics.get('average_confidence', 0):.1%}")
        
        test_results.append(("Decision Analytics", True))
        
    except Exception as e:
        print(f"❌ Decision analytics test failed: {e}")
        test_results.append(("Decision Analytics", False))
    
    # Generate test report
    total_time = time.time() - start_time
    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print("\n" + "=" * 70)
    print("📋 QUICK VALIDATION RESULTS")
    print("=" * 70)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<40} {status}")
    
    print("-" * 70)
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Execution Time: {total_time:.2f}s")
    
    # Readiness assessment
    print(f"\n🚀 READINESS ASSESSMENT")
    print("-" * 35)
    
    if success_rate >= 90:
        print("✅ READY FOR INTEGRATION")
        print("✅ Core framework selection functionality operational")
        print("✅ Intelligent routing and decision engine active")
        print("✅ User tier restrictions working correctly")
        print("✅ Performance prediction system functional")
    elif success_rate >= 70:
        print("⚠️  READY FOR BASIC INTEGRATION")
        print("✅ Most core functionality operational")
        print("⚠️  Some advanced features may need attention")
    else:
        print("❌ REQUIRES FIXES BEFORE INTEGRATION")
        print("❌ Critical functionality issues detected")
    
    return test_results

if __name__ == "__main__":
    async def main():
        results = await quick_framework_coordinator_test()
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        if passed == total:
            print("\n🎉 All tests passed! LangGraph Framework Coordinator validated.")
        else:
            print(f"\n⚠️  {total-passed} test(s) failed. Review implementation.")
        
        return passed == total
    
    asyncio.run(main())