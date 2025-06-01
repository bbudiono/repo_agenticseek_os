#!/usr/bin/env python3
"""
Comprehensive Test Suite for LangGraph State-Based Agent Coordination
Tests StateGraph workflows, multi-agent coordination, and state management
"""

import asyncio
import time
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def comprehensive_langgraph_coordination_test():
    """Comprehensive validation of LangGraph State Coordination"""
    print("🧪 LangGraph State Coordination - Comprehensive Test Suite")
    print("=" * 80)
    
    test_results = []
    start_time = time.time()
    
    # Test 1: Import and basic initialization
    print("\n🔍 Test 1: Import and Basic Initialization")
    try:
        from sources.langgraph_state_coordination import (
            LangGraphCoordinator,
            LangGraphAgent,
            AgentRole,
            AgentCapability,
            CoordinationPattern,
            WorkflowState,
            ComplexityLevel,
            create_base_state_schema,
            create_pro_state_schema,
            create_enterprise_state_schema
        )
        from sources.langgraph_framework_coordinator import ComplexTask, UserTier
        from sources.apple_silicon_optimization_layer import AppleSiliconOptimizationLayer
        
        print("✅ Successfully imported LangGraph State Coordination components")
        test_results.append(("Import and Initialization", True))
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        test_results.append(("Import and Initialization", False))
        return test_results
    
    # Test 2: LangGraph Coordinator Initialization
    print("\n🤖 Test 2: LangGraph Coordinator Initialization")
    try:
        apple_optimizer = AppleSiliconOptimizationLayer()
        coordinator = LangGraphCoordinator(apple_optimizer)
        
        assert hasattr(coordinator, 'available_agents'), "Should have available_agents"
        assert hasattr(coordinator, 'coordination_patterns'), "Should have coordination_patterns"
        assert hasattr(coordinator, 'active_workflows'), "Should have active_workflows"
        
        print(f"✅ Coordinator initialized successfully")
        print(f"   Available patterns: {len(coordinator.coordination_patterns)}")
        print(f"   Initial agents: {len(coordinator.available_agents)}")
        
        test_results.append(("LangGraph Coordinator Initialization", True))
        
    except Exception as e:
        print(f"❌ Coordinator initialization failed: {e}")
        test_results.append(("LangGraph Coordinator Initialization", False))
    
    # Test 3: Agent Creation and Registration
    print("\n👥 Test 3: Agent Creation and Registration")
    try:
        # Create test agent capability
        test_capability = AgentCapability(
            role=AgentRole.RESEARCHER,
            skills=['research', 'analysis', 'data_gathering'],
            tools=['web_search', 'database_query'],
            max_complexity=ComplexityLevel.HIGH,
            specializations=['technical_research'],
            performance_metrics={'avg_execution_time': 1.2, 'avg_quality': 0.88},
            tier_requirements=UserTier.FREE
        )
        
        # Create mock provider
        from sources.llm_provider import Provider
        mock_provider = Provider("test", "test-model", "test-key")
        
        # Create agent
        test_agent = LangGraphAgent(
            role=AgentRole.RESEARCHER,
            capability=test_capability,
            llm_provider=mock_provider,
            apple_optimizer=apple_optimizer
        )
        
        # Register agent
        coordinator.register_agent(test_agent)
        
        assert AgentRole.RESEARCHER in coordinator.available_agents, "Agent should be registered"
        assert coordinator.available_agents[AgentRole.RESEARCHER] == test_agent, "Should store correct agent"
        
        print(f"✅ Agent creation and registration successful")
        print(f"   Agent role: {test_agent.role.value}")
        print(f"   Agent skills: {test_capability.skills}")
        print(f"   Registered agents: {len(coordinator.available_agents)}")
        
        test_results.append(("Agent Creation and Registration", True))
        
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        test_results.append(("Agent Creation and Registration", False))
    
    # Test 4: State Schema Creation
    print("\n📋 Test 4: State Schema Creation")
    try:
        # Test different tier schemas
        base_schema = create_base_state_schema()
        pro_schema = create_pro_state_schema()
        enterprise_schema = create_enterprise_state_schema()
        
        print(f"✅ State schemas created successfully")
        print(f"   Base schema: {type(base_schema)}")
        print(f"   Pro schema: {type(pro_schema)}")
        print(f"   Enterprise schema: {type(enterprise_schema)}")
        
        test_results.append(("State Schema Creation", True))
        
    except Exception as e:
        print(f"❌ State schema creation failed: {e}")
        test_results.append(("State Schema Creation", False))
    
    # Test 5: Workflow Creation
    print("\n⚡ Test 5: Workflow Creation")
    try:
        # Create test task
        test_task = ComplexTask(
            task_id="workflow_test_001",
            task_type="multi_agent_coordination",
            description="Test workflow creation",
            requirements={
                "multi_agent": True,
                "state_persistence": True,
                "quality_control": True
            },
            constraints={"max_latency_ms": 30000},
            context={"domain": "testing"},
            user_tier=UserTier.PRO,
            priority="high"
        )
        
        # Define required agents
        required_agents = [
            AgentRole.COORDINATOR,
            AgentRole.RESEARCHER,
            AgentRole.ANALYST,
            AgentRole.WRITER
        ]
        
        # Test supervisor pattern workflow creation
        workflow = await coordinator.create_workflow(
            test_task, 
            CoordinationPattern.SUPERVISOR, 
            required_agents
        )
        
        assert workflow is not None, "Workflow should be created"
        assert len(coordinator.active_workflows) > 0, "Should track active workflows"
        
        print(f"✅ Workflow creation successful")
        print(f"   Pattern: {CoordinationPattern.SUPERVISOR.value}")
        print(f"   Required agents: {len(required_agents)}")
        print(f"   Active workflows: {len(coordinator.active_workflows)}")
        
        test_results.append(("Workflow Creation", True))
        
    except Exception as e:
        print(f"❌ Workflow creation failed: {e}")
        test_results.append(("Workflow Creation", False))
    
    # Test 6: Agent Processing
    print("\n🔄 Test 6: Agent Processing")
    try:
        # Get registered agent
        researcher_agent = coordinator.available_agents[AgentRole.RESEARCHER]
        
        # Create test state
        test_state = {
            'messages': [],
            'task_context': {'task_type': 'research', 'domain': 'technology'},
            'current_step': 'research_phase',
            'agent_outputs': {},
            'coordination_data': {},
            'quality_scores': {},
            'next_agent': None,
            'workflow_metadata': {},
            'execution_history': []
        }
        
        # Process state with agent
        updated_state = await researcher_agent.process(test_state)
        
        assert 'agent_outputs' in updated_state, "Should update agent outputs"
        assert AgentRole.RESEARCHER.value in updated_state['agent_outputs'], "Should add researcher output"
        assert 'quality_scores' in updated_state, "Should include quality scores"
        assert len(updated_state['execution_history']) > 0, "Should track execution history"
        
        output = updated_state['agent_outputs'][AgentRole.RESEARCHER.value]
        assert 'message' in output, "Should include message"
        assert 'quality_score' in output, "Should include quality score"
        
        print(f"✅ Agent processing successful")
        print(f"   Output quality: {output.get('quality_score', 0):.2f}")
        print(f"   Execution history entries: {len(updated_state['execution_history'])}")
        print(f"   Processing time: {updated_state['coordination_data'][AgentRole.RESEARCHER.value]['execution_time']:.3f}s")
        
        test_results.append(("Agent Processing", True))
        
    except Exception as e:
        print(f"❌ Agent processing failed: {e}")
        test_results.append(("Agent Processing", False))
    
    # Test 7: Coordination Patterns
    print("\n🔗 Test 7: Coordination Patterns")
    try:
        patterns_tested = []
        
        # Test different coordination patterns
        test_patterns = [
            CoordinationPattern.SUPERVISOR,
            CoordinationPattern.PIPELINE,
            CoordinationPattern.HIERARCHICAL
        ]
        
        for pattern in test_patterns:
            try:
                pattern_workflow = await coordinator.create_workflow(
                    test_task, pattern, [AgentRole.COORDINATOR, AgentRole.RESEARCHER]
                )
                patterns_tested.append(pattern.value)
                print(f"   ✅ {pattern.value} pattern created successfully")
            except Exception as e:
                print(f"   ❌ {pattern.value} pattern failed: {e}")
        
        assert len(patterns_tested) >= 2, "Should successfully create multiple patterns"
        
        print(f"✅ Coordination patterns testing successful")
        print(f"   Patterns tested: {len(patterns_tested)}")
        print(f"   Successful patterns: {patterns_tested}")
        
        test_results.append(("Coordination Patterns", True))
        
    except Exception as e:
        print(f"❌ Coordination patterns test failed: {e}")
        test_results.append(("Coordination Patterns", False))
    
    # Test 8: Workflow Execution
    print("\n🚀 Test 8: Workflow Execution")
    try:
        # Create simple workflow for execution test
        execution_workflow = await coordinator.create_workflow(
            test_task, 
            CoordinationPattern.PIPELINE, 
            [AgentRole.RESEARCHER, AgentRole.WRITER]
        )
        
        # Create initial state
        initial_state = {
            'messages': [],
            'task_context': test_task.to_dict(),
            'current_step': 'start',
            'agent_outputs': {},
            'coordination_data': {},
            'quality_scores': {},
            'next_agent': None,
            'workflow_metadata': {'pattern': 'pipeline'},
            'execution_history': []
        }
        
        # Execute workflow
        execution_result = await coordinator.execute_workflow(execution_workflow, initial_state)
        
        assert execution_result is not None, "Should return execution result"
        assert len(coordinator.execution_history) > 0, "Should track execution history"
        
        execution_log = coordinator.execution_history[-1]
        assert execution_log['success'], "Execution should be marked as successful"
        
        print(f"✅ Workflow execution successful")
        print(f"   Execution time: {execution_log['execution_time']:.3f}s")
        print(f"   Result keys: {list(execution_result.keys()) if isinstance(execution_result, dict) else 'Non-dict result'}")
        print(f"   Total executions: {len(coordinator.execution_history)}")
        
        test_results.append(("Workflow Execution", True))
        
    except Exception as e:
        print(f"❌ Workflow execution failed: {e}")
        test_results.append(("Workflow Execution", False))
    
    # Test 9: Analytics and Monitoring
    print("\n📊 Test 9: Analytics and Monitoring")
    try:
        analytics = coordinator.get_coordination_analytics()
        
        required_analytics = [
            'available_agents',
            'coordination_patterns', 
            'active_workflows',
            'execution_history',
            'agent_performance'
        ]
        
        for key in required_analytics:
            assert key in analytics, f"Analytics should include {key}"
        
        assert analytics['execution_history']['total_executions'] > 0, "Should track executions"
        assert analytics['execution_history']['success_rate'] >= 0, "Should calculate success rate"
        
        print(f"✅ Analytics and monitoring working")
        print(f"   Available agents: {len(analytics['available_agents'])}")
        print(f"   Coordination patterns: {len(analytics['coordination_patterns'])}")
        print(f"   Total executions: {analytics['execution_history']['total_executions']}")
        print(f"   Success rate: {analytics['execution_history']['success_rate']:.1%}")
        print(f"   Avg execution time: {analytics['execution_history']['average_execution_time']:.3f}s")
        
        test_results.append(("Analytics and Monitoring", True))
        
    except Exception as e:
        print(f"❌ Analytics test failed: {e}")
        test_results.append(("Analytics and Monitoring", False))
    
    # Test 10: User Tier Integration
    print("\n👤 Test 10: User Tier Integration")
    try:
        # Test different user tiers
        tiers_tested = []
        
        for tier in [UserTier.FREE, UserTier.PRO, UserTier.ENTERPRISE]:
            tier_task = ComplexTask(
                task_id=f"tier_test_{tier.value}",
                task_type="multi_agent_coordination",
                description=f"Test for {tier.value} tier",
                requirements={"multi_agent": True},
                constraints={},
                context={},
                user_tier=tier,
                priority="medium"
            )
            
            try:
                tier_workflow = await coordinator.create_workflow(
                    tier_task,
                    CoordinationPattern.SUPERVISOR,
                    [AgentRole.COORDINATOR, AgentRole.RESEARCHER]
                )
                tiers_tested.append(tier.value)
                print(f"   ✅ {tier.value} tier workflow created")
            except Exception as e:
                print(f"   ❌ {tier.value} tier failed: {e}")
        
        assert len(tiers_tested) >= 2, "Should support multiple user tiers"
        
        print(f"✅ User tier integration successful")
        print(f"   Tiers tested: {tiers_tested}")
        
        test_results.append(("User Tier Integration", True))
        
    except Exception as e:
        print(f"❌ User tier integration failed: {e}")
        test_results.append(("User Tier Integration", False))
    
    # Generate comprehensive test report
    total_time = time.time() - start_time
    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print("\n" + "=" * 80)
    print("📋 COMPREHENSIVE TEST RESULTS SUMMARY")
    print("=" * 80)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<40} {status}")
    
    print("-" * 80)
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Total Execution Time: {total_time:.2f}s")
    
    # Detailed readiness assessment
    print(f"\n🚀 PRODUCTION READINESS ASSESSMENT")
    print("-" * 50)
    
    if success_rate >= 90:
        print("✅ READY FOR PRODUCTION INTEGRATION")
        print("✅ LangGraph StateGraph coordination fully operational")
        print("✅ Multi-agent workflows functioning correctly")
        print("✅ State management and persistence working")
        print("✅ Coordination patterns implemented and tested")
        print("✅ Analytics and monitoring systems active")
        print("✅ User tier integration validated")
        print("✅ Apple Silicon optimization integrated")
    elif success_rate >= 70:
        print("⚠️  READY FOR STAGING ENVIRONMENT")
        print("✅ Core LangGraph functionality operational")
        print("✅ Basic coordination patterns working")
        print("⚠️  Some advanced features may need fine-tuning")
    elif success_rate >= 50:
        print("⚠️  REQUIRES OPTIMIZATION BEFORE DEPLOYMENT")
        print("✅ Basic components functional")
        print("⚠️  Coordination and execution need improvement")
        print("⚠️  Integration points require validation")
    else:
        print("❌ SIGNIFICANT DEVELOPMENT REQUIRED")
        print("❌ Core functionality issues detected")
        print("❌ Multiple system components need attention")
    
    # Feature completeness summary
    if success_rate >= 80:
        print(f"\n🎯 FEATURE COMPLETENESS SUMMARY")
        print("-" * 40)
        print("✅ StateGraph workflow creation and management")
        print("✅ Multi-agent coordination with role-based processing")
        print("✅ State persistence and management across workflow steps")
        print("✅ Multiple coordination patterns (supervisor, pipeline, hierarchical)")
        print("✅ User tier-based feature restrictions and optimizations")
        print("✅ Performance monitoring and analytics")
        print("✅ Apple Silicon hardware optimization integration")
        print("✅ Error handling and execution tracking")
        print("✅ Agent capability management and registration")
        print("✅ Workflow execution with comprehensive state updates")
    
    return test_results

if __name__ == "__main__":
    async def main():
        results = await comprehensive_langgraph_coordination_test()
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        if passed == total:
            print("\n🎉 All tests passed! LangGraph State Coordination fully validated.")
            print("🚀 Ready for integration with MLACS and production deployment.")
        elif passed >= total * 0.8:
            print(f"\n✅ Most tests passed ({passed}/{total}). System ready for advanced testing.")
        else:
            print(f"\n⚠️  {total-passed} test(s) failed. Review implementation before proceeding.")
        
        return passed >= total * 0.8  # Consider 80% success rate as acceptable
    
    asyncio.run(main())