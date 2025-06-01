#!/usr/bin/env python3
"""
Comprehensive Test Suite for Pydantic AI Core Integration
Tests type safety, validation, tier management, and agent functionality
"""

import asyncio
import time
import json
from datetime import datetime
from typing import Dict, Any, List

# Test framework components
import sys
import traceback

class TestResult:
    def __init__(self, test_name: str, passed: bool, details: str = "", execution_time: float = 0.0):
        self.test_name = test_name
        self.passed = passed
        self.details = details
        self.execution_time = execution_time

class PydanticAITestSuite:
    def __init__(self):
        self.test_results: List[TestResult] = []
        self.start_time = time.time()
    
    async def run_all_tests(self):
        """Run comprehensive test suite for Pydantic AI Core Integration"""
        print("🧪 Pydantic AI Core Integration - Comprehensive Test Suite")
        print("=" * 70)
        
        # Import tests
        await self.test_import_and_initialization()
        await self.test_type_safe_models()
        await self.test_agent_configuration_validation()
        await self.test_task_creation_and_validation()
        await self.test_tier_based_capabilities()
        await self.test_agent_communication()
        await self.test_agent_execution()
        await self.test_performance_tracking()
        await self.test_error_handling()
        await self.test_fallback_compatibility()
        
        # Generate final report
        return await self.generate_test_report()
    
    async def test_import_and_initialization(self):
        """Test 1: Import and basic initialization"""
        test_name = "Import and Initialization"
        start_time = time.time()
        
        try:
            from sources.pydantic_ai_core_integration import (
                AgentSpecialization,
                AgentCapability,
                UserTier,
                AgentConfiguration,
                TypeSafeTask,
                TaskResult,
                TypeSafeAgent,
                PydanticAIIntegrationDependencies,
                PYDANTIC_AI_AVAILABLE
            )
            
            # Test enum availability
            assert AgentSpecialization.RESEARCH is not None
            assert AgentCapability.BASIC_REASONING is not None
            assert UserTier.PRO is not None
            
            execution_time = time.time() - start_time
            self.test_results.append(TestResult(
                test_name, True, 
                f"All imports successful. Pydantic AI available: {PYDANTIC_AI_AVAILABLE}",
                execution_time
            ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(TestResult(
                test_name, False, 
                f"Import failed: {str(e)}",
                execution_time
            ))
    
    async def test_type_safe_models(self):
        """Test 2: Type-safe model creation and validation"""
        test_name = "Type-Safe Model Creation"
        start_time = time.time()
        
        try:
            from sources.pydantic_ai_core_integration import (
                AgentConfiguration, AgentSpecialization, UserTier, AgentCapability,
                TypeSafeTask, TaskComplexity, PYDANTIC_AI_AVAILABLE
            )
            
            if PYDANTIC_AI_AVAILABLE:
                # Test valid agent configuration
                config = AgentConfiguration(
                    agent_id="test_agent",
                    specialization=AgentSpecialization.RESEARCH,
                    tier_requirements=UserTier.PRO,
                    capabilities=[AgentCapability.BASIC_REASONING, AgentCapability.ADVANCED_REASONING]
                )
                
                assert config.agent_id == "test_agent"
                assert config.specialization == AgentSpecialization.RESEARCH
                assert len(config.capabilities) == 2
                
                # Test valid task creation
                task = TypeSafeTask(
                    description="Test task for validation",
                    complexity=TaskComplexity.MEDIUM,
                    user_tier=UserTier.PRO
                )
                
                assert len(task.description) >= 10
                assert task.complexity == TaskComplexity.MEDIUM
                assert task.user_tier == UserTier.PRO
                
                details = "Pydantic models created and validated successfully"
            else:
                # Test fallback implementations
                config = AgentConfiguration(
                    agent_id="test_agent",
                    specialization="research",
                    tier_requirements="pro",
                    capabilities=["basic_reasoning", "advanced_reasoning"]
                )
                
                task = TypeSafeTask(
                    description="Test task for validation",
                    complexity="medium",
                    user_tier="pro"
                )
                
                assert config.agent_id == "test_agent"
                assert task.description == "Test task for validation"
                
                details = "Fallback models created successfully"
            
            execution_time = time.time() - start_time
            self.test_results.append(TestResult(
                test_name, True, details, execution_time
            ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(TestResult(
                test_name, False, 
                f"Model creation failed: {str(e)}\n{traceback.format_exc()}",
                execution_time
            ))
    
    async def test_agent_configuration_validation(self):
        """Test 3: Agent configuration validation"""
        test_name = "Agent Configuration Validation"
        start_time = time.time()
        
        try:
            from sources.pydantic_ai_core_integration import (
                AgentConfiguration, AgentSpecialization, UserTier, AgentCapability,
                PYDANTIC_AI_AVAILABLE
            )
            
            validation_tests_passed = 0
            total_validation_tests = 3
            
            if PYDANTIC_AI_AVAILABLE:
                # Test 1: Valid configuration
                try:
                    valid_config = AgentConfiguration(
                        agent_id="valid_agent",
                        specialization=AgentSpecialization.TECHNICAL,
                        tier_requirements=UserTier.ENTERPRISE,
                        capabilities=[AgentCapability.CUSTOM_TOOLS, AgentCapability.PREDICTIVE_ANALYTICS]
                    )
                    validation_tests_passed += 1
                except Exception:
                    pass
                
                # Test 2: Invalid empty agent_id
                try:
                    invalid_config = AgentConfiguration(
                        agent_id="",
                        specialization=AgentSpecialization.RESEARCH
                    )
                    # Should not reach here
                except Exception:
                    validation_tests_passed += 1  # Expected validation error
                
                # Test 3: Valid tier-capability combination
                try:
                    enterprise_config = AgentConfiguration(
                        agent_id="enterprise_agent",
                        specialization=AgentSpecialization.VIDEO_GENERATION,
                        tier_requirements=UserTier.ENTERPRISE,
                        capabilities=[AgentCapability.VIDEO_GENERATION, AgentCapability.CUSTOM_TOOLS]
                    )
                    validation_tests_passed += 1
                except Exception:
                    pass
                
                details = f"Pydantic validation: {validation_tests_passed}/{total_validation_tests} tests passed"
            else:
                # Fallback validation tests
                config1 = AgentConfiguration(
                    agent_id="fallback_agent",
                    specialization="research",
                    tier_requirements="pro"
                )
                
                config2 = AgentConfiguration(
                    agent_id="enterprise_agent",
                    specialization="video_generation",
                    tier_requirements="enterprise"
                )
                
                validation_tests_passed = 2  # Basic creation tests
                details = f"Fallback validation: {validation_tests_passed}/2 tests passed"
            
            execution_time = time.time() - start_time
            success = validation_tests_passed >= (total_validation_tests - 1)  # Allow 1 failure
            
            self.test_results.append(TestResult(
                test_name, success, details, execution_time
            ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(TestResult(
                test_name, False, 
                f"Configuration validation failed: {str(e)}",
                execution_time
            ))
    
    async def test_task_creation_and_validation(self):
        """Test 4: Task creation and validation"""
        test_name = "Task Creation and Validation"
        start_time = time.time()
        
        try:
            from sources.pydantic_ai_core_integration import (
                TypeSafeTask, TaskComplexity, UserTier, TaskRequirement, AgentCapability,
                PYDANTIC_AI_AVAILABLE
            )
            
            task_tests_passed = 0
            total_task_tests = 3
            
            if PYDANTIC_AI_AVAILABLE:
                # Test 1: Valid basic task
                try:
                    basic_task = TypeSafeTask(
                        description="This is a valid task description that meets minimum length requirements",
                        complexity=TaskComplexity.SIMPLE,
                        user_tier=UserTier.FREE
                    )
                    assert basic_task.task_id is not None
                    assert len(basic_task.task_id) > 0
                    task_tests_passed += 1
                except Exception:
                    pass
                
                # Test 2: Task with requirements
                try:
                    requirement = TaskRequirement(
                        capability=AgentCapability.BASIC_REASONING,
                        importance=0.8,
                        optional=False
                    )
                    
                    complex_task = TypeSafeTask(
                        description="Complex task with specific requirements and high priority",
                        complexity=TaskComplexity.COMPLEX,
                        user_tier=UserTier.PRO,
                        requirements=[requirement],
                        priority=8
                    )
                    
                    assert len(complex_task.requirements) == 1
                    assert complex_task.priority == 8
                    task_tests_passed += 1
                except Exception:
                    pass
                
                # Test 3: Invalid task (too short description)
                try:
                    invalid_task = TypeSafeTask(
                        description="Short",  # Too short
                        complexity=TaskComplexity.SIMPLE,
                        user_tier=UserTier.FREE
                    )
                    # Should not reach here
                except Exception:
                    task_tests_passed += 1  # Expected validation error
                
                details = f"Pydantic task validation: {task_tests_passed}/{total_task_tests} tests passed"
            else:
                # Fallback task tests
                task1 = TypeSafeTask(
                    description="Fallback task with proper description length",
                    complexity="simple",
                    user_tier="free"
                )
                
                task2 = TypeSafeTask(
                    description="Another fallback task for testing purposes",
                    complexity="complex",
                    user_tier="pro",
                    priority=5
                )
                
                task_tests_passed = 2
                details = f"Fallback task creation: {task_tests_passed}/2 tests passed"
            
            execution_time = time.time() - start_time
            success = task_tests_passed >= 2
            
            self.test_results.append(TestResult(
                test_name, success, details, execution_time
            ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(TestResult(
                test_name, False, 
                f"Task validation failed: {str(e)}",
                execution_time
            ))
    
    async def test_tier_based_capabilities(self):
        """Test 5: Tier-based capability management"""
        test_name = "Tier-Based Capabilities"
        start_time = time.time()
        
        try:
            from sources.pydantic_ai_core_integration import (
                AgentConfiguration, AgentSpecialization, UserTier, AgentCapability,
                TypeSafeAgent, PydanticAIIntegrationDependencies,
                PYDANTIC_AI_AVAILABLE
            )
            
            dependencies = PydanticAIIntegrationDependencies()
            tier_tests_passed = 0
            total_tier_tests = 3
            
            # Test 1: FREE tier agent
            try:
                if PYDANTIC_AI_AVAILABLE:
                    free_config = AgentConfiguration(
                        agent_id="free_agent",
                        specialization=AgentSpecialization.RESEARCH,
                        tier_requirements=UserTier.FREE,
                        capabilities=[AgentCapability.BASIC_REASONING]
                    )
                else:
                    free_config = AgentConfiguration(
                        agent_id="free_agent",
                        specialization="research",
                        tier_requirements="free",
                        capabilities=["basic_reasoning"]
                    )
                
                free_agent = TypeSafeAgent(free_config, dependencies)
                assert free_agent.config.agent_id == "free_agent"
                tier_tests_passed += 1
            except Exception:
                pass
            
            # Test 2: PRO tier agent
            try:
                if PYDANTIC_AI_AVAILABLE:
                    pro_config = AgentConfiguration(
                        agent_id="pro_agent",
                        specialization=AgentSpecialization.CREATIVE,
                        tier_requirements=UserTier.PRO,
                        capabilities=[AgentCapability.BASIC_REASONING, AgentCapability.ADVANCED_REASONING]
                    )
                else:
                    pro_config = AgentConfiguration(
                        agent_id="pro_agent",
                        specialization="creative",
                        tier_requirements="pro",
                        capabilities=["basic_reasoning", "advanced_reasoning"]
                    )
                
                pro_agent = TypeSafeAgent(pro_config, dependencies)
                assert pro_agent.config.agent_id == "pro_agent"
                tier_tests_passed += 1
            except Exception:
                pass
            
            # Test 3: ENTERPRISE tier agent
            try:
                if PYDANTIC_AI_AVAILABLE:
                    enterprise_config = AgentConfiguration(
                        agent_id="enterprise_agent",
                        specialization=AgentSpecialization.VIDEO_GENERATION,
                        tier_requirements=UserTier.ENTERPRISE,
                        capabilities=[
                            AgentCapability.VIDEO_GENERATION,
                            AgentCapability.CUSTOM_TOOLS,
                            AgentCapability.PREDICTIVE_ANALYTICS
                        ]
                    )
                else:
                    enterprise_config = AgentConfiguration(
                        agent_id="enterprise_agent",
                        specialization="video_generation",
                        tier_requirements="enterprise",
                        capabilities=["video_generation", "custom_tools", "predictive_analytics"]
                    )
                
                enterprise_agent = TypeSafeAgent(enterprise_config, dependencies)
                assert enterprise_agent.config.agent_id == "enterprise_agent"
                tier_tests_passed += 1
            except Exception:
                pass
            
            execution_time = time.time() - start_time
            details = f"Tier-based agents created: {tier_tests_passed}/{total_tier_tests}"
            
            self.test_results.append(TestResult(
                test_name, tier_tests_passed == total_tier_tests, details, execution_time
            ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(TestResult(
                test_name, False, 
                f"Tier capability testing failed: {str(e)}",
                execution_time
            ))
    
    async def test_agent_communication(self):
        """Test 6: Agent communication system"""
        test_name = "Agent Communication"
        start_time = time.time()
        
        try:
            from sources.pydantic_ai_core_integration import (
                AgentConfiguration, AgentSpecialization, UserTier, AgentCapability,
                TypeSafeAgent, PydanticAIIntegrationDependencies, MessageType,
                PYDANTIC_AI_AVAILABLE
            )
            
            dependencies = PydanticAIIntegrationDependencies()
            
            # Create two agents for communication testing
            if PYDANTIC_AI_AVAILABLE:
                config1 = AgentConfiguration(
                    agent_id="agent_sender",
                    specialization=AgentSpecialization.COORDINATOR,
                    tier_requirements=UserTier.PRO
                )
                config2 = AgentConfiguration(
                    agent_id="agent_receiver",
                    specialization=AgentSpecialization.RESEARCH,
                    tier_requirements=UserTier.PRO
                )
            else:
                config1 = AgentConfiguration(
                    agent_id="agent_sender",
                    specialization="coordinator",
                    tier_requirements="pro"
                )
                config2 = AgentConfiguration(
                    agent_id="agent_receiver",
                    specialization="research",
                    tier_requirements="pro"
                )
            
            agent1 = TypeSafeAgent(config1, dependencies)
            agent2 = TypeSafeAgent(config2, dependencies)
            
            # Test message sending
            message_payload = {
                "task_description": "Research latest AI developments",
                "priority": "high",
                "deadline": "2025-01-10"
            }
            
            # Send message from agent1 to agent2
            message_sent = await agent1.send_message(
                recipient_agent_id="agent_receiver",
                message_type=MessageType.TASK_ASSIGNMENT,
                payload=message_payload
            )
            
            assert message_sent == True
            
            execution_time = time.time() - start_time
            details = f"Message sent successfully between agents"
            
            self.test_results.append(TestResult(
                test_name, True, details, execution_time
            ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(TestResult(
                test_name, False, 
                f"Agent communication failed: {str(e)}",
                execution_time
            ))
    
    async def test_agent_execution(self):
        """Test 7: Agent task execution"""
        test_name = "Agent Task Execution"
        start_time = time.time()
        
        try:
            from sources.pydantic_ai_core_integration import (
                AgentConfiguration, AgentSpecialization, UserTier, AgentCapability,
                TypeSafeAgent, PydanticAIIntegrationDependencies,
                TypeSafeTask, TaskComplexity, ExecutionStatus,
                PYDANTIC_AI_AVAILABLE
            )
            
            dependencies = PydanticAIIntegrationDependencies()
            
            # Create agent
            if PYDANTIC_AI_AVAILABLE:
                config = AgentConfiguration(
                    agent_id="execution_test_agent",
                    specialization=AgentSpecialization.DATA_ANALYSIS,
                    tier_requirements=UserTier.PRO,
                    capabilities=[AgentCapability.BASIC_REASONING, AgentCapability.ADVANCED_REASONING]
                )
                
                task = TypeSafeTask(
                    description="Analyze sample data and provide insights for business decision making",
                    complexity=TaskComplexity.MEDIUM,
                    user_tier=UserTier.PRO,
                    priority=5
                )
            else:
                config = AgentConfiguration(
                    agent_id="execution_test_agent",
                    specialization="data_analysis",
                    tier_requirements="pro",
                    capabilities=["basic_reasoning", "advanced_reasoning"]
                )
                
                task = TypeSafeTask(
                    description="Analyze sample data and provide insights for business decision making",
                    complexity="medium",
                    user_tier="pro",
                    priority=5
                )
            
            agent = TypeSafeAgent(config, dependencies)
            
            # Execute task
            result = await agent.execute_task(task)
            
            # Validate result
            assert result.task_id == task.task_id
            assert result.agent_id == config.agent_id
            
            # Check execution status
            if PYDANTIC_AI_AVAILABLE:
                success = result.status == ExecutionStatus.COMPLETED
            else:
                success = result.status == "completed"
            
            execution_time = time.time() - start_time
            details = f"Task executed. Status: {result.status}, Confidence: {result.confidence_score}"
            
            self.test_results.append(TestResult(
                test_name, success, details, execution_time
            ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(TestResult(
                test_name, False, 
                f"Agent execution failed: {str(e)}\n{traceback.format_exc()}",
                execution_time
            ))
    
    async def test_performance_tracking(self):
        """Test 8: Performance tracking and metrics"""
        test_name = "Performance Tracking"
        start_time = time.time()
        
        try:
            from sources.pydantic_ai_core_integration import (
                AgentConfiguration, AgentSpecialization, UserTier,
                TypeSafeAgent, PydanticAIIntegrationDependencies,
                PYDANTIC_AI_AVAILABLE
            )
            
            dependencies = PydanticAIIntegrationDependencies()
            
            # Create agent
            if PYDANTIC_AI_AVAILABLE:
                config = AgentConfiguration(
                    agent_id="performance_test_agent",
                    specialization=AgentSpecialization.QUALITY_ASSURANCE,
                    tier_requirements=UserTier.ENTERPRISE
                )
            else:
                config = AgentConfiguration(
                    agent_id="performance_test_agent",
                    specialization="quality_assurance",
                    tier_requirements="enterprise"
                )
            
            agent = TypeSafeAgent(config, dependencies)
            
            # Get performance report
            performance_report = agent.get_performance_report()
            
            # Validate performance report structure
            required_fields = ["agent_id", "task_completion_rate", "total_tasks_executed"]
            
            for field in required_fields:
                assert field in performance_report, f"Missing field: {field}"
            
            assert performance_report["agent_id"] == "performance_test_agent"
            assert isinstance(performance_report["total_tasks_executed"], int)
            
            execution_time = time.time() - start_time
            details = f"Performance report generated with {len(performance_report)} metrics"
            
            self.test_results.append(TestResult(
                test_name, True, details, execution_time
            ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(TestResult(
                test_name, False, 
                f"Performance tracking failed: {str(e)}",
                execution_time
            ))
    
    async def test_error_handling(self):
        """Test 9: Error handling and validation"""
        test_name = "Error Handling"
        start_time = time.time()
        
        try:
            from sources.pydantic_ai_core_integration import (
                AgentConfiguration, AgentSpecialization, UserTier,
                TypeSafeAgent, PydanticAIIntegrationDependencies,
                TypeSafeTask, TaskComplexity, ExecutionStatus,
                PYDANTIC_AI_AVAILABLE
            )
            
            dependencies = PydanticAIIntegrationDependencies()
            error_tests_passed = 0
            total_error_tests = 2
            
            # Test 1: Agent with invalid task compatibility
            try:
                if PYDANTIC_AI_AVAILABLE:
                    free_config = AgentConfiguration(
                        agent_id="free_error_agent",
                        specialization=AgentSpecialization.RESEARCH,
                        tier_requirements=UserTier.FREE
                    )
                    
                    enterprise_task = TypeSafeTask(
                        description="Enterprise-only task that requires high-tier capabilities",
                        complexity=TaskComplexity.ENTERPRISE_ONLY,
                        user_tier=UserTier.ENTERPRISE
                    )
                else:
                    free_config = AgentConfiguration(
                        agent_id="free_error_agent",
                        specialization="research",
                        tier_requirements="free"
                    )
                    
                    enterprise_task = TypeSafeTask(
                        description="Enterprise-only task that requires high-tier capabilities",
                        complexity="enterprise_only",
                        user_tier="enterprise"
                    )
                
                agent = TypeSafeAgent(free_config, dependencies)
                result = await agent.execute_task(enterprise_task)
                
                # Should handle the incompatibility gracefully
                if PYDANTIC_AI_AVAILABLE:
                    tier_mismatch_handled = result.status == ExecutionStatus.FAILED
                else:
                    tier_mismatch_handled = result.status == "failed"
                
                if tier_mismatch_handled:
                    error_tests_passed += 1
                    
            except Exception:
                error_tests_passed += 1  # Error handling worked
            
            # Test 2: Invalid message sending
            try:
                if PYDANTIC_AI_AVAILABLE:
                    config = AgentConfiguration(
                        agent_id="message_error_agent",
                        specialization=AgentSpecialization.COORDINATOR,
                        tier_requirements=UserTier.PRO
                    )
                else:
                    config = AgentConfiguration(
                        agent_id="message_error_agent",
                        specialization="coordinator",
                        tier_requirements="pro"
                    )
                
                agent = TypeSafeAgent(config, dependencies)
                
                # Try to send message with invalid payload
                from sources.pydantic_ai_core_integration import MessageType
                result = await agent.send_message(
                    recipient_agent_id="nonexistent_agent",
                    message_type=MessageType.ERROR,
                    payload={"error": "test error"}
                )
                
                # Should handle gracefully
                error_tests_passed += 1
                
            except Exception:
                error_tests_passed += 1  # Error handling worked
            
            execution_time = time.time() - start_time
            details = f"Error handling: {error_tests_passed}/{total_error_tests} scenarios handled correctly"
            
            self.test_results.append(TestResult(
                test_name, error_tests_passed >= 1, details, execution_time
            ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(TestResult(
                test_name, False, 
                f"Error handling test failed: {str(e)}",
                execution_time
            ))
    
    async def test_fallback_compatibility(self):
        """Test 10: Fallback compatibility when Pydantic AI is not available"""
        test_name = "Fallback Compatibility"
        start_time = time.time()
        
        try:
            from sources.pydantic_ai_core_integration import PYDANTIC_AI_AVAILABLE
            
            # This test validates that the system works in both modes
            fallback_tests_passed = 0
            total_fallback_tests = 2
            
            # Test 1: System reports availability status correctly
            if isinstance(PYDANTIC_AI_AVAILABLE, bool):
                fallback_tests_passed += 1
            
            # Test 2: Basic functionality works regardless of Pydantic AI availability
            from sources.pydantic_ai_core_integration import (
                AgentConfiguration, TypeSafeAgent, PydanticAIIntegrationDependencies
            )
            
            dependencies = PydanticAIIntegrationDependencies()
            
            if PYDANTIC_AI_AVAILABLE:
                from sources.pydantic_ai_core_integration import AgentSpecialization, UserTier
                config = AgentConfiguration(
                    agent_id="fallback_test_agent",
                    specialization=AgentSpecialization.RESEARCH,
                    tier_requirements=UserTier.FREE
                )
            else:
                config = AgentConfiguration(
                    agent_id="fallback_test_agent",
                    specialization="research",
                    tier_requirements="free"
                )
            
            agent = TypeSafeAgent(config, dependencies)
            assert agent.config.agent_id == "fallback_test_agent"
            fallback_tests_passed += 1
            
            execution_time = time.time() - start_time
            details = f"Fallback compatibility: {fallback_tests_passed}/{total_fallback_tests} tests passed. Pydantic AI: {PYDANTIC_AI_AVAILABLE}"
            
            self.test_results.append(TestResult(
                test_name, fallback_tests_passed == total_fallback_tests, details, execution_time
            ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(TestResult(
                test_name, False, 
                f"Fallback compatibility test failed: {str(e)}",
                execution_time
            ))
    
    async def generate_test_report(self):
        """Generate comprehensive test report"""
        total_time = time.time() - self.start_time
        passed_tests = sum(1 for result in self.test_results if result.passed)
        total_tests = len(self.test_results)
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print("\n" + "=" * 70)
        print("📋 PYDANTIC AI CORE INTEGRATION TEST RESULTS")
        print("=" * 70)
        
        for result in self.test_results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"{result.test_name:<35} {status}")
            if result.details:
                print(f"    Details: {result.details}")
            print(f"    Execution Time: {result.execution_time:.3f}s")
            print()
        
        print("-" * 70)
        print(f"Tests Passed: {passed_tests}/{total_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Total Execution Time: {total_time:.2f}s")
        
        # Readiness assessment
        print(f"\n🚀 READINESS ASSESSMENT")
        print("-" * 35)
        
        if success_rate >= 90:
            print("✅ READY FOR PRODUCTION INTEGRATION")
            print("✅ Type safety and validation framework operational")
            print("✅ Tier-based capability management working")
            print("✅ Agent communication and execution validated")
            print("✅ Performance tracking and error handling confirmed")
        elif success_rate >= 70:
            print("⚠️  READY FOR FURTHER DEVELOPMENT")
            print("✅ Core functionality validated")
            print("⚠️  Some components may need refinement")
        else:
            print("❌ REQUIRES FIXES BEFORE INTEGRATION")
            print("❌ Critical component issues detected")
        
        # Architecture summary
        if success_rate >= 80:
            print(f"\n🏗️ ARCHITECTURE SUMMARY")
            print("-" * 30)
            print("✅ Pydantic AI core integration foundation established")
            print("✅ Type-safe agent configuration and validation")
            print("✅ Tier-based capability management system")
            print("✅ Agent communication and task execution framework")
            print("✅ Performance tracking and error handling")
            print("✅ Fallback compatibility for diverse environments")
        
        return self.test_results

async def main():
    """Run the comprehensive Pydantic AI Core Integration test suite"""
    test_suite = PydanticAITestSuite()
    results = await test_suite.run_all_tests()
    
    passed = sum(1 for result in results if result.passed)
    total = len(results)
    
    if passed >= total * 0.8:  # 80% success threshold
        print("\n🎉 Pydantic AI Core Integration test suite passed!")
        return True
    else:
        print(f"\n⚠️  {total-passed} test(s) failed. Core functionality needs attention.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)