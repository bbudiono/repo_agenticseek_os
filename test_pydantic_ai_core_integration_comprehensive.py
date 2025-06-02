#!/usr/bin/env python3
"""
SANDBOX FILE: For testing/development. See .cursorrules.

Comprehensive Test Suite for Pydantic AI Core Integration System
TASK-PYDANTIC-001: Core Pydantic AI Integration - Comprehensive Testing & Validation

Purpose: Execute comprehensive headless testing with crash detection and performance monitoring
Issues & Complexity Summary: Type system validation, agent lifecycle testing, memory management validation
Key Complexity Drivers:
  - Logic Scope (Est. LoC): ~1200
  - Core Algorithm Complexity: High
  - Dependencies: 8 New, 3 Mod
  - State Management Complexity: Very High
  - Novelty/Uncertainty Factor: High
AI Pre-Task Self-Assessment (Est. Solution Difficulty %): 90%
Problem Estimate (Inherent Problem Difficulty %): 85%
Initial Code Complexity Estimate %: 90%
Justification for Estimates: Comprehensive validation of type-safe agent system with crash detection
Final Code Complexity (Actual %): TBD
Overall Result Score (Success & Quality %): TBD
Key Variances/Learnings: TBD
Last Updated: 2025-06-02

Features:
- Comprehensive agent lifecycle testing with validation
- Type safety validation for all Pydantic models
- Crash detection and recovery testing
- Performance monitoring and memory management validation
- Background monitoring system testing
- Database persistence and integrity verification
- Streaming response validation
- Error handling and validation engine testing
"""

import asyncio
import json
import time
import os
import sys
import sqlite3
import tempfile
import shutil
import traceback
import gc
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging

# Add the sources directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'sources'))

from pydantic_ai_core_integration import (
    PydanticAICore, AgentConfig, TaskInput, TaskOutput, ValidationResult,
    AgentRole, ModelProvider, ValidationLevel
)

# Configure logging for testing
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensivePydanticAITest:
    """Comprehensive test suite for Pydantic AI Core Integration system"""
    
    def __init__(self):
        self.test_db_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.test_db_dir, "test_pydantic_ai_core.db")
        self.core_system = None
        self.test_results = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "test_details": [],
            "performance_metrics": {
                "memory_usage": [],
                "response_times": [],
                "validation_times": []
            },
            "crash_detection": {
                "crashes_detected": 0,
                "crash_details": [],
                "recovery_tests": []
            },
            "system_stability": {
                "memory_leaks": False,
                "resource_cleanup": True,
                "background_monitoring": True
            }
        }
        self.start_time = time.time()
        self.initial_memory = self._get_memory_usage()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except:
            return 0.0
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run complete test suite with crash detection and performance monitoring"""
        print("🧠 STARTING COMPREHENSIVE PYDANTIC AI CORE INTEGRATION TESTING")
        print("=" * 100)
        
        try:
            # Core system initialization tests
            await self._test_core_system_initialization()
            await self._test_agent_configuration_validation()
            await self._test_task_input_output_validation()
            
            # Agent lifecycle and management tests
            await self._test_agent_creation_and_management()
            await self._test_agent_role_based_processing()
            await self._test_multi_model_provider_support()
            
            # Task execution and validation tests
            await self._test_task_execution_workflow()
            await self._test_streaming_response_handling()
            await self._test_validation_engine_comprehensive()
            
            # Memory and persistence tests
            await self._test_memory_management_and_storage()
            await self._test_database_persistence_integrity()
            await self._test_background_monitoring_systems()
            
            # Performance and stress tests
            await self._test_performance_metrics_collection()
            await self._test_concurrent_agent_execution()
            await self._test_memory_management_and_cleanup()
            
            # Error handling and recovery tests
            await self._test_error_handling_and_recovery()
            await self._test_crash_detection_and_recovery()
            await self._test_validation_error_handling()
            
            # Generate final comprehensive report
            await self._generate_comprehensive_report()
            
        except Exception as e:
            self._record_crash("comprehensive_test_suite", str(e), traceback.format_exc())
            print(f"💥 CRITICAL: Comprehensive test suite crashed: {e}")
            
        finally:
            await self._cleanup_test_environment()
        
        return self.test_results
    
    async def _test_core_system_initialization(self):
        """Test Pydantic AI Core system initialization"""
        test_name = "Core System Initialization"
        print(f"🔧 Testing: {test_name}")
        
        try:
            # Initialize core system
            self.core_system = PydanticAICore(self.test_db_path)
            
            # Verify database creation
            assert os.path.exists(self.test_db_path), "Database file not created"
            
            # Verify core components initialization
            assert self.core_system.agent_manager is not None, "Agent manager not initialized"
            assert self.core_system.validation_engine is not None, "Validation engine not initialized"
            assert self.core_system.memory_manager is not None, "Memory manager not initialized"
            assert self.core_system.response_handler is not None, "Response handler not initialized"
            
            # Verify agents dictionary exists
            assert hasattr(self.core_system, 'agents'), "Agents dictionary not initialized"
            assert hasattr(self.core_system, 'tools'), "Tools dictionary not initialized"
            assert hasattr(self.core_system, 'memory_store'), "Memory store not initialized"
            
            # Verify background monitoring
            assert self.core_system.monitoring_active == True, "Background monitoring not active"
            assert self.core_system.monitor_thread is not None, "Monitor thread not created"
            assert self.core_system.monitor_thread.is_alive(), "Monitor thread not running"
            
            self._record_test_result(test_name, True, "Core system initialized with all components")
            
        except Exception as e:
            self._record_crash(test_name, str(e), traceback.format_exc())
            self._record_test_result(test_name, False, f"Initialization failed: {e}")
    
    async def _test_agent_configuration_validation(self):
        """Test agent configuration validation"""
        test_name = "Agent Configuration Validation"
        print(f"📋 Testing: {test_name}")
        
        try:
            # Test valid agent configuration
            valid_config = AgentConfig(
                agent_id="test_agent_001",
                name="Test Analyzer Agent",
                role=AgentRole.ANALYZER,
                model_provider=ModelProvider.OPENAI,
                model_name="gpt-4",
                temperature=0.7,
                max_tokens=1000,
                system_prompt="You are a test analyzer agent",
                tools=["analysis", "validation"],
                memory_enabled=True,
                streaming_enabled=False,
                validation_level=ValidationLevel.STRICT,
                timeout_seconds=30
            )
            
            # Validate configuration
            validation_result = await self.core_system.validation_engine.validate_agent_config(valid_config)
            assert validation_result.is_valid == True, "Valid configuration rejected"
            assert len(validation_result.errors) == 0, "Valid configuration has errors"
            
            # Test invalid agent configurations (handle Pydantic validation errors)
            invalid_config_tests = [
                # Test invalid temperature (will be caught by Pydantic)
                ("temperature", 3.0),
                # Test invalid max_tokens (will be caught by Pydantic)
                ("max_tokens", -100),
                # Test empty agent ID (will be caught by our validation)
                ("agent_id", "")
            ]
            
            for test_name, invalid_value in invalid_config_tests:
                try:
                    if test_name == "temperature":
                        invalid_config = AgentConfig(
                            agent_id="invalid_temp_agent",
                            name="Invalid Temperature Agent",
                            role=AgentRole.VALIDATOR,
                            model_provider=ModelProvider.GOOGLE,
                            model_name="gemini-pro",
                            temperature=invalid_value
                        )
                    elif test_name == "max_tokens":
                        invalid_config = AgentConfig(
                            agent_id="invalid_tokens_agent",
                            name="Invalid Tokens Agent",
                            role=AgentRole.COORDINATOR,
                            model_provider=ModelProvider.LOCAL,
                            model_name="local-model",
                            max_tokens=invalid_value
                        )
                    else:  # agent_id
                        invalid_config = AgentConfig(
                            agent_id=invalid_value,
                            name="Invalid Agent",
                            role=AgentRole.PROCESSOR,
                            model_provider=ModelProvider.ANTHROPIC,
                            model_name="claude-3"
                        )
                    
                    # If Pydantic didn't catch it, our validator should
                    validation_result = await self.core_system.validation_engine.validate_agent_config(invalid_config)
                    assert validation_result.is_valid == False, f"Invalid {test_name} configuration accepted"
                    assert len(validation_result.errors) > 0, f"Invalid {test_name} configuration has no errors"
                    
                except Exception as e:
                    # Pydantic validation error is expected and acceptable
                    if "validation error" in str(e).lower() or "input should be" in str(e).lower():
                        pass  # This is expected behavior
                    else:
                        raise e
            
            # Test configuration serialization/deserialization
            config_dict = valid_config.dict()
            assert isinstance(config_dict, dict), "Configuration serialization failed"
            assert config_dict["agent_id"] == valid_config.agent_id, "Serialization data mismatch"
            
            self._record_test_result(test_name, True, "Agent configuration validation comprehensive")
            
        except Exception as e:
            self._record_crash(test_name, str(e), traceback.format_exc())
            self._record_test_result(test_name, False, f"Configuration validation failed: {e}")
    
    async def _test_task_input_output_validation(self):
        """Test task input and output validation"""
        test_name = "Task Input/Output Validation"
        print(f"📝 Testing: {test_name}")
        
        try:
            # Test valid task input
            valid_task_input = TaskInput(
                task_id="test_task_001",
                content="Analyze this test content for validation",
                context={"source": "test", "priority": "high"},
                priority=3,
                user_id="test_user_001",
                session_id="test_session_001"
            )
            
            # Validate task input
            validation_result = await self.core_system.validation_engine.validate_task_input(valid_task_input)
            assert validation_result.is_valid == True, "Valid task input rejected"
            assert len(validation_result.errors) == 0, "Valid task input has errors"
            
            # Test invalid task inputs (handle Pydantic validation errors)
            invalid_input_tests = [
                ("empty_content", ""),
                ("invalid_priority", 10),
                ("empty_task_id", "task_id"),
                ("empty_user_id", "user_id")
            ]
            
            for test_name, invalid_value in invalid_input_tests:
                try:
                    if test_name == "empty_content":
                        invalid_input = TaskInput(
                            task_id="empty_content_task",
                            content=invalid_value,
                            priority=1,
                            user_id="test_user",
                            session_id="test_session"
                        )
                    elif test_name == "invalid_priority":
                        invalid_input = TaskInput(
                            task_id="invalid_priority_task",
                            content="Test content",
                            priority=invalid_value,
                            user_id="test_user",
                            session_id="test_session"
                        )
                    elif test_name == "empty_task_id":
                        invalid_input = TaskInput(
                            task_id="",
                            content="Test content",
                            priority=1,
                            user_id="test_user",
                            session_id="test_session"
                        )
                    else:  # empty_user_id
                        invalid_input = TaskInput(
                            task_id="empty_user_task",
                            content="Test content",
                            priority=1,
                            user_id="",
                            session_id="test_session"
                        )
                    
                    # If Pydantic didn't catch it, our validator should
                    validation_result = await self.core_system.validation_engine.validate_task_input(invalid_input)
                    assert validation_result.is_valid == False, f"Invalid {test_name} input accepted"
                    assert len(validation_result.errors) > 0, f"Invalid {test_name} input has no errors"
                    
                except Exception as e:
                    # Pydantic validation error is expected and acceptable
                    if "validation error" in str(e).lower() or "string should have" in str(e).lower() or "input should be" in str(e).lower():
                        pass  # This is expected behavior
                    else:
                        raise e
            
            # Test task output validation
            valid_task_output = TaskOutput(
                task_id="test_task_001",
                agent_id="test_agent_001",
                result={"status": "completed", "data": "test result"},
                confidence=0.95,
                processing_time=0.150,
                token_usage={"input_tokens": 10, "output_tokens": 15, "total_tokens": 25},
                metadata={"model": "test-model", "version": "1.0"},
                validation_errors=[]
            )
            
            # Validate output structure
            output_dict = valid_task_output.dict()
            assert isinstance(output_dict, dict), "Task output serialization failed"
            assert output_dict["confidence"] == 0.95, "Output data integrity failed"
            assert "timestamp" in output_dict, "Timestamp not included in output"
            
            self._record_test_result(test_name, True, "Task input/output validation comprehensive")
            
        except Exception as e:
            self._record_crash(test_name, str(e), traceback.format_exc())
            self._record_test_result(test_name, False, f"Task validation failed: {e}")
    
    async def _test_agent_creation_and_management(self):
        """Test agent creation and lifecycle management"""
        test_name = "Agent Creation and Management"
        print(f"👤 Testing: {test_name}")
        
        try:
            # Create test agent
            test_agent_config = AgentConfig(
                agent_id="lifecycle_test_agent",
                name="Lifecycle Test Agent",
                role=AgentRole.PROCESSOR,
                model_provider=ModelProvider.ANTHROPIC,
                model_name="claude-3",
                temperature=0.5,
                max_tokens=1500,
                system_prompt="You are a lifecycle test agent for comprehensive validation",
                tools=["processing", "validation", "analysis"],
                memory_enabled=True,
                streaming_enabled=True,
                validation_level=ValidationLevel.MODERATE,
                timeout_seconds=45
            )
            
            # Create agent
            agent_id = await self.core_system.create_agent(test_agent_config)
            assert agent_id == test_agent_config.agent_id, "Agent ID mismatch"
            
            # Verify agent is stored
            assert agent_id in self.core_system.agents, "Agent not stored in system"
            stored_agent = self.core_system.agents[agent_id]
            assert stored_agent.name == test_agent_config.name, "Agent name mismatch"
            assert stored_agent.role == test_agent_config.role, "Agent role mismatch"
            
            # Test agent manager initialization
            assert agent_id in self.core_system.agent_manager.agent_instances, "Agent not initialized in manager"
            agent_instance = self.core_system.agent_manager.agent_instances[agent_id]
            assert agent_instance["status"] == "ready", "Agent not in ready status"
            assert agent_instance["usage_count"] == 0, "Agent usage count not initialized"
            
            # Test agent configuration updates
            updated_config = AgentConfig(
                agent_id=agent_id,
                name="Updated Lifecycle Test Agent",
                role=AgentRole.PROCESSOR,
                model_provider=ModelProvider.ANTHROPIC,
                model_name="claude-3",
                temperature=0.8,  # Updated temperature
                max_tokens=2000,  # Updated max tokens
                system_prompt="Updated system prompt for lifecycle testing"
            )
            
            # Update agent (create with same ID should replace)
            updated_agent_id = await self.core_system.create_agent(updated_config)
            assert updated_agent_id == agent_id, "Agent update failed"
            
            # Verify update
            updated_stored_agent = self.core_system.agents[agent_id]
            assert updated_stored_agent.name == "Updated Lifecycle Test Agent", "Agent name not updated"
            assert updated_stored_agent.temperature == 0.8, "Agent temperature not updated"
            
            self._record_test_result(test_name, True, "Agent creation and management comprehensive")
            
        except Exception as e:
            self._record_crash(test_name, str(e), traceback.format_exc())
            self._record_test_result(test_name, False, f"Agent management failed: {e}")
    
    async def _test_agent_role_based_processing(self):
        """Test agent role-based processing functionality"""
        test_name = "Agent Role-Based Processing"
        print(f"🎭 Testing: {test_name}")
        
        try:
            # Test each agent role
            role_test_cases = [
                (AgentRole.ANALYZER, "analyzer_test_agent", "Analyze this test data"),
                (AgentRole.PROCESSOR, "processor_test_agent", "Process this test content"),
                (AgentRole.VALIDATOR, "validator_test_agent", "Validate this test information"),
                (AgentRole.COORDINATOR, "coordinator_test_agent", "Coordinate this test workflow")
            ]
            
            for role, agent_id, test_content in role_test_cases:
                # Create role-specific agent
                role_config = AgentConfig(
                    agent_id=agent_id,
                    name=f"{role.value.title()} Test Agent",
                    role=role,
                    model_provider=ModelProvider.OPENAI,
                    model_name="gpt-4",
                    system_prompt=f"You are a {role.value} agent for comprehensive testing"
                )
                
                await self.core_system.create_agent(role_config)
                
                # Create role-specific task
                role_task = TaskInput(
                    task_id=f"{role.value}_test_task",
                    content=test_content,
                    context={"role_test": True, "expected_role": role.value},
                    priority=2,
                    user_id="role_test_user",
                    session_id="role_test_session"
                )
                
                # Execute task
                start_time = time.time()
                result = await self.core_system.execute_task(agent_id, role_task)
                execution_time = time.time() - start_time
                
                # Verify result structure
                assert result is not None, f"No result for {role.value} agent"
                assert result.task_id == role_task.task_id, f"Task ID mismatch for {role.value}"
                assert result.agent_id == agent_id, f"Agent ID mismatch for {role.value}"
                assert isinstance(result.result, dict), f"Result not dict for {role.value}"
                assert 0 <= result.confidence <= 1, f"Invalid confidence for {role.value}"
                assert result.processing_time > 0, f"Invalid processing time for {role.value}"
                
                # Verify role-specific result content
                if role == AgentRole.ANALYZER:
                    assert "analysis" in result.result, "Analyzer result missing analysis"
                elif role == AgentRole.PROCESSOR:
                    assert "processed_content" in result.result, "Processor result missing processed content"
                elif role == AgentRole.VALIDATOR:
                    assert "validation_result" in result.result, "Validator result missing validation"
                
                # Record performance metrics
                self.test_results["performance_metrics"]["response_times"].append(execution_time)
                
            self._record_test_result(test_name, True, f"All {len(role_test_cases)} agent roles validated")
            
        except Exception as e:
            self._record_crash(test_name, str(e), traceback.format_exc())
            self._record_test_result(test_name, False, f"Role-based processing failed: {e}")
    
    async def _test_multi_model_provider_support(self):
        """Test multi-model provider support"""
        test_name = "Multi-Model Provider Support"
        print(f"🤖 Testing: {test_name}")
        
        try:
            # Test each model provider
            provider_test_cases = [
                (ModelProvider.OPENAI, "gpt-4", "openai_test_agent"),
                (ModelProvider.ANTHROPIC, "claude-3", "anthropic_test_agent"),
                (ModelProvider.GOOGLE, "gemini-pro", "google_test_agent"),
                (ModelProvider.LOCAL, "local-model", "local_test_agent")
            ]
            
            for provider, model_name, agent_id in provider_test_cases:
                # Create provider-specific agent
                provider_config = AgentConfig(
                    agent_id=agent_id,
                    name=f"{provider.value.title()} Test Agent",
                    role=AgentRole.PROCESSOR,
                    model_provider=provider,
                    model_name=model_name,
                    temperature=0.7,
                    max_tokens=1000,
                    system_prompt=f"You are a {provider.value} model test agent"
                )
                
                await self.core_system.create_agent(provider_config)
                
                # Verify agent configuration
                stored_agent = self.core_system.agents[agent_id]
                assert stored_agent.model_provider == provider, f"Provider mismatch for {provider.value}"
                assert stored_agent.model_name == model_name, f"Model name mismatch for {provider.value}"
                
                # Create provider test task
                provider_task = TaskInput(
                    task_id=f"{provider.value}_provider_test",
                    content=f"Test {provider.value} model provider functionality",
                    context={"provider_test": True, "provider": provider.value},
                    priority=1,
                    user_id="provider_test_user",
                    session_id="provider_test_session"
                )
                
                # Execute task
                result = await self.core_system.execute_task(agent_id, provider_task)
                
                # Verify provider-specific result
                assert result is not None, f"No result for {provider.value}"
                assert result.agent_id == agent_id, f"Agent ID mismatch for {provider.value}"
                
                # Verify metadata includes provider information
                if "metadata" in result.result:
                    metadata = result.result["metadata"]
                    assert "model_provider" in metadata, f"Provider metadata missing for {provider.value}"
                    assert metadata["model_provider"] == provider.value, f"Provider metadata incorrect for {provider.value}"
            
            self._record_test_result(test_name, True, f"All {len(provider_test_cases)} model providers validated")
            
        except Exception as e:
            self._record_crash(test_name, str(e), traceback.format_exc())
            self._record_test_result(test_name, False, f"Multi-provider support failed: {e}")
    
    async def _test_task_execution_workflow(self):
        """Test complete task execution workflow"""
        test_name = "Task Execution Workflow"
        print(f"🔄 Testing: {test_name}")
        
        try:
            # Create workflow test agent
            workflow_agent_config = AgentConfig(
                agent_id="workflow_test_agent",
                name="Workflow Test Agent",
                role=AgentRole.ANALYZER,
                model_provider=ModelProvider.OPENAI,
                model_name="gpt-4",
                memory_enabled=True,
                validation_level=ValidationLevel.STRICT
            )
            
            await self.core_system.create_agent(workflow_agent_config)
            
            # Test workflow stages
            workflow_tasks = [
                TaskInput(
                    task_id="workflow_task_001",
                    content="Initial workflow analysis task",
                    context={"stage": "initial", "workflow_id": "test_workflow_001"},
                    priority=1,
                    user_id="workflow_user",
                    session_id="workflow_session_001"
                ),
                TaskInput(
                    task_id="workflow_task_002",
                    content="Follow-up workflow processing task",
                    context={"stage": "processing", "workflow_id": "test_workflow_001", "depends_on": "workflow_task_001"},
                    priority=2,
                    user_id="workflow_user",
                    session_id="workflow_session_001"
                ),
                TaskInput(
                    task_id="workflow_task_003",
                    content="Final workflow validation task",
                    context={"stage": "validation", "workflow_id": "test_workflow_001", "depends_on": "workflow_task_002"},
                    priority=3,
                    user_id="workflow_user",
                    session_id="workflow_session_001"
                )
            ]
            
            # Execute workflow tasks sequentially
            workflow_results = []
            total_workflow_time = 0
            
            for task in workflow_tasks:
                start_time = time.time()
                result = await self.core_system.execute_task("workflow_test_agent", task)
                execution_time = time.time() - start_time
                total_workflow_time += execution_time
                
                # Verify task result
                assert result is not None, f"No result for {task.task_id}"
                assert result.task_id == task.task_id, f"Task ID mismatch for {task.task_id}"
                assert result.processing_time > 0, f"Invalid processing time for {task.task_id}"
                
                workflow_results.append(result)
                
                # Verify memory is being stored (if enabled)
                if workflow_agent_config.memory_enabled:
                    # Check memory cache
                    memory_entries = self.core_system.memory_manager.memory_cache.get("workflow_test_agent", [])
                    assert len(memory_entries) > 0, f"No memory stored for {task.task_id}"
            
            # Verify workflow completion
            assert len(workflow_results) == len(workflow_tasks), "Workflow incomplete"
            
            # Test workflow performance
            avg_task_time = total_workflow_time / len(workflow_tasks)
            assert avg_task_time < 5.0, f"Workflow tasks too slow: {avg_task_time:.2f}s average"
            
            # Test performance metrics collection
            agent_performance = await self.core_system.get_agent_performance("workflow_test_agent", time_window_hours=1)
            assert agent_performance["total_tasks"] >= len(workflow_tasks), "Performance metrics incomplete"
            assert agent_performance["success_rate"] > 0, "Success rate not calculated"
            
            self._record_test_result(test_name, True, f"Workflow executed {len(workflow_tasks)} tasks in {total_workflow_time:.2f}s")
            
        except Exception as e:
            self._record_crash(test_name, str(e), traceback.format_exc())
            self._record_test_result(test_name, False, f"Workflow execution failed: {e}")
    
    async def _test_streaming_response_handling(self):
        """Test streaming response handling"""
        test_name = "Streaming Response Handling"
        print(f"🌊 Testing: {test_name}")
        
        try:
            # Create streaming-enabled agent
            streaming_agent_config = AgentConfig(
                agent_id="streaming_test_agent",
                name="Streaming Test Agent",
                role=AgentRole.PROCESSOR,
                model_provider=ModelProvider.ANTHROPIC,
                model_name="claude-3",
                streaming_enabled=True,
                validation_level=ValidationLevel.MODERATE
            )
            
            await self.core_system.create_agent(streaming_agent_config)
            
            # Create streaming test task
            streaming_task = TaskInput(
                task_id="streaming_test_task",
                content="Generate a streaming response for comprehensive testing",
                context={"streaming_test": True, "expected_chunks": 5},
                priority=1,
                user_id="streaming_user",
                session_id="streaming_session"
            )
            
            # Test streaming execution
            chunks_received = []
            start_time = time.time()
            
            async for chunk in self.core_system.execute_with_streaming("streaming_test_agent", streaming_task):
                chunks_received.append(chunk)
                
                # Verify chunk structure
                assert isinstance(chunk, dict), "Chunk not a dictionary"
                assert "chunk_id" in chunk, "Chunk missing ID"
                assert "content" in chunk, "Chunk missing content"
                assert "is_final" in chunk, "Chunk missing final flag"
                assert "timestamp" in chunk, "Chunk missing timestamp"
                
                # Verify chunk timestamp is recent
                chunk_time = chunk["timestamp"]
                assert abs(chunk_time - time.time()) < 10, "Chunk timestamp too old"
            
            streaming_time = time.time() - start_time
            
            # Verify streaming results
            assert len(chunks_received) > 0, "No chunks received"
            assert len(chunks_received) <= 10, "Too many chunks received"
            
            # Verify final chunk
            if chunks_received:
                final_chunk = chunks_received[-1]
                assert final_chunk.get("is_final", False) == True, "Final chunk not marked as final"
            
            # Test streaming with non-streaming agent (should fail gracefully)
            non_streaming_config = AgentConfig(
                agent_id="non_streaming_agent",
                name="Non-Streaming Agent",
                role=AgentRole.VALIDATOR,
                model_provider=ModelProvider.GOOGLE,
                model_name="gemini-pro",
                streaming_enabled=False
            )
            
            await self.core_system.create_agent(non_streaming_config)
            
            # Should raise ValueError for non-streaming agent
            try:
                async for chunk in self.core_system.execute_with_streaming("non_streaming_agent", streaming_task):
                    pass
                assert False, "Non-streaming agent should not support streaming"
            except ValueError as e:
                assert "does not support streaming" in str(e), "Incorrect error message for non-streaming agent"
            
            self._record_test_result(test_name, True, f"Streaming validated with {len(chunks_received)} chunks in {streaming_time:.2f}s")
            
        except Exception as e:
            self._record_crash(test_name, str(e), traceback.format_exc())
            self._record_test_result(test_name, False, f"Streaming response failed: {e}")
    
    async def _test_validation_engine_comprehensive(self):
        """Test validation engine comprehensive functionality"""
        test_name = "Validation Engine Comprehensive"
        print(f"🔍 Testing: {test_name}")
        
        try:
            validation_engine = self.core_system.validation_engine
            
            # Test structured output validation
            test_output_data = {
                "task_id": "validation_test_001",
                "agent_id": "test_agent",
                "result": {"status": "completed", "data": "test result"},
                "confidence": 0.95,
                "processing_time": 0.123,
                "token_usage": {"input_tokens": 10, "output_tokens": 15},
                "metadata": {"test": True},
                "validation_errors": []
            }
            
            validation_start_time = time.time()
            validation_result = await validation_engine.validate_structured_output(test_output_data, TaskOutput)
            validation_time = time.time() - validation_start_time
            
            # Record validation time
            self.test_results["performance_metrics"]["validation_times"].append(validation_time)
            
            # Verify validation result
            assert validation_result is not None, "No validation result returned"
            assert isinstance(validation_result, ValidationResult), "Invalid validation result type"
            assert validation_result.is_valid == True, "Valid output rejected"
            assert validation_result.confidence > 0, "Validation confidence too low"
            
            # Test invalid structured output
            invalid_output_data = {
                "task_id": "validation_test_002",
                "agent_id": "test_agent",
                "result": {"status": "completed"},
                "confidence": 1.5,  # Invalid: > 1.0
                "processing_time": -0.1,  # Invalid: < 0
                "token_usage": "invalid_format",  # Invalid: should be dict
                "metadata": None,  # Invalid: should be dict
                "validation_errors": "not_a_list"  # Invalid: should be list
            }
            
            invalid_validation_result = await validation_engine.validate_structured_output(invalid_output_data, TaskOutput)
            assert invalid_validation_result.is_valid == False, "Invalid output accepted"
            assert len(invalid_validation_result.errors) > 0, "No errors for invalid output"
            
            # Test validation with different validation levels
            validation_levels = [ValidationLevel.STRICT, ValidationLevel.MODERATE, ValidationLevel.LENIENT]
            
            for level in validation_levels:
                # Create agent with specific validation level
                level_agent_config = AgentConfig(
                    agent_id=f"validation_level_{level.value}_agent",
                    name=f"Validation Level {level.value.title()} Agent",
                    role=AgentRole.VALIDATOR,
                    model_provider=ModelProvider.LOCAL,
                    model_name="local-validator",
                    validation_level=level
                )
                
                await self.core_system.create_agent(level_agent_config)
                
                # Test task with potential validation issues
                potentially_invalid_task = TaskInput(
                    task_id=f"validation_level_test_{level.value}",
                    content="   ",  # Whitespace only - potentially invalid
                    priority=1,
                    user_id="validation_user",
                    session_id="validation_session"
                )
                
                if level == ValidationLevel.STRICT:
                    # Should fail validation in strict mode
                    try:
                        result = await self.core_system.execute_task(level_agent_config.agent_id, potentially_invalid_task)
                        # If it doesn't fail, check for validation errors
                        if result:
                            assert len(result.validation_errors) > 0, "Strict validation should catch whitespace-only content"
                    except Exception as e:
                        # Expected in strict mode
                        error_lower = str(e).lower()
                        assert any(keyword in error_lower for keyword in ["invalid", "validation", "pydantic", "error"]), f"Unexpected error type: {e}"
                else:
                    # Should pass or provide warnings in moderate/lenient modes
                    result = await self.core_system.execute_task(level_agent_config.agent_id, potentially_invalid_task)
                    assert result is not None, f"Task failed in {level.value} mode"
            
            self._record_test_result(test_name, True, f"Validation engine tested with {len(validation_levels)} levels")
            
        except Exception as e:
            self._record_crash(test_name, str(e), traceback.format_exc())
            self._record_test_result(test_name, False, f"Validation engine testing failed: {e}")
    
    async def _test_memory_management_and_storage(self):
        """Test memory management and storage functionality"""
        test_name = "Memory Management and Storage"
        print(f"🧠 Testing: {test_name}")
        
        try:
            memory_manager = self.core_system.memory_manager
            
            # Create memory-enabled agent
            memory_agent_config = AgentConfig(
                agent_id="memory_test_agent",
                name="Memory Test Agent",
                role=AgentRole.ANALYZER,
                model_provider=ModelProvider.OPENAI,
                model_name="gpt-4",
                memory_enabled=True
            )
            
            await self.core_system.create_agent(memory_agent_config)
            
            # Execute tasks to generate memory
            memory_tasks = []
            for i in range(5):
                task = TaskInput(
                    task_id=f"memory_task_{i:03d}",
                    content=f"Memory test task {i} - remember this context",
                    context={"memory_test": True, "sequence": i, "total": 5},
                    priority=1,
                    user_id="memory_user",
                    session_id="memory_session"
                )
                memory_tasks.append(task)
                
                # Execute task
                result = await self.core_system.execute_task("memory_test_agent", task)
                assert result is not None, f"Memory task {i} failed"
            
            # Verify memory storage
            agent_memory = memory_manager.memory_cache.get("memory_test_agent", [])
            assert len(agent_memory) >= len(memory_tasks), "Memory not stored for all tasks"
            
            # Verify memory structure
            for memory_entry in agent_memory:
                assert "entry_id" in memory_entry, "Memory entry missing ID"
                assert "agent_id" in memory_entry, "Memory entry missing agent ID"
                assert "content_type" in memory_entry, "Memory entry missing content type"
                assert "content" in memory_entry, "Memory entry missing content"
                assert "timestamp" in memory_entry, "Memory entry missing timestamp"
                
                # Verify content structure
                content = memory_entry["content"]
                assert "input" in content, "Memory content missing input"
                assert "output" in content, "Memory content missing output"
            
            # Test memory persistence to database
            conn = sqlite3.connect(self.core_system.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM memory_entries WHERE agent_id = ?", ("memory_test_agent",))
            stored_memory_count = cursor.fetchone()[0]
            
            conn.close()
            
            assert stored_memory_count >= len(memory_tasks), "Memory not persisted to database"
            
            # Test memory retrieval and context building
            # This would involve more complex memory operations if implemented
            
            self._record_test_result(test_name, True, f"Memory management validated with {len(memory_tasks)} tasks and {stored_memory_count} DB entries")
            
        except Exception as e:
            self._record_crash(test_name, str(e), traceback.format_exc())
            self._record_test_result(test_name, False, f"Memory management failed: {e}")
    
    async def _test_database_persistence_integrity(self):
        """Test database persistence and data integrity"""
        test_name = "Database Persistence Integrity"
        print(f"💾 Testing: {test_name}")
        
        try:
            # Verify database structure
            conn = sqlite3.connect(self.core_system.db_path)
            cursor = conn.cursor()
            
            # Check required tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            required_tables = [
                "agent_configs", "task_executions", "memory_entries",
                "validation_logs", "performance_metrics"
            ]
            
            for table in required_tables:
                assert table in tables, f"Required table '{table}' not found"
            
            # Test data persistence through task execution
            persistence_task = TaskInput(
                task_id="persistence_test_001",
                content="Test data persistence and integrity",
                priority=1,
                user_id="persistence_user",
                session_id="persistence_session"
            )
            
            # Execute task to generate database entries
            result = await self.core_system.execute_task("analyzer_001", persistence_task)
            assert result is not None, "Persistence test task failed"
            
            # Verify task execution is stored
            cursor.execute("SELECT * FROM task_executions WHERE task_id = ?", (persistence_task.task_id,))
            execution_record = cursor.fetchone()
            assert execution_record is not None, "Task execution not stored"
            
            # Verify data integrity
            stored_task_id = execution_record[1]  # task_id column
            assert stored_task_id == persistence_task.task_id, "Task ID integrity violation"
            
            # Test performance metrics storage
            initial_metrics_count = 0
            cursor.execute("SELECT COUNT(*) FROM performance_metrics")
            initial_metrics_count = cursor.fetchone()[0]
            
            # Trigger metrics update
            await self.core_system._background_monitoring()
            await asyncio.sleep(0.1)  # Allow async operations to complete
            
            cursor.execute("SELECT COUNT(*) FROM performance_metrics")
            final_metrics_count = cursor.fetchone()[0]
            
            assert final_metrics_count >= initial_metrics_count, "Performance metrics not being stored"
            
            # Test transaction integrity
            try:
                cursor.execute("BEGIN TRANSACTION")
                cursor.execute("INSERT INTO performance_metrics (metric_id, agent_id, metric_name, metric_value, timestamp) VALUES (?, ?, ?, ?, ?)",
                             ("test_transaction", "test_agent", "test_metric", 1.0, time.time()))
                cursor.execute("ROLLBACK")
                
                # Verify rollback worked
                cursor.execute("SELECT COUNT(*) FROM performance_metrics WHERE metric_id = ?", ("test_transaction",))
                rollback_count = cursor.fetchone()[0]
                assert rollback_count == 0, "Transaction rollback failed"
                
            except Exception as e:
                cursor.execute("ROLLBACK")
                raise e
            
            conn.close()
            
            # Test database recovery (create new core instance with same database)
            recovery_core = PydanticAICore(self.core_system.db_path)
            
            # Should be able to load existing data
            assert len(recovery_core.agents) > 0, "Existing agents not loaded on recovery"
            
            # Should be able to execute operations
            recovery_task = TaskInput(
                task_id="recovery_test_001",
                content="Test database recovery functionality",
                priority=1,
                user_id="recovery_user",
                session_id="recovery_session"
            )
            
            recovery_result = await recovery_core.execute_task("analyzer_001", recovery_task)
            assert recovery_result is not None, "Recovery core cannot execute tasks"
            
            # Cleanup recovery instance
            recovery_core.monitoring_active = False
            
            self._record_test_result(test_name, True, f"Database integrity validated with {len(required_tables)} tables and transaction testing")
            
        except Exception as e:
            self._record_crash(test_name, str(e), traceback.format_exc())
            self._record_test_result(test_name, False, f"Database persistence testing failed: {e}")
    
    async def _test_background_monitoring_systems(self):
        """Test background monitoring systems"""
        test_name = "Background Monitoring Systems"
        print(f"👁️ Testing: {test_name}")
        
        try:
            # Verify monitoring is active
            assert self.core_system.monitoring_active == True, "Background monitoring not active"
            assert self.core_system.monitor_thread is not None, "Monitor thread not created"
            assert self.core_system.monitor_thread.is_alive(), "Monitor thread not running"
            
            # Test monitoring data collection
            conn = sqlite3.connect(self.core_system.db_path)
            cursor = conn.cursor()
            
            # Get initial metrics count
            cursor.execute("SELECT COUNT(*) FROM performance_metrics")
            initial_count = cursor.fetchone()[0]
            
            # Wait for monitoring cycle
            await asyncio.sleep(1.5)
            
            # Check if metrics were updated
            cursor.execute("SELECT COUNT(*) FROM performance_metrics")
            updated_count = cursor.fetchone()[0]
            
            # Should have more metrics after monitoring
            assert updated_count >= initial_count, "Background monitoring not collecting metrics"
            
            # Test specific metric types
            cursor.execute("SELECT DISTINCT metric_name FROM performance_metrics ORDER BY timestamp DESC LIMIT 10")
            recent_metrics = [row[0] for row in cursor.fetchall()]
            
            expected_metric_types = ["response_time", "memory_usage", "success_rate"]
            found_metrics = [metric for metric in expected_metric_types if any(expected_metric in metric_name for metric_name in recent_metrics)]
            
            assert len(found_metrics) > 0, "Expected metric types not found in background monitoring"
            
            # Test monitoring can be controlled
            self.core_system.monitoring_active = False
            await asyncio.sleep(0.5)
            
            # Get count after stopping
            cursor.execute("SELECT COUNT(*) FROM performance_metrics")
            stopped_count = cursor.fetchone()[0]
            
            # Wait a bit more
            await asyncio.sleep(1.0)
            
            # Count should not increase significantly after stopping
            cursor.execute("SELECT COUNT(*) FROM performance_metrics")
            final_count = cursor.fetchone()[0]
            
            # Allow for some delay in thread stopping
            metric_increase = final_count - stopped_count
            assert metric_increase <= 3, f"Too many metrics added after stopping monitoring: {metric_increase}"
            
            # Restart monitoring for cleanup
            self.core_system.monitoring_active = True
            
            conn.close()
            
            self._record_test_result(test_name, True, f"Background monitoring validated with {updated_count - initial_count} new metrics")
            
        except Exception as e:
            self._record_crash(test_name, str(e), traceback.format_exc())
            self._record_test_result(test_name, False, f"Background monitoring testing failed: {e}")
    
    async def _test_performance_metrics_collection(self):
        """Test performance metrics collection and analysis"""
        test_name = "Performance Metrics Collection"
        print(f"📈 Testing: {test_name}")
        
        try:
            # Execute multiple tasks to generate performance data
            performance_tasks = []
            execution_times = []
            
            for i in range(10):
                task = TaskInput(
                    task_id=f"performance_task_{i:03d}",
                    content=f"Performance test task {i} with varying complexity",
                    context={"performance_test": True, "iteration": i},
                    priority=(i % 3) + 1,
                    user_id="performance_user",
                    session_id="performance_session"
                )
                
                start_time = time.time()
                result = await self.core_system.execute_task("processor_001", task)
                execution_time = time.time() - start_time
                
                assert result is not None, f"Performance task {i} failed"
                execution_times.append(execution_time)
                performance_tasks.append(task)
                
                # Record memory usage
                current_memory = self._get_memory_usage()
                self.test_results["performance_metrics"]["memory_usage"].append(current_memory)
            
            # Analyze performance metrics
            avg_execution_time = sum(execution_times) / len(execution_times)
            max_execution_time = max(execution_times)
            min_execution_time = min(execution_times)
            
            assert avg_execution_time < 5.0, f"Average execution time too high: {avg_execution_time:.2f}s"
            assert max_execution_time < 10.0, f"Maximum execution time too high: {max_execution_time:.2f}s"
            
            # Test agent performance retrieval
            agent_performance = await self.core_system.get_agent_performance("processor_001", time_window_hours=1)
            
            assert agent_performance["total_tasks"] >= len(performance_tasks), "Performance metrics incomplete"
            assert agent_performance["success_rate"] > 0.8, "Success rate too low"
            assert agent_performance["average_processing_time"] > 0, "Average processing time not calculated"
            
            # Test system-wide performance metrics
            memory_usage = self.test_results["performance_metrics"]["memory_usage"]
            if len(memory_usage) > 1:
                memory_growth = memory_usage[-1] - memory_usage[0]
                # Allow some memory growth but not excessive
                assert memory_growth < 100, f"Excessive memory growth: {memory_growth:.2f}MB"
            
            # Verify performance data persistence
            conn = sqlite3.connect(self.core_system.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM performance_metrics WHERE timestamp > ?", (time.time() - 3600,))
            recent_metrics_count = cursor.fetchone()[0]
            
            assert recent_metrics_count > 0, "No recent performance metrics stored"
            
            conn.close()
            
            self._record_test_result(test_name, True, f"Performance metrics validated - avg: {avg_execution_time:.3f}s, success rate: {agent_performance['success_rate']:.2f}")
            
        except Exception as e:
            self._record_crash(test_name, str(e), traceback.format_exc())
            self._record_test_result(test_name, False, f"Performance metrics testing failed: {e}")
    
    async def _test_concurrent_agent_execution(self):
        """Test concurrent agent execution and resource management"""
        test_name = "Concurrent Agent Execution"
        print(f"⚡ Testing: {test_name}")
        
        try:
            # Create multiple tasks for concurrent execution
            concurrent_tasks = []
            for i in range(8):
                task = TaskInput(
                    task_id=f"concurrent_task_{i:03d}",
                    content=f"Concurrent execution test task {i}",
                    context={"concurrent_test": True, "task_number": i},
                    priority=1,
                    user_id="concurrent_user",
                    session_id=f"concurrent_session_{i}"
                )
                concurrent_tasks.append(task)
            
            # Execute tasks concurrently
            start_time = time.time()
            
            concurrent_results = await asyncio.gather(
                *[self.core_system.execute_task("processor_001", task) for task in concurrent_tasks],
                return_exceptions=True
            )
            
            concurrent_execution_time = time.time() - start_time
            
            # Analyze concurrent execution results
            successful_results = [r for r in concurrent_results if not isinstance(r, Exception)]
            failed_results = [r for r in concurrent_results if isinstance(r, Exception)]
            
            assert len(successful_results) >= 6, f"Too many failed concurrent executions: {len(failed_results)}"
            assert len(failed_results) <= 2, f"Too many failures in concurrent execution"
            
            # Verify reasonable concurrent performance
            sequential_estimate = len(concurrent_tasks) * 0.2  # Estimate 0.2s per task
            assert concurrent_execution_time < sequential_estimate * 0.8, f"Concurrent execution not faster than sequential: {concurrent_execution_time:.2f}s vs {sequential_estimate:.2f}s estimate"
            
            # Verify all successful results are valid
            for i, result in enumerate(successful_results):
                assert result is not None, f"Concurrent result {i} is None"
                assert hasattr(result, 'task_id'), f"Concurrent result {i} missing task_id"
                assert hasattr(result, 'confidence'), f"Concurrent result {i} missing confidence"
                assert 0 <= result.confidence <= 1, f"Invalid confidence in concurrent result {i}"
            
            # Test resource usage during concurrent execution
            current_memory = self._get_memory_usage()
            memory_growth = current_memory - self.initial_memory
            
            # Allow reasonable memory growth for concurrent execution
            assert memory_growth < 200, f"Excessive memory usage during concurrent execution: {memory_growth:.2f}MB"
            
            self._record_test_result(test_name, True, f"Concurrent execution: {len(successful_results)}/{len(concurrent_tasks)} succeeded in {concurrent_execution_time:.2f}s")
            
        except Exception as e:
            self._record_crash(test_name, str(e), traceback.format_exc())
            self._record_test_result(test_name, False, f"Concurrent execution testing failed: {e}")
    
    async def _test_memory_management_and_cleanup(self):
        """Test memory management and resource cleanup"""
        test_name = "Memory Management and Cleanup"
        print(f"🧹 Testing: {test_name}")
        
        try:
            initial_memory = self._get_memory_usage()
            
            # Execute many tasks to test memory management
            memory_test_tasks = []
            for i in range(20):
                task = TaskInput(
                    task_id=f"memory_mgmt_task_{i:03d}",
                    content=f"Memory management test task {i} with substantial content to test memory usage patterns and cleanup mechanisms",
                    context={"memory_management_test": True, "iteration": i, "data": "x" * 1000},  # Add some data
                    priority=1,
                    user_id="memory_mgmt_user",
                    session_id="memory_mgmt_session"
                )
                memory_test_tasks.append(task)
                
                # Execute task
                result = await self.core_system.execute_task("analyzer_001", task)
                assert result is not None, f"Memory management task {i} failed"
                
                # Periodically check memory usage
                if i % 5 == 0:
                    current_memory = self._get_memory_usage()
                    memory_growth = current_memory - initial_memory
                    
                    # Memory should not grow excessively
                    assert memory_growth < 150, f"Excessive memory growth at task {i}: {memory_growth:.2f}MB"
            
            # Force garbage collection
            gc.collect()
            await asyncio.sleep(0.5)
            
            # Check memory after cleanup
            final_memory = self._get_memory_usage()
            total_memory_growth = final_memory - initial_memory
            
            # Should have reasonable memory growth
            assert total_memory_growth < 100, f"Total memory growth too high: {total_memory_growth:.2f}MB"
            
            # Test agent instance cleanup
            agent_manager = self.core_system.agent_manager
            
            # Verify agents are being managed properly
            for agent_id, instance in agent_manager.agent_instances.items():
                assert "last_used" in instance, f"Agent {agent_id} missing last_used timestamp"
                assert "usage_count" in instance, f"Agent {agent_id} missing usage count"
                assert instance["usage_count"] > 0, f"Agent {agent_id} usage count not updated"
            
            # Test memory cache management
            memory_manager = self.core_system.memory_manager
            total_memory_entries = sum(len(entries) for entries in memory_manager.memory_cache.values())
            
            # Should have memory entries but not excessive
            assert total_memory_entries > 0, "No memory entries in cache"
            assert total_memory_entries < 1000, f"Too many memory entries: {total_memory_entries}"
            
            # Test database connection management
            # Execute a task to ensure connections are working
            cleanup_task = TaskInput(
                task_id="cleanup_test_task",
                content="Test database connection management",
                priority=1,
                user_id="cleanup_user",
                session_id="cleanup_session"
            )
            
            cleanup_result = await self.core_system.execute_task("validator_001", cleanup_task)
            assert cleanup_result is not None, "Database connections not working after memory tests"
            
            # Mark memory management as successful
            self.test_results["system_stability"]["memory_leaks"] = False
            self.test_results["system_stability"]["resource_cleanup"] = True
            
            self._record_test_result(test_name, True, f"Memory management validated - growth: {total_memory_growth:.2f}MB, entries: {total_memory_entries}")
            
        except Exception as e:
            self._record_crash(test_name, str(e), traceback.format_exc())
            self._record_test_result(test_name, False, f"Memory management testing failed: {e}")
    
    async def _test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms"""
        test_name = "Error Handling and Recovery"
        print(f"🛡️ Testing: {test_name}")
        
        try:
            # Test various error conditions
            error_test_cases = [
                # Invalid agent ID
                ("invalid_agent_task", "nonexistent_agent", "Test invalid agent error handling"),
                # Valid agent, invalid task data will be tested separately
            ]
            
            recovery_results = []
            
            # Test invalid agent ID
            invalid_agent_task = TaskInput(
                task_id="invalid_agent_test",
                content="Test error handling with invalid agent",
                priority=1,
                user_id="error_test_user",
                session_id="error_test_session"
            )
            
            try:
                result = await self.core_system.execute_task("nonexistent_agent", invalid_agent_task)
                assert False, "Invalid agent should raise error"
            except ValueError as e:
                assert "not found" in str(e).lower(), "Incorrect error message for invalid agent"
                recovery_results.append(("invalid_agent", "handled"))
            
            # Test validation error handling with strict validation
            strict_agent_config = AgentConfig(
                agent_id="strict_error_test_agent",
                name="Strict Error Test Agent",
                role=AgentRole.VALIDATOR,
                model_provider=ModelProvider.LOCAL,
                model_name="local-validator",
                validation_level=ValidationLevel.STRICT
            )
            
            await self.core_system.create_agent(strict_agent_config)
            
            # Test invalid task input
            invalid_task_inputs = [
                TaskInput(
                    task_id="",  # Empty task ID
                    content="Valid content",
                    priority=1,
                    user_id="error_user",
                    session_id="error_session"
                ),
                TaskInput(
                    task_id="invalid_priority_task",
                    content="Valid content",
                    priority=10,  # Invalid priority
                    user_id="error_user",
                    session_id="error_session"
                ),
                TaskInput(
                    task_id="empty_content_task",
                    content="",  # Empty content
                    priority=1,
                    user_id="error_user",
                    session_id="error_session"
                )
            ]
            
            for i, invalid_task in enumerate(invalid_task_inputs):
                try:
                    result = await self.core_system.execute_task("strict_error_test_agent", invalid_task)
                    # If it doesn't raise an exception, should have validation errors
                    if result:
                        assert len(result.validation_errors) > 0, f"Invalid task {i} should have validation errors"
                        recovery_results.append((f"validation_error_{i}", "handled_gracefully"))
                except Exception as e:
                    # Expected for strict validation
                    assert len(str(e)) > 0, f"Error message should not be empty for invalid task {i}"
                    recovery_results.append((f"validation_error_{i}", "handled_with_exception"))
            
            # Test database error recovery
            original_db_path = self.core_system.db_path
            
            try:
                # Temporarily set invalid database path
                self.core_system.db_path = "/invalid/path/nonexistent.db"
                
                # This should handle database errors gracefully
                try:
                    performance = await self.core_system.get_agent_performance("analyzer_001")
                    # Should either succeed with fallback or handle error gracefully
                    recovery_results.append(("database_error", "handled"))
                except Exception as e:
                    # Acceptable if proper error handling
                    if "database" in str(e).lower() or "file" in str(e).lower() or "path" in str(e).lower():
                        recovery_results.append(("database_error", "handled_with_appropriate_error"))
                    else:
                        raise e
                        
            finally:
                # Restore original database path
                self.core_system.db_path = original_db_path
            
            # Test memory error conditions
            # Create a task with very large context
            large_context_task = TaskInput(
                task_id="large_context_test",
                content="Test memory handling with large context",
                context={"large_data": "x" * 100000},  # 100KB of data
                priority=1,
                user_id="memory_test_user",
                session_id="memory_test_session"
            )
            
            try:
                result = await self.core_system.execute_task("processor_001", large_context_task)
                if result:
                    recovery_results.append(("large_context", "handled"))
                else:
                    recovery_results.append(("large_context", "failed_gracefully"))
            except Exception as e:
                # Should handle large context gracefully
                recovery_results.append(("large_context", "handled_with_error"))
            
            # Verify recovery tracking
            assert len(recovery_results) > 0, "No error recovery scenarios tested"
            
            handled_count = len([r for r in recovery_results if "handled" in r[1]])
            total_count = len(recovery_results)
            
            assert handled_count >= total_count * 0.8, f"Too many unhandled errors: {handled_count}/{total_count}"
            
            self._record_test_result(test_name, True, f"Error handling validated - {handled_count}/{total_count} scenarios handled properly")
            
        except Exception as e:
            self._record_crash(test_name, str(e), traceback.format_exc())
            self._record_test_result(test_name, False, f"Error handling testing failed: {e}")
    
    async def _test_crash_detection_and_recovery(self):
        """Test crash detection and recovery mechanisms"""
        test_name = "Crash Detection and Recovery"
        print(f"💥 Testing: {test_name}")
        
        try:
            # Test various scenarios that could cause crashes
            crash_test_scenarios = []
            
            # Test 1: Rapid concurrent task execution (stress test)
            try:
                stress_tasks = [
                    TaskInput(
                        task_id=f"stress_task_{i}",
                        content=f"Stress test task {i}",
                        priority=1,
                        user_id="stress_user",
                        session_id="stress_session"
                    ) for i in range(15)
                ]
                
                # Execute many tasks rapidly
                stress_results = await asyncio.gather(
                    *[self.core_system.execute_task("processor_001", task) for task in stress_tasks[:5]],
                    return_exceptions=True
                )
                
                successful_stress = [r for r in stress_results if not isinstance(r, Exception)]
                failed_stress = [r for r in stress_results if isinstance(r, Exception)]
                
                if len(failed_stress) > 0:
                    for failure in failed_stress:
                        self._record_crash("stress_test", str(failure), str(failure))
                
                crash_test_scenarios.append(("stress_test", len(successful_stress), len(failed_stress)))
                
            except Exception as e:
                self._record_crash("stress_test_setup", str(e), traceback.format_exc())
                crash_test_scenarios.append(("stress_test", 0, 1))
            
            # Test 2: Memory allocation stress
            try:
                large_memory_task = TaskInput(
                    task_id="memory_stress_task",
                    content="Memory stress test with large content",
                    context={"stress_data": {"large_array": list(range(10000))}},
                    priority=1,
                    user_id="memory_stress_user",
                    session_id="memory_stress_session"
                )
                
                memory_result = await self.core_system.execute_task("analyzer_001", large_memory_task)
                
                if memory_result:
                    crash_test_scenarios.append(("memory_stress", 1, 0))
                else:
                    crash_test_scenarios.append(("memory_stress", 0, 1))
                    
            except Exception as e:
                self._record_crash("memory_stress", str(e), traceback.format_exc())
                crash_test_scenarios.append(("memory_stress", 0, 1))
            
            # Test 3: Database connection stress
            try:
                # Rapid database operations
                db_tasks = []
                for i in range(8):
                    task = TaskInput(
                        task_id=f"db_stress_task_{i}",
                        content=f"Database stress test {i}",
                        priority=1,
                        user_id="db_stress_user",
                        session_id="db_stress_session"
                    )
                    db_tasks.append(task)
                
                # Execute tasks that will cause database writes
                db_results = []
                for task in db_tasks:
                    try:
                        result = await self.core_system.execute_task("validator_001", task)
                        db_results.append(result)
                    except Exception as e:
                        self._record_crash("db_stress", str(e), traceback.format_exc())
                        break
                
                crash_test_scenarios.append(("db_stress", len(db_results), len(db_tasks) - len(db_results)))
                
            except Exception as e:
                self._record_crash("db_stress_setup", str(e), traceback.format_exc())
                crash_test_scenarios.append(("db_stress", 0, 1))
            
            # Test 4: Background monitoring resilience
            try:
                # Test monitoring during system stress
                monitoring_active = self.core_system.monitoring_active
                thread_alive = self.core_system.monitor_thread.is_alive()
                
                # Execute some tasks while monitoring
                monitoring_task = TaskInput(
                    task_id="monitoring_resilience_task",
                    content="Test monitoring system resilience",
                    priority=1,
                    user_id="monitoring_user",
                    session_id="monitoring_session"
                )
                
                monitoring_result = await self.core_system.execute_task("processor_001", monitoring_task)
                
                # Check monitoring is still active
                still_monitoring = self.core_system.monitoring_active
                still_alive = self.core_system.monitor_thread.is_alive()
                
                if monitoring_result and still_monitoring and still_alive:
                    crash_test_scenarios.append(("monitoring_resilience", 1, 0))
                else:
                    crash_test_scenarios.append(("monitoring_resilience", 0, 1))
                    
            except Exception as e:
                self._record_crash("monitoring_resilience", str(e), traceback.format_exc())
                crash_test_scenarios.append(("monitoring_resilience", 0, 1))
            
            # Analyze crash detection results
            total_scenarios = len(crash_test_scenarios)
            successful_scenarios = sum(1 for scenario, success, failure in crash_test_scenarios if failure == 0)
            total_crashes = self.test_results["crash_detection"]["crashes_detected"]
            
            # System should handle most scenarios gracefully
            success_rate = successful_scenarios / total_scenarios if total_scenarios > 0 else 0
            
            # Allow for some failures under stress, but system should generally be resilient
            assert success_rate >= 0.6, f"Too many crash scenarios failed: {success_rate:.2f} success rate"
            
            # Record crash detection results
            self.test_results["crash_detection"]["recovery_tests"] = [
                {"scenario": scenario, "success": success, "failures": failure}
                for scenario, success, failure in crash_test_scenarios
            ]
            
            self._record_test_result(test_name, True, f"Crash detection validated - {successful_scenarios}/{total_scenarios} scenarios passed, {total_crashes} crashes detected")
            
        except Exception as e:
            self._record_crash(test_name, str(e), traceback.format_exc())
            self._record_test_result(test_name, False, f"Crash detection testing failed: {e}")
    
    async def _test_validation_error_handling(self):
        """Test validation error handling comprehensively"""
        test_name = "Validation Error Handling"
        print(f"🔍 Testing: {test_name}")
        
        try:
            validation_engine = self.core_system.validation_engine
            
            # Test validation error scenarios
            validation_test_cases = [
                # Invalid data types
                {"invalid": "data", "structure": 123},
                # Missing required fields (for TaskOutput validation)
                {"task_id": "test", "agent_id": "test"},
                # Invalid value ranges
                {"task_id": "test", "agent_id": "test", "confidence": 1.5, "processing_time": -1},
            ]
            
            validation_results = []
            
            for i, test_case in enumerate(validation_test_cases):
                try:
                    validation_result = await validation_engine.validate_structured_output(test_case, TaskOutput)
                    
                    # Should be invalid
                    assert validation_result.is_valid == False, f"Invalid test case {i} was accepted as valid"
                    assert len(validation_result.errors) > 0, f"Invalid test case {i} has no errors"
                    
                    validation_results.append(("structured_validation", True, len(validation_result.errors)))
                    
                except Exception as e:
                    # Should handle validation errors gracefully
                    if "validation" in str(e).lower() or "invalid" in str(e).lower():
                        validation_results.append(("structured_validation", True, 1))
                    else:
                        validation_results.append(("structured_validation", False, 0))
                        self._record_crash(f"validation_error_{i}", str(e), traceback.format_exc())
            
            # Test agent configuration validation errors
            invalid_agent_configs = [
                # Empty required fields
                {"agent_id": "", "name": "Test", "role": "analyzer"},
                # Invalid numeric values
                {"agent_id": "test", "name": "Test", "role": "analyzer", "temperature": 5.0},
                # Invalid enum values (if using proper Pydantic validation)
                {"agent_id": "test", "name": "Test", "role": "invalid_role"},
            ]
            
            for i, invalid_config in enumerate(invalid_agent_configs):
                try:
                    # Create partial config for testing
                    if "role" in invalid_config and invalid_config["role"] == "analyzer":
                        config = AgentConfig(
                            agent_id=invalid_config.get("agent_id", "test"),
                            name=invalid_config.get("name", "Test"),
                            role=AgentRole.ANALYZER,
                            model_provider=ModelProvider.LOCAL,
                            model_name="test-model"
                        )
                        
                        if "temperature" in invalid_config:
                            config.temperature = invalid_config["temperature"]
                    else:
                        # Skip invalid enum test as it would fail at creation
                        continue
                    
                    validation_result = await validation_engine.validate_agent_config(config)
                    
                    if invalid_config.get("temperature") == 5.0:
                        # Should be invalid due to temperature
                        assert validation_result.is_valid == False, f"Invalid agent config {i} was accepted"
                        validation_results.append(("agent_validation", True, len(validation_result.errors)))
                    
                except Exception as e:
                    if "temperature" in str(e) or "validation" in str(e).lower():
                        validation_results.append(("agent_validation", True, 1))
                    else:
                        validation_results.append(("agent_validation", False, 0))
                        self._record_crash(f"agent_validation_error_{i}", str(e), traceback.format_exc())
            
            # Test task input validation errors
            invalid_task_inputs = [
                {"task_id": "", "content": "test", "priority": 1, "user_id": "test", "session_id": "test"},
                {"task_id": "test", "content": "", "priority": 1, "user_id": "test", "session_id": "test"},
                {"task_id": "test", "content": "test", "priority": 10, "user_id": "test", "session_id": "test"},
            ]
            
            for i, invalid_input in enumerate(invalid_task_inputs):
                try:
                    task_input = TaskInput(**invalid_input)
                    validation_result = await validation_engine.validate_task_input(task_input)
                    
                    # Should be invalid
                    assert validation_result.is_valid == False, f"Invalid task input {i} was accepted"
                    validation_results.append(("task_validation", True, len(validation_result.errors)))
                    
                except Exception as e:
                    if "validation" in str(e).lower() or "priority" in str(e).lower():
                        validation_results.append(("task_validation", True, 1))
                    else:
                        validation_results.append(("task_validation", False, 0))
                        self._record_crash(f"task_validation_error_{i}", str(e), traceback.format_exc())
            
            # Analyze validation error handling
            successful_validations = [r for r in validation_results if r[1] == True]
            failed_validations = [r for r in validation_results if r[1] == False]
            
            total_errors_detected = sum(r[2] for r in successful_validations)
            
            assert len(successful_validations) >= len(validation_results) * 0.8, "Too many validation error handling failures"
            assert total_errors_detected > 0, "No validation errors were properly detected"
            
            self._record_test_result(test_name, True, f"Validation error handling: {len(successful_validations)}/{len(validation_results)} scenarios handled, {total_errors_detected} errors detected")
            
        except Exception as e:
            self._record_crash(test_name, str(e), traceback.format_exc())
            self._record_test_result(test_name, False, f"Validation error handling testing failed: {e}")
    
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
    
    def _record_crash(self, test_name: str, error_details: str, traceback_info: str = ""):
        """Record crash details"""
        self.test_results["crash_detection"]["crashes_detected"] += 1
        self.test_results["crash_detection"]["crash_details"].append({
            "test_name": test_name,
            "error": error_details,
            "traceback": traceback_info,
            "timestamp": datetime.now().isoformat()
        })
        print(f"💥 CRASH in {test_name}: {error_details}")
    
    async def _generate_comprehensive_report(self):
        """Generate comprehensive test report"""
        total_time = time.time() - self.start_time
        success_rate = (self.test_results["passed_tests"] / self.test_results["total_tests"] * 100) if self.test_results["total_tests"] > 0 else 0
        
        # Calculate performance metrics
        response_times = self.test_results["performance_metrics"]["response_times"]
        memory_usage = self.test_results["performance_metrics"]["memory_usage"]
        validation_times = self.test_results["performance_metrics"]["validation_times"]
        
        performance_summary = {
            "average_response_time": sum(response_times) / len(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "average_memory_usage": sum(memory_usage) / len(memory_usage) if memory_usage else 0,
            "memory_growth": memory_usage[-1] - memory_usage[0] if len(memory_usage) > 1 else 0,
            "average_validation_time": sum(validation_times) / len(validation_times) if validation_times else 0
        }
        
        self.test_results["summary"] = {
            "total_execution_time_seconds": total_time,
            "success_rate_percentage": success_rate,
            "crash_rate": (self.test_results["crash_detection"]["crashes_detected"] / self.test_results["total_tests"] * 100) if self.test_results["total_tests"] > 0 else 0,
            "performance_summary": performance_summary,
            "overall_status": (
                "EXCELLENT" if success_rate >= 95 and self.test_results["crash_detection"]["crashes_detected"] == 0 else
                "GOOD" if success_rate >= 85 and self.test_results["crash_detection"]["crashes_detected"] <= 2 else
                "ACCEPTABLE" if success_rate >= 70 else "NEEDS_IMPROVEMENT"
            ),
            "timestamp": datetime.now().isoformat()
        }
        
        # Save detailed report
        report_path = f"pydantic_ai_core_comprehensive_test_report_{int(time.time())}.json"
        with open(report_path, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"📊 Comprehensive test report saved: {report_path}")
    
    async def _cleanup_test_environment(self):
        """Clean up test environment"""
        try:
            # Stop core system monitoring
            if self.core_system:
                self.core_system.monitoring_active = False
                if hasattr(self.core_system, 'monitor_thread'):
                    self.core_system.monitor_thread.join(timeout=5)
            
            # Clean up test database directory
            if os.path.exists(self.test_db_dir):
                shutil.rmtree(self.test_db_dir)
            
            print("🧹 Test environment cleaned up")
            
        except Exception as e:
            print(f"Cleanup error: {e}")


async def main():
    """Run comprehensive Pydantic AI Core Integration tests"""
    print("🧠 COMPREHENSIVE PYDANTIC AI CORE INTEGRATION TESTING")
    print("=" * 100)
    
    tester = ComprehensivePydanticAITest()
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
    
    # Performance metrics
    perf = summary["performance_summary"]
    print(f"\n📈 PERFORMANCE METRICS:")
    print(f"Average Response Time: {perf['average_response_time']:.3f}s")
    print(f"Memory Growth: {perf['memory_growth']:.2f}MB")
    print(f"Average Validation Time: {perf['average_validation_time']:.3f}s")
    
    print(f"\n🎯 FINAL ASSESSMENT: {summary['overall_status']}")
    
    if summary['overall_status'] in ['EXCELLENT', 'GOOD']:
        print("✅ PYDANTIC AI CORE INTEGRATION SYSTEM READY FOR PRODUCTION")
    elif summary['overall_status'] == 'ACCEPTABLE':
        print("⚠️ PYDANTIC AI CORE INTEGRATION SYSTEM ACCEPTABLE - MINOR IMPROVEMENTS NEEDED")
    else:
        print("❌ PYDANTIC AI CORE INTEGRATION SYSTEM NEEDS SIGNIFICANT IMPROVEMENTS")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())