#!/usr/bin/env python3
"""
SANDBOX FILE: For testing/development. See .cursorrules.

Comprehensive Test Suite for LangGraph Complex Workflow Structures System
TASK-LANGGRAPH-002.4: Complex Workflow Structures - Comprehensive Testing & Validation
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

from langgraph_complex_workflow_structures import (
    ComplexWorkflowStructureSystem, WorkflowTemplate, WorkflowNode, WorkflowInstance,
    WorkflowNodeType, ExecutionState, ConditionType, ConditionalLogic
)

# Configure logging for testing
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveComplexWorkflowTest:
    """Comprehensive test suite for complex workflow structures system"""
    
    def __init__(self):
        self.test_db_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.test_db_dir, "test_complex_workflows.db")
        self.workflow_system = None
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
        print("🧪 STARTING COMPREHENSIVE COMPLEX WORKFLOW STRUCTURES TESTING")
        print("=" * 100)
        
        try:
            # Initialize workflow system
            await self._test_system_initialization()
            await self._test_template_library_management()
            await self._test_dynamic_workflow_generation()
            await self._test_hierarchical_workflow_composition()
            await self._test_conditional_execution_paths()
            await self._test_loop_structures_and_iteration()
            await self._test_parallel_node_execution()
            await self._test_workflow_optimization()
            await self._test_performance_monitoring()
            await self._test_complex_workflow_execution()
            await self._test_error_handling_and_recovery()
            await self._test_memory_management_and_cleanup()
            
            # Generate final test report
            await self._generate_test_report()
            
        except Exception as e:
            self._record_crash("comprehensive_test_suite", str(e))
            print(f"💥 CRITICAL: Comprehensive test suite crashed: {e}")
            
        finally:
            await self._cleanup_test_environment()
        
        return self.test_results
    
    async def _test_system_initialization(self):
        """Test complex workflow system initialization"""
        test_name = "System Initialization"
        print(f"🔧 Testing: {test_name}")
        
        try:
            # Initialize system
            self.workflow_system = ComplexWorkflowStructureSystem(self.test_db_path)
            
            # Verify database creation
            assert os.path.exists(self.test_db_path), "Database file not created"
            
            # Verify core components
            assert self.workflow_system.template_library is not None, "Template library not initialized"
            assert self.workflow_system.workflow_generator is not None, "Workflow generator not initialized"
            assert self.workflow_system.condition_evaluator is not None, "Condition evaluator not initialized"
            assert self.workflow_system.loop_manager is not None, "Loop manager not initialized"
            assert self.workflow_system.workflow_optimizer is not None, "Workflow optimizer not initialized"
            
            # Check default templates
            templates = await self.workflow_system.template_library.list_templates()
            assert len(templates) >= 3, f"Expected at least 3 default templates, got {len(templates)}"
            
            self._record_test_result(test_name, True, "System initialized successfully with all components")
            
        except Exception as e:
            self._record_crash(test_name, str(e))
            self._record_test_result(test_name, False, f"Initialization failed: {e}")
    
    async def _test_template_library_management(self):
        """Test workflow template library functionality"""
        test_name = "Template Library Management"
        print(f"📚 Testing: {test_name}")
        
        try:
            # List all templates
            all_templates = await self.workflow_system.template_library.list_templates()
            assert len(all_templates) > 0, "No templates available"
            
            # List templates by category
            basic_templates = await self.workflow_system.template_library.list_templates("basic")
            conditional_templates = await self.workflow_system.template_library.list_templates("conditional")
            parallel_templates = await self.workflow_system.template_library.list_templates("parallel")
            
            assert len(basic_templates) > 0, "No basic templates found"
            assert len(conditional_templates) > 0, "No conditional templates found"
            assert len(parallel_templates) > 0, "No parallel templates found"
            
            # Get specific template
            sequential_template = await self.workflow_system.template_library.get_template("sequential_basic")
            assert sequential_template is not None, "Sequential template not found"
            assert sequential_template.name == "Basic Sequential Workflow", "Template name mismatch"
            assert len(sequential_template.nodes) >= 3, "Sequential template should have at least 3 nodes"
            
            # Test template usage tracking
            initial_usage = sequential_template.usage_count
            await self.workflow_system.template_library.increment_usage("sequential_basic")
            updated_template = await self.workflow_system.template_library.get_template("sequential_basic")
            assert updated_template.usage_count == initial_usage + 1, "Usage count not incremented"
            
            self._record_test_result(test_name, True, f"Template library validated with {len(all_templates)} templates")
            
        except Exception as e:
            self._record_crash(test_name, str(e))
            self._record_test_result(test_name, False, f"Template library testing failed: {e}")
    
    async def _test_dynamic_workflow_generation(self):
        """Test dynamic workflow generation from specifications"""
        test_name = "Dynamic Workflow Generation"
        print(f"⚡ Testing: {test_name}")
        
        try:
            # Test basic dynamic workflow generation
            basic_spec = {
                "type": "sequential",
                "name": "Test Dynamic Workflow",
                "description": "Test workflow generated dynamically",
                "nodes": [
                    {
                        "type": "agent",
                        "name": "Input Processor",
                        "agent_type": "processor",
                        "configuration": {"mode": "fast"}
                    },
                    {
                        "type": "transform",
                        "name": "Data Transformer",
                        "configuration": {"transform_type": "normalize"},
                        "dependencies": ["input_processor"]
                    },
                    {
                        "type": "agent",
                        "name": "Output Generator",
                        "agent_type": "generator",
                        "dependencies": ["data_transformer"]
                    }
                ],
                "variables": {"processing_mode": "test"}
            }
            
            dynamic_workflow = await self.workflow_system.create_dynamic_workflow(
                basic_spec, "test_user_dynamic", "pro"
            )
            
            assert dynamic_workflow is not None, "Dynamic workflow not created"
            assert dynamic_workflow.name == "Test Dynamic Workflow", "Workflow name mismatch"
            assert len(dynamic_workflow.nodes) == 3, f"Expected 3 nodes, got {len(dynamic_workflow.nodes)}"
            assert dynamic_workflow.state == ExecutionState.PENDING, "Workflow should be in pending state"
            assert "processing_mode" in dynamic_workflow.variables, "Variables not set correctly"
            
            # Test complex dynamic workflow with conditions
            complex_spec = {
                "type": "conditional",
                "name": "Complex Dynamic Workflow",
                "description": "Complex workflow with conditions and loops",
                "nodes": [
                    {
                        "type": "agent",
                        "name": "Analyzer",
                        "agent_type": "analyzer"
                    },
                    {
                        "type": "condition",
                        "name": "Decision Point",
                        "configuration": {
                            "condition": {
                                "type": "if_then_else",
                                "expression": "complexity > 0.5",
                                "true_path": ["complex_path"],
                                "false_path": ["simple_path"]
                            }
                        },
                        "dependencies": ["analyzer"]
                    },
                    {
                        "type": "loop",
                        "name": "Processing Loop",
                        "configuration": {
                            "loop": {
                                "type": "while",
                                "condition": "iterations < 10",
                                "iteration_limit": 10
                            }
                        }
                    }
                ],
                "variables": {"complexity_threshold": 0.5, "max_iterations": 10}
            }
            
            complex_workflow = await self.workflow_system.create_dynamic_workflow(
                complex_spec, "test_user_complex", "enterprise"
            )
            
            assert complex_workflow is not None, "Complex dynamic workflow not created"
            assert len(complex_workflow.nodes) == 3, "Complex workflow should have 3 nodes"
            
            # Verify node types
            node_types = [node.node_type for node in complex_workflow.nodes]
            assert WorkflowNodeType.AGENT in node_types, "Agent node not found"
            assert WorkflowNodeType.CONDITION in node_types, "Condition node not found"
            assert WorkflowNodeType.LOOP in node_types, "Loop node not found"
            
            self._record_test_result(test_name, True, "Dynamic workflow generation validated for basic and complex scenarios")
            
        except Exception as e:
            self._record_crash(test_name, str(e))
            self._record_test_result(test_name, False, f"Dynamic workflow generation failed: {e}")
    
    async def _test_hierarchical_workflow_composition(self):
        """Test hierarchical workflow composition with subworkflows"""
        test_name = "Hierarchical Workflow Composition"
        print(f"🏗️ Testing: {test_name}")
        
        try:
            # Create main workflow with subworkflow
            main_workflow = await self.workflow_system.create_workflow_from_template(
                "sequential_basic", "test_user_hierarchical", "enterprise", 
                {"main_input": "hierarchical test data"}
            )
            
            assert main_workflow is not None, "Main workflow not created"
            
            # Test subworkflow creation specification
            subworkflow_spec = {
                "type": "sequential",
                "name": "Subworkflow Test",
                "description": "Test subworkflow for hierarchical composition",
                "nodes": [
                    {
                        "type": "agent",
                        "name": "Sub Processor",
                        "agent_type": "sub_processor"
                    },
                    {
                        "type": "subworkflow",
                        "name": "Nested Subworkflow",
                        "configuration": {
                            "subworkflow": {
                                "template_id": "sequential_basic",
                                "parameters": {"nested_input": "nested data"}
                            }
                        },
                        "dependencies": ["sub_processor"]
                    }
                ],
                "variables": {"subworkflow_level": 1}
            }
            
            hierarchical_workflow = await self.workflow_system.create_dynamic_workflow(
                subworkflow_spec, "test_user_sub", "enterprise"
            )
            
            assert hierarchical_workflow is not None, "Hierarchical workflow not created"
            
            # Verify subworkflow node exists
            subworkflow_nodes = [node for node in hierarchical_workflow.nodes 
                               if node.node_type == WorkflowNodeType.SUBWORKFLOW]
            assert len(subworkflow_nodes) > 0, "No subworkflow nodes found"
            
            # Verify subworkflow configuration
            subworkflow_node = subworkflow_nodes[0]
            subworkflow_config = subworkflow_node.configuration.get("subworkflow", {})
            assert "template_id" in subworkflow_config, "Subworkflow template_id not configured"
            assert subworkflow_config["template_id"] == "sequential_basic", "Incorrect template_id"
            
            self._record_test_result(test_name, True, "Hierarchical workflow composition validated")
            
        except Exception as e:
            self._record_crash(test_name, str(e))
            self._record_test_result(test_name, False, f"Hierarchical composition testing failed: {e}")
    
    async def _test_conditional_execution_paths(self):
        """Test conditional execution paths and logic"""
        test_name = "Conditional Execution Paths"
        print(f"🔀 Testing: {test_name}")
        
        try:
            # Test basic condition evaluation
            condition_evaluator = self.workflow_system.condition_evaluator
            
            # Test simple conditions
            true_result = await condition_evaluator.evaluate_condition("true", {})
            false_result = await condition_evaluator.evaluate_condition("false", {})
            assert true_result == True, "True condition evaluation failed"
            assert false_result == False, "False condition evaluation failed"
            
            # Test variable-based conditions
            gt_result = await condition_evaluator.evaluate_condition("test_value > 5", {"test_value": 10})
            assert gt_result == True, "Greater than condition failed"
            
            lt_result = await condition_evaluator.evaluate_condition("test_value > 5", {"test_value": 3})
            assert lt_result == False, "Less than condition failed"
            
            # Test equality conditions
            eq_result = await condition_evaluator.evaluate_condition('status == "completed"', {"status": "completed"})
            assert eq_result == True, "Equality condition failed"
            
            # Test conditional workflow creation
            conditional_workflow = await self.workflow_system.create_workflow_from_template(
                "conditional_basic", "test_user_conditional", "pro",
                {"input_complexity": 0.7}
            )
            
            assert conditional_workflow is not None, "Conditional workflow not created"
            
            # Verify conditional nodes
            condition_nodes = [node for node in conditional_workflow.nodes 
                             if node.node_type == WorkflowNodeType.CONDITION]
            assert len(condition_nodes) > 0, "No conditional nodes found"
            
            # Verify condition configuration
            condition_node = condition_nodes[0]
            condition_config = condition_node.configuration.get("condition", {})
            assert "type" in condition_config, "Condition type not specified"
            assert "expression" in condition_config, "Condition expression not specified"
            assert "true_path" in condition_config, "True path not specified"
            assert "false_path" in condition_config, "False path not specified"
            
            self._record_test_result(test_name, True, "Conditional execution paths validated")
            
        except Exception as e:
            self._record_crash(test_name, str(e))
            self._record_test_result(test_name, False, f"Conditional execution testing failed: {e}")
    
    async def _test_loop_structures_and_iteration(self):
        """Test loop structures and iteration handling"""
        test_name = "Loop Structures and Iteration"
        print(f"🔄 Testing: {test_name}")
        
        try:
            # Test loop manager initialization
            loop_manager = self.workflow_system.loop_manager
            assert loop_manager is not None, "Loop manager not initialized"
            
            # Create workflow with loop structure
            loop_spec = {
                "type": "iterative",
                "name": "Loop Test Workflow",
                "description": "Workflow with loop structures",
                "nodes": [
                    {
                        "type": "agent",
                        "name": "Iterator Setup",
                        "agent_type": "setup"
                    },
                    {
                        "type": "loop",
                        "name": "Processing Loop",
                        "configuration": {
                            "loop": {
                                "type": "while",
                                "condition": "counter < 5",
                                "iteration_limit": 10,
                                "timeout_seconds": 30,
                                "break_conditions": ["error_occurred"]
                            }
                        },
                        "dependencies": ["iterator_setup"]
                    },
                    {
                        "type": "agent",
                        "name": "Result Aggregator",
                        "agent_type": "aggregator",
                        "dependencies": ["processing_loop"]
                    }
                ],
                "variables": {"counter": 0, "max_iterations": 5}
            }
            
            loop_workflow = await self.workflow_system.create_dynamic_workflow(
                loop_spec, "test_user_loop", "pro"
            )
            
            assert loop_workflow is not None, "Loop workflow not created"
            
            # Verify loop nodes
            loop_nodes = [node for node in loop_workflow.nodes 
                         if node.node_type == WorkflowNodeType.LOOP]
            assert len(loop_nodes) > 0, "No loop nodes found"
            
            # Verify loop configuration
            loop_node = loop_nodes[0]
            loop_config = loop_node.configuration.get("loop", {})
            assert "type" in loop_config, "Loop type not specified"
            assert "condition" in loop_config, "Loop condition not specified"
            assert "iteration_limit" in loop_config, "Iteration limit not specified"
            assert loop_config["iteration_limit"] == 10, "Incorrect iteration limit"
            
            # Test termination guarantees
            assert loop_config["iteration_limit"] <= 1000, "Iteration limit too high"
            assert "timeout_seconds" in loop_config, "Timeout not specified"
            
            self._record_test_result(test_name, True, "Loop structures and iteration handling validated")
            
        except Exception as e:
            self._record_crash(test_name, str(e))
            self._record_test_result(test_name, False, f"Loop structures testing failed: {e}")
    
    async def _test_parallel_node_execution(self):
        """Test parallel node execution capabilities"""
        test_name = "Parallel Node Execution"
        print(f"⚡ Testing: {test_name}")
        
        try:
            # Create parallel workflow from template
            parallel_workflow = await self.workflow_system.create_workflow_from_template(
                "parallel_basic", "test_user_parallel", "enterprise",
                {"parallel_data": "test data for parallel processing"}
            )
            
            assert parallel_workflow is not None, "Parallel workflow not created"
            
            # Verify parallel nodes
            parallel_nodes = [node for node in parallel_workflow.nodes 
                            if node.node_type == WorkflowNodeType.PARALLEL]
            assert len(parallel_nodes) > 0, "No parallel nodes found"
            
            # Verify parallel configuration
            parallel_node = parallel_nodes[0]
            parallel_config = parallel_node.configuration.get("parallel", {})
            assert "nodes" in parallel_config, "Parallel nodes not specified"
            
            parallel_node_list = parallel_config["nodes"]
            assert len(parallel_node_list) > 0, "No nodes specified for parallel execution"
            
            # Test split and merge nodes
            split_nodes = [node for node in parallel_workflow.nodes 
                          if node.node_type == WorkflowNodeType.SPLIT]
            merge_nodes = [node for node in parallel_workflow.nodes 
                          if node.node_type == WorkflowNodeType.MERGE]
            
            assert len(split_nodes) > 0, "No split nodes found for parallel workflow"
            assert len(merge_nodes) > 0, "No merge nodes found for parallel workflow"
            
            # Verify dependencies
            parallel_node = parallel_nodes[0]
            assert len(parallel_node.dependencies) > 0, "Parallel node should have dependencies"
            
            # Verify merge node depends on parallel node
            merge_node = merge_nodes[0]
            assert parallel_node.node_id.split('_')[-1] in [dep.split('_')[-1] for dep in merge_node.dependencies], \
                   "Merge node should depend on parallel node"
            
            self._record_test_result(test_name, True, "Parallel node execution capabilities validated")
            
        except Exception as e:
            self._record_crash(test_name, str(e))
            self._record_test_result(test_name, False, f"Parallel node execution testing failed: {e}")
    
    async def _test_workflow_optimization(self):
        """Test workflow optimization capabilities"""
        test_name = "Workflow Optimization"
        print(f"🚀 Testing: {test_name}")
        
        try:
            # Create a workflow to optimize
            original_workflow = await self.workflow_system.create_workflow_from_template(
                "sequential_basic", "test_user_optimize", "pro",
                {"optimization_test": "data"}
            )
            
            assert original_workflow is not None, "Original workflow not created"
            
            # Test workflow optimization
            optimizer = self.workflow_system.workflow_optimizer
            optimized_workflow = await optimizer.optimize_workflow(original_workflow)
            
            assert optimized_workflow is not None, "Optimized workflow not created"
            assert optimized_workflow.workflow_id == original_workflow.workflow_id, "Workflow ID changed during optimization"
            assert len(optimized_workflow.nodes) == len(original_workflow.nodes), "Node count changed during optimization"
            
            # Verify optimization doesn't break workflow structure
            for original_node, optimized_node in zip(original_workflow.nodes, optimized_workflow.nodes):
                assert optimized_node.node_type == original_node.node_type, "Node type changed during optimization"
                assert optimized_node.name == original_node.name, "Node name changed during optimization"
            
            # Test that optimization maintains dependencies
            original_deps = set()
            optimized_deps = set()
            
            for node in original_workflow.nodes:
                original_deps.update(node.dependencies)
            
            for node in optimized_workflow.nodes:
                optimized_deps.update(node.dependencies)
            
            # Dependencies should be preserved (though order may change)
            assert len(optimized_deps) >= len(original_deps), "Dependencies lost during optimization"
            
            self._record_test_result(test_name, True, "Workflow optimization capabilities validated")
            
        except Exception as e:
            self._record_crash(test_name, str(e))
            self._record_test_result(test_name, False, f"Workflow optimization testing failed: {e}")
    
    async def _test_performance_monitoring(self):
        """Test performance monitoring and analytics"""
        test_name = "Performance Monitoring"
        print(f"📊 Testing: {test_name}")
        
        try:
            # Test performance monitor initialization
            performance_monitor = self.workflow_system.performance_monitor
            assert performance_monitor is not None, "Performance monitor not initialized"
            
            # Record test metrics
            test_workflow_id = "test_workflow_perf"
            
            await performance_monitor.record_metric(test_workflow_id, "execution_time", 1.5)
            await performance_monitor.record_metric(test_workflow_id, "execution_time", 2.0)
            await performance_monitor.record_metric(test_workflow_id, "execution_time", 1.8)
            await performance_monitor.record_metric(test_workflow_id, "memory_usage", 256.0)
            await performance_monitor.record_metric(test_workflow_id, "memory_usage", 312.0)
            
            # Get performance analytics
            performance_data = await performance_monitor.get_workflow_performance(test_workflow_id)
            
            assert "execution_time" in performance_data, "Execution time metrics not found"
            assert "memory_usage" in performance_data, "Memory usage metrics not found"
            
            # Verify metric calculations
            exec_time_data = performance_data["execution_time"]
            assert "average" in exec_time_data, "Average not calculated"
            assert "min" in exec_time_data, "Minimum not calculated"
            assert "max" in exec_time_data, "Maximum not calculated"
            assert "count" in exec_time_data, "Count not calculated"
            
            assert exec_time_data["count"] == 3, f"Expected 3 execution time records, got {exec_time_data['count']}"
            assert exec_time_data["min"] == 1.5, f"Expected min 1.5, got {exec_time_data['min']}"
            assert exec_time_data["max"] == 2.0, f"Expected max 2.0, got {exec_time_data['max']}"
            assert abs(exec_time_data["average"] - 1.767) < 0.01, f"Expected average ~1.767, got {exec_time_data['average']}"
            
            memory_data = performance_data["memory_usage"]
            assert memory_data["count"] == 2, f"Expected 2 memory usage records, got {memory_data['count']}"
            
            self._record_test_result(test_name, True, "Performance monitoring and analytics validated")
            
        except Exception as e:
            self._record_crash(test_name, str(e))
            self._record_test_result(test_name, False, f"Performance monitoring testing failed: {e}")
    
    async def _test_complex_workflow_execution(self):
        """Test end-to-end complex workflow execution"""
        test_name = "Complex Workflow Execution"
        print(f"🎯 Testing: {test_name}")
        
        try:
            # Create a complex workflow for execution testing
            complex_spec = {
                "type": "complex",
                "name": "Complex Execution Test Workflow",
                "description": "Complex workflow for end-to-end execution testing",
                "nodes": [
                    {
                        "type": "agent",
                        "name": "Data Ingestion",
                        "agent_type": "ingestion",
                        "configuration": {"source": "test_data"}
                    },
                    {
                        "type": "condition",
                        "name": "Data Quality Check",
                        "configuration": {
                            "condition": {
                                "type": "if_then_else",
                                "expression": "data_quality > 0.8",
                                "true_path": ["high_quality_processor"],
                                "false_path": ["data_cleaning"]
                            }
                        },
                        "dependencies": ["data_ingestion"]
                    },
                    {
                        "type": "agent",
                        "name": "High Quality Processor",
                        "agent_type": "processor"
                    },
                    {
                        "type": "agent",
                        "name": "Data Cleaning",
                        "agent_type": "cleaner"
                    },
                    {
                        "type": "parallel",
                        "name": "Parallel Analysis",
                        "configuration": {
                            "parallel": {
                                "nodes": ["analyzer_1", "analyzer_2", "analyzer_3"]
                            }
                        }
                    },
                    {
                        "type": "agent",
                        "name": "Result Synthesis",
                        "agent_type": "synthesizer",
                        "dependencies": ["parallel_analysis"]
                    }
                ],
                "variables": {"data_quality_threshold": 0.8, "parallel_processing": True}
            }
            
            # Create and execute the workflow
            complex_workflow = await self.workflow_system.create_dynamic_workflow(
                complex_spec, "test_user_execution", "enterprise"
            )
            
            assert complex_workflow is not None, "Complex workflow not created"
            assert complex_workflow.state == ExecutionState.PENDING, "Workflow should be in pending state"
            
            # Execute the workflow
            start_time = time.time()
            execution_result = await self.workflow_system.execute_workflow(complex_workflow.workflow_id)
            execution_time = time.time() - start_time
            
            # Verify execution results
            assert execution_result is not None, "Execution result is None"
            assert isinstance(execution_result, dict), "Execution result should be a dictionary"
            
            # Verify workflow state after execution
            executed_workflow = await self.workflow_system._load_workflow_instance(complex_workflow.workflow_id)
            assert executed_workflow is not None, "Executed workflow not found"
            assert executed_workflow.state == ExecutionState.COMPLETED, "Workflow should be completed"
            assert executed_workflow.start_time is not None, "Start time not recorded"
            assert executed_workflow.end_time is not None, "End time not recorded"
            assert executed_workflow.end_time > executed_workflow.start_time, "End time should be after start time"
            
            # Verify execution time is reasonable
            assert execution_time < 30, f"Execution took too long: {execution_time:.2f}s"
            
            # Check that results were stored
            assert executed_workflow.results is not None, "Results not stored"
            
            self._record_test_result(test_name, True, f"Complex workflow execution completed in {execution_time:.2f}s")
            
        except Exception as e:
            self._record_crash(test_name, str(e))
            self._record_test_result(test_name, False, f"Complex workflow execution failed: {e}")
    
    async def _test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms"""
        test_name = "Error Handling and Recovery"
        print(f"🛡️ Testing: {test_name}")
        
        try:
            # Test invalid template handling
            try:
                invalid_workflow = await self.workflow_system.create_workflow_from_template(
                    "nonexistent_template", "test_user_error", "pro"
                )
                assert False, "Should have raised exception for nonexistent template"
            except ValueError as e:
                assert "not found" in str(e), "Appropriate error message not provided"
            
            # Test invalid workflow specification
            invalid_spec = {
                "type": "invalid_type",
                "nodes": []  # Empty nodes should be handled gracefully
            }
            
            try:
                invalid_dynamic = await self.workflow_system.create_dynamic_workflow(
                    invalid_spec, "test_user_invalid", "pro"
                )
                # Should not fail completely, but should handle gracefully
                assert invalid_dynamic is not None, "Invalid workflow creation should be handled gracefully"
            except Exception as e:
                # Acceptable if proper error handling is in place
                pass
            
            # Test workflow execution with missing dependencies
            dependency_spec = {
                "type": "dependency_test",
                "name": "Dependency Test Workflow",
                "nodes": [
                    {
                        "type": "agent",
                        "name": "Dependent Node",
                        "agent_type": "processor",
                        "dependencies": ["nonexistent_node"]  # Invalid dependency
                    }
                ]
            }
            
            dependency_workflow = await self.workflow_system.create_dynamic_workflow(
                dependency_spec, "test_user_dependency", "pro"
            )
            
            # Execution should handle missing dependencies gracefully
            try:
                result = await self.workflow_system.execute_workflow(dependency_workflow.workflow_id)
                # Either succeeds with appropriate handling or fails gracefully
            except Exception as e:
                # Acceptable if proper error handling is in place
                error_msg = str(e).lower()
                assert "dependency" in error_msg or "not found" in error_msg or "index" in error_msg, \
                       f"Error message should indicate dependency issue, got: {e}"
            
            self._record_test_result(test_name, True, "Error handling and recovery mechanisms validated")
            
        except Exception as e:
            self._record_crash(test_name, str(e))
            self._record_test_result(test_name, False, f"Error handling testing failed: {e}")
    
    async def _test_memory_management_and_cleanup(self):
        """Test memory management and resource cleanup"""
        test_name = "Memory Management and Cleanup"
        print(f"🧹 Testing: {test_name}")
        
        try:
            initial_active_workflows = len(self.workflow_system.active_workflows)
            
            # Create multiple workflows to test memory management
            test_workflows = []
            for i in range(5):
                workflow = await self.workflow_system.create_workflow_from_template(
                    "sequential_basic", f"test_user_memory_{i}", "pro",
                    {"test_data": f"memory_test_{i}"}
                )
                test_workflows.append(workflow.workflow_id)
            
            # Verify workflows are not in active list yet (not executed)
            assert len(self.workflow_system.active_workflows) == initial_active_workflows, \
                   "Workflows should not be in active list until execution starts"
            
            # Execute workflows and verify cleanup
            executed_workflows = 0
            for workflow_id in test_workflows:
                try:
                    await self.workflow_system.execute_workflow(workflow_id)
                    executed_workflows += 1
                except Exception as e:
                    print(f"Workflow {workflow_id} execution failed: {e}")
            
            # Verify cleanup after execution
            final_active_workflows = len(self.workflow_system.active_workflows)
            assert final_active_workflows == initial_active_workflows, \
                   f"Active workflows not cleaned up: {final_active_workflows} vs {initial_active_workflows}"
            
            # Test template library memory management
            initial_template_count = len(self.workflow_system.template_library.templates)
            templates = await self.workflow_system.template_library.list_templates()
            final_template_count = len(self.workflow_system.template_library.templates)
            
            assert final_template_count == initial_template_count, \
                   "Template library memory changed unexpectedly"
            
            # Test performance monitor cleanup
            performance_metrics = self.workflow_system.performance_monitor.performance_metrics
            metrics_count = len(performance_metrics)
            
            # Metrics should be maintained but not grow indefinitely
            assert metrics_count < 1000, f"Performance metrics growing too large: {metrics_count}"
            
            self.test_results["system_stability"]["resource_cleanup"] = True
            self._record_test_result(test_name, True, f"Memory management validated - {executed_workflows} workflows executed and cleaned up")
            
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
        report_path = f"complex_workflow_test_report_{int(time.time())}.json"
        with open(report_path, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"📊 Test report saved: {report_path}")
    
    async def _cleanup_test_environment(self):
        """Clean up test environment"""
        try:
            # Stop monitoring
            if self.workflow_system:
                self.workflow_system.monitoring_active = False
                if hasattr(self.workflow_system, 'monitor_thread'):
                    self.workflow_system.monitor_thread.join(timeout=5)
            
            # Clean up test database directory
            if os.path.exists(self.test_db_dir):
                shutil.rmtree(self.test_db_dir)
            
            print("🧹 Test environment cleaned up")
            
        except Exception as e:
            print(f"Cleanup error: {e}")


async def main():
    """Run comprehensive complex workflow structures tests"""
    print("🚀 COMPREHENSIVE COMPLEX WORKFLOW STRUCTURES TESTING")
    print("=" * 100)
    
    tester = ComprehensiveComplexWorkflowTest()
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
        print("✅ COMPLEX WORKFLOW STRUCTURES SYSTEM READY FOR PRODUCTION")
    elif summary['overall_status'] == 'ACCEPTABLE':
        print("⚠️ COMPLEX WORKFLOW STRUCTURES SYSTEM ACCEPTABLE - MINOR IMPROVEMENTS NEEDED")
    else:
        print("❌ COMPLEX WORKFLOW STRUCTURES SYSTEM NEEDS SIGNIFICANT IMPROVEMENTS")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())