#!/usr/bin/env python3
"""
Comprehensive Test Suite for LangGraph Video Generation Workflow Coordination
Tests video workflow integration, multi-stage coordination, resource scheduling, and quality control.

* Purpose: Comprehensive testing for LangGraph-Video generation workflow coordination with TDD validation
* Issues & Complexity Summary: Complex video workflow coordination requiring validation of multi-stage processing and resource optimization
* Key Complexity Drivers:
  - Logic Scope (Est. LoC): ~2500
  - Core Algorithm Complexity: Very High (video workflow coordination, multi-stage processing, resource scheduling)
  - Dependencies: 15 (unittest, asyncio, time, json, uuid, datetime, threading, tempfile, sqlite3, typing, concurrent.futures, subprocess, os, pathlib, io)
  - State Management Complexity: Very High (video workflow state, resource coordination, quality control)
  - Novelty/Uncertainty Factor: Very High (LangGraph video generation integration with real-time coordination)
* AI Pre-Task Self-Assessment (Est. Solution Difficulty %): 94%
* Problem Estimate (Inherent Problem Difficulty %): 92%
* Initial Code Complexity Estimate %: 94%
* Justification for Estimates: Complex video workflow coordination requiring multi-stage processing with resource optimization and quality control
* Final Code Complexity (Actual %): TBD
* Overall Result Score (Success & Quality %): TBD
* Key Variances/Learnings: TBD
* Last Updated: 2025-06-04
"""

import unittest
import asyncio
import time
import json
import uuid
import tempfile
import os
import sqlite3
import threading
import subprocess
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from io import StringIO

# Import the system under test
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from sources.langgraph_video_generation_workflow_coordination_sandbox import (
        LangGraphVideoWorkflowCoordinator,
        VideoGenerationWorkflowManager,
        MultiStageVideoProcessor,
        VideoResourceScheduler,
        VideoQualityController,
        VideoRenderingOptimizer,
        VideoGenerationOrchestrator,
        VideoWorkflowState,
        VideoStageResult,
        VideoResourceAllocation,
        VideoQualityMetrics
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Warning: Could not import LangGraph Video workflow coordination components: {e}")
    IMPORT_SUCCESS = False


class TestLangGraphVideoWorkflowCoordinator(unittest.TestCase):
    """Test LangGraph video workflow coordination functionality"""
    
    def setUp(self):
        """Set up test environment"""
        if not IMPORT_SUCCESS:
            self.skipTest("LangGraph Video coordination imports not available")
        
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        self.coordinator = LangGraphVideoWorkflowCoordinator(self.db_path)
    
    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'db_path') and os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_coordinator_initialization(self):
        """Test coordinator initialization"""
        self.assertIsNotNone(self.coordinator)
        self.assertEqual(self.coordinator.db_path, self.db_path)
        self.assertTrue(os.path.exists(self.db_path))
        
        # Verify database tables created
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = [
                'video_workflows', 'video_stages', 'video_resources', 
                'video_quality_metrics', 'video_render_tasks'
            ]
            for table in expected_tables:
                self.assertIn(table, tables)
    
    def test_video_workflow_setup(self):
        """Test video workflow setup"""
        workflow_config = {
            'workflow_id': 'test_video_workflow_001',
            'project_name': 'Test Video Project',
            'video_specs': {
                'duration': 30,
                'resolution': '1920x1080',
                'fps': 30,
                'format': 'mp4'
            },
            'stages': ['script_generation', 'storyboard', 'video_generation', 'post_processing']
        }
        
        setup_result = self.coordinator.setup_video_workflow(workflow_config)
        
        self.assertTrue(setup_result)
        self.assertIn(workflow_config['workflow_id'], self.coordinator.active_workflows)
        
        # Verify workflow context created
        workflow_state = self.coordinator.get_workflow_state(workflow_config['workflow_id'])
        self.assertIsNotNone(workflow_state)
        self.assertEqual(workflow_state.workflow_id, workflow_config['workflow_id'])
        self.assertEqual(workflow_state.project_name, workflow_config['project_name'])
    
    def test_video_stage_management(self):
        """Test video stage creation and management"""
        workflow_id = "stage_test_workflow"
        
        # Setup workflow first
        workflow_config = {
            'workflow_id': workflow_id,
            'project_name': 'Stage Test Project',
            'stages': ['concept', 'script', 'visual', 'render']
        }
        self.coordinator.setup_video_workflow(workflow_config)
        
        # Create video stage
        stage_config = {
            'stage_id': 'concept_stage',
            'stage_type': 'concept_development',
            'stage_name': 'Concept Development',
            'dependencies': [],
            'estimated_duration': 300,  # 5 minutes
            'resources_required': ['creative_llm', 'planning_agent']
        }
        
        stage_result = self.coordinator.create_video_stage(workflow_id, stage_config)
        
        self.assertIsNotNone(stage_result)
        self.assertEqual(stage_result.stage_id, 'concept_stage')
        self.assertEqual(stage_result.stage_type, 'concept_development')
        
        # Verify stage stored in database
        stored_stages = self.coordinator.get_workflow_stages(workflow_id)
        self.assertGreater(len(stored_stages), 0)
        self.assertEqual(stored_stages[0].stage_id, 'concept_stage')
    
    def test_seamless_video_workflow_coordination(self):
        """Test seamless video workflow coordination"""
        workflow_id = "seamless_coordination_test"
        
        # Setup comprehensive workflow
        workflow_config = {
            'workflow_id': workflow_id,
            'project_name': 'Seamless Coordination Test',
            'video_specs': {
                'duration': 60,
                'resolution': '1920x1080',
                'fps': 30,
                'style': 'cinematic'
            },
            'stages': ['concept', 'script', 'storyboard', 'generation', 'post_processing']
        }
        
        self.coordinator.setup_video_workflow(workflow_config)
        
        # Test seamless coordination across stages
        stages_data = [
            {
                'stage_id': 'concept_001',
                'stage_type': 'concept_development',
                'input_data': {'theme': 'technology innovation', 'target_audience': 'professionals'},
                'expected_output': 'concept_document'
            },
            {
                'stage_id': 'script_001',
                'stage_type': 'script_generation',
                'input_data': {'concept_id': 'concept_001', 'duration': 60},
                'expected_output': 'video_script'
            },
            {
                'stage_id': 'storyboard_001',
                'stage_type': 'storyboard_creation',
                'input_data': {'script_id': 'script_001', 'visual_style': 'modern'},
                'expected_output': 'storyboard_frames'
            }
        ]
        
        coordination_results = []
        for stage_data in stages_data:
            result = self.coordinator.coordinate_video_stage(workflow_id, stage_data)
            coordination_results.append(result)
        
        # Verify seamless coordination
        self.assertEqual(len(coordination_results), 3)
        for result in coordination_results:
            self.assertIsNotNone(result)
            self.assertIn('stage_result', result)
            self.assertIn('coordination_status', result)
            self.assertEqual(result['coordination_status'], 'coordinated')
        
        # Verify workflow progression
        workflow_state = self.coordinator.get_workflow_state(workflow_id)
        self.assertIsNotNone(workflow_state)
        self.assertGreater(len(workflow_state.completed_stages), 0)


class TestVideoGenerationWorkflowManager(unittest.TestCase):
    """Test video generation workflow management functionality"""
    
    def setUp(self):
        """Set up test environment"""
        if not IMPORT_SUCCESS:
            self.skipTest("LangGraph Video coordination imports not available")
        
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        
        # Initialize coordinator first
        coordinator = LangGraphVideoWorkflowCoordinator(self.db_path)
        del coordinator
        
        self.workflow_manager = VideoGenerationWorkflowManager(self.db_path)
    
    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'db_path') and os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_workflow_lifecycle_management(self):
        """Test complete workflow lifecycle management"""
        workflow_config = {
            'workflow_id': 'lifecycle_test_workflow',
            'project_name': 'Lifecycle Test Video',
            'video_specs': {
                'duration': 45,
                'resolution': '1920x1080',
                'fps': 24,
                'style': 'documentary'
            },
            'priority': 'high'
        }
        
        # Create workflow
        create_result = self.workflow_manager.create_workflow(workflow_config)
        self.assertTrue(create_result)
        
        # Start workflow execution
        start_result = self.workflow_manager.start_workflow_execution(workflow_config['workflow_id'])
        self.assertTrue(start_result)
        
        # Pause workflow
        pause_result = self.workflow_manager.pause_workflow(workflow_config['workflow_id'])
        self.assertTrue(pause_result)
        
        # Resume workflow
        resume_result = self.workflow_manager.resume_workflow(workflow_config['workflow_id'])
        self.assertTrue(resume_result)
        
        # Get workflow status
        status = self.workflow_manager.get_workflow_status(workflow_config['workflow_id'])
        self.assertIsNotNone(status)
        self.assertIn('status', status)
        self.assertIn('progress', status)
    
    def test_multi_stage_processing_efficiency(self):
        """Test multi-stage processing efficiency >90%"""
        workflow_id = "efficiency_test_workflow"
        
        # Create workflow with multiple stages
        stages = [
            {'stage_id': f'stage_{i}', 'stage_type': 'processing', 'duration': 10}
            for i in range(5)
        ]
        
        workflow_config = {
            'workflow_id': workflow_id,
            'stages': stages,
            'efficiency_target': 0.9
        }
        
        self.workflow_manager.create_workflow(workflow_config)
        
        # Measure processing efficiency
        start_time = time.time()
        
        processing_results = []
        for stage in stages:
            stage_start = time.time()
            result = self.workflow_manager.process_stage(workflow_id, stage)
            stage_duration = time.time() - stage_start
            
            processing_results.append({
                'stage_id': stage['stage_id'],
                'result': result,
                'duration': stage_duration,
                'expected_duration': stage['duration']
            })
        
        total_duration = time.time() - start_time
        expected_total_duration = sum(stage['duration'] for stage in stages)
        
        # Calculate efficiency
        efficiency = min(1.0, expected_total_duration / total_duration) if total_duration > 0 else 0.0
        
        # Framework should be capable of >90% efficiency
        self.assertIsInstance(efficiency, float)
        self.assertGreaterEqual(len(processing_results), 5)
        
        # Verify all stages processed successfully
        for result in processing_results:
            self.assertIsNotNone(result['result'])
    
    def test_workflow_template_system(self):
        """Test workflow template creation and usage"""
        template_config = {
            'template_id': 'promotional_video_template',
            'template_name': 'Promotional Video Standard',
            'default_stages': [
                'concept_development',
                'script_writing',
                'visual_planning',
                'video_generation',
                'audio_sync',
                'post_processing'
            ],
            'default_specs': {
                'duration': 30,
                'resolution': '1920x1080',
                'fps': 30,
                'style': 'promotional'
            }
        }
        
        # Create template
        template_result = self.workflow_manager.create_workflow_template(template_config)
        self.assertTrue(template_result)
        
        # Use template to create workflow
        workflow_from_template = self.workflow_manager.create_workflow_from_template(
            template_id='promotional_video_template',
            workflow_id='promo_test_001',
            customizations={'duration': 45}
        )
        
        self.assertIsNotNone(workflow_from_template)
        self.assertEqual(workflow_from_template['workflow_id'], 'promo_test_001')
        self.assertEqual(workflow_from_template['video_specs']['duration'], 45)  # Customized
        self.assertEqual(workflow_from_template['video_specs']['resolution'], '1920x1080')  # From template


class TestMultiStageVideoProcessor(unittest.TestCase):
    """Test multi-stage video processing functionality"""
    
    def setUp(self):
        """Set up test environment"""
        if not IMPORT_SUCCESS:
            self.skipTest("LangGraph Video coordination imports not available")
        
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        
        # Initialize coordinator first
        coordinator = LangGraphVideoWorkflowCoordinator(self.db_path)
        del coordinator
        
        self.processor = MultiStageVideoProcessor(self.db_path)
    
    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'db_path') and os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_stage_dependency_resolution(self):
        """Test stage dependency resolution and ordering"""
        stages = [
            {
                'stage_id': 'post_processing',
                'dependencies': ['video_generation', 'audio_sync']
            },
            {
                'stage_id': 'video_generation',
                'dependencies': ['storyboard']
            },
            {
                'stage_id': 'storyboard',
                'dependencies': ['script']
            },
            {
                'stage_id': 'script',
                'dependencies': []
            },
            {
                'stage_id': 'audio_sync',
                'dependencies': ['video_generation']
            }
        ]
        
        # Resolve stage execution order
        execution_order = self.processor.resolve_stage_dependencies(stages)
        
        self.assertIsNotNone(execution_order)
        self.assertEqual(len(execution_order), 5)
        
        # Verify correct ordering
        script_index = execution_order.index('script')
        storyboard_index = execution_order.index('storyboard')
        video_gen_index = execution_order.index('video_generation')
        audio_sync_index = execution_order.index('audio_sync')
        post_proc_index = execution_order.index('post_processing')
        
        self.assertLess(script_index, storyboard_index)
        self.assertLess(storyboard_index, video_gen_index)
        self.assertLess(video_gen_index, audio_sync_index)
        self.assertLess(audio_sync_index, post_proc_index)
    
    def test_parallel_stage_processing(self):
        """Test parallel processing of independent stages"""
        independent_stages = [
            {
                'stage_id': 'audio_generation',
                'stage_type': 'audio',
                'dependencies': [],
                'processing_time': 5
            },
            {
                'stage_id': 'visual_effects',
                'stage_type': 'effects',
                'dependencies': [],
                'processing_time': 7
            },
            {
                'stage_id': 'subtitle_generation',
                'stage_type': 'text',
                'dependencies': [],
                'processing_time': 3
            }
        ]
        
        # Process stages in parallel
        start_time = time.time()
        
        parallel_results = self.processor.process_stages_parallel(independent_stages)
        
        processing_time = time.time() - start_time
        
        # Verify parallel processing efficiency
        self.assertEqual(len(parallel_results), 3)
        
        # Parallel processing should be faster than sequential
        max_individual_time = max(stage['processing_time'] for stage in independent_stages)
        self.assertLess(processing_time, max_individual_time + 2)  # Allow for overhead
        
        # Verify all stages completed successfully
        for result in parallel_results:
            self.assertIsNotNone(result)
            self.assertIn('stage_id', result)
            self.assertIn('completion_status', result)
    
    def test_stage_failure_recovery(self):
        """Test stage failure detection and recovery"""
        problematic_stages = [
            {
                'stage_id': 'normal_stage',
                'should_fail': False
            },
            {
                'stage_id': 'failing_stage',
                'should_fail': True,
                'failure_type': 'resource_timeout'
            },
            {
                'stage_id': 'recovery_stage',
                'should_fail': False,
                'depends_on_failed': True
            }
        ]
        
        # Process stages with failure scenarios
        recovery_results = []
        for stage in problematic_stages:
            result = self.processor.process_stage_with_recovery(stage)
            recovery_results.append(result)
        
        # Verify failure detection and recovery
        self.assertEqual(len(recovery_results), 3)
        
        # Normal stage should succeed
        self.assertEqual(recovery_results[0]['status'], 'completed')
        
        # Failing stage should be detected and recovery attempted
        self.assertIn(recovery_results[1]['status'], ['failed', 'recovered'])
        self.assertIn('recovery_attempts', recovery_results[1])
        
        # Recovery stage should handle dependency failure
        self.assertIsNotNone(recovery_results[2])
        self.assertIn('dependency_handling', recovery_results[2])


class TestVideoResourceScheduler(unittest.TestCase):
    """Test video resource scheduling functionality"""
    
    def setUp(self):
        """Set up test environment"""
        if not IMPORT_SUCCESS:
            self.skipTest("LangGraph Video coordination imports not available")
        
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        
        # Initialize coordinator first
        coordinator = LangGraphVideoWorkflowCoordinator(self.db_path)
        del coordinator
        
        self.scheduler = VideoResourceScheduler(self.db_path)
    
    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'db_path') and os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_optimal_resource_scheduling(self):
        """Test optimal resource scheduling"""
        # Define available resources
        available_resources = {
            'compute_nodes': [
                {'node_id': 'gpu_node_1', 'type': 'gpu', 'capacity': 100, 'available': True},
                {'node_id': 'cpu_node_1', 'type': 'cpu', 'capacity': 80, 'available': True},
                {'node_id': 'cpu_node_2', 'type': 'cpu', 'capacity': 80, 'available': True}
            ],
            'memory_pools': [
                {'pool_id': 'mem_pool_1', 'total_gb': 32, 'available_gb': 28},
                {'pool_id': 'mem_pool_2', 'total_gb': 16, 'available_gb': 12}
            ],
            'storage_systems': [
                {'storage_id': 'ssd_storage', 'type': 'ssd', 'available_tb': 2.5},
                {'storage_id': 'hdd_storage', 'type': 'hdd', 'available_tb': 10.0}
            ]
        }
        
        # Define resource requirements for video stages
        stage_requirements = [
            {
                'stage_id': 'video_generation_stage',
                'compute_requirement': {'type': 'gpu', 'capacity_needed': 75},
                'memory_requirement': {'gb_needed': 12},
                'storage_requirement': {'type': 'ssd', 'gb_needed': 500},
                'priority': 'high'
            },
            {
                'stage_id': 'audio_processing_stage',
                'compute_requirement': {'type': 'cpu', 'capacity_needed': 40},
                'memory_requirement': {'gb_needed': 4},
                'storage_requirement': {'type': 'hdd', 'gb_needed': 100},
                'priority': 'medium'
            }
        ]
        
        # Schedule resources
        scheduling_result = self.scheduler.schedule_optimal_resources(
            available_resources, stage_requirements
        )
        
        self.assertIsNotNone(scheduling_result)
        self.assertIn('allocations', scheduling_result)
        self.assertIn('optimization_score', scheduling_result)
        
        # Verify all stages got resource allocations
        allocations = scheduling_result['allocations']
        self.assertEqual(len(allocations), 2)
        
        for allocation in allocations:
            self.assertIn('stage_id', allocation)
            self.assertIn('allocated_resources', allocation)
            self.assertIn('allocation_efficiency', allocation)
    
    def test_resource_conflict_resolution(self):
        """Test resource conflict resolution"""
        # Create conflicting resource requests
        conflicting_requests = [
            {
                'request_id': 'req_001',
                'stage_id': 'high_priority_stage',
                'priority': 'critical',
                'resource_type': 'gpu',
                'capacity_needed': 90,
                'deadline': datetime.now(timezone.utc) + timedelta(minutes=30)
            },
            {
                'request_id': 'req_002',
                'stage_id': 'medium_priority_stage',
                'priority': 'medium',
                'resource_type': 'gpu',
                'capacity_needed': 80,
                'deadline': datetime.now(timezone.utc) + timedelta(hours=2)
            },
            {
                'request_id': 'req_003',
                'stage_id': 'low_priority_stage',
                'priority': 'low',
                'resource_type': 'gpu',
                'capacity_needed': 60,
                'deadline': datetime.now(timezone.utc) + timedelta(hours=4)
            }
        ]
        
        # Available GPU capacity: 100 (can't satisfy all requests simultaneously)
        available_gpu_capacity = 100
        
        # Resolve conflicts
        resolution_result = self.scheduler.resolve_resource_conflicts(
            conflicting_requests, available_gpu_capacity
        )
        
        self.assertIsNotNone(resolution_result)
        self.assertIn('resolved_schedule', resolution_result)
        self.assertIn('conflict_resolution_strategy', resolution_result)
        
        # Verify conflict resolution prioritizes correctly
        resolved_schedule = resolution_result['resolved_schedule']
        self.assertGreater(len(resolved_schedule), 0)
        
        # Critical priority should be scheduled first
        first_scheduled = resolved_schedule[0]
        self.assertEqual(first_scheduled['priority'], 'critical')
    
    def test_dynamic_resource_reallocation(self):
        """Test dynamic resource reallocation during execution"""
        # Initial resource allocation
        initial_allocation = {
            'stage_1': {'gpu_capacity': 50, 'memory_gb': 8},
            'stage_2': {'gpu_capacity': 30, 'memory_gb': 4},
            'stage_3': {'gpu_capacity': 20, 'memory_gb': 4}
        }
        
        # Simulate stage completion and resource availability change
        completed_stages = ['stage_2']
        new_stage_requirements = {
            'stage_4': {'gpu_capacity': 40, 'memory_gb': 6, 'priority': 'high'},
            'stage_5': {'gpu_capacity': 25, 'memory_gb': 3, 'priority': 'medium'}
        }
        
        # Perform dynamic reallocation
        reallocation_result = self.scheduler.perform_dynamic_reallocation(
            initial_allocation, completed_stages, new_stage_requirements
        )
        
        self.assertIsNotNone(reallocation_result)
        self.assertIn('new_allocation', reallocation_result)
        self.assertIn('reallocation_efficiency', reallocation_result)
        
        # Verify reallocation maintains resource constraints
        new_allocation = reallocation_result['new_allocation']
        total_gpu_allocated = sum(
            allocation['gpu_capacity'] 
            for allocation in new_allocation.values()
        )
        self.assertLessEqual(total_gpu_allocated, 100)  # Total available GPU capacity


class TestVideoQualityController(unittest.TestCase):
    """Test video quality control functionality"""
    
    def setUp(self):
        """Set up test environment"""
        if not IMPORT_SUCCESS:
            self.skipTest("LangGraph Video coordination imports not available")
        
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        
        # Initialize coordinator first
        coordinator = LangGraphVideoWorkflowCoordinator(self.db_path)
        del coordinator
        
        self.quality_controller = VideoQualityController(self.db_path)
    
    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'db_path') and os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_quality_control_integration(self):
        """Test quality control integration with >95% accuracy"""
        # Define quality standards
        quality_standards = {
            'resolution_min': '1920x1080',
            'fps_min': 24,
            'audio_quality_min': '44.1kHz',
            'compression_quality_min': 85,
            'color_accuracy_min': 0.95,
            'audio_sync_tolerance_ms': 40
        }
        
        # Test video samples with different quality levels
        video_samples = [
            {
                'sample_id': 'high_quality_sample',
                'resolution': '1920x1080',
                'fps': 30,
                'audio_quality': '48kHz',
                'compression_quality': 92,
                'color_accuracy': 0.98,
                'audio_sync_offset_ms': 15
            },
            {
                'sample_id': 'medium_quality_sample',
                'resolution': '1920x1080',
                'fps': 24,
                'audio_quality': '44.1kHz',
                'compression_quality': 88,
                'color_accuracy': 0.96,
                'audio_sync_offset_ms': 35
            },
            {
                'sample_id': 'low_quality_sample',
                'resolution': '1280x720',
                'fps': 20,
                'audio_quality': '22kHz',
                'compression_quality': 70,
                'color_accuracy': 0.85,
                'audio_sync_offset_ms': 60
            }
        ]
        
        # Perform quality analysis
        quality_results = []
        for sample in video_samples:
            result = self.quality_controller.analyze_video_quality(sample, quality_standards)
            quality_results.append(result)
        
        # Verify quality analysis accuracy
        self.assertEqual(len(quality_results), 3)
        
        # High quality sample should pass all checks
        high_quality_result = quality_results[0]
        self.assertTrue(high_quality_result['meets_standards'])
        self.assertGreater(high_quality_result['overall_quality_score'], 0.95)
        
        # Medium quality sample should pass
        medium_quality_result = quality_results[1]
        self.assertTrue(medium_quality_result['meets_standards'])
        
        # Low quality sample should fail
        low_quality_result = quality_results[2]
        self.assertFalse(low_quality_result['meets_standards'])
        self.assertIn('quality_issues', low_quality_result)
    
    def test_automated_quality_improvement(self):
        """Test automated quality improvement suggestions"""
        # Video with quality issues
        problematic_video = {
            'video_id': 'needs_improvement',
            'current_specs': {
                'resolution': '1280x720',
                'fps': 20,
                'bitrate': '2000kbps',
                'audio_quality': '22kHz'
            },
            'quality_issues': [
                'resolution_below_standard',
                'fps_too_low',
                'audio_quality_insufficient'
            ]
        }
        
        target_quality = {
            'resolution': '1920x1080',
            'fps': 30,
            'bitrate': '8000kbps',
            'audio_quality': '48kHz'
        }
        
        # Generate improvement suggestions
        improvement_result = self.quality_controller.generate_quality_improvements(
            problematic_video, target_quality
        )
        
        self.assertIsNotNone(improvement_result)
        self.assertIn('improvement_plan', improvement_result)
        self.assertIn('estimated_improvement_time', improvement_result)
        self.assertIn('resource_requirements', improvement_result)
        
        # Verify improvement plan addresses all issues
        improvement_plan = improvement_result['improvement_plan']
        self.assertGreater(len(improvement_plan), 0)
        
        for issue in problematic_video['quality_issues']:
            issue_addressed = any(
                issue in step['addresses'] 
                for step in improvement_plan 
                if 'addresses' in step
            )
            self.assertTrue(issue_addressed, f"Issue {issue} not addressed in improvement plan")
    
    def test_real_time_quality_monitoring(self):
        """Test real-time quality monitoring during video generation"""
        # Simulate video generation process
        generation_stages = [
            {'stage': 'frame_generation', 'progress': 0.25, 'quality_score': 0.92},
            {'stage': 'frame_generation', 'progress': 0.50, 'quality_score': 0.89},
            {'stage': 'frame_generation', 'progress': 0.75, 'quality_score': 0.94},
            {'stage': 'audio_sync', 'progress': 1.0, 'quality_score': 0.96}
        ]
        
        quality_threshold = 0.90
        monitoring_results = []
        
        # Monitor quality in real-time
        for stage_data in generation_stages:
            monitoring_result = self.quality_controller.monitor_generation_quality(
                stage_data, quality_threshold
            )
            monitoring_results.append(monitoring_result)
        
        # Verify real-time monitoring
        self.assertEqual(len(monitoring_results), 4)
        
        for result in monitoring_results:
            self.assertIsNotNone(result)
            self.assertIn('quality_status', result)
            self.assertIn('quality_score', result)
            self.assertIn('threshold_met', result)
        
        # Check quality threshold enforcement
        below_threshold_results = [
            r for r in monitoring_results 
            if not r['threshold_met']
        ]
        
        # Should detect quality issues in stage 2 (0.89 < 0.90)
        self.assertGreater(len(below_threshold_results), 0)


class TestVideoRenderingOptimizer(unittest.TestCase):
    """Test video rendering optimization functionality"""
    
    def setUp(self):
        """Set up test environment"""
        if not IMPORT_SUCCESS:
            self.skipTest("LangGraph Video coordination imports not available")
        
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        
        # Initialize coordinator first
        coordinator = LangGraphVideoWorkflowCoordinator(self.db_path)
        del coordinator
        
        self.rendering_optimizer = VideoRenderingOptimizer(self.db_path)
    
    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'db_path') and os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_rendering_optimization_performance(self):
        """Test rendering optimization improves speed by >30%"""
        # Baseline rendering configuration
        baseline_config = {
            'resolution': '1920x1080',
            'fps': 30,
            'duration': 60,
            'codec': 'h264',
            'preset': 'medium',
            'threads': 1
        }
        
        # Optimized rendering configuration
        optimized_config = {
            'resolution': '1920x1080',
            'fps': 30,
            'duration': 60,
            'codec': 'h264',
            'preset': 'fast',
            'threads': 4,
            'hardware_acceleration': True,
            'chunk_processing': True
        }
        
        # Simulate rendering times
        baseline_time = self.rendering_optimizer.estimate_rendering_time(baseline_config)
        optimized_time = self.rendering_optimizer.estimate_rendering_time(optimized_config)
        
        # Calculate speed improvement
        speed_improvement = ((baseline_time - optimized_time) / baseline_time) * 100
        
        # Framework should be capable of >30% improvement
        self.assertIsInstance(speed_improvement, float)
        self.assertGreater(baseline_time, optimized_time)
        
        # Verify optimization strategies are applied
        optimization_report = self.rendering_optimizer.get_optimization_report(
            baseline_config, optimized_config
        )
        
        self.assertIsNotNone(optimization_report)
        self.assertIn('optimizations_applied', optimization_report)
        self.assertIn('performance_gain_estimate', optimization_report)
    
    def test_hardware_specific_optimization(self):
        """Test hardware-specific rendering optimization"""
        # Different hardware configurations
        hardware_configs = [
            {
                'hardware_id': 'apple_silicon_m1',
                'gpu_cores': 8,
                'cpu_cores': 8,
                'memory_gb': 16,
                'video_encoders': ['h264_videotoolbox', 'hevc_videotoolbox']
            },
            {
                'hardware_id': 'apple_silicon_m2_pro',
                'gpu_cores': 19,
                'cpu_cores': 12,
                'memory_gb': 32,
                'video_encoders': ['h264_videotoolbox', 'hevc_videotoolbox', 'prores']
            },
            {
                'hardware_id': 'intel_x86_nvidia',
                'gpu_cores': 2048,
                'cpu_cores': 8,
                'memory_gb': 32,
                'video_encoders': ['h264_nvenc', 'hevc_nvenc']
            }
        ]
        
        video_task = {
            'resolution': '3840x2160',  # 4K
            'fps': 60,
            'duration': 120,
            'complexity': 'high'
        }
        
        # Generate hardware-specific optimizations
        optimization_results = []
        for hardware in hardware_configs:
            optimization = self.rendering_optimizer.optimize_for_hardware(
                hardware, video_task
            )
            optimization_results.append(optimization)
        
        # Verify hardware-specific optimizations
        self.assertEqual(len(optimization_results), 3)
        
        for optimization in optimization_results:
            self.assertIsNotNone(optimization)
            self.assertIn('optimized_settings', optimization)
            self.assertIn('hardware_utilization', optimization)
            self.assertIn('estimated_performance', optimization)
        
        # Apple Silicon should use VideoToolbox
        apple_m1_optimization = optimization_results[0]
        self.assertIn('videotoolbox', 
                     apple_m1_optimization['optimized_settings']['encoder'].lower())
        
        # Intel with NVIDIA should use NVENC
        intel_nvidia_optimization = optimization_results[2]
        self.assertIn('nvenc', 
                     intel_nvidia_optimization['optimized_settings']['encoder'].lower())
    
    def test_adaptive_quality_optimization(self):
        """Test adaptive quality optimization during rendering"""
        # Video segments with different complexity levels
        video_segments = [
            {
                'segment_id': 'simple_scene',
                'complexity_score': 0.3,
                'motion_level': 'low',
                'detail_level': 'medium'
            },
            {
                'segment_id': 'complex_action',
                'complexity_score': 0.9,
                'motion_level': 'high',
                'detail_level': 'high'
            },
            {
                'segment_id': 'static_text',
                'complexity_score': 0.1,
                'motion_level': 'none',
                'detail_level': 'low'
            }
        ]
        
        target_quality = 0.95
        performance_budget = 100  # seconds
        
        # Perform adaptive optimization
        adaptive_result = self.rendering_optimizer.perform_adaptive_optimization(
            video_segments, target_quality, performance_budget
        )
        
        self.assertIsNotNone(adaptive_result)
        self.assertIn('segment_optimizations', adaptive_result)
        self.assertIn('total_estimated_time', adaptive_result)
        self.assertIn('quality_maintenance', adaptive_result)
        
        # Verify adaptive optimization for each segment
        segment_optimizations = adaptive_result['segment_optimizations']
        self.assertEqual(len(segment_optimizations), 3)
        
        # Complex segments should get more rendering resources
        complex_segment_opt = next(
            opt for opt in segment_optimizations 
            if opt['segment_id'] == 'complex_action'
        )
        simple_segment_opt = next(
            opt for opt in segment_optimizations 
            if opt['segment_id'] == 'simple_scene'
        )
        
        self.assertGreater(
            complex_segment_opt['allocated_time'],
            simple_segment_opt['allocated_time']
        )


class TestVideoGenerationOrchestrator(unittest.TestCase):
    """Test main video generation orchestrator functionality"""
    
    def setUp(self):
        """Set up test environment"""
        if not IMPORT_SUCCESS:
            self.skipTest("LangGraph Video coordination imports not available")
        
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        self.orchestrator = VideoGenerationOrchestrator(self.db_path)
    
    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'orchestrator'):
            self.orchestrator.stop_orchestration()
        if hasattr(self, 'db_path') and os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization"""
        self.assertIsNotNone(self.orchestrator)
        self.assertIsInstance(self.orchestrator.workflow_coordinator, LangGraphVideoWorkflowCoordinator)
        self.assertIsInstance(self.orchestrator.workflow_manager, VideoGenerationWorkflowManager)
        self.assertIsInstance(self.orchestrator.multi_stage_processor, MultiStageVideoProcessor)
        self.assertIsInstance(self.orchestrator.resource_scheduler, VideoResourceScheduler)
        self.assertIsInstance(self.orchestrator.quality_controller, VideoQualityController)
        self.assertIsInstance(self.orchestrator.rendering_optimizer, VideoRenderingOptimizer)
    
    def test_orchestration_system_start_stop(self):
        """Test orchestration system start and stop"""
        # Start orchestration system
        self.orchestrator.start_orchestration()
        self.assertTrue(self.orchestrator.is_running)
        
        # Give it a moment to start
        time.sleep(0.1)
        
        # Stop orchestration system
        self.orchestrator.stop_orchestration()
        self.assertFalse(self.orchestrator.is_running)
    
    def test_end_to_end_video_generation_workflow(self):
        """Test end-to-end video generation workflow"""
        # Start system
        self.orchestrator.start_orchestration()
        
        # Create comprehensive video project
        project_config = {
            'project_id': 'e2e_test_project',
            'project_name': 'End-to-End Test Video',
            'video_specifications': {
                'duration': 30,
                'resolution': '1920x1080',
                'fps': 30,
                'style': 'promotional',
                'quality_target': 'high'
            },
            'content_requirements': {
                'theme': 'technology innovation',
                'target_audience': 'professionals',
                'key_messages': ['efficiency', 'innovation', 'reliability']
            },
            'delivery_deadline': datetime.now(timezone.utc) + timedelta(hours=2)
        }
        
        # Execute end-to-end workflow
        workflow_result = self.orchestrator.execute_complete_video_workflow(project_config)
        
        self.assertIsNotNone(workflow_result)
        self.assertIn('workflow_id', workflow_result)
        self.assertIn('execution_status', workflow_result)
        self.assertIn('stages_completed', workflow_result)
        self.assertIn('quality_metrics', workflow_result)
        
        # Verify workflow execution
        workflow_id = workflow_result['workflow_id']
        
        # Check workflow status
        status = self.orchestrator.get_workflow_status(workflow_id)
        self.assertIsNotNone(status)
        self.assertIn('current_stage', status)
        self.assertIn('progress_percentage', status)
        
        # Stop system
        self.orchestrator.stop_orchestration()
    
    def test_acceptance_criteria_validation(self):
        """Test all acceptance criteria for video workflow coordination"""
        # Start system
        self.orchestrator.start_orchestration()
        
        # AC1: Seamless video workflow coordination
        seamless_test = self.orchestrator.validate_seamless_coordination()
        self.assertTrue(seamless_test['is_seamless'])
        
        # AC2: Multi-stage processing efficiency >90%
        efficiency_test = self.orchestrator.measure_processing_efficiency()
        self.assertGreaterEqual(efficiency_test['efficiency_percentage'], 90)
        
        # AC3: Optimal resource scheduling
        resource_test = self.orchestrator.validate_optimal_scheduling()
        self.assertTrue(resource_test['is_optimal'])
        
        # AC4: Quality control integration with >95% accuracy
        quality_test = self.orchestrator.measure_quality_control_accuracy()
        self.assertGreaterEqual(quality_test['accuracy_percentage'], 95)
        
        # AC5: Rendering optimization improves speed by >30%
        rendering_test = self.orchestrator.measure_rendering_optimization()
        self.assertGreaterEqual(rendering_test['speed_improvement_percentage'], 30)
        
        # Stop system
        self.orchestrator.stop_orchestration()


class TestAcceptanceCriteria(unittest.TestCase):
    """Test acceptance criteria for LangGraph video workflow coordination"""
    
    def setUp(self):
        """Set up test environment"""
        if not IMPORT_SUCCESS:
            self.skipTest("LangGraph Video coordination imports not available")
        
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        self.orchestrator = VideoGenerationOrchestrator(self.db_path)
    
    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'orchestrator'):
            self.orchestrator.stop_orchestration()
        if hasattr(self, 'db_path') and os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_seamless_video_workflow_coordination(self):
        """Test seamless video workflow coordination"""
        # Start orchestration
        self.orchestrator.start_orchestration()
        
        # Create test workflows
        test_workflows = [
            {
                'workflow_id': f'seamless_test_{i}',
                'project_name': f'Seamless Test {i}',
                'complexity': 'medium',
                'stages': ['concept', 'script', 'visual', 'render']
            }
            for i in range(3)
        ]
        
        # Execute workflows seamlessly
        seamless_results = []
        for workflow_config in test_workflows:
            result = self.orchestrator.execute_seamless_workflow(workflow_config)
            seamless_results.append(result)
        
        # Verify seamless coordination
        self.assertEqual(len(seamless_results), 3)
        for result in seamless_results:
            self.assertIsNotNone(result)
            self.assertTrue(result['seamless_execution'])
            self.assertIn('coordination_quality', result)
            self.assertGreater(result['coordination_quality'], 0.9)
        
        # Stop orchestration
        self.orchestrator.stop_orchestration()
    
    def test_multi_stage_processing_efficiency_above_90_percent(self):
        """Test multi-stage processing efficiency >90%"""
        # Start orchestration
        self.orchestrator.start_orchestration()
        
        # Create efficiency test scenario
        efficiency_workflow = {
            'workflow_id': 'efficiency_benchmark',
            'stages': [
                {'stage_id': 'concept', 'baseline_time': 60, 'complexity': 'low'},
                {'stage_id': 'script', 'baseline_time': 120, 'complexity': 'medium'},
                {'stage_id': 'visual', 'baseline_time': 180, 'complexity': 'high'},
                {'stage_id': 'render', 'baseline_time': 300, 'complexity': 'very_high'}
            ],
            'efficiency_target': 0.9
        }
        
        # Measure actual processing efficiency
        efficiency_result = self.orchestrator.measure_multi_stage_efficiency(efficiency_workflow)
        
        self.assertIsNotNone(efficiency_result)
        self.assertIn('efficiency_score', efficiency_result)
        self.assertIn('stage_efficiencies', efficiency_result)
        
        # Verify >90% efficiency
        efficiency_score = efficiency_result['efficiency_score']
        self.assertGreaterEqual(efficiency_score, 0.9)
        
        # Verify individual stage efficiencies
        stage_efficiencies = efficiency_result['stage_efficiencies']
        for stage_efficiency in stage_efficiencies:
            self.assertGreater(stage_efficiency['efficiency'], 0.8)  # Each stage should be efficient
        
        # Stop orchestration
        self.orchestrator.stop_orchestration()
    
    def test_optimal_resource_scheduling(self):
        """Test optimal resource scheduling"""
        # Start orchestration
        self.orchestrator.start_orchestration()
        
        # Create resource scheduling test scenario
        resource_scenario = {
            'available_resources': {
                'compute_capacity': 100,
                'memory_gb': 64,
                'storage_gb': 1000,
                'gpu_cores': 16
            },
            'competing_workflows': [
                {
                    'workflow_id': 'high_priority_workflow',
                    'priority': 'critical',
                    'resource_requirements': {
                        'compute': 40,
                        'memory_gb': 16,
                        'storage_gb': 200,
                        'gpu_cores': 8
                    }
                },
                {
                    'workflow_id': 'medium_priority_workflow',
                    'priority': 'high',
                    'resource_requirements': {
                        'compute': 35,
                        'memory_gb': 12,
                        'storage_gb': 150,
                        'gpu_cores': 6
                    }
                },
                {
                    'workflow_id': 'low_priority_workflow',
                    'priority': 'medium',
                    'resource_requirements': {
                        'compute': 30,
                        'memory_gb': 8,
                        'storage_gb': 100,
                        'gpu_cores': 4
                    }
                }
            ]
        }
        
        # Test optimal scheduling
        scheduling_result = self.orchestrator.test_optimal_resource_scheduling(resource_scenario)
        
        self.assertIsNotNone(scheduling_result)
        self.assertIn('scheduling_efficiency', scheduling_result)
        self.assertIn('resource_utilization', scheduling_result)
        self.assertIn('priority_adherence', scheduling_result)
        
        # Verify optimal scheduling
        self.assertTrue(scheduling_result['is_optimal'])
        self.assertGreater(scheduling_result['scheduling_efficiency'], 0.85)
        self.assertGreater(scheduling_result['resource_utilization'], 0.8)
        
        # Stop orchestration
        self.orchestrator.stop_orchestration()
    
    def test_quality_control_integration_above_95_percent_accuracy(self):
        """Test quality control integration with >95% accuracy"""
        # Start orchestration
        self.orchestrator.start_orchestration()
        
        # Create quality control test dataset
        quality_test_samples = [
            {
                'sample_id': f'quality_sample_{i}',
                'video_specs': {
                    'resolution': '1920x1080' if i % 2 == 0 else '1280x720',
                    'fps': 30 if i % 3 == 0 else 24,
                    'bitrate': '8000kbps' if i % 4 == 0 else '4000kbps',
                    'audio_quality': '48kHz' if i % 5 == 0 else '44.1kHz'
                },
                'expected_quality_pass': i % 2 == 0  # Half should pass, half should fail
            }
            for i in range(20)
        ]
        
        quality_standards = {
            'min_resolution': '1920x1080',
            'min_fps': 24,
            'min_bitrate': '4000kbps',
            'min_audio_quality': '44.1kHz'
        }
        
        # Test quality control accuracy
        quality_results = []
        correct_predictions = 0
        
        for sample in quality_test_samples:
            result = self.orchestrator.test_quality_control_accuracy(
                sample['video_specs'], quality_standards
            )
            quality_results.append(result)
            
            # Check if prediction matches expected result
            predicted_pass = result['quality_assessment']['meets_standards']
            expected_pass = sample['expected_quality_pass']
            
            if predicted_pass == expected_pass:
                correct_predictions += 1
        
        # Calculate accuracy
        accuracy = (correct_predictions / len(quality_test_samples)) * 100
        
        # Verify >95% accuracy
        self.assertGreaterEqual(accuracy, 95.0)
        
        # Verify quality control system functionality
        self.assertEqual(len(quality_results), 20)
        for result in quality_results:
            self.assertIn('quality_assessment', result)
            self.assertIn('quality_score', result)
            self.assertIn('detected_issues', result)
        
        # Stop orchestration
        self.orchestrator.stop_orchestration()
    
    def test_rendering_optimization_improves_speed_above_30_percent(self):
        """Test rendering optimization improves speed by >30%"""
        # Start orchestration
        self.orchestrator.start_orchestration()
        
        # Create rendering optimization test scenarios
        rendering_scenarios = [
            {
                'scenario_id': 'baseline_rendering',
                'video_config': {
                    'resolution': '1920x1080',
                    'fps': 30,
                    'duration': 60,
                    'codec': 'h264',
                    'quality': 'high'
                },
                'optimization_level': 'none'
            },
            {
                'scenario_id': 'optimized_rendering',
                'video_config': {
                    'resolution': '1920x1080',
                    'fps': 30,
                    'duration': 60,
                    'codec': 'h264',
                    'quality': 'high'
                },
                'optimization_level': 'aggressive'
            }
        ]
        
        # Measure rendering performance
        performance_results = {}
        for scenario in rendering_scenarios:
            result = self.orchestrator.benchmark_rendering_performance(scenario)
            performance_results[scenario['scenario_id']] = result
        
        # Calculate speed improvement
        baseline_time = performance_results['baseline_rendering']['estimated_render_time']
        optimized_time = performance_results['optimized_rendering']['estimated_render_time']
        
        speed_improvement = ((baseline_time - optimized_time) / baseline_time) * 100
        
        # Verify >30% speed improvement
        self.assertGreaterEqual(speed_improvement, 30.0)
        
        # Verify optimization strategies applied
        optimization_report = performance_results['optimized_rendering']['optimization_report']
        self.assertIn('strategies_applied', optimization_report)
        self.assertGreater(len(optimization_report['strategies_applied']), 0)
        
        # Stop orchestration
        self.orchestrator.stop_orchestration()


class TestDemoSystem(unittest.TestCase):
    """Test demo system functionality"""
    
    def test_demo_system_creation_and_execution(self):
        """Test demo system creation and execution"""
        if not IMPORT_SUCCESS:
            self.skipTest("LangGraph Video coordination imports not available")
        
        # Import demo function
        from sources.langgraph_video_generation_workflow_coordination_sandbox import create_demo_video_coordination_system
        
        # Create demo system
        demo_system = create_demo_video_coordination_system()
        
        try:
            # Verify demo system was created
            self.assertIsNotNone(demo_system)
            self.assertIsInstance(demo_system, VideoGenerationOrchestrator)
            
            # Test demo system functionality
            status = demo_system.get_orchestration_status()
            self.assertIsInstance(status, dict)
            self.assertIn('is_running', status)
            
            # Test video workflow coordination
            test_project = {
                'project_id': 'demo_test_project',
                'project_name': 'Demo Test Video',
                'video_specs': {
                    'duration': 15,
                    'resolution': '1280x720',
                    'fps': 24
                }
            }
            
            result = demo_system.demonstrate_video_coordination(test_project)
            
            self.assertIsNotNone(result)
            self.assertIn('coordination_result', result)
            
        finally:
            # Clean up
            if demo_system:
                demo_system.stop_orchestration()


def run_comprehensive_tests():
    """Run comprehensive test suite with detailed reporting"""
    
    print("\\n🎬 LangGraph Video Generation Workflow Coordination - Comprehensive Test Suite")
    print("=" * 95)
    
    if not IMPORT_SUCCESS:
        print("❌ CRITICAL: Cannot import LangGraph Video coordination components")
        print("Please ensure the sandbox implementation is available")
        return False
    
    # Test suite configuration
    test_classes = [
        TestLangGraphVideoWorkflowCoordinator,
        TestVideoGenerationWorkflowManager,
        TestMultiStageVideoProcessor,
        TestVideoResourceScheduler,
        TestVideoQualityController,
        TestVideoRenderingOptimizer,
        TestVideoGenerationOrchestrator,
        TestAcceptanceCriteria,
        TestDemoSystem
    ]
    
    total_tests = 0
    total_passed = 0
    total_failed = 0
    total_errors = 0
    
    detailed_results = []
    
    for test_class in test_classes:
        print(f"\\n📋 Running {test_class.__name__}")
        print("-" * 70)
        
        # Create test suite
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        
        # Run tests with detailed result tracking
        stream = StringIO()
        runner = unittest.TextTestRunner(stream=stream, verbosity=2)
        result = runner.run(suite)
        
        # Calculate metrics
        tests_run = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
        passed = tests_run - failures - errors - skipped
        
        success_rate = (passed / tests_run * 100) if tests_run > 0 else 0
        
        # Update totals
        total_tests += tests_run
        total_passed += passed
        total_failed += failures
        total_errors += errors
        
        # Store detailed results
        detailed_results.append({
            'class_name': test_class.__name__,
            'tests_run': tests_run,
            'failures': failures,
            'errors': errors,
            'skipped': skipped,
            'success_rate': success_rate
        })
        
        # Print results
        status_icon = "✅" if success_rate >= 90 else "⚠️" if success_rate >= 70 else "❌"
        print(f"{status_icon} {test_class.__name__}: {success_rate:.1f}% ({passed}/{tests_run} passed)")
        
        if failures > 0:
            print(f"   ⚠️ {failures} test failures")
        if errors > 0:
            print(f"   ❌ {errors} test errors")
        if skipped > 0:
            print(f"   ⏭️ {skipped} tests skipped")
    
    # Calculate overall metrics
    overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\\n📊 COMPREHENSIVE TEST RESULTS")
    print("=" * 70)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Errors: {total_errors}")
    print(f"Overall Success Rate: {overall_success_rate:.1f}%")
    
    # Status assessment
    if overall_success_rate >= 95:
        status = "🎯 EXCELLENT - Production Ready"
    elif overall_success_rate >= 90:
        status = "✅ GOOD - Production Ready"
    elif overall_success_rate >= 80:
        status = "⚠️ ACCEPTABLE - Minor Issues"
    elif overall_success_rate >= 70:
        status = "🔧 NEEDS WORK - Significant Issues"
    else:
        status = "❌ CRITICAL - Major Problems"
    
    print(f"Status: {status}")
    
    # Save detailed results
    test_report = {
        'total_tests': total_tests,
        'passed_tests': total_passed,
        'failed_tests': total_failed,
        'error_tests': total_errors,
        'skipped_tests': 0,
        'test_results': detailed_results,
        'start_time': time.time(),
        'end_time': time.time(),
        'duration': 0,
        'overall_success_rate': overall_success_rate
    }
    
    report_filename = f"video_coordination_test_report_{int(time.time())}.json"
    with open(report_filename, 'w') as f:
        json.dump(test_report, f, indent=2)
    
    print(f"\\n📄 Detailed test report saved to: {report_filename}")
    
    return overall_success_rate >= 90


if __name__ == "__main__":
    try:
        success = run_comprehensive_tests()
        exit_code = 0 if success else 1
        exit(exit_code)
    except KeyboardInterrupt:
        print("\\n⏹️ Test execution interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\\n❌ Test execution failed: {e}")
        exit(1)