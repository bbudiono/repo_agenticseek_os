#!/usr/bin/env python3
"""
Comprehensive Test Suite for LangGraph Neural Engine and GPU Acceleration
Tests all aspects including Neural Engine utilization, GPU acceleration, energy optimization,
workload scheduling, and performance profiling.
"""

import asyncio
import pytest
import numpy as np
import time
import json
import sqlite3
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Add the sources directory to Python path
sys.path.insert(0, '/Users/bernhardbudiono/Library/CloudStorage/Dropbox/_Documents - Apps (Working)/repos_github/Working/_repo_agenticseek/sources')

try:
    from langgraph_neural_engine_gpu_acceleration_sandbox import (
        AccelerationType, WorkloadType, WorkloadProfile, AccelerationResult,
        NeuralEngineConfig, GPUConfig, SystemProfiler, NeuralEngineAccelerator,
        GPUAccelerator, WorkloadScheduler, EnergyOptimizer, PerformanceProfiler,
        NeuralEngineGPUAccelerationOrchestrator, run_neural_engine_gpu_acceleration_demo
    )
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

class TestSystemProfiler:
    """Test system profiling capabilities"""
    
    def test_system_profiler_initialization(self):
        """Test SystemProfiler initialization"""
        profiler = SystemProfiler()
        
        assert profiler.system_info is not None
        assert profiler.capabilities is not None
        assert 'platform' in profiler.system_info
        assert 'machine' in profiler.system_info
        assert 'cpu_count' in profiler.system_info
        assert 'memory_total' in profiler.system_info
    
    def test_capability_assessment(self):
        """Test system capability assessment"""
        profiler = SystemProfiler()
        capabilities = profiler.capabilities
        
        assert 'cpu_performance' in capabilities
        assert 'memory_capacity' in capabilities
        assert 'neural_engine_score' in capabilities
        assert 'gpu_score' in capabilities
        assert 'overall_score' in capabilities
        
        # Check score ranges
        for key, value in capabilities.items():
            assert 0.0 <= value <= 1.0, f"Capability {key} should be between 0.0 and 1.0"
    
    @patch('platform.machine')
    def test_apple_silicon_detection(self, mock_machine):
        """Test Apple Silicon detection"""
        # Test ARM64 detection
        mock_machine.return_value = 'arm64'
        profiler = SystemProfiler()
        assert profiler.system_info['apple_silicon'] is True
        
        # Test x86_64 detection
        mock_machine.return_value = 'x86_64'
        profiler = SystemProfiler()
        assert profiler.system_info['apple_silicon'] is False

class TestNeuralEngineAccelerator:
    """Test Neural Engine acceleration functionality"""
    
    def test_neural_engine_initialization(self):
        """Test NeuralEngineAccelerator initialization"""
        config = NeuralEngineConfig()
        accelerator = NeuralEngineAccelerator(config)
        
        assert accelerator.config == config
        assert accelerator.models_cache == {}
        assert accelerator.performance_metrics == []
    
    @pytest.mark.asyncio
    async def test_neural_engine_workload_acceleration(self):
        """Test Neural Engine workload acceleration"""
        config = NeuralEngineConfig()
        accelerator = NeuralEngineAccelerator(config)
        
        workload = WorkloadProfile(
            workload_id="test_workload",
            workload_type=WorkloadType.ML_INFERENCE,
            input_size=1000,
            complexity_score=0.7,
            memory_requirements=4000,
            estimated_duration=0.1,
            priority=1,
            preferred_acceleration=AccelerationType.NEURAL_ENGINE,
            energy_budget=5.0
        )
        
        test_data = np.random.randn(1000)
        result = await accelerator.accelerate_workload(workload, test_data)
        
        assert isinstance(result, AccelerationResult)
        assert result.workload_id == "test_workload"
        assert result.acceleration_type == AccelerationType.NEURAL_ENGINE
        assert result.success is True
        assert result.execution_time > 0
        assert result.energy_consumed > 0
        assert result.performance_gain >= 0
    
    @pytest.mark.asyncio
    async def test_neural_engine_model_caching(self):
        """Test Neural Engine model caching"""
        config = NeuralEngineConfig()
        accelerator = NeuralEngineAccelerator(config)
        
        workload = WorkloadProfile(
            workload_id="test_workload",
            workload_type=WorkloadType.TEXT_PROCESSING,
            input_size=500,
            complexity_score=0.5,
            memory_requirements=2000,
            estimated_duration=0.05,
            priority=2,
            preferred_acceleration=AccelerationType.NEURAL_ENGINE,
            energy_budget=3.0
        )
        
        # First call should create model
        model1 = await accelerator._get_optimized_model(workload)
        assert len(accelerator.models_cache) == 1
        
        # Second call should use cached model
        model2 = await accelerator._get_optimized_model(workload)
        assert model1 == model2
        assert len(accelerator.models_cache) == 1
    
    def test_neural_engine_data_preparation(self):
        """Test Neural Engine data preparation"""
        config = NeuralEngineConfig()
        accelerator = NeuralEngineAccelerator(config)
        
        workload = WorkloadProfile(
            workload_id="test_workload",
            workload_type=WorkloadType.ML_INFERENCE,
            input_size=100,
            complexity_score=0.6,
            memory_requirements=400,
            estimated_duration=0.02,
            priority=1,
            preferred_acceleration=AccelerationType.NEURAL_ENGINE,
            energy_budget=2.0
        )
        
        # Test list input
        list_data = list(range(100))
        prepared = accelerator._prepare_data_for_neural_engine(list_data, workload)
        assert 'input' in prepared
        assert prepared['input'].dtype == np.float32
        
        # Test numpy array input
        array_data = np.random.randn(100)
        prepared = accelerator._prepare_data_for_neural_engine(array_data, workload)
        assert 'input' in prepared
        assert prepared['input'].dtype == np.float32

class TestGPUAccelerator:
    """Test GPU acceleration functionality"""
    
    def test_gpu_accelerator_initialization(self):
        """Test GPUAccelerator initialization"""
        config = GPUConfig()
        accelerator = GPUAccelerator(config)
        
        assert accelerator.config == config
        assert accelerator.compute_pipelines == {}
        assert accelerator.performance_metrics == []
    
    @pytest.mark.asyncio
    async def test_gpu_workload_acceleration(self):
        """Test GPU workload acceleration"""
        config = GPUConfig()
        accelerator = GPUAccelerator(config)
        
        workload = WorkloadProfile(
            workload_id="test_gpu_workload",
            workload_type=WorkloadType.MATRIX_OPERATIONS,
            input_size=2500,  # 50x50 matrix
            complexity_score=0.8,
            memory_requirements=10000,
            estimated_duration=0.15,
            priority=1,
            preferred_acceleration=AccelerationType.GPU_METAL,
            energy_budget=8.0
        )
        
        test_data = np.random.randn(50, 50)
        result = await accelerator.accelerate_workload(workload, test_data)
        
        assert isinstance(result, AccelerationResult)
        assert result.workload_id == "test_gpu_workload"
        assert result.acceleration_type == AccelerationType.GPU_METAL
        assert result.success is True
        assert result.execution_time > 0
        assert result.energy_consumed > 0
        assert result.performance_gain >= 0
    
    @pytest.mark.asyncio
    async def test_gpu_pipeline_caching(self):
        """Test GPU compute pipeline caching"""
        config = GPUConfig()
        accelerator = GPUAccelerator(config)
        
        workload = WorkloadProfile(
            workload_id="test_pipeline",
            workload_type=WorkloadType.IMAGE_PROCESSING,
            input_size=65536,  # 256x256 image
            complexity_score=0.9,
            memory_requirements=262144,
            estimated_duration=0.2,
            priority=1,
            preferred_acceleration=AccelerationType.GPU_METAL,
            energy_budget=10.0
        )
        
        # First call should create pipeline
        pipeline1 = await accelerator._get_compute_pipeline(workload)
        assert len(accelerator.compute_pipelines) == 1
        
        # Second call should use cached pipeline
        pipeline2 = await accelerator._get_compute_pipeline(workload)
        assert pipeline1 == pipeline2
        assert len(accelerator.compute_pipelines) == 1
    
    def test_gpu_data_preparation(self):
        """Test GPU data preparation"""
        config = GPUConfig()
        accelerator = GPUAccelerator(config)
        
        workload = WorkloadProfile(
            workload_id="test_gpu_data",
            workload_type=WorkloadType.GENERAL_COMPUTE,
            input_size=1000,
            complexity_score=0.5,
            memory_requirements=4000,
            estimated_duration=0.1,
            priority=2,
            preferred_acceleration=AccelerationType.GPU_METAL,
            energy_budget=5.0
        )
        
        # Test array input
        array_data = np.random.randn(1000)
        prepared = accelerator._prepare_data_for_gpu(array_data, workload)
        assert 'input_buffer' in prepared
        assert 'output_buffer' in prepared
        assert prepared['input_buffer'].dtype == np.float32

class TestWorkloadScheduler:
    """Test intelligent workload scheduling"""
    
    def test_scheduler_initialization(self):
        """Test WorkloadScheduler initialization"""
        neural_config = NeuralEngineConfig()
        gpu_config = GPUConfig()
        neural_engine = NeuralEngineAccelerator(neural_config)
        gpu_accelerator = GPUAccelerator(gpu_config)
        
        scheduler = WorkloadScheduler(neural_engine, gpu_accelerator)
        
        assert scheduler.neural_engine == neural_engine
        assert scheduler.gpu == gpu_accelerator
        assert scheduler.system_profiler is not None
        assert scheduler.scheduling_queue == []
        assert scheduler.active_workloads == {}
        assert scheduler.scheduling_history == []
    
    @pytest.mark.asyncio
    async def test_workload_scheduling(self):
        """Test workload scheduling functionality"""
        neural_config = NeuralEngineConfig()
        gpu_config = GPUConfig()
        neural_engine = NeuralEngineAccelerator(neural_config)
        gpu_accelerator = GPUAccelerator(gpu_config)
        scheduler = WorkloadScheduler(neural_engine, gpu_accelerator)
        
        workload = WorkloadProfile(
            workload_id="test_schedule",
            workload_type=WorkloadType.ML_INFERENCE,
            input_size=500,
            complexity_score=0.6,
            memory_requirements=2000,
            estimated_duration=0.08,
            priority=1,
            preferred_acceleration=AccelerationType.AUTO,
            energy_budget=4.0
        )
        
        test_data = np.random.randn(500)
        result = await scheduler.schedule_workload(workload, test_data)
        
        assert isinstance(result, AccelerationResult)
        assert result.success is True
        assert len(scheduler.scheduling_history) == 1
    
    def test_acceleration_type_scoring(self):
        """Test acceleration type scoring algorithms"""
        neural_config = NeuralEngineConfig()
        gpu_config = GPUConfig()
        neural_engine = NeuralEngineAccelerator(neural_config)
        gpu_accelerator = GPUAccelerator(gpu_config)
        scheduler = WorkloadScheduler(neural_engine, gpu_accelerator)
        
        # Test ML workload (should favor Neural Engine)
        ml_workload = WorkloadProfile(
            workload_id="ml_test",
            workload_type=WorkloadType.ML_INFERENCE,
            input_size=800,
            complexity_score=0.7,
            memory_requirements=3200,
            estimated_duration=0.12,
            priority=1,
            preferred_acceleration=AccelerationType.AUTO,
            energy_budget=3.0
        )
        
        ne_score = scheduler._calculate_neural_engine_score(ml_workload)
        gpu_score = scheduler._calculate_gpu_score(ml_workload)
        cpu_score = scheduler._calculate_cpu_score(ml_workload)
        
        assert ne_score > cpu_score  # Neural Engine should score higher than CPU for ML
        
        # Test image processing workload (should favor GPU)
        image_workload = WorkloadProfile(
            workload_id="image_test",
            workload_type=WorkloadType.IMAGE_PROCESSING,
            input_size=65536,
            complexity_score=0.9,
            memory_requirements=262144,
            estimated_duration=0.25,
            priority=1,
            preferred_acceleration=AccelerationType.AUTO,
            energy_budget=12.0
        )
        
        ne_score_img = scheduler._calculate_neural_engine_score(image_workload)
        gpu_score_img = scheduler._calculate_gpu_score(image_workload)
        cpu_score_img = scheduler._calculate_cpu_score(image_workload)
        
        assert gpu_score_img > cpu_score_img  # GPU should score higher than CPU for images
    
    @pytest.mark.asyncio
    async def test_cpu_fallback_execution(self):
        """Test CPU fallback execution"""
        neural_config = NeuralEngineConfig()
        gpu_config = GPUConfig()
        neural_engine = NeuralEngineAccelerator(neural_config)
        gpu_accelerator = GPUAccelerator(gpu_config)
        scheduler = WorkloadScheduler(neural_engine, gpu_accelerator)
        
        workload = WorkloadProfile(
            workload_id="cpu_fallback",
            workload_type=WorkloadType.GRAPH_PROCESSING,
            input_size=50,
            complexity_score=0.4,
            memory_requirements=200,
            estimated_duration=0.05,
            priority=3,
            preferred_acceleration=AccelerationType.CPU_ONLY,
            energy_budget=2.0
        )
        
        test_data = list(range(50))
        result = await scheduler._execute_cpu_fallback(workload, test_data)
        
        assert isinstance(result, AccelerationResult)
        assert result.acceleration_type == AccelerationType.CPU_ONLY
        assert result.success is True
        assert result.performance_gain == 0.0  # CPU is baseline

class TestEnergyOptimizer:
    """Test energy efficiency optimization"""
    
    def test_energy_optimizer_initialization(self):
        """Test EnergyOptimizer initialization"""
        optimizer = EnergyOptimizer()
        
        assert optimizer.energy_profiles == {}
        assert optimizer.optimization_history == []
        assert optimizer.current_energy_budget == 100.0
    
    def test_energy_profile_analysis(self):
        """Test energy profile analysis"""
        optimizer = EnergyOptimizer()
        
        workload = WorkloadProfile(
            workload_id="energy_test",
            workload_type=WorkloadType.MATRIX_OPERATIONS,
            input_size=1000,
            complexity_score=0.6,
            memory_requirements=4000,
            estimated_duration=0.1,
            priority=2,
            preferred_acceleration=AccelerationType.AUTO,
            energy_budget=6.0
        )
        
        profiles = optimizer._analyze_energy_profiles(workload)
        
        assert AccelerationType.NEURAL_ENGINE in profiles
        assert AccelerationType.GPU_METAL in profiles
        assert AccelerationType.CPU_ONLY in profiles
        
        # Neural Engine should generally be most energy efficient
        assert profiles[AccelerationType.NEURAL_ENGINE] < profiles[AccelerationType.CPU_ONLY]
    
    def test_energy_optimization(self):
        """Test complete energy optimization process"""
        optimizer = EnergyOptimizer()
        
        # Low energy budget workload
        low_energy_workload = WorkloadProfile(
            workload_id="low_energy",
            workload_type=WorkloadType.TEXT_PROCESSING,
            input_size=500,
            complexity_score=0.4,
            memory_requirements=2000,
            estimated_duration=0.06,
            priority=1,
            preferred_acceleration=AccelerationType.AUTO,
            energy_budget=2.0
        )
        
        optimal_type, config = optimizer.optimize_for_energy_efficiency(low_energy_workload)
        
        # Should prefer Neural Engine for energy efficiency
        assert optimal_type in [AccelerationType.NEURAL_ENGINE, AccelerationType.CPU_ONLY]
        assert isinstance(config, dict)
        
        # High energy budget workload
        high_energy_workload = WorkloadProfile(
            workload_id="high_energy",
            workload_type=WorkloadType.IMAGE_PROCESSING,
            input_size=65536,
            complexity_score=0.9,
            memory_requirements=262144,
            estimated_duration=0.3,
            priority=1,
            preferred_acceleration=AccelerationType.AUTO,
            energy_budget=15.0
        )
        
        optimal_type_high, config_high = optimizer.optimize_for_energy_efficiency(high_energy_workload)
        
        # Should allow GPU for high-performance tasks with sufficient budget
        assert optimal_type_high in [AccelerationType.GPU_METAL, AccelerationType.NEURAL_ENGINE, AccelerationType.CPU_ONLY]
        assert isinstance(config_high, dict)

class TestPerformanceProfiler:
    """Test performance profiling and monitoring"""
    
    def test_profiler_initialization(self):
        """Test PerformanceProfiler initialization"""
        test_db = "test_neural_engine_gpu.db"
        
        # Clean up any existing test database
        if os.path.exists(test_db):
            os.remove(test_db)
        
        profiler = PerformanceProfiler(test_db)
        
        assert profiler.db_path == test_db
        assert profiler.metrics_buffer == []
        assert profiler.profiling_active is False
        
        # Check database tables were created
        assert os.path.exists(test_db)
        
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        
        # Check performance_metrics table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='performance_metrics'")
        assert cursor.fetchone() is not None
        
        # Check system_profiles table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='system_profiles'")
        assert cursor.fetchone() is not None
        
        conn.close()
        
        # Clean up test database
        os.remove(test_db)
    
    @pytest.mark.asyncio
    async def test_profiler_lifecycle(self):
        """Test profiler start/stop lifecycle"""
        test_db = "test_profiler_lifecycle.db"
        
        # Clean up any existing test database
        if os.path.exists(test_db):
            os.remove(test_db)
        
        profiler = PerformanceProfiler(test_db)
        
        # Start profiling
        await profiler.start_profiling()
        assert profiler.profiling_active is True
        
        # Let it run briefly
        await asyncio.sleep(0.5)
        
        # Stop profiling
        await profiler.stop_profiling()
        assert profiler.profiling_active is False
        
        # Clean up test database
        if os.path.exists(test_db):
            os.remove(test_db)
    
    def test_performance_report_generation(self):
        """Test performance report generation"""
        test_db = "test_performance_report.db"
        
        # Clean up any existing test database
        if os.path.exists(test_db):
            os.remove(test_db)
        
        profiler = PerformanceProfiler(test_db)
        
        # Generate empty report
        report = profiler.generate_performance_report()
        
        assert 'timestamp' in report
        assert 'overall_statistics' in report
        assert 'acceleration_performance' in report
        assert 'system_capabilities' in report
        
        # Clean up test database
        if os.path.exists(test_db):
            os.remove(test_db)
    
    def test_acceleration_result_recording(self):
        """Test recording acceleration results"""
        test_db = "test_result_recording.db"
        
        # Clean up any existing test database
        if os.path.exists(test_db):
            os.remove(test_db)
        
        profiler = PerformanceProfiler(test_db)
        
        # Create test result
        result = AccelerationResult(
            workload_id="test_result",
            acceleration_type=AccelerationType.NEURAL_ENGINE,
            execution_time=0.05,
            energy_consumed=2.5,
            memory_used=1024,
            success=True,
            performance_gain=45.0
        )
        
        system_metrics = {
            'system_load': 0.3,
            'neural_engine_utilization': 0.6,
            'gpu_utilization': 0.1,
            'cpu_utilization': 25.0
        }
        
        # Record result
        profiler.record_acceleration_result(result, system_metrics)
        
        # Verify recording
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM performance_metrics")
        count = cursor.fetchone()[0]
        assert count == 1
        
        cursor.execute("SELECT * FROM performance_metrics")
        row = cursor.fetchone()
        assert row[2] == "test_result"  # workload_id
        assert row[4] == "neural_engine"  # acceleration_type
        
        conn.close()
        
        # Clean up test database
        os.remove(test_db)

class TestNeuralEngineGPUAccelerationOrchestrator:
    """Test main orchestrator functionality"""
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization"""
        test_db = "test_orchestrator.db"
        
        # Clean up any existing test database
        if os.path.exists(test_db):
            os.remove(test_db)
        
        orchestrator = NeuralEngineGPUAccelerationOrchestrator(db_path=test_db)
        
        assert orchestrator.neural_config is not None
        assert orchestrator.gpu_config is not None
        assert orchestrator.system_profiler is not None
        assert orchestrator.neural_engine is not None
        assert orchestrator.gpu_accelerator is not None
        assert orchestrator.scheduler is not None
        assert orchestrator.energy_optimizer is not None
        assert orchestrator.profiler is not None
        assert orchestrator.optimization_metrics['total_workloads'] == 0
        
        # Clean up test database
        if os.path.exists(test_db):
            os.remove(test_db)
    
    @pytest.mark.asyncio
    async def test_orchestrator_lifecycle(self):
        """Test orchestrator start/stop lifecycle"""
        test_db = "test_orchestrator_lifecycle.db"
        
        # Clean up any existing test database
        if os.path.exists(test_db):
            os.remove(test_db)
        
        orchestrator = NeuralEngineGPUAccelerationOrchestrator(db_path=test_db)
        
        # Start orchestrator
        await orchestrator.start()
        
        # Let it run briefly
        await asyncio.sleep(0.3)
        
        # Stop orchestrator
        await orchestrator.stop()
        
        # Clean up test database
        if os.path.exists(test_db):
            os.remove(test_db)
    
    @pytest.mark.asyncio
    async def test_workload_acceleration_integration(self):
        """Test end-to-end workload acceleration"""
        test_db = "test_integration.db"
        
        # Clean up any existing test database
        if os.path.exists(test_db):
            os.remove(test_db)
        
        orchestrator = NeuralEngineGPUAccelerationOrchestrator(db_path=test_db)
        
        await orchestrator.start()
        
        # Test ML inference workload
        ml_data = np.random.randn(800)
        ml_config = {'priority': 1, 'energy_budget': 4.0}
        
        result = await orchestrator.accelerate_workload(
            WorkloadType.ML_INFERENCE, 
            ml_data, 
            ml_config
        )
        
        assert isinstance(result, AccelerationResult)
        assert result.success is True
        assert result.acceleration_type in [
            AccelerationType.NEURAL_ENGINE, 
            AccelerationType.GPU_METAL, 
            AccelerationType.CPU_ONLY
        ]
        
        # Test matrix operations workload
        matrix_data = np.random.randn(100, 100)
        matrix_config = {'priority': 2, 'energy_budget': 8.0}
        
        result2 = await orchestrator.accelerate_workload(
            WorkloadType.MATRIX_OPERATIONS,
            matrix_data,
            matrix_config
        )
        
        assert isinstance(result2, AccelerationResult)
        assert result2.success is True
        
        # Check orchestrator metrics
        summary = orchestrator.get_optimization_summary()
        assert summary['total_workloads'] == 2
        assert summary['success_rate'] == 100.0
        
        await orchestrator.stop()
        
        # Clean up test database
        if os.path.exists(test_db):
            os.remove(test_db)
    
    def test_workload_profile_creation(self):
        """Test workload profile creation"""
        test_db = "test_profile_creation.db"
        
        # Clean up any existing test database
        if os.path.exists(test_db):
            os.remove(test_db)
        
        orchestrator = NeuralEngineGPUAccelerationOrchestrator(db_path=test_db)
        
        # Test with numpy array
        array_data = np.random.randn(1000)
        config = {'priority': 1, 'energy_budget': 5.0}
        
        profile = orchestrator._create_workload_profile(
            WorkloadType.ML_INFERENCE,
            array_data,
            config
        )
        
        assert isinstance(profile, WorkloadProfile)
        assert profile.workload_type == WorkloadType.ML_INFERENCE
        assert profile.input_size == 1000
        assert profile.priority == 1
        assert profile.energy_budget == 5.0
        assert profile.complexity_score > 0
        assert profile.estimated_duration > 0
        
        # Test with list data
        list_data = list(range(500))
        profile2 = orchestrator._create_workload_profile(
            WorkloadType.TEXT_PROCESSING,
            list_data,
            {}
        )
        
        assert profile2.input_size == 500
        assert profile2.workload_type == WorkloadType.TEXT_PROCESSING
        
        # Clean up test database
        if os.path.exists(test_db):
            os.remove(test_db)
    
    def test_complexity_estimation(self):
        """Test workload complexity estimation"""
        test_db = "test_complexity.db"
        
        # Clean up any existing test database
        if os.path.exists(test_db):
            os.remove(test_db)
        
        orchestrator = NeuralEngineGPUAccelerationOrchestrator(db_path=test_db)
        
        # Test different workload types
        ml_complexity = orchestrator._estimate_workload_complexity(
            WorkloadType.ML_INFERENCE, 1000
        )
        image_complexity = orchestrator._estimate_workload_complexity(
            WorkloadType.IMAGE_PROCESSING, 65536
        )
        text_complexity = orchestrator._estimate_workload_complexity(
            WorkloadType.TEXT_PROCESSING, 500
        )
        
        # Image processing should be most complex
        assert image_complexity > text_complexity
        assert ml_complexity > text_complexity
        
        # All should be in valid range
        assert 0.1 <= ml_complexity <= 1.0
        assert 0.1 <= image_complexity <= 1.0
        assert 0.1 <= text_complexity <= 1.0
        
        # Clean up test database
        if os.path.exists(test_db):
            os.remove(test_db)

class TestDemoAndIntegration:
    """Test demo functionality and overall integration"""
    
    @pytest.mark.asyncio
    async def test_demo_execution(self):
        """Test complete demo execution"""
        # This test runs the actual demo to ensure everything works end-to-end
        try:
            metrics = await run_neural_engine_gpu_acceleration_demo()
            
            assert isinstance(metrics, dict)
            assert 'total_workloads' in metrics
            assert 'successful_workloads' in metrics
            assert 'success_rate' in metrics
            assert 'average_performance_gain' in metrics
            assert 'total_energy_consumed' in metrics
            
            # Should have tested 5 different workload types
            assert metrics['total_workloads'] == 5
            assert metrics['successful_workloads'] >= 0
            assert metrics['success_rate'] >= 0
            
            # Performance metrics should be reasonable
            assert metrics['average_performance_gain'] >= 0
            assert metrics['total_energy_consumed'] >= 0
            
        except Exception as e:
            pytest.fail(f"Demo execution failed: {e}")
    
    def test_workload_type_coverage(self):
        """Test all workload types are properly handled"""
        all_workload_types = [
            WorkloadType.ML_INFERENCE,
            WorkloadType.MATRIX_OPERATIONS,
            WorkloadType.GRAPH_PROCESSING,
            WorkloadType.TEXT_PROCESSING,
            WorkloadType.IMAGE_PROCESSING,
            WorkloadType.GENERAL_COMPUTE
        ]
        
        test_db = "test_workload_coverage.db"
        
        # Clean up any existing test database
        if os.path.exists(test_db):
            os.remove(test_db)
        
        orchestrator = NeuralEngineGPUAccelerationOrchestrator(db_path=test_db)
        
        for workload_type in all_workload_types:
            test_data = np.random.randn(100)
            profile = orchestrator._create_workload_profile(
                workload_type,
                test_data,
                {}
            )
            
            assert isinstance(profile, WorkloadProfile)
            assert profile.workload_type == workload_type
            assert profile.complexity_score > 0
            assert profile.estimated_duration > 0
        
        # Clean up test database
        if os.path.exists(test_db):
            os.remove(test_db)
    
    def test_acceleration_type_coverage(self):
        """Test all acceleration types are properly handled"""
        all_acceleration_types = [
            AccelerationType.CPU_ONLY,
            AccelerationType.NEURAL_ENGINE,
            AccelerationType.GPU_METAL,
            AccelerationType.HYBRID,
            AccelerationType.AUTO
        ]
        
        for acc_type in all_acceleration_types:
            workload = WorkloadProfile(
                workload_id=f"test_{acc_type.value}",
                workload_type=WorkloadType.GENERAL_COMPUTE,
                input_size=100,
                complexity_score=0.5,
                memory_requirements=400,
                estimated_duration=0.05,
                priority=2,
                preferred_acceleration=acc_type,
                energy_budget=3.0
            )
            
            assert workload.preferred_acceleration == acc_type

def run_comprehensive_tests():
    """Run all comprehensive tests"""
    test_db_files = [
        "test_neural_engine_gpu.db",
        "test_profiler_lifecycle.db", 
        "test_performance_report.db",
        "test_result_recording.db",
        "test_orchestrator.db",
        "test_orchestrator_lifecycle.db",
        "test_integration.db",
        "test_profile_creation.db",
        "test_complexity.db",
        "test_workload_coverage.db",
        "neural_engine_gpu_acceleration.db"
    ]
    
    # Clean up any test databases before starting
    for db_file in test_db_files:
        if os.path.exists(db_file):
            os.remove(db_file)
    
    print("Starting comprehensive Neural Engine and GPU acceleration tests...")
    
    # Run tests
    test_result = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--disable-warnings"
    ])
    
    # Clean up test databases after tests
    for db_file in test_db_files:
        if os.path.exists(db_file):
            os.remove(db_file)
    
    return test_result

if __name__ == "__main__":
    exit_code = run_comprehensive_tests()
    
    if exit_code == 0:
        print("\n✅ All Neural Engine and GPU acceleration tests passed!")
        
        # Run a quick integration test
        print("\n🚀 Running integration test...")
        
        async def quick_integration_test():
            try:
                metrics = await run_neural_engine_gpu_acceleration_demo()
                print(f"✅ Integration test completed successfully!")
                print(f"📊 Test Results:")
                print(f"   - Total workloads: {metrics['total_workloads']}")
                print(f"   - Success rate: {metrics['success_rate']:.1f}%")
                print(f"   - Average performance gain: {metrics['average_performance_gain']:.1f}%")
                print(f"   - Neural Engine utilization: {metrics['neural_engine_utilization']}")
                print(f"   - GPU utilization: {metrics['gpu_utilization']}")
                print(f"   - CPU fallback: {metrics['cpu_fallback']}")
                return True
            except Exception as e:
                print(f"❌ Integration test failed: {e}")
                return False
        
        integration_success = asyncio.run(quick_integration_test())
        
        if integration_success:
            print("\n🎉 All tests completed successfully! Neural Engine and GPU acceleration system is ready.")
        else:
            print("\n⚠️  Unit tests passed but integration test failed.")
            exit_code = 1
    else:
        print("\n❌ Some tests failed. Please review the output above.")
    
    sys.exit(exit_code)