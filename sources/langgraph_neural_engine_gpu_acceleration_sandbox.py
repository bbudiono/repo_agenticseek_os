#!/usr/bin/env python3
# SANDBOX FILE: For testing/development. See .cursorrules.

"""
* Purpose: Neural Engine and GPU Acceleration for LangGraph on Apple Silicon
* Issues & Complexity Summary: Complex hardware acceleration with ML workload optimization
* Key Complexity Drivers:
  - Logic Scope (Est. LoC): ~2500
  - Core Algorithm Complexity: High
  - Dependencies: 4 New (CoreML, Metal, MLCompute, GPUImage), 2 Mod
  - State Management Complexity: High
  - Novelty/Uncertainty Factor: High
* AI Pre-Task Self-Assessment (Est. Solution Difficulty %): 85%
* Problem Estimate (Inherent Problem Difficulty %): 90%
* Initial Code Complexity Estimate %: 85%
* Justification for Estimates: Complex hardware acceleration, ML workload scheduling, energy optimization
* Final Code Complexity (Actual %): To be updated
* Overall Result Score (Success & Quality %): To be updated
* Key Variances/Learnings: To be updated
* Last Updated: 2025-01-06
"""

import asyncio
import sqlite3
import json
import time
import logging
import threading
import psutil
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import subprocess
import platform
import uuid

# Apple Silicon specific imports (with fallbacks)
try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False
    logging.warning("CoreML not available - Neural Engine features will be limited")

try:
    import metal
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False
    logging.warning("Metal not available - GPU acceleration will be limited")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AccelerationType(Enum):
    """Types of hardware acceleration available"""
    CPU_ONLY = "cpu_only"
    NEURAL_ENGINE = "neural_engine"
    GPU_METAL = "gpu_metal"
    HYBRID = "hybrid"
    AUTO = "auto"

class WorkloadType(Enum):
    """Types of workloads for optimization"""
    ML_INFERENCE = "ml_inference"
    MATRIX_OPERATIONS = "matrix_operations"
    GRAPH_PROCESSING = "graph_processing"
    TEXT_PROCESSING = "text_processing"
    IMAGE_PROCESSING = "image_processing"
    GENERAL_COMPUTE = "general_compute"

@dataclass
class WorkloadProfile:
    """Profile for workload characteristics"""
    workload_id: str
    workload_type: WorkloadType
    input_size: int
    complexity_score: float
    memory_requirements: int
    estimated_duration: float
    priority: int
    preferred_acceleration: AccelerationType
    energy_budget: float

@dataclass
class AccelerationResult:
    """Result of hardware acceleration"""
    workload_id: str
    acceleration_type: AccelerationType
    execution_time: float
    energy_consumed: float
    memory_used: int
    success: bool
    performance_gain: float
    error_message: Optional[str] = None

@dataclass
class NeuralEngineConfig:
    """Configuration for Neural Engine utilization"""
    model_precision: str = "float16"
    batch_size: int = 1
    compute_units: str = "cpuAndNeuralEngine"
    optimization_level: str = "high"
    memory_limit: int = 1024  # MB

@dataclass
class GPUConfig:
    """Configuration for GPU acceleration"""
    device_preference: str = "gpu"
    memory_pool_size: int = 2048  # MB
    concurrent_operations: int = 4
    precision_mode: str = "float32"
    shader_optimization: bool = True

class SystemProfiler:
    """System profiler for Apple Silicon capabilities"""
    
    def __init__(self):
        self.system_info = self._gather_system_info()
        self.capabilities = self._assess_capabilities()
    
    def _gather_system_info(self) -> Dict[str, Any]:
        """Gather detailed system information"""
        info = {
            'platform': platform.platform(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'timestamp': time.time()
        }
        
        # Check for Apple Silicon specific features
        if platform.machine() == 'arm64':
            info['apple_silicon'] = True
            info['neural_engine_available'] = self._check_neural_engine()
            info['metal_gpu_available'] = self._check_metal_gpu()
        else:
            info['apple_silicon'] = False
            info['neural_engine_available'] = False
            info['metal_gpu_available'] = False
        
        return info
    
    def _check_neural_engine(self) -> bool:
        """Check if Neural Engine is available"""
        try:
            if COREML_AVAILABLE:
                # Simple test to verify Neural Engine availability
                test_model = ct.models.neural_network.NeuralNetworkBuilder(
                    input_features=[('input', (1,))],
                    output_features=[('output', (1,))]
                )
                test_model.add_elementwise('add', ['input'], 'output', alpha=1.0)
                model = ct.models.MLModel(test_model.spec)
                
                # Test prediction with Neural Engine preference
                prediction = model.predict({'input': np.array([1.0])})
                return True
            return False
        except Exception as e:
            logger.warning(f"Neural Engine check failed: {e}")
            return False
    
    def _check_metal_gpu(self) -> bool:
        """Check if Metal GPU is available"""
        try:
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                  capture_output=True, text=True, timeout=10)
            return 'Metal' in result.stdout
        except Exception as e:
            logger.warning(f"Metal GPU check failed: {e}")
            return False
    
    def _assess_capabilities(self) -> Dict[str, float]:
        """Assess system capabilities with scoring"""
        capabilities = {
            'cpu_performance': min(self.system_info['cpu_count'] / 8.0, 1.0),
            'memory_capacity': min(self.system_info['memory_total'] / (32 * 1024**3), 1.0),
            'neural_engine_score': 1.0 if self.system_info.get('neural_engine_available') else 0.0,
            'gpu_score': 1.0 if self.system_info.get('metal_gpu_available') else 0.0
        }
        
        # Overall system score
        capabilities['overall_score'] = (
            capabilities['cpu_performance'] * 0.3 +
            capabilities['memory_capacity'] * 0.2 +
            capabilities['neural_engine_score'] * 0.3 +
            capabilities['gpu_score'] * 0.2
        )
        
        return capabilities

class NeuralEngineAccelerator:
    """Neural Engine acceleration handler"""
    
    def __init__(self, config: NeuralEngineConfig):
        self.config = config
        self.models_cache = {}
        self.performance_metrics = []
        
    async def accelerate_workload(self, workload: WorkloadProfile, data: Any) -> AccelerationResult:
        """Accelerate workload using Neural Engine"""
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss
        
        try:
            # Create or retrieve optimized model for workload
            model = await self._get_optimized_model(workload)
            
            # Prepare data for Neural Engine
            prepared_data = self._prepare_data_for_neural_engine(data, workload)
            
            # Execute on Neural Engine (simulated)
            result = await self._execute_on_neural_engine(model, prepared_data)
            
            execution_time = time.time() - start_time
            final_memory = psutil.Process().memory_info().rss
            memory_used = final_memory - initial_memory
            
            # Estimate energy consumption (simplified model)
            energy_consumed = self._estimate_energy_consumption(execution_time, workload)
            
            # Calculate performance gain (compared to CPU baseline)
            performance_gain = await self._calculate_performance_gain(workload, execution_time)
            
            acceleration_result = AccelerationResult(
                workload_id=workload.workload_id,
                acceleration_type=AccelerationType.NEURAL_ENGINE,
                execution_time=execution_time,
                energy_consumed=energy_consumed,
                memory_used=memory_used,
                success=True,
                performance_gain=performance_gain
            )
            
            self.performance_metrics.append(acceleration_result)
            return acceleration_result
            
        except Exception as e:
            logger.error(f"Neural Engine acceleration failed: {e}")
            return AccelerationResult(
                workload_id=workload.workload_id,
                acceleration_type=AccelerationType.NEURAL_ENGINE,
                execution_time=time.time() - start_time,
                energy_consumed=0.0,
                memory_used=0,
                success=False,
                performance_gain=0.0,
                error_message=str(e)
            )
    
    async def _get_optimized_model(self, workload: WorkloadProfile) -> Dict[str, Any]:
        """Get or create optimized model for workload"""
        cache_key = f"{workload.workload_type.value}_{workload.input_size}"
        
        if cache_key in self.models_cache:
            return self.models_cache[cache_key]
        
        # Create workload-specific optimized model (simulated)
        model_config = {
            'type': workload.workload_type.value,
            'input_size': workload.input_size,
            'optimization_level': self.config.optimization_level,
            'precision': self.config.model_precision,
            'neural_engine_optimized': True
        }
        
        self.models_cache[cache_key] = model_config
        return model_config
    
    def _prepare_data_for_neural_engine(self, data: Any, workload: WorkloadProfile) -> Dict[str, Any]:
        """Prepare data for Neural Engine execution"""
        if isinstance(data, (list, tuple)):
            data = np.array(data, dtype=np.float32)
        elif isinstance(data, dict):
            # Handle dictionary input
            prepared = {}
            for key, value in data.items():
                if isinstance(value, (list, tuple)):
                    prepared[key] = np.array(value, dtype=np.float32)
                else:
                    prepared[key] = value
            return prepared
        
        # Ensure data matches expected input size
        if hasattr(data, 'shape'):
            if data.size != workload.input_size:
                if data.size > workload.input_size:
                    data = data.flatten()[:workload.input_size]
                else:
                    padding = np.zeros(workload.input_size - data.size)
                    data = np.concatenate([data.flatten(), padding])
        
        return {'input': data.astype(np.float32)}
    
    async def _execute_on_neural_engine(self, model: Dict[str, Any], data: Dict[str, Any]) -> Any:
        """Execute model on Neural Engine (simulated)"""
        # Simulate Neural Engine execution
        loop = asyncio.get_event_loop()
        
        def neural_engine_compute():
            # Simulate Neural Engine computation time (faster than CPU)
            neural_engine_speed_factor = 3.0  # 3x faster than CPU baseline
            base_compute_time = len(data.get('input', [])) / 10000.0  # Base computation
            compute_time = base_compute_time / neural_engine_speed_factor
            time.sleep(max(0.001, compute_time))  # Minimum 1ms
            
            # Return processed data (Neural Engine optimized)
            if 'input' in data:
                result = data['input'] * 1.5 + 0.1  # Example Neural Engine computation
                return {'output': result}
            return data
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = await loop.run_in_executor(executor, neural_engine_compute)
        
        return result
    
    def _estimate_energy_consumption(self, execution_time: float, workload: WorkloadProfile) -> float:
        """Estimate energy consumption for Neural Engine execution"""
        # Simplified energy model for Neural Engine
        # Based on Apple Silicon power characteristics
        base_power = 2.0  # Watts for Neural Engine
        workload_factor = workload.complexity_score * 0.5
        energy = (base_power + workload_factor) * execution_time
        return energy
    
    async def _calculate_performance_gain(self, workload: WorkloadProfile, execution_time: float) -> float:
        """Calculate performance gain compared to CPU baseline"""
        # Estimate CPU execution time for comparison
        cpu_baseline = workload.estimated_duration
        if cpu_baseline > 0:
            gain = (cpu_baseline - execution_time) / cpu_baseline * 100
            return max(0, gain)
        return 40.0  # Default 40% improvement for Neural Engine

class GPUAccelerator:
    """GPU acceleration handler using Metal"""
    
    def __init__(self, config: GPUConfig):
        self.config = config
        self.compute_pipelines = {}
        self.performance_metrics = []
        
    async def accelerate_workload(self, workload: WorkloadProfile, data: Any) -> AccelerationResult:
        """Accelerate workload using Metal GPU"""
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss
        
        try:
            # Create or retrieve compute pipeline
            pipeline = await self._get_compute_pipeline(workload)
            
            # Prepare data for GPU
            gpu_data = self._prepare_data_for_gpu(data, workload)
            
            # Execute on GPU
            result = await self._execute_on_gpu(pipeline, gpu_data, workload)
            
            execution_time = time.time() - start_time
            final_memory = psutil.Process().memory_info().rss
            memory_used = final_memory - initial_memory
            
            # Estimate energy consumption
            energy_consumed = self._estimate_gpu_energy_consumption(execution_time, workload)
            
            # Calculate performance gain
            performance_gain = await self._calculate_gpu_performance_gain(workload, execution_time)
            
            acceleration_result = AccelerationResult(
                workload_id=workload.workload_id,
                acceleration_type=AccelerationType.GPU_METAL,
                execution_time=execution_time,
                energy_consumed=energy_consumed,
                memory_used=memory_used,
                success=True,
                performance_gain=performance_gain
            )
            
            self.performance_metrics.append(acceleration_result)
            return acceleration_result
            
        except Exception as e:
            logger.error(f"GPU acceleration failed: {e}")
            return AccelerationResult(
                workload_id=workload.workload_id,
                acceleration_type=AccelerationType.GPU_METAL,
                execution_time=time.time() - start_time,
                energy_consumed=0.0,
                memory_used=0,
                success=False,
                performance_gain=0.0,
                error_message=str(e)
            )
    
    async def _get_compute_pipeline(self, workload: WorkloadProfile) -> Dict[str, Any]:
        """Get or create Metal compute pipeline for workload"""
        cache_key = f"{workload.workload_type.value}_{workload.input_size}"
        
        if cache_key in self.compute_pipelines:
            return self.compute_pipelines[cache_key]
        
        # Create workload-specific Metal shader (simulated)
        pipeline_config = {
            'shader_type': workload.workload_type.value,
            'input_size': workload.input_size,
            'precision_mode': self.config.precision_mode,
            'shader_optimization': self.config.shader_optimization,
            'metal_optimized': True
        }
        
        self.compute_pipelines[cache_key] = pipeline_config
        return pipeline_config
    
    def _prepare_data_for_gpu(self, data: Any, workload: WorkloadProfile) -> Dict[str, Any]:
        """Prepare data for GPU execution"""
        if isinstance(data, (list, tuple)):
            data = np.array(data, dtype=np.float32)
        
        # Ensure data is properly formatted for GPU
        if hasattr(data, 'shape'):
            if data.size != workload.input_size:
                if data.size > workload.input_size:
                    data = data.flatten()[:workload.input_size]
                else:
                    padding = np.zeros(workload.input_size - data.size)
                    data = np.concatenate([data.flatten(), padding])
        
        return {
            'input_buffer': data.astype(np.float32),
            'output_buffer': np.zeros_like(data, dtype=np.float32)
        }
    
    async def _execute_on_gpu(self, pipeline: Dict[str, Any], data: Dict[str, Any], workload: WorkloadProfile) -> Any:
        """Execute compute pipeline on GPU (simulated)"""
        loop = asyncio.get_event_loop()
        
        def gpu_compute():
            # Simulate GPU computation time based on workload (faster for parallel tasks)
            gpu_speed_factor = 5.0 if workload.workload_type in [
                WorkloadType.MATRIX_OPERATIONS, 
                WorkloadType.IMAGE_PROCESSING
            ] else 2.0
            
            base_compute_time = workload.complexity_score * 0.02
            compute_time = base_compute_time / gpu_speed_factor
            time.sleep(max(0.001, compute_time))
            
            # Return processed data
            input_data = data['input_buffer']
            output_data = input_data * 2.0 + 1.0  # Example GPU computation
            return output_data
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = await loop.run_in_executor(executor, gpu_compute)
        
        return result
    
    def _estimate_gpu_energy_consumption(self, execution_time: float, workload: WorkloadProfile) -> float:
        """Estimate energy consumption for GPU execution"""
        # Simplified energy model for Apple Silicon GPU
        base_power = 8.0  # Watts for GPU
        workload_factor = workload.complexity_score * 1.5
        energy = (base_power + workload_factor) * execution_time
        return energy
    
    async def _calculate_gpu_performance_gain(self, workload: WorkloadProfile, execution_time: float) -> float:
        """Calculate GPU performance gain compared to CPU baseline"""
        cpu_baseline = workload.estimated_duration
        if cpu_baseline > 0:
            gain = (cpu_baseline - execution_time) / cpu_baseline * 100
            return max(0, gain)
        return 60.0  # Default 60% improvement for GPU on parallel workloads

class WorkloadScheduler:
    """Intelligent workload scheduler for optimal hardware utilization"""
    
    def __init__(self, neural_engine: NeuralEngineAccelerator, gpu: GPUAccelerator):
        self.neural_engine = neural_engine
        self.gpu = gpu
        self.system_profiler = SystemProfiler()
        self.scheduling_queue = []
        self.active_workloads = {}
        self.scheduling_history = []
        
    async def schedule_workload(self, workload: WorkloadProfile, data: Any) -> AccelerationResult:
        """Schedule workload for optimal execution"""
        # Determine optimal acceleration type
        optimal_acceleration = await self._determine_optimal_acceleration(workload)
        
        # Schedule based on system load and capabilities
        scheduled_time = await self._calculate_optimal_scheduling_time(workload, optimal_acceleration)
        
        if scheduled_time > 0:
            await asyncio.sleep(scheduled_time)
        
        # Execute workload
        if optimal_acceleration == AccelerationType.NEURAL_ENGINE:
            result = await self.neural_engine.accelerate_workload(workload, data)
        elif optimal_acceleration == AccelerationType.GPU_METAL:
            result = await self.gpu.accelerate_workload(workload, data)
        else:
            result = await self._execute_cpu_fallback(workload, data)
        
        # Record scheduling decision
        self._record_scheduling_decision(workload, optimal_acceleration, result)
        
        return result
    
    async def _determine_optimal_acceleration(self, workload: WorkloadProfile) -> AccelerationType:
        """Determine optimal acceleration type for workload"""
        if workload.preferred_acceleration != AccelerationType.AUTO:
            return workload.preferred_acceleration
        
        # Score different acceleration types
        scores = {}
        
        # Neural Engine score
        if self.system_profiler.capabilities['neural_engine_score'] > 0:
            ne_score = self._calculate_neural_engine_score(workload)
            scores[AccelerationType.NEURAL_ENGINE] = ne_score
        
        # GPU score
        if self.system_profiler.capabilities['gpu_score'] > 0:
            gpu_score = self._calculate_gpu_score(workload)
            scores[AccelerationType.GPU_METAL] = gpu_score
        
        # CPU fallback score
        cpu_score = self._calculate_cpu_score(workload)
        scores[AccelerationType.CPU_ONLY] = cpu_score
        
        # Return highest scoring acceleration type
        if scores:
            optimal = max(scores.items(), key=lambda x: x[1])
            return optimal[0]
        
        return AccelerationType.CPU_ONLY
    
    def _calculate_neural_engine_score(self, workload: WorkloadProfile) -> float:
        """Calculate Neural Engine suitability score"""
        base_score = 0.5
        
        # ML workloads are highly suitable
        if workload.workload_type == WorkloadType.ML_INFERENCE:
            base_score += 0.4
        elif workload.workload_type == WorkloadType.TEXT_PROCESSING:
            base_score += 0.3
        elif workload.workload_type == WorkloadType.MATRIX_OPERATIONS:
            base_score += 0.2
        
        # Consider workload size (Neural Engine prefers smaller, focused tasks)
        if workload.input_size < 1000:
            base_score += 0.1
        elif workload.input_size > 10000:
            base_score -= 0.1
        
        # Energy efficiency bonus
        if workload.energy_budget < 5.0:
            base_score += 0.1
        
        return min(1.0, base_score)
    
    def _calculate_gpu_score(self, workload: WorkloadProfile) -> float:
        """Calculate GPU suitability score"""
        base_score = 0.4
        
        # Graphics and parallel workloads are highly suitable
        if workload.workload_type == WorkloadType.IMAGE_PROCESSING:
            base_score += 0.4
        elif workload.workload_type == WorkloadType.MATRIX_OPERATIONS:
            base_score += 0.3
        elif workload.workload_type == WorkloadType.GENERAL_COMPUTE:
            base_score += 0.2
        
        # Larger workloads benefit more from GPU
        if workload.input_size > 5000:
            base_score += 0.2
        elif workload.input_size < 500:
            base_score -= 0.1
        
        # High complexity workloads benefit from GPU
        if workload.complexity_score > 0.7:
            base_score += 0.1
        
        return min(1.0, base_score)
    
    def _calculate_cpu_score(self, workload: WorkloadProfile) -> float:
        """Calculate CPU suitability score"""
        # CPU is always available as fallback
        base_score = 0.3
        
        # Some workloads are better suited for CPU
        if workload.workload_type == WorkloadType.GRAPH_PROCESSING:
            base_score += 0.2
        
        # Small workloads may be more efficient on CPU
        if workload.input_size < 100:
            base_score += 0.2
        
        # High CPU core count improves score
        cpu_factor = min(self.system_profiler.capabilities['cpu_performance'], 1.0)
        base_score += cpu_factor * 0.2
        
        return min(1.0, base_score)
    
    async def _calculate_optimal_scheduling_time(self, workload: WorkloadProfile, 
                                               acceleration_type: AccelerationType) -> float:
        """Calculate optimal delay before executing workload"""
        current_load = await self._assess_current_system_load()
        
        # If system is under heavy load, introduce delay
        if current_load > 0.8:
            delay = workload.priority * 0.1  # Lower priority = longer delay
            return delay
        
        # If many workloads are queued for same acceleration type, add delay
        same_type_count = sum(1 for w in self.active_workloads.values() 
                             if w.get('acceleration_type') == acceleration_type)
        
        if same_type_count > 2:
            return 0.05 * same_type_count
        
        return 0.0
    
    async def _assess_current_system_load(self) -> float:
        """Assess current system load"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        
        # Combine CPU and memory usage for overall load score
        load_score = (cpu_percent + memory_percent) / 200.0
        return min(load_score, 1.0)
    
    async def _execute_cpu_fallback(self, workload: WorkloadProfile, data: Any) -> AccelerationResult:
        """Execute workload on CPU as fallback"""
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss
        
        try:
            # Simulate CPU computation
            def cpu_compute():
                compute_time = workload.estimated_duration
                time.sleep(compute_time)
                
                # Return processed data (example computation)
                if isinstance(data, (list, tuple, np.ndarray)):
                    return np.array(data) * 1.1
                return data
            
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                result = await loop.run_in_executor(executor, cpu_compute)
            
            execution_time = time.time() - start_time
            final_memory = psutil.Process().memory_info().rss
            memory_used = final_memory - initial_memory
            
            # Estimate energy consumption for CPU
            energy_consumed = self._estimate_cpu_energy_consumption(execution_time, workload)
            
            return AccelerationResult(
                workload_id=workload.workload_id,
                acceleration_type=AccelerationType.CPU_ONLY,
                execution_time=execution_time,
                energy_consumed=energy_consumed,
                memory_used=memory_used,
                success=True,
                performance_gain=0.0  # Baseline performance
            )
            
        except Exception as e:
            logger.error(f"CPU fallback execution failed: {e}")
            return AccelerationResult(
                workload_id=workload.workload_id,
                acceleration_type=AccelerationType.CPU_ONLY,
                execution_time=time.time() - start_time,
                energy_consumed=0.0,
                memory_used=0,
                success=False,
                performance_gain=0.0,
                error_message=str(e)
            )
    
    def _estimate_cpu_energy_consumption(self, execution_time: float, workload: WorkloadProfile) -> float:
        """Estimate energy consumption for CPU execution"""
        # Simplified energy model for CPU
        base_power = 15.0  # Watts for CPU under load
        workload_factor = workload.complexity_score * 5.0
        energy = (base_power + workload_factor) * execution_time
        return energy
    
    def _record_scheduling_decision(self, workload: WorkloadProfile, 
                                  acceleration_type: AccelerationType, 
                                  result: AccelerationResult):
        """Record scheduling decision for optimization"""
        decision_record = {
            'timestamp': time.time(),
            'workload_id': workload.workload_id,
            'workload_type': workload.workload_type.value,
            'acceleration_type': acceleration_type.value,
            'success': result.success,
            'performance_gain': result.performance_gain,
            'energy_consumed': result.energy_consumed,
            'execution_time': result.execution_time
        }
        
        self.scheduling_history.append(decision_record)
        
        # Keep only recent history (last 1000 decisions)
        if len(self.scheduling_history) > 1000:
            self.scheduling_history = self.scheduling_history[-1000:]

class EnergyOptimizer:
    """Energy efficiency optimizer for Apple Silicon"""
    
    def __init__(self):
        self.energy_profiles = {}
        self.optimization_history = []
        self.current_energy_budget = 100.0  # Watts
        
    def optimize_for_energy_efficiency(self, workload: WorkloadProfile) -> Tuple[AccelerationType, Dict[str, Any]]:
        """Optimize workload execution for energy efficiency"""
        # Analyze energy profiles for different acceleration types
        energy_analysis = self._analyze_energy_profiles(workload)
        
        # Select most energy-efficient option
        optimal_acceleration = self._select_energy_optimal_acceleration(energy_analysis, workload)
        
        # Generate configuration recommendations
        config_recommendations = self._generate_energy_config(optimal_acceleration, workload)
        
        return optimal_acceleration, config_recommendations
    
    def _analyze_energy_profiles(self, workload: WorkloadProfile) -> Dict[AccelerationType, float]:
        """Analyze energy consumption profiles for different acceleration types"""
        profiles = {}
        
        # Neural Engine energy profile
        ne_energy = self._estimate_neural_engine_energy(workload)
        profiles[AccelerationType.NEURAL_ENGINE] = ne_energy
        
        # GPU energy profile
        gpu_energy = self._estimate_gpu_energy(workload)
        profiles[AccelerationType.GPU_METAL] = gpu_energy
        
        # CPU energy profile
        cpu_energy = self._estimate_cpu_energy(workload)
        profiles[AccelerationType.CPU_ONLY] = cpu_energy
        
        return profiles
    
    def _estimate_neural_engine_energy(self, workload: WorkloadProfile) -> float:
        """Estimate Neural Engine energy consumption"""
        base_energy = 2.0 * workload.estimated_duration
        complexity_factor = workload.complexity_score * 1.0
        size_factor = min(workload.input_size / 1000.0, 2.0)
        
        return base_energy + complexity_factor + size_factor
    
    def _estimate_gpu_energy(self, workload: WorkloadProfile) -> float:
        """Estimate GPU energy consumption"""
        base_energy = 8.0 * workload.estimated_duration
        complexity_factor = workload.complexity_score * 3.0
        size_factor = workload.input_size / 500.0
        
        return base_energy + complexity_factor + size_factor
    
    def _estimate_cpu_energy(self, workload: WorkloadProfile) -> float:
        """Estimate CPU energy consumption"""
        base_energy = 15.0 * workload.estimated_duration
        complexity_factor = workload.complexity_score * 5.0
        
        return base_energy + complexity_factor
    
    def _select_energy_optimal_acceleration(self, energy_analysis: Dict[AccelerationType, float], 
                                          workload: WorkloadProfile) -> AccelerationType:
        """Select most energy-efficient acceleration type"""
        # Filter by energy budget
        viable_options = {acc_type: energy for acc_type, energy in energy_analysis.items() 
                         if energy <= workload.energy_budget}
        
        if not viable_options:
            # If no option fits budget, select least energy-consuming
            return min(energy_analysis.items(), key=lambda x: x[1])[0]
        
        # Select most energy-efficient viable option
        return min(viable_options.items(), key=lambda x: x[1])[0]
    
    def _generate_energy_config(self, acceleration_type: AccelerationType, 
                              workload: WorkloadProfile) -> Dict[str, Any]:
        """Generate energy-optimized configuration"""
        config = {}
        
        if acceleration_type == AccelerationType.NEURAL_ENGINE:
            config.update({
                'model_precision': 'float16',  # Lower precision for energy savings
                'batch_size': 1,
                'compute_units': 'cpuAndNeuralEngine',
                'optimization_level': 'high'
            })
        elif acceleration_type == AccelerationType.GPU_METAL:
            config.update({
                'precision_mode': 'float16' if workload.energy_budget < 10 else 'float32',
                'concurrent_operations': min(2, 4),  # Reduce concurrency for energy savings
                'shader_optimization': True
            })
        else:
            config.update({
                'cpu_affinity': 'efficiency_cores',  # Use efficiency cores when possible
                'thread_count': min(psutil.cpu_count() // 2, 4)
            })
        
        return config

class PerformanceProfiler:
    """Performance profiler and monitoring system"""
    
    def __init__(self, db_path: str = "neural_engine_gpu_acceleration.db"):
        self.db_path = db_path
        self.metrics_buffer = []
        self.profiling_active = False
        self._init_database()
        
    def _init_database(self):
        """Initialize performance database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                workload_id TEXT,
                workload_type TEXT,
                acceleration_type TEXT,
                execution_time REAL,
                energy_consumed REAL,
                memory_used INTEGER,
                performance_gain REAL,
                success BOOLEAN,
                error_message TEXT,
                system_load REAL,
                neural_engine_utilization REAL,
                gpu_utilization REAL,
                cpu_utilization REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                apple_silicon BOOLEAN,
                neural_engine_available BOOLEAN,
                metal_gpu_available BOOLEAN,
                cpu_count INTEGER,
                memory_total INTEGER,
                capabilities_score REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def start_profiling(self):
        """Start performance profiling"""
        self.profiling_active = True
        
        # Start background profiling task
        asyncio.create_task(self._continuous_profiling())
        
        logger.info("Performance profiling started")
    
    async def stop_profiling(self):
        """Stop performance profiling"""
        self.profiling_active = False
        
        # Flush remaining metrics
        await self._flush_metrics()
        
        logger.info("Performance profiling stopped")
    
    async def _continuous_profiling(self):
        """Continuous system profiling"""
        while self.profiling_active:
            try:
                # Collect system metrics
                system_metrics = await self._collect_system_metrics()
                
                # Store metrics
                self.metrics_buffer.append(system_metrics)
                
                # Flush buffer if it gets too large
                if len(self.metrics_buffer) > 100:
                    await self._flush_metrics()
                
                # Wait before next collection
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in continuous profiling: {e}")
                await asyncio.sleep(5.0)
    
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        metrics = {
            'timestamp': time.time(),
            'cpu_utilization': cpu_percent,
            'memory_utilization': memory.percent,
            'memory_available': memory.available,
            'system_load': (cpu_percent + memory.percent) / 200.0
        }
        
        # Estimate Neural Engine and GPU utilization (simplified)
        metrics['neural_engine_utilization'] = self._estimate_neural_engine_utilization()
        metrics['gpu_utilization'] = self._estimate_gpu_utilization()
        
        return metrics
    
    def _estimate_neural_engine_utilization(self) -> float:
        """Estimate Neural Engine utilization"""
        # Simplified estimation based on system activity
        # In real implementation, would use system profiling APIs
        base_utilization = 0.1
        return min(base_utilization + (psutil.cpu_percent() / 100.0) * 0.3, 1.0)
    
    def _estimate_gpu_utilization(self) -> float:
        """Estimate GPU utilization"""
        # Simplified estimation
        base_utilization = 0.05
        return min(base_utilization + (psutil.cpu_percent() / 100.0) * 0.2, 1.0)
    
    async def _flush_metrics(self):
        """Flush metrics buffer to database"""
        if not self.metrics_buffer:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for metrics in self.metrics_buffer:
                cursor.execute('''
                    INSERT INTO system_profiles 
                    (timestamp, apple_silicon, neural_engine_available, metal_gpu_available,
                     cpu_count, memory_total, capabilities_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics['timestamp'],
                    platform.machine() == 'arm64',
                    COREML_AVAILABLE,
                    METAL_AVAILABLE,
                    psutil.cpu_count(),
                    psutil.virtual_memory().total,
                    metrics['system_load']
                ))
            
            conn.commit()
            conn.close()
            
            self.metrics_buffer.clear()
            
        except Exception as e:
            logger.error(f"Error flushing metrics: {e}")
    
    def record_acceleration_result(self, result: AccelerationResult, system_metrics: Dict[str, Any]):
        """Record acceleration result with system context"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_metrics 
                (timestamp, workload_id, workload_type, acceleration_type, execution_time,
                 energy_consumed, memory_used, performance_gain, success, error_message,
                 system_load, neural_engine_utilization, gpu_utilization, cpu_utilization)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                time.time(),
                result.workload_id,
                'unknown',  # Would be passed from workload
                result.acceleration_type.value,
                result.execution_time,
                result.energy_consumed,
                result.memory_used,
                result.performance_gain,
                result.success,
                result.error_message,
                system_metrics.get('system_load', 0.0),
                system_metrics.get('neural_engine_utilization', 0.0),
                system_metrics.get('gpu_utilization', 0.0),
                system_metrics.get('cpu_utilization', 0.0)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error recording acceleration result: {e}")
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Overall statistics
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_executions,
                    AVG(execution_time) as avg_execution_time,
                    AVG(energy_consumed) as avg_energy_consumed,
                    AVG(performance_gain) as avg_performance_gain,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
                FROM performance_metrics
            ''')
            overall_stats = cursor.fetchone()
            
            # Performance by acceleration type
            cursor.execute('''
                SELECT 
                    acceleration_type,
                    COUNT(*) as executions,
                    AVG(execution_time) as avg_execution_time,
                    AVG(energy_consumed) as avg_energy_consumed,
                    AVG(performance_gain) as avg_performance_gain,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
                FROM performance_metrics
                GROUP BY acceleration_type
            ''')
            acceleration_stats = cursor.fetchall()
            
            # System capability assessment
            cursor.execute('''
                SELECT 
                    apple_silicon,
                    neural_engine_available,
                    metal_gpu_available,
                    AVG(capabilities_score) as avg_capabilities_score
                FROM system_profiles
                ORDER BY timestamp DESC
                LIMIT 1
            ''')
            system_capabilities = cursor.fetchone()
            
            conn.close()
            
            report = {
                'timestamp': time.time(),
                'overall_statistics': {
                    'total_executions': overall_stats[0] if overall_stats else 0,
                    'average_execution_time': overall_stats[1] if overall_stats else 0.0,
                    'average_energy_consumed': overall_stats[2] if overall_stats else 0.0,
                    'average_performance_gain': overall_stats[3] if overall_stats else 0.0,
                    'success_rate': overall_stats[4] if overall_stats else 0.0
                },
                'acceleration_performance': {},
                'system_capabilities': {
                    'apple_silicon': bool(system_capabilities[0]) if system_capabilities else False,
                    'neural_engine_available': bool(system_capabilities[1]) if system_capabilities else False,
                    'metal_gpu_available': bool(system_capabilities[2]) if system_capabilities else False,
                    'capabilities_score': system_capabilities[3] if system_capabilities else 0.0
                }
            }
            
            # Process acceleration statistics
            for stat in acceleration_stats:
                acc_type = stat[0]
                report['acceleration_performance'][acc_type] = {
                    'executions': stat[1],
                    'average_execution_time': stat[2],
                    'average_energy_consumed': stat[3],
                    'average_performance_gain': stat[4],
                    'success_rate': stat[5]
                }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {
                'timestamp': time.time(),
                'error': str(e),
                'overall_statistics': {},
                'acceleration_performance': {},
                'system_capabilities': {}
            }

class NeuralEngineGPUAccelerationOrchestrator:
    """Main orchestrator for Neural Engine and GPU acceleration"""
    
    def __init__(self, neural_config: Optional[NeuralEngineConfig] = None,
                 gpu_config: Optional[GPUConfig] = None,
                 db_path: str = "neural_engine_gpu_acceleration.db"):
        
        # Initialize configurations with defaults
        self.neural_config = neural_config or NeuralEngineConfig()
        self.gpu_config = gpu_config or GPUConfig()
        
        # Initialize components
        self.system_profiler = SystemProfiler()
        self.neural_engine = NeuralEngineAccelerator(self.neural_config)
        self.gpu_accelerator = GPUAccelerator(self.gpu_config)
        self.scheduler = WorkloadScheduler(self.neural_engine, self.gpu_accelerator)
        self.energy_optimizer = EnergyOptimizer()
        self.profiler = PerformanceProfiler(db_path)
        
        # Orchestrator state
        self.active_workloads = {}
        self.completed_workloads = []
        self.optimization_metrics = {
            'total_workloads': 0,
            'successful_accelerations': 0,
            'total_energy_saved': 0.0,
            'total_performance_gain': 0.0
        }
        
        logger.info("Neural Engine and GPU Acceleration Orchestrator initialized")
    
    async def start(self):
        """Start the acceleration orchestrator"""
        # Start performance profiling
        await self.profiler.start_profiling()
        
        # Log system capabilities
        capabilities = self.system_profiler.capabilities
        logger.info(f"System capabilities: {capabilities}")
        
        # Verify hardware availability
        if not capabilities['neural_engine_score'] and not capabilities['gpu_score']:
            logger.warning("Neither Neural Engine nor GPU acceleration available")
        
        logger.info("Neural Engine and GPU Acceleration Orchestrator started")
    
    async def stop(self):
        """Stop the acceleration orchestrator"""
        # Stop profiling
        await self.profiler.stop_profiling()
        
        # Generate final performance report
        final_report = self.profiler.generate_performance_report()
        logger.info(f"Final performance report: {json.dumps(final_report, indent=2)}")
        
        logger.info("Neural Engine and GPU Acceleration Orchestrator stopped")
    
    async def accelerate_workload(self, workload_type: WorkloadType, data: Any,
                                 workload_config: Optional[Dict[str, Any]] = None) -> AccelerationResult:
        """Main entry point for workload acceleration"""
        
        # Create workload profile
        workload = self._create_workload_profile(workload_type, data, workload_config)
        
        # Optimize for energy efficiency
        optimal_acceleration, energy_config = self.energy_optimizer.optimize_for_energy_efficiency(workload)
        
        # Update workload with energy optimization
        workload.preferred_acceleration = optimal_acceleration
        
        # Collect system metrics before execution
        system_metrics = await self.profiler._collect_system_metrics()
        
        # Schedule and execute workload
        try:
            result = await self.scheduler.schedule_workload(workload, data)
            
            # Record result with system context
            self.profiler.record_acceleration_result(result, system_metrics)
            
            # Update orchestrator metrics
            self._update_orchestrator_metrics(result)
            
            # Log successful acceleration
            if result.success:
                logger.info(f"Workload {workload.workload_id} accelerated successfully: "
                           f"{result.acceleration_type.value}, "
                           f"gain: {result.performance_gain:.1f}%, "
                           f"energy: {result.energy_consumed:.2f}J")
            else:
                logger.error(f"Workload {workload.workload_id} failed: {result.error_message}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error accelerating workload: {e}")
            return AccelerationResult(
                workload_id=workload.workload_id,
                acceleration_type=AccelerationType.CPU_ONLY,
                execution_time=0.0,
                energy_consumed=0.0,
                memory_used=0,
                success=False,
                performance_gain=0.0,
                error_message=str(e)
            )
    
    def _create_workload_profile(self, workload_type: WorkloadType, data: Any,
                               config: Optional[Dict[str, Any]] = None) -> WorkloadProfile:
        """Create workload profile from input parameters"""
        config = config or {}
        
        # Calculate input size
        if isinstance(data, (list, tuple)):
            input_size = len(data)
        elif isinstance(data, np.ndarray):
            input_size = data.size
        elif isinstance(data, dict):
            input_size = sum(len(v) if isinstance(v, (list, tuple)) else 1 for v in data.values())
        else:
            input_size = 1
        
        # Estimate complexity based on workload type and size
        complexity_score = self._estimate_workload_complexity(workload_type, input_size)
        
        # Estimate duration
        estimated_duration = self._estimate_workload_duration(workload_type, input_size, complexity_score)
        
        return WorkloadProfile(
            workload_id=str(uuid.uuid4()),
            workload_type=workload_type,
            input_size=input_size,
            complexity_score=complexity_score,
            memory_requirements=config.get('memory_requirements', input_size * 4),  # 4 bytes per element
            estimated_duration=estimated_duration,
            priority=config.get('priority', 5),
            preferred_acceleration=AccelerationType(config.get('preferred_acceleration', 'auto')),
            energy_budget=config.get('energy_budget', 10.0)
        )
    
    def _estimate_workload_complexity(self, workload_type: WorkloadType, input_size: int) -> float:
        """Estimate workload complexity score"""
        base_complexity = {
            WorkloadType.ML_INFERENCE: 0.8,
            WorkloadType.MATRIX_OPERATIONS: 0.6,
            WorkloadType.GRAPH_PROCESSING: 0.7,
            WorkloadType.TEXT_PROCESSING: 0.5,
            WorkloadType.IMAGE_PROCESSING: 0.9,
            WorkloadType.GENERAL_COMPUTE: 0.4
        }
        
        complexity = base_complexity.get(workload_type, 0.5)
        
        # Adjust based on input size
        if input_size > 10000:
            complexity += 0.2
        elif input_size > 1000:
            complexity += 0.1
        elif input_size < 100:
            complexity -= 0.1
        
        return min(1.0, max(0.1, complexity))
    
    def _estimate_workload_duration(self, workload_type: WorkloadType, input_size: int, 
                                  complexity_score: float) -> float:
        """Estimate workload execution duration"""
        base_duration = {
            WorkloadType.ML_INFERENCE: 0.1,
            WorkloadType.MATRIX_OPERATIONS: 0.05,
            WorkloadType.GRAPH_PROCESSING: 0.2,
            WorkloadType.TEXT_PROCESSING: 0.03,
            WorkloadType.IMAGE_PROCESSING: 0.15,
            WorkloadType.GENERAL_COMPUTE: 0.02
        }
        
        duration = base_duration.get(workload_type, 0.05)
        
        # Scale by input size and complexity
        size_factor = max(1.0, input_size / 1000.0)
        complexity_factor = 1.0 + complexity_score
        
        return duration * size_factor * complexity_factor
    
    def _update_orchestrator_metrics(self, result: AccelerationResult):
        """Update orchestrator-level metrics"""
        self.optimization_metrics['total_workloads'] += 1
        
        if result.success:
            self.optimization_metrics['successful_accelerations'] += 1
            self.optimization_metrics['total_performance_gain'] += result.performance_gain
            
            # Estimate energy savings compared to CPU baseline
            if result.acceleration_type != AccelerationType.CPU_ONLY:
                estimated_cpu_energy = result.energy_consumed * 2.0  # Simplified estimate
                energy_saved = max(0, estimated_cpu_energy - result.energy_consumed)
                self.optimization_metrics['total_energy_saved'] += energy_saved
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get orchestrator optimization summary"""
        total = self.optimization_metrics['total_workloads']
        
        return {
            'total_workloads': total,
            'success_rate': (self.optimization_metrics['successful_accelerations'] / max(1, total)) * 100,
            'average_performance_gain': (self.optimization_metrics['total_performance_gain'] / max(1, total)),
            'total_energy_saved': self.optimization_metrics['total_energy_saved'],
            'system_capabilities': self.system_profiler.capabilities,
            'neural_engine_available': self.system_profiler.system_info.get('neural_engine_available', False),
            'metal_gpu_available': self.system_profiler.system_info.get('metal_gpu_available', False)
        }

# Example usage and testing functions
async def run_neural_engine_gpu_acceleration_demo():
    """Run demonstration of Neural Engine and GPU acceleration"""
    
    # Initialize orchestrator
    orchestrator = NeuralEngineGPUAccelerationOrchestrator()
    
    try:
        # Start orchestrator
        await orchestrator.start()
        
        # Test different workload types
        test_workloads = [
            (WorkloadType.ML_INFERENCE, np.random.randn(1000), {'priority': 1, 'energy_budget': 5.0}),
            (WorkloadType.MATRIX_OPERATIONS, np.random.randn(500, 500), {'priority': 2, 'energy_budget': 8.0}),
            (WorkloadType.TEXT_PROCESSING, list(range(2000)), {'priority': 3, 'energy_budget': 3.0}),
            (WorkloadType.IMAGE_PROCESSING, np.random.randn(256, 256, 3), {'priority': 1, 'energy_budget': 10.0}),
            (WorkloadType.GENERAL_COMPUTE, list(range(5000)), {'priority': 4, 'energy_budget': 6.0})
        ]
        
        results = []
        
        # Execute workloads
        for workload_type, data, config in test_workloads:
            logger.info(f"Testing {workload_type.value} workload...")
            result = await orchestrator.accelerate_workload(workload_type, data, config)
            results.append(result)
            
            # Small delay between workloads
            await asyncio.sleep(0.1)
        
        # Wait for all processing to complete
        await asyncio.sleep(1.0)
        
        # Generate performance report
        performance_report = orchestrator.profiler.generate_performance_report()
        optimization_summary = orchestrator.get_optimization_summary()
        
        # Log results
        logger.info("=== Neural Engine and GPU Acceleration Demo Results ===")
        logger.info(f"Performance Report: {json.dumps(performance_report, indent=2)}")
        logger.info(f"Optimization Summary: {json.dumps(optimization_summary, indent=2)}")
        
        # Calculate overall success metrics
        successful_results = [r for r in results if r.success]
        total_performance_gain = sum(r.performance_gain for r in successful_results)
        total_energy_consumed = sum(r.energy_consumed for r in successful_results)
        
        final_metrics = {
            'total_workloads': len(results),
            'successful_workloads': len(successful_results),
            'success_rate': len(successful_results) / len(results) * 100,
            'total_performance_gain': total_performance_gain,
            'average_performance_gain': total_performance_gain / max(1, len(successful_results)),
            'total_energy_consumed': total_energy_consumed,
            'neural_engine_utilization': len([r for r in successful_results if r.acceleration_type == AccelerationType.NEURAL_ENGINE]),
            'gpu_utilization': len([r for r in successful_results if r.acceleration_type == AccelerationType.GPU_METAL]),
            'cpu_fallback': len([r for r in successful_results if r.acceleration_type == AccelerationType.CPU_ONLY])
        }
        
        logger.info(f"Final Demo Metrics: {json.dumps(final_metrics, indent=2)}")
        
        return final_metrics
        
    finally:
        # Stop orchestrator
        await orchestrator.stop()

if __name__ == "__main__":
    # Run the demo
    asyncio.run(run_neural_engine_gpu_acceleration_demo())