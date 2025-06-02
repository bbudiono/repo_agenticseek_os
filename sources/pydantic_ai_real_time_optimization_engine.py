#!/usr/bin/env python3
"""
SANDBOX FILE: For testing/development. See .cursorrules.

Pydantic AI Real-Time Optimization Engine System
===============================================

* Purpose: Build real-time optimization engine with predictive workflow performance optimization,
  dynamic resource allocation, and intelligent workload balancing for MLACS
* Issues & Complexity Summary: Real-time performance optimization, predictive analytics,
  resource allocation algorithms, and cross-framework performance coordination
* Key Complexity Drivers:
  - Logic Scope (Est. LoC): ~2,000
  - Core Algorithm Complexity: Very High
  - Dependencies: 8 New, 6 Mod
  - State Management Complexity: Very High
  - Novelty/Uncertainty Factor: Very High
* AI Pre-Task Self-Assessment (Est. Solution Difficulty %): 92%
* Problem Estimate (Inherent Problem Difficulty %): 95%
* Initial Code Complexity Estimate %: 92%
* Justification for Estimates: Real-time optimization with predictive analytics and resource allocation
* Final Code Complexity (Actual %): TBD
* Overall Result Score (Success & Quality %): TBD
* Key Variances/Learnings: TBD
* Last Updated: 2025-01-06

Provides:
- Real-time performance optimization engine
- Predictive workflow performance analytics
- Dynamic resource allocation and scaling
- Intelligent workload balancing
- Cross-framework optimization coordination
"""

import asyncio
import json
import logging
import sqlite3
import time
import uuid
import threading
import os
import statistics
import heapq
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set, Tuple, Callable, Type, AsyncGenerator
from dataclasses import dataclass, field
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import numpy as np
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Timer decorator for performance monitoring
def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} execution time: {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# Async timer decorator
def async_timer_decorator(func):
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} async execution time: {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# Try to import advanced frameworks, fall back to basic implementations
try:
    from pydantic import BaseModel, Field, ValidationError, validator
    from pydantic_ai import Agent
    PYDANTIC_AI_AVAILABLE = True
    logger.info("Pydantic AI successfully imported")
except ImportError:
    logger.warning("Pydantic AI not available, using fallback implementations")
    PYDANTIC_AI_AVAILABLE = False
    
    # Fallback BaseModel
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        def dict(self):
            return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        
        def json(self):
            def json_serializer(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            return json.dumps(self.dict(), default=json_serializer)
    
    def Field(**kwargs):
        return kwargs.get('default', None)
    
    class ValidationError(Exception):
        pass
    
    def validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Try to import enterprise plugins and communication workflows
try:
    from sources.pydantic_ai_enterprise_workflow_plugins_production import (
        EnterpriseWorkflowPluginSystem,
        PluginMetadata,
        WorkflowTemplate,
        PluginType,
        IndustryDomain,
        SecurityLevel
    )
    ENTERPRISE_PLUGINS_AVAILABLE = True
    logger.info("Enterprise Workflow Plugins System available")
except ImportError:
    logger.warning("Enterprise plugins not available, using fallback")
    ENTERPRISE_PLUGINS_AVAILABLE = False

try:
    from sources.pydantic_ai_production_communication_workflows_production import (
        ProductionCommunicationWorkflowsSystem,
        WorkflowDefinition,
        WorkflowExecution,
        WorkflowStatus,
        CommunicationMessage
    )
    COMMUNICATION_WORKFLOWS_AVAILABLE = True
    logger.info("Production Communication Workflows System available")
except ImportError:
    logger.warning("Communication workflows not available, using fallback")
    COMMUNICATION_WORKFLOWS_AVAILABLE = False

# ================================
# Optimization Engine Enums and Models
# ================================

class OptimizationStrategy(Enum):
    """Optimization strategies for different scenarios"""
    THROUGHPUT_MAXIMIZATION = "throughput_maximization"
    LATENCY_MINIMIZATION = "latency_minimization"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    COST_OPTIMIZATION = "cost_optimization"
    QUALITY_ASSURANCE = "quality_assurance"
    BALANCED_PERFORMANCE = "balanced_performance"
    PREDICTIVE_SCALING = "predictive_scaling"
    ADAPTIVE_LEARNING = "adaptive_learning"

class MetricType(Enum):
    """Types of performance metrics"""
    EXECUTION_TIME = "execution_time"
    THROUGHPUT = "throughput"
    RESOURCE_USAGE = "resource_usage"
    ERROR_RATE = "error_rate"
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_USAGE = "memory_usage"
    NETWORK_LATENCY = "network_latency"
    SUCCESS_RATE = "success_rate"

class ResourceType(Enum):
    """Types of system resources"""
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    STORAGE = "storage"
    THREAD_POOL = "thread_pool"
    CONNECTION_POOL = "connection_pool"
    CACHE = "cache"
    DATABASE = "database"

class OptimizationPriority(Enum):
    """Priority levels for optimization decisions"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKGROUND = "background"

class PredictionModel(Enum):
    """Predictive models for performance forecasting"""
    LINEAR_REGRESSION = "linear_regression"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    MOVING_AVERAGE = "moving_average"
    ARIMA = "arima"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"

# ================================
# Optimization Data Models
# ================================

class PerformanceMetric(BaseModel):
    """Performance metric data point"""
    
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', str(uuid.uuid4()))
        self.metric_type = MetricType(kwargs.get('metric_type', 'execution_time'))
        self.value = kwargs.get('value', 0.0)
        self.timestamp = kwargs.get('timestamp', datetime.now())
        self.source_component = kwargs.get('source_component', '')
        self.workflow_id = kwargs.get('workflow_id', '')
        self.agent_id = kwargs.get('agent_id', '')
        self.context = kwargs.get('context', {})
        self.tags = kwargs.get('tags', [])
        self.severity = kwargs.get('severity', 'normal')
        self.prediction_confidence = kwargs.get('prediction_confidence', 0.0)

class ResourceAllocation(BaseModel):
    """Resource allocation specification"""
    
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', str(uuid.uuid4()))
        self.resource_type = ResourceType(kwargs.get('resource_type', 'cpu'))
        self.allocated_amount = kwargs.get('allocated_amount', 0.0)
        self.max_amount = kwargs.get('max_amount', 100.0)
        self.target_component = kwargs.get('target_component', '')
        self.priority = OptimizationPriority(kwargs.get('priority', 'medium'))
        self.allocation_strategy = kwargs.get('allocation_strategy', 'proportional')
        self.constraints = kwargs.get('constraints', {})
        self.created_at = kwargs.get('created_at', datetime.now())
        self.expires_at = kwargs.get('expires_at')
        self.effectiveness_score = kwargs.get('effectiveness_score', 0.0)

class OptimizationRecommendation(BaseModel):
    """Optimization recommendation with implementation details"""
    
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', str(uuid.uuid4()))
        self.strategy = OptimizationStrategy(kwargs.get('strategy', 'balanced_performance'))
        self.target_component = kwargs.get('target_component', '')
        self.description = kwargs.get('description', '')
        self.expected_improvement = kwargs.get('expected_improvement', 0.0)
        self.confidence_score = kwargs.get('confidence_score', 0.0)
        self.implementation_complexity = kwargs.get('implementation_complexity', 'medium')
        self.resource_requirements = kwargs.get('resource_requirements', {})
        self.risk_assessment = kwargs.get('risk_assessment', 'low')
        self.action_items = kwargs.get('action_items', [])
        self.priority = OptimizationPriority(kwargs.get('priority', 'medium'))
        self.created_at = kwargs.get('created_at', datetime.now())
        self.implemented_at = kwargs.get('implemented_at')
        self.effectiveness_rating = kwargs.get('effectiveness_rating', 0.0)

class PredictiveModel(BaseModel):
    """Predictive model for performance forecasting"""
    
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', str(uuid.uuid4()))
        self.model_type = PredictionModel(kwargs.get('model_type', 'linear_regression'))
        self.target_metric = MetricType(kwargs.get('target_metric', 'execution_time'))
        self.model_parameters = kwargs.get('model_parameters', {})
        self.training_data_size = kwargs.get('training_data_size', 0)
        self.accuracy_score = kwargs.get('accuracy_score', 0.0)
        self.last_trained = kwargs.get('last_trained', datetime.now())
        self.prediction_horizon = kwargs.get('prediction_horizon', 3600)  # seconds
        self.feature_importance = kwargs.get('feature_importance', {})
        self.validation_results = kwargs.get('validation_results', {})
        self.is_active = kwargs.get('is_active', True)

class WorkloadProfile(BaseModel):
    """Workload profile for optimization planning"""
    
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', str(uuid.uuid4()))
        self.component_id = kwargs.get('component_id', '')
        self.workload_type = kwargs.get('workload_type', 'standard')
        self.average_throughput = kwargs.get('average_throughput', 0.0)
        self.peak_throughput = kwargs.get('peak_throughput', 0.0)
        self.average_latency = kwargs.get('average_latency', 0.0)
        self.resource_patterns = kwargs.get('resource_patterns', {})
        self.seasonal_patterns = kwargs.get('seasonal_patterns', {})
        self.growth_trends = kwargs.get('growth_trends', {})
        self.optimization_targets = kwargs.get('optimization_targets', {})
        self.constraints = kwargs.get('constraints', {})
        self.last_updated = kwargs.get('last_updated', datetime.now())

# ================================
# Real-Time Optimization Engine
# ================================

class RealTimeOptimizationEngine:
    """
    Real-time optimization engine with predictive performance analytics
    """
    
    def __init__(
        self,
        db_path: str = "optimization_engine.db",
        optimization_interval: int = 30,  # seconds
        prediction_horizon: int = 3600,  # seconds
        enable_predictive_scaling: bool = True,
        enable_adaptive_learning: bool = True
    ):
        self.db_path = db_path
        self.optimization_interval = optimization_interval
        self.prediction_horizon = prediction_horizon
        self.enable_predictive_scaling = enable_predictive_scaling
        self.enable_adaptive_learning = enable_adaptive_learning
        
        # Core system components
        self.performance_metrics: defaultdict = defaultdict(deque)
        self.resource_allocations: Dict[str, ResourceAllocation] = {}
        self.optimization_recommendations: Dict[str, OptimizationRecommendation] = {}
        self.predictive_models: Dict[str, PredictiveModel] = {}
        self.workload_profiles: Dict[str, WorkloadProfile] = {}
        
        # Optimization engine state
        self.optimization_strategies: Dict[str, OptimizationStrategy] = {}
        self.component_priorities: Dict[str, OptimizationPriority] = {}
        self.optimization_history: List[Dict[str, Any]] = []
        self.active_optimizations: Set[str] = set()
        
        # Performance tracking
        self.optimization_effectiveness: defaultdict = defaultdict(list)
        self.system_health_metrics: Dict[str, float] = {}
        self.prediction_accuracy: defaultdict = defaultdict(list)
        
        # Predictive analytics
        self.feature_extractors: Dict[str, Callable] = {}
        self.model_trainers: Dict[str, Callable] = {}
        self.prediction_cache: Dict[str, Dict[str, Any]] = {}
        
        # Integration components
        self.enterprise_plugin_system = None
        self.communication_system = None
        
        # Threading and async management
        self.optimization_loop_task = None
        self._lock = threading.RLock()
        self._initialized = False
        
        # Initialize system
        self._initialize_system()

    def _initialize_system(self):
        """Initialize the real-time optimization engine"""
        try:
            # Initialize database
            self._initialize_database()
            
            # Initialize predictive models
            self._initialize_predictive_models()
            
            # Initialize feature extractors
            self._initialize_feature_extractors()
            
            # Initialize optimization strategies
            self._initialize_optimization_strategies()
            
            # Load existing data
            self._load_existing_data()
            
            # Initialize integrations
            if ENTERPRISE_PLUGINS_AVAILABLE:
                try:
                    self.enterprise_plugin_system = EnterpriseWorkflowPluginSystem(
                        db_path="optimization_plugins.db",
                        enable_security_scanning=False
                    )
                    logger.info("Enterprise plugin system integration initialized")
                except Exception as e:
                    logger.warning(f"Enterprise plugin integration failed: {e}")
            
            if COMMUNICATION_WORKFLOWS_AVAILABLE:
                try:
                    self.communication_system = ProductionCommunicationWorkflowsSystem(
                        db_path="optimization_communication.db",
                        enable_persistence=True
                    )
                    logger.info("Communication system integration initialized")
                except Exception as e:
                    logger.warning(f"Communication system integration failed: {e}")
            
            self._initialized = True
            logger.info("Real-Time Optimization Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize optimization engine: {e}")
            raise

    def _initialize_database(self):
        """Initialize SQLite database for optimization data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id TEXT PRIMARY KEY,
                    metric_type TEXT,
                    value REAL,
                    timestamp TEXT,
                    source_component TEXT,
                    workflow_id TEXT,
                    agent_id TEXT,
                    context TEXT,
                    tags TEXT,
                    severity TEXT,
                    prediction_confidence REAL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS resource_allocations (
                    id TEXT PRIMARY KEY,
                    resource_type TEXT,
                    allocated_amount REAL,
                    max_amount REAL,
                    target_component TEXT,
                    priority TEXT,
                    allocation_strategy TEXT,
                    constraints TEXT,
                    created_at TEXT,
                    expires_at TEXT,
                    effectiveness_score REAL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS optimization_recommendations (
                    id TEXT PRIMARY KEY,
                    strategy TEXT,
                    target_component TEXT,
                    description TEXT,
                    expected_improvement REAL,
                    confidence_score REAL,
                    implementation_complexity TEXT,
                    resource_requirements TEXT,
                    risk_assessment TEXT,
                    action_items TEXT,
                    priority TEXT,
                    created_at TEXT,
                    implemented_at TEXT,
                    effectiveness_rating REAL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictive_models (
                    id TEXT PRIMARY KEY,
                    model_type TEXT,
                    target_metric TEXT,
                    model_parameters TEXT,
                    training_data_size INTEGER,
                    accuracy_score REAL,
                    last_trained TEXT,
                    prediction_horizon INTEGER,
                    feature_importance TEXT,
                    validation_results TEXT,
                    is_active BOOLEAN
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS workload_profiles (
                    id TEXT PRIMARY KEY,
                    component_id TEXT,
                    workload_type TEXT,
                    average_throughput REAL,
                    peak_throughput REAL,
                    average_latency REAL,
                    resource_patterns TEXT,
                    seasonal_patterns TEXT,
                    growth_trends TEXT,
                    optimization_targets TEXT,
                    constraints TEXT,
                    last_updated TEXT
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metric_timestamp ON performance_metrics(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metric_component ON performance_metrics(source_component)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metric_type ON performance_metrics(metric_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_allocation_component ON resource_allocations(target_component)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_recommendation_component ON optimization_recommendations(target_component)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_metric ON predictive_models(target_metric)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_profile_component ON workload_profiles(component_id)')
            
            conn.commit()
            conn.close()
            
            logger.info("Optimization database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    def _initialize_predictive_models(self):
        """Initialize predictive models for performance forecasting"""
        try:
            # Initialize simple predictive models
            for metric_type in MetricType:
                for model_type in PredictionModel:
                    if model_type in [PredictionModel.LINEAR_REGRESSION, PredictionModel.MOVING_AVERAGE, PredictionModel.EXPONENTIAL_SMOOTHING]:
                        model = PredictiveModel(
                            model_type=model_type,
                            target_metric=metric_type,
                            model_parameters={
                                'window_size': 100,
                                'alpha': 0.3,
                                'learning_rate': 0.01
                            },
                            prediction_horizon=self.prediction_horizon,
                            is_active=True
                        )
                        self.predictive_models[f"{metric_type.value}_{model_type.value}"] = model
            
            logger.info("Predictive models initialized")
            
        except Exception as e:
            logger.warning(f"Predictive model initialization failed: {e}")

    def _initialize_feature_extractors(self):
        """Initialize feature extractors for predictive modeling"""
        try:
            self.feature_extractors = {
                'temporal_features': self._extract_temporal_features,
                'statistical_features': self._extract_statistical_features,
                'trend_features': self._extract_trend_features,
                'pattern_features': self._extract_pattern_features,
                'workload_features': self._extract_workload_features
            }
            
            logger.info("Feature extractors initialized")
            
        except Exception as e:
            logger.warning(f"Feature extractor initialization failed: {e}")

    def _initialize_optimization_strategies(self):
        """Initialize optimization strategies"""
        try:
            # Set default optimization strategies for different components
            self.optimization_strategies = {
                'workflow_engine': OptimizationStrategy.BALANCED_PERFORMANCE,
                'communication_system': OptimizationStrategy.LATENCY_MINIMIZATION,
                'enterprise_plugins': OptimizationStrategy.RESOURCE_EFFICIENCY,
                'database': OptimizationStrategy.THROUGHPUT_MAXIMIZATION,
                'cache': OptimizationStrategy.RESOURCE_EFFICIENCY,
                'network': OptimizationStrategy.LATENCY_MINIMIZATION
            }
            
            # Set default priorities
            self.component_priorities = {
                'workflow_engine': OptimizationPriority.HIGH,
                'communication_system': OptimizationPriority.HIGH,
                'enterprise_plugins': OptimizationPriority.MEDIUM,
                'database': OptimizationPriority.MEDIUM,
                'cache': OptimizationPriority.LOW,
                'network': OptimizationPriority.MEDIUM
            }
            
            logger.info("Optimization strategies initialized")
            
        except Exception as e:
            logger.warning(f"Optimization strategy initialization failed: {e}")

    def _load_existing_data(self):
        """Load existing optimization data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Load recent performance metrics
            cursor.execute('''
                SELECT * FROM performance_metrics 
                WHERE datetime(timestamp) > datetime('now', '-1 hour')
                ORDER BY timestamp DESC
                LIMIT 1000
            ''')
            
            for row in cursor.fetchall():
                try:
                    metric = PerformanceMetric(
                        id=row[0], metric_type=row[1], value=row[2],
                        timestamp=datetime.fromisoformat(row[3]) if row[3] else datetime.now(),
                        source_component=row[4], workflow_id=row[5], agent_id=row[6],
                        context=json.loads(row[7]) if row[7] else {},
                        tags=json.loads(row[8]) if row[8] else [],
                        severity=row[9], prediction_confidence=row[10] or 0.0
                    )
                    self.performance_metrics[metric.source_component].append(metric)
                except Exception as e:
                    logger.warning(f"Failed to load performance metric {row[0]}: {e}")
            
            # Load active resource allocations
            cursor.execute('SELECT * FROM resource_allocations WHERE expires_at IS NULL OR expires_at > datetime("now")')
            for row in cursor.fetchall():
                try:
                    allocation = ResourceAllocation(
                        id=row[0], resource_type=row[1], allocated_amount=row[2],
                        max_amount=row[3], target_component=row[4], priority=row[5],
                        allocation_strategy=row[6], constraints=json.loads(row[7]) if row[7] else {},
                        created_at=datetime.fromisoformat(row[8]) if row[8] else datetime.now(),
                        expires_at=datetime.fromisoformat(row[9]) if row[9] else None,
                        effectiveness_score=row[10] or 0.0
                    )
                    self.resource_allocations[allocation.id] = allocation
                except Exception as e:
                    logger.warning(f"Failed to load resource allocation {row[0]}: {e}")
            
            # Load predictive models
            cursor.execute('SELECT * FROM predictive_models WHERE is_active = 1')
            for row in cursor.fetchall():
                try:
                    model = PredictiveModel(
                        id=row[0], model_type=row[1], target_metric=row[2],
                        model_parameters=json.loads(row[3]) if row[3] else {},
                        training_data_size=row[4], accuracy_score=row[5] or 0.0,
                        last_trained=datetime.fromisoformat(row[6]) if row[6] else datetime.now(),
                        prediction_horizon=row[7], feature_importance=json.loads(row[8]) if row[8] else {},
                        validation_results=json.loads(row[9]) if row[9] else {},
                        is_active=bool(row[10])
                    )
                    self.predictive_models[model.id] = model
                except Exception as e:
                    logger.warning(f"Failed to load predictive model {row[0]}: {e}")
            
            conn.close()
            logger.info(f"Loaded {len(self.performance_metrics)} components with metrics, {len(self.resource_allocations)} allocations, {len(self.predictive_models)} models")
            
        except Exception as e:
            logger.warning(f"Failed to load existing data: {e}")

    @timer_decorator
    def record_performance_metric(self, metric: PerformanceMetric) -> bool:
        """Record a performance metric for optimization analysis"""
        try:
            # Store in memory for real-time processing
            self.performance_metrics[metric.source_component].append(metric)
            
            # Keep only recent metrics in memory (last 1000 per component)
            if len(self.performance_metrics[metric.source_component]) > 1000:
                self.performance_metrics[metric.source_component].popleft()
            
            # Persist to database
            self._persist_performance_metric(metric)
            
            # Trigger real-time optimization if needed
            if self._should_trigger_optimization(metric):
                asyncio.create_task(self._trigger_optimization(metric.source_component))
            
            logger.info(f"Recorded performance metric: {metric.metric_type.value} = {metric.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to record performance metric: {e}")
            return False

    @timer_decorator
    def generate_optimization_recommendations(self, component_id: str = None) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on current performance data"""
        try:
            recommendations = []
            components = [component_id] if component_id else list(self.performance_metrics.keys())
            
            for comp_id in components:
                if comp_id not in self.performance_metrics:
                    continue
                
                metrics = list(self.performance_metrics[comp_id])
                if len(metrics) < 10:  # Need sufficient data
                    continue
                
                # Analyze performance trends
                trend_analysis = self._analyze_performance_trends(comp_id, metrics)
                
                # Generate recommendations based on analysis
                comp_recommendations = self._generate_component_recommendations(comp_id, trend_analysis)
                recommendations.extend(comp_recommendations)
            
            # Sort by priority and confidence
            recommendations.sort(key=lambda r: (r.priority.value, -r.confidence_score))
            
            # Store recommendations
            for recommendation in recommendations:
                self.optimization_recommendations[recommendation.id] = recommendation
                self._persist_optimization_recommendation(recommendation)
            
            logger.info(f"Generated {len(recommendations)} optimization recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate optimization recommendations: {e}")
            return []

    async def predict_performance(self, component_id: str, metric_type: MetricType, horizon_minutes: int = 60) -> Dict[str, Any]:
        """Predict future performance for a component"""
        try:
            if component_id not in self.performance_metrics:
                return {'error': 'No performance data available for component'}
            
            metrics = [m for m in self.performance_metrics[component_id] if m.metric_type == metric_type]
            if len(metrics) < 20:  # Need sufficient data for prediction
                return {'error': 'Insufficient data for prediction'}
            
            # Extract features
            features = self._extract_prediction_features(metrics)
            
            # Select best model for this metric
            model_key = f"{metric_type.value}_linear_regression"
            if model_key not in self.predictive_models:
                model_key = f"{metric_type.value}_moving_average"
            
            if model_key not in self.predictive_models:
                return {'error': 'No suitable prediction model available'}
            
            model = self.predictive_models[model_key]
            
            # Generate predictions
            predictions = self._generate_predictions(model, features, horizon_minutes)
            
            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(predictions, model.accuracy_score)
            
            result = {
                'component_id': component_id,
                'metric_type': metric_type.value,
                'horizon_minutes': horizon_minutes,
                'predictions': predictions,
                'confidence_intervals': confidence_intervals,
                'model_accuracy': model.accuracy_score,
                'prediction_confidence': np.mean([p.get('confidence', 0.5) for p in predictions])
            }
            
            logger.info(f"Generated {len(predictions)} predictions for {component_id}:{metric_type.value}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to predict performance: {e}")
            return {'error': str(e)}

    async def optimize_resource_allocation(self, target_component: str = None) -> Dict[str, ResourceAllocation]:
        """Optimize resource allocation based on current performance and predictions"""
        try:
            components = [target_component] if target_component else list(self.performance_metrics.keys())
            allocations = {}
            
            for component_id in components:
                if component_id not in self.performance_metrics:
                    continue
                
                # Analyze current resource usage
                usage_analysis = self._analyze_resource_usage(component_id)
                
                # Predict future resource needs
                resource_predictions = await self._predict_resource_needs(component_id)
                
                # Calculate optimal allocation
                optimal_allocation = self._calculate_optimal_allocation(
                    component_id, usage_analysis, resource_predictions
                )
                
                if optimal_allocation:
                    allocations[component_id] = optimal_allocation
                    self.resource_allocations[optimal_allocation.id] = optimal_allocation
                    self._persist_resource_allocation(optimal_allocation)
            
            logger.info(f"Optimized resource allocation for {len(allocations)} components")
            return allocations
            
        except Exception as e:
            logger.error(f"Failed to optimize resource allocation: {e}")
            return {}

    async def start_optimization_loop(self):
        """Start the continuous optimization loop"""
        try:
            if self.optimization_loop_task and not self.optimization_loop_task.done():
                logger.warning("Optimization loop already running")
                return
            
            self.optimization_loop_task = asyncio.create_task(self._optimization_loop())
            logger.info("Started optimization loop")
            
        except Exception as e:
            logger.error(f"Failed to start optimization loop: {e}")

    async def stop_optimization_loop(self):
        """Stop the continuous optimization loop"""
        try:
            if self.optimization_loop_task:
                self.optimization_loop_task.cancel()
                try:
                    await self.optimization_loop_task
                except asyncio.CancelledError:
                    pass
                logger.info("Stopped optimization loop")
            
        except Exception as e:
            logger.error(f"Failed to stop optimization loop: {e}")

    async def _optimization_loop(self):
        """Main optimization loop that runs continuously"""
        try:
            while True:
                loop_start = time.time()
                
                # Generate optimization recommendations
                recommendations = self.generate_optimization_recommendations()
                
                # Optimize resource allocations
                allocations = await self.optimize_resource_allocation()
                
                # Update predictive models if needed
                await self._update_predictive_models()
                
                # Execute high-priority optimizations
                await self._execute_optimizations(recommendations)
                
                # Update system health metrics
                self._update_system_health_metrics()
                
                # Log optimization cycle
                cycle_time = time.time() - loop_start
                logger.info(f"Optimization cycle completed in {cycle_time:.3f}s")
                
                # Wait for next optimization interval
                await asyncio.sleep(max(0, self.optimization_interval - cycle_time))
                
        except asyncio.CancelledError:
            logger.info("Optimization loop cancelled")
        except Exception as e:
            logger.error(f"Optimization loop error: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive optimization engine status"""
        try:
            total_metrics = sum(len(metrics) for metrics in self.performance_metrics.values())
            active_components = len(self.performance_metrics)
            
            return {
                'engine_status': 'operational' if self._initialized else 'initializing',
                'optimization_loop_active': bool(self.optimization_loop_task and not self.optimization_loop_task.done()),
                'performance_monitoring': {
                    'active_components': active_components,
                    'total_metrics': total_metrics,
                    'metrics_per_component': {k: len(v) for k, v in self.performance_metrics.items()},
                    'recent_metric_rate': self._calculate_recent_metric_rate()
                },
                'resource_management': {
                    'active_allocations': len(self.resource_allocations),
                    'allocation_effectiveness': self._calculate_allocation_effectiveness(),
                    'resource_utilization': self._calculate_resource_utilization()
                },
                'optimization': {
                    'active_recommendations': len(self.optimization_recommendations),
                    'optimization_strategies': {k: v.value for k, v in self.optimization_strategies.items()},
                    'optimization_effectiveness': self._calculate_optimization_effectiveness()
                },
                'prediction': {
                    'active_models': len([m for m in self.predictive_models.values() if m.is_active]),
                    'model_accuracy': self._calculate_average_model_accuracy(),
                    'prediction_cache_size': len(self.prediction_cache)
                },
                'system_health': self.system_health_metrics,
                'configuration': {
                    'optimization_interval': self.optimization_interval,
                    'prediction_horizon': self.prediction_horizon,
                    'predictive_scaling_enabled': self.enable_predictive_scaling,
                    'adaptive_learning_enabled': self.enable_adaptive_learning
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {'error': str(e)}

    # Helper methods for internal processing
    
    def _should_trigger_optimization(self, metric: PerformanceMetric) -> bool:
        """Determine if a metric should trigger immediate optimization"""
        # Trigger optimization for critical metrics or significant deviations
        if metric.severity == 'critical':
            return True
        
        # Check for significant performance degradation
        recent_metrics = list(self.performance_metrics[metric.source_component])[-10:]
        if len(recent_metrics) >= 10:
            recent_avg = statistics.mean([m.value for m in recent_metrics])
            if metric.value > recent_avg * 1.5:  # 50% degradation
                return True
        
        return False

    async def _trigger_optimization(self, component_id: str):
        """Trigger immediate optimization for a component"""
        try:
            logger.info(f"Triggering immediate optimization for {component_id}")
            
            # Generate recommendations
            recommendations = self.generate_optimization_recommendations(component_id)
            
            # Execute high-priority recommendations
            await self._execute_optimizations(recommendations[:3])  # Execute top 3
            
        except Exception as e:
            logger.error(f"Failed to trigger optimization for {component_id}: {e}")

    def _analyze_performance_trends(self, component_id: str, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Analyze performance trends for a component"""
        try:
            analysis = {
                'component_id': component_id,
                'metric_count': len(metrics),
                'time_range': (metrics[0].timestamp, metrics[-1].timestamp) if metrics else (None, None),
                'trends': {},
                'anomalies': [],
                'patterns': {}
            }
            
            # Group metrics by type
            by_type = defaultdict(list)
            for metric in metrics:
                by_type[metric.metric_type].append(metric.value)
            
            # Analyze trends for each metric type
            for metric_type, values in by_type.items():
                if len(values) >= 10:
                    trend_analysis = self._calculate_trend(values)
                    analysis['trends'][metric_type.value] = trend_analysis
                    
                    # Detect anomalies
                    anomalies = self._detect_anomalies(values)
                    if anomalies:
                        analysis['anomalies'].extend(anomalies)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze performance trends: {e}")
            return {'error': str(e)}

    def _generate_component_recommendations(self, component_id: str, trend_analysis: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations for a specific component"""
        try:
            recommendations = []
            
            # Get component's optimization strategy
            strategy = self.optimization_strategies.get(component_id, OptimizationStrategy.BALANCED_PERFORMANCE)
            priority = self.component_priorities.get(component_id, OptimizationPriority.MEDIUM)
            
            # Analyze trends and generate recommendations
            for metric_type, trend in trend_analysis.get('trends', {}).items():
                if trend.get('slope', 0) > 0.1:  # Increasing trend (degradation)
                    recommendation = OptimizationRecommendation(
                        strategy=strategy,
                        target_component=component_id,
                        description=f"Performance degradation detected in {metric_type}",
                        expected_improvement=min(trend['slope'] * 0.5, 0.3),  # Cap at 30%
                        confidence_score=min(trend.get('r_squared', 0.5) + 0.2, 0.9),
                        implementation_complexity='medium',
                        resource_requirements={
                            'cpu': trend['slope'] * 10,
                            'memory': trend['slope'] * 20
                        },
                        risk_assessment='low',
                        action_items=[
                            f"Increase {component_id} resource allocation",
                            f"Optimize {metric_type} processing",
                            "Monitor performance closely"
                        ],
                        priority=priority
                    )
                    recommendations.append(recommendation)
            
            # Check for anomalies
            if trend_analysis.get('anomalies'):
                recommendation = OptimizationRecommendation(
                    strategy=OptimizationStrategy.QUALITY_ASSURANCE,
                    target_component=component_id,
                    description=f"Performance anomalies detected in {component_id}",
                    expected_improvement=0.2,
                    confidence_score=0.7,
                    implementation_complexity='low',
                    resource_requirements={},
                    risk_assessment='medium',
                    action_items=[
                        "Investigate performance anomalies",
                        "Review recent changes",
                        "Increase monitoring frequency"
                    ],
                    priority=OptimizationPriority.HIGH
                )
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate component recommendations: {e}")
            return []

    def _extract_prediction_features(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Extract features for predictive modeling"""
        try:
            features = {}
            
            # Extract temporal features
            features.update(self.feature_extractors['temporal_features'](metrics))
            
            # Extract statistical features
            features.update(self.feature_extractors['statistical_features'](metrics))
            
            # Extract trend features
            features.update(self.feature_extractors['trend_features'](metrics))
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract prediction features: {e}")
            return {}

    def _extract_temporal_features(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Extract temporal features from metrics"""
        if not metrics:
            return {}
        
        timestamps = [m.timestamp for m in metrics]
        values = [m.value for m in metrics]
        
        return {
            'time_span': (timestamps[-1] - timestamps[0]).total_seconds(),
            'sample_rate': len(metrics) / max((timestamps[-1] - timestamps[0]).total_seconds() / 60, 1),
            'hour_of_day': timestamps[-1].hour,
            'day_of_week': timestamps[-1].weekday(),
            'recent_trend': self._calculate_recent_trend(values[-10:] if len(values) >= 10 else values)
        }

    def _extract_statistical_features(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Extract statistical features from metrics"""
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        
        return {
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
            'min': min(values),
            'max': max(values),
            'range': max(values) - min(values),
            'percentile_95': np.percentile(values, 95),
            'percentile_5': np.percentile(values, 5)
        }

    def _extract_trend_features(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Extract trend features from metrics"""
        if len(metrics) < 2:
            return {}
        
        values = [m.value for m in metrics]
        trend = self._calculate_trend(values)
        
        return {
            'slope': trend.get('slope', 0),
            'r_squared': trend.get('r_squared', 0),
            'is_increasing': trend.get('slope', 0) > 0,
            'trend_strength': abs(trend.get('slope', 0)) * trend.get('r_squared', 0)
        }

    def _extract_pattern_features(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Extract pattern features from metrics"""
        # Simplified pattern detection
        return {'pattern_detected': False}

    def _extract_workload_features(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Extract workload-related features from metrics"""
        # Simplified workload analysis
        return {'workload_intensity': 'medium'}

    def _generate_predictions(self, model: PredictiveModel, features: Dict[str, Any], horizon_minutes: int) -> List[Dict[str, Any]]:
        """Generate predictions using a predictive model"""
        try:
            predictions = []
            
            # Simple prediction based on model type
            if model.model_type == PredictionModel.LINEAR_REGRESSION:
                predictions = self._linear_regression_predict(features, horizon_minutes)
            elif model.model_type == PredictionModel.MOVING_AVERAGE:
                predictions = self._moving_average_predict(features, horizon_minutes)
            elif model.model_type == PredictionModel.EXPONENTIAL_SMOOTHING:
                predictions = self._exponential_smoothing_predict(features, horizon_minutes)
            else:
                # Default simple prediction
                base_value = features.get('mean', 1.0)
                trend = features.get('slope', 0)
                
                for i in range(horizon_minutes):
                    predicted_value = base_value + (trend * i)
                    predictions.append({
                        'timestamp': datetime.now() + timedelta(minutes=i),
                        'predicted_value': max(0, predicted_value),
                        'confidence': max(0.1, model.accuracy_score - (i * 0.01))
                    })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to generate predictions: {e}")
            return []

    def _linear_regression_predict(self, features: Dict[str, Any], horizon_minutes: int) -> List[Dict[str, Any]]:
        """Simple linear regression prediction"""
        predictions = []
        base_value = features.get('mean', 1.0)
        trend = features.get('slope', 0)
        
        for i in range(0, horizon_minutes, 5):  # Every 5 minutes
            predicted_value = base_value + (trend * i)
            predictions.append({
                'timestamp': datetime.now() + timedelta(minutes=i),
                'predicted_value': max(0, predicted_value),
                'confidence': max(0.1, 0.8 - (i * 0.005))
            })
        
        return predictions

    def _moving_average_predict(self, features: Dict[str, Any], horizon_minutes: int) -> List[Dict[str, Any]]:
        """Moving average prediction"""
        predictions = []
        base_value = features.get('mean', 1.0)
        
        for i in range(0, horizon_minutes, 5):
            predictions.append({
                'timestamp': datetime.now() + timedelta(minutes=i),
                'predicted_value': base_value,
                'confidence': 0.6
            })
        
        return predictions

    def _exponential_smoothing_predict(self, features: Dict[str, Any], horizon_minutes: int) -> List[Dict[str, Any]]:
        """Exponential smoothing prediction"""
        predictions = []
        base_value = features.get('mean', 1.0)
        trend = features.get('recent_trend', 0)
        alpha = 0.3
        
        for i in range(0, horizon_minutes, 5):
            smoothed_value = base_value + (alpha * trend * i)
            predictions.append({
                'timestamp': datetime.now() + timedelta(minutes=i),
                'predicted_value': max(0, smoothed_value),
                'confidence': max(0.2, 0.7 - (i * 0.003))
            })
        
        return predictions

    def _calculate_confidence_intervals(self, predictions: List[Dict[str, Any]], model_accuracy: float) -> Dict[str, List[float]]:
        """Calculate confidence intervals for predictions"""
        try:
            if not predictions:
                return {}
            
            values = [p['predicted_value'] for p in predictions]
            confidence_factor = model_accuracy * 0.2  # 20% of accuracy as confidence interval
            
            lower_bounds = [v * (1 - confidence_factor) for v in values]
            upper_bounds = [v * (1 + confidence_factor) for v in values]
            
            return {
                'lower_bounds': lower_bounds,
                'upper_bounds': upper_bounds,
                'confidence_level': model_accuracy
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate confidence intervals: {e}")
            return {}

    def _analyze_resource_usage(self, component_id: str) -> Dict[str, Any]:
        """Analyze current resource usage for a component"""
        try:
            # Get recent metrics
            recent_metrics = list(self.performance_metrics[component_id])[-100:]
            
            # Analyze by resource type
            resource_usage = {}
            for resource_type in ResourceType:
                usage_metrics = [m for m in recent_metrics if resource_type.value in str(m.context)]
                if usage_metrics:
                    values = [m.value for m in usage_metrics]
                    resource_usage[resource_type.value] = {
                        'current': values[-1] if values else 0,
                        'average': statistics.mean(values),
                        'peak': max(values),
                        'trend': self._calculate_recent_trend(values[-10:] if len(values) >= 10 else values)
                    }
            
            return {
                'component_id': component_id,
                'resource_usage': resource_usage,
                'overall_utilization': self._calculate_overall_utilization(resource_usage)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze resource usage: {e}")
            return {}

    async def _predict_resource_needs(self, component_id: str) -> Dict[str, Any]:
        """Predict future resource needs for a component"""
        try:
            predictions = {}
            
            # Predict for each resource type
            for resource_type in ResourceType:
                # Use CPU utilization as proxy for other resources
                prediction = await self.predict_performance(
                    component_id, 
                    MetricType.CPU_UTILIZATION, 
                    horizon_minutes=60
                )
                
                if 'error' not in prediction:
                    predictions[resource_type.value] = prediction
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to predict resource needs: {e}")
            return {}

    def _calculate_optimal_allocation(self, component_id: str, usage_analysis: Dict[str, Any], predictions: Dict[str, Any]) -> Optional[ResourceAllocation]:
        """Calculate optimal resource allocation for a component"""
        try:
            # Simple allocation strategy based on current usage and predictions
            resource_usage = usage_analysis.get('resource_usage', {})
            
            # Find the most constrained resource
            max_utilization = 0
            constrained_resource = ResourceType.CPU
            
            for resource_type_str, usage_data in resource_usage.items():
                utilization = usage_data.get('current', 0) / 100.0  # Assume percentage
                if utilization > max_utilization:
                    max_utilization = utilization
                    constrained_resource = ResourceType(resource_type_str)
            
            # If utilization is high, recommend allocation increase
            if max_utilization > 0.8:  # 80% threshold
                recommended_increase = min(max_utilization * 0.2, 0.5)  # 20% increase, max 50%
                
                allocation = ResourceAllocation(
                    resource_type=constrained_resource,
                    allocated_amount=100 * (1 + recommended_increase),
                    max_amount=200,  # Double the normal allocation as max
                    target_component=component_id,
                    priority=self.component_priorities.get(component_id, OptimizationPriority.MEDIUM),
                    allocation_strategy='performance_based',
                    constraints={'max_increase': 0.5},
                    expires_at=datetime.now() + timedelta(hours=1)
                )
                
                return allocation
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to calculate optimal allocation: {e}")
            return None

    async def _update_predictive_models(self):
        """Update predictive models with new data"""
        try:
            for model_id, model in self.predictive_models.items():
                if not model.is_active:
                    continue
                
                # Check if model needs retraining
                if datetime.now() - model.last_trained > timedelta(hours=1):
                    await self._retrain_model(model)
            
        except Exception as e:
            logger.error(f"Failed to update predictive models: {e}")

    async def _retrain_model(self, model: PredictiveModel):
        """Retrain a predictive model with recent data"""
        try:
            # Collect training data
            training_data = []
            for component_metrics in self.performance_metrics.values():
                relevant_metrics = [m for m in component_metrics if m.metric_type == model.target_metric]
                if len(relevant_metrics) >= 50:  # Need sufficient data
                    training_data.extend(relevant_metrics[-50:])  # Use recent 50 metrics
            
            if len(training_data) < 20:
                return
            
            # Simple model update (placeholder for more sophisticated training)
            values = [m.value for m in training_data]
            model.accuracy_score = min(0.9, model.accuracy_score + 0.01)  # Gradual improvement
            model.training_data_size = len(training_data)
            model.last_trained = datetime.now()
            
            # Update model parameters based on data
            if model.model_type == PredictionModel.EXPONENTIAL_SMOOTHING:
                model.model_parameters['alpha'] = min(0.5, statistics.stdev(values) / statistics.mean(values))
            
            self._persist_predictive_model(model)
            logger.info(f"Retrained model {model.id} with {len(training_data)} samples")
            
        except Exception as e:
            logger.error(f"Failed to retrain model {model.id}: {e}")

    async def _execute_optimizations(self, recommendations: List[OptimizationRecommendation]):
        """Execute optimization recommendations"""
        try:
            executed_count = 0
            
            for recommendation in recommendations:
                if recommendation.priority in [OptimizationPriority.CRITICAL, OptimizationPriority.HIGH]:
                    success = await self._execute_recommendation(recommendation)
                    if success:
                        executed_count += 1
                        recommendation.implemented_at = datetime.now()
                        self._persist_optimization_recommendation(recommendation)
            
            logger.info(f"Executed {executed_count} optimization recommendations")
            
        except Exception as e:
            logger.error(f"Failed to execute optimizations: {e}")

    async def _execute_recommendation(self, recommendation: OptimizationRecommendation) -> bool:
        """Execute a single optimization recommendation"""
        try:
            # Simulate optimization execution
            logger.info(f"Executing optimization: {recommendation.description}")
            
            # Add to active optimizations
            self.active_optimizations.add(recommendation.id)
            
            # Simulate implementation delay
            await asyncio.sleep(0.1)
            
            # Remove from active optimizations
            self.active_optimizations.discard(recommendation.id)
            
            # Record effectiveness (placeholder)
            recommendation.effectiveness_rating = min(0.9, recommendation.confidence_score + 0.1)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute recommendation {recommendation.id}: {e}")
            return False

    def _update_system_health_metrics(self):
        """Update overall system health metrics"""
        try:
            # Calculate system-wide metrics
            total_components = len(self.performance_metrics)
            if total_components == 0:
                return
            
            # Average response time across all components
            all_execution_times = []
            for component_metrics in self.performance_metrics.values():
                execution_times = [m.value for m in component_metrics if m.metric_type == MetricType.EXECUTION_TIME]
                all_execution_times.extend(execution_times[-10:])  # Recent 10 per component
            
            if all_execution_times:
                self.system_health_metrics['avg_response_time'] = statistics.mean(all_execution_times)
                self.system_health_metrics['response_time_p95'] = np.percentile(all_execution_times, 95)
            
            # System utilization
            self.system_health_metrics['active_components'] = total_components
            self.system_health_metrics['optimization_efficiency'] = self._calculate_optimization_effectiveness()
            self.system_health_metrics['prediction_accuracy'] = self._calculate_average_model_accuracy()
            
            # Overall health score
            health_factors = [
                min(1.0, 1.0 / max(0.1, self.system_health_metrics.get('avg_response_time', 1.0))),
                self.system_health_metrics.get('optimization_efficiency', 0.5),
                self.system_health_metrics.get('prediction_accuracy', 0.5)
            ]
            self.system_health_metrics['overall_health'] = statistics.mean(health_factors)
            
        except Exception as e:
            logger.error(f"Failed to update system health metrics: {e}")

    # Helper calculation methods
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, float]:
        """Calculate trend analysis for a series of values"""
        if len(values) < 2:
            return {'slope': 0, 'r_squared': 0}
        
        try:
            x = list(range(len(values)))
            n = len(values)
            
            # Calculate linear regression
            sum_x = sum(x)
            sum_y = sum(values)
            sum_xy = sum(x[i] * values[i] for i in range(n))
            sum_x2 = sum(xi * xi for xi in x)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            
            # Calculate R-squared
            y_mean = sum_y / n
            ss_tot = sum((yi - y_mean) ** 2 for yi in values)
            ss_res = sum((values[i] - (slope * x[i])) ** 2 for i in range(n))
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {'slope': slope, 'r_squared': max(0, r_squared)}
            
        except Exception:
            return {'slope': 0, 'r_squared': 0}

    def _calculate_recent_trend(self, values: List[float]) -> float:
        """Calculate recent trend (simplified)"""
        if len(values) < 2:
            return 0
        return (values[-1] - values[0]) / len(values)

    def _detect_anomalies(self, values: List[float]) -> List[Dict[str, Any]]:
        """Detect anomalies in performance data"""
        if len(values) < 10:
            return []
        
        try:
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values)
            threshold = 2 * std_val  # 2 standard deviations
            
            anomalies = []
            for i, value in enumerate(values):
                if abs(value - mean_val) > threshold:
                    anomalies.append({
                        'index': i,
                        'value': value,
                        'deviation': abs(value - mean_val),
                        'severity': 'high' if abs(value - mean_val) > 3 * std_val else 'medium'
                    })
            
            return anomalies
            
        except Exception:
            return []

    def _calculate_recent_metric_rate(self) -> float:
        """Calculate the rate of recent metrics per minute"""
        try:
            total_recent = 0
            cutoff_time = datetime.now() - timedelta(minutes=10)
            
            for component_metrics in self.performance_metrics.values():
                recent_count = sum(1 for m in component_metrics if m.timestamp > cutoff_time)
                total_recent += recent_count
            
            return total_recent / 10.0  # per minute
            
        except Exception:
            return 0.0

    def _calculate_allocation_effectiveness(self) -> float:
        """Calculate the effectiveness of resource allocations"""
        if not self.resource_allocations:
            return 0.0
        
        try:
            effectiveness_scores = [a.effectiveness_score for a in self.resource_allocations.values()]
            return statistics.mean(effectiveness_scores) if effectiveness_scores else 0.0
        except Exception:
            return 0.0

    def _calculate_resource_utilization(self) -> Dict[str, float]:
        """Calculate current resource utilization"""
        try:
            utilization = {}
            
            for resource_type in ResourceType:
                allocations = [a for a in self.resource_allocations.values() if a.resource_type == resource_type]
                if allocations:
                    total_allocated = sum(a.allocated_amount for a in allocations)
                    total_max = sum(a.max_amount for a in allocations)
                    utilization[resource_type.value] = total_allocated / max(total_max, 1)
            
            return utilization
            
        except Exception:
            return {}

    def _calculate_optimization_effectiveness(self) -> float:
        """Calculate the overall effectiveness of optimizations"""
        try:
            implemented_recommendations = [r for r in self.optimization_recommendations.values() if r.implemented_at]
            if not implemented_recommendations:
                return 0.0
            
            effectiveness_scores = [r.effectiveness_rating for r in implemented_recommendations if r.effectiveness_rating > 0]
            return statistics.mean(effectiveness_scores) if effectiveness_scores else 0.0
            
        except Exception:
            return 0.0

    def _calculate_average_model_accuracy(self) -> float:
        """Calculate average accuracy across all predictive models"""
        try:
            active_models = [m for m in self.predictive_models.values() if m.is_active]
            if not active_models:
                return 0.0
            
            accuracy_scores = [m.accuracy_score for m in active_models]
            return statistics.mean(accuracy_scores)
            
        except Exception:
            return 0.0

    def _calculate_overall_utilization(self, resource_usage: Dict[str, Any]) -> float:
        """Calculate overall utilization from resource usage data"""
        try:
            utilizations = []
            for usage_data in resource_usage.values():
                current = usage_data.get('current', 0)
                utilizations.append(min(current / 100.0, 1.0))  # Assume percentage, cap at 100%
            
            return statistics.mean(utilizations) if utilizations else 0.0
            
        except Exception:
            return 0.0

    # Persistence methods
    
    def _persist_performance_metric(self, metric: PerformanceMetric):
        """Persist performance metric to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO performance_metrics 
                (id, metric_type, value, timestamp, source_component, workflow_id, agent_id,
                 context, tags, severity, prediction_confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metric.id, metric.metric_type.value, metric.value, metric.timestamp.isoformat(),
                metric.source_component, metric.workflow_id, metric.agent_id,
                json.dumps(metric.context), json.dumps(metric.tags),
                metric.severity, metric.prediction_confidence
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to persist performance metric: {e}")

    def _persist_resource_allocation(self, allocation: ResourceAllocation):
        """Persist resource allocation to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO resource_allocations 
                (id, resource_type, allocated_amount, max_amount, target_component, priority,
                 allocation_strategy, constraints, created_at, expires_at, effectiveness_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                allocation.id, allocation.resource_type.value, allocation.allocated_amount,
                allocation.max_amount, allocation.target_component, allocation.priority.value,
                allocation.allocation_strategy, json.dumps(allocation.constraints),
                allocation.created_at.isoformat(),
                allocation.expires_at.isoformat() if allocation.expires_at else None,
                allocation.effectiveness_score
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to persist resource allocation: {e}")

    def _persist_optimization_recommendation(self, recommendation: OptimizationRecommendation):
        """Persist optimization recommendation to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO optimization_recommendations 
                (id, strategy, target_component, description, expected_improvement, confidence_score,
                 implementation_complexity, resource_requirements, risk_assessment, action_items,
                 priority, created_at, implemented_at, effectiveness_rating)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                recommendation.id, recommendation.strategy.value, recommendation.target_component,
                recommendation.description, recommendation.expected_improvement, recommendation.confidence_score,
                recommendation.implementation_complexity, json.dumps(recommendation.resource_requirements),
                recommendation.risk_assessment, json.dumps(recommendation.action_items),
                recommendation.priority.value, recommendation.created_at.isoformat(),
                recommendation.implemented_at.isoformat() if recommendation.implemented_at else None,
                recommendation.effectiveness_rating
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to persist optimization recommendation: {e}")

    def _persist_predictive_model(self, model: PredictiveModel):
        """Persist predictive model to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO predictive_models 
                (id, model_type, target_metric, model_parameters, training_data_size, accuracy_score,
                 last_trained, prediction_horizon, feature_importance, validation_results, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model.id, model.model_type.value, model.target_metric.value,
                json.dumps(model.model_parameters), model.training_data_size, model.accuracy_score,
                model.last_trained.isoformat(), model.prediction_horizon,
                json.dumps(model.feature_importance), json.dumps(model.validation_results),
                model.is_active
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to persist predictive model: {e}")

# ================================
# Optimization Engine Factory
# ================================

class OptimizationEngineFactory:
    """Factory for creating optimization engine instances"""
    
    @staticmethod
    def create_optimization_engine(
        config: Optional[Dict[str, Any]] = None
    ) -> RealTimeOptimizationEngine:
        """Create a configured optimization engine"""
        
        default_config = {
            'db_path': 'optimization_engine.db',
            'optimization_interval': 30,
            'prediction_horizon': 3600,
            'enable_predictive_scaling': True,
            'enable_adaptive_learning': True
        }
        
        if config:
            default_config.update(config)
        
        return RealTimeOptimizationEngine(**default_config)

# ================================
# Export Classes
# ================================

__all__ = [
    'RealTimeOptimizationEngine',
    'OptimizationEngineFactory',
    'PerformanceMetric',
    'ResourceAllocation',
    'OptimizationRecommendation',
    'PredictiveModel',
    'WorkloadProfile',
    'OptimizationStrategy',
    'MetricType',
    'ResourceType',
    'OptimizationPriority',
    'PredictionModel',
    'timer_decorator',
    'async_timer_decorator'
]

# ================================
# Demo Functions
# ================================

async def demo_real_time_optimization_engine():
    """Demonstrate real-time optimization engine capabilities"""
    
    print("🚀 Real-Time Optimization Engine Demo")
    print("=" * 50)
    
    # Create optimization engine
    engine = OptimizationEngineFactory.create_optimization_engine({
        'optimization_interval': 10,  # 10 seconds for demo
        'enable_predictive_scaling': True
    })
    
    print("\n1. Recording performance metrics...")
    
    # Record sample performance metrics
    for i in range(20):
        metric = PerformanceMetric(
            metric_type=MetricType.EXECUTION_TIME,
            value=1.0 + (i * 0.1) + (np.random.random() * 0.2),  # Increasing trend with noise
            source_component='workflow_engine',
            context={'operation': 'process_workflow'},
            tags=['performance', 'monitoring']
        )
        engine.record_performance_metric(metric)
        
        # Add CPU utilization metrics
        cpu_metric = PerformanceMetric(
            metric_type=MetricType.CPU_UTILIZATION,
            value=50 + (i * 2) + (np.random.random() * 10),  # Increasing CPU usage
            source_component='workflow_engine',
            context={'resource': 'cpu'},
            tags=['resource', 'monitoring']
        )
        engine.record_performance_metric(cpu_metric)
    
    print(f"✅ Recorded 40 performance metrics")
    
    print("\n2. Generating optimization recommendations...")
    
    # Generate recommendations
    recommendations = engine.generate_optimization_recommendations()
    print(f"✅ Generated {len(recommendations)} recommendations")
    
    for i, rec in enumerate(recommendations[:3]):  # Show top 3
        print(f"   {i+1}. {rec.description}")
        print(f"      Expected improvement: {rec.expected_improvement:.2%}")
        print(f"      Confidence: {rec.confidence_score:.2%}")
        print(f"      Priority: {rec.priority.value}")
    
    print("\n3. Predicting future performance...")
    
    # Generate performance predictions
    prediction = await engine.predict_performance(
        'workflow_engine', 
        MetricType.EXECUTION_TIME, 
        horizon_minutes=30
    )
    
    if 'error' not in prediction:
        print(f"✅ Generated {len(prediction['predictions'])} predictions")
        print(f"   Model accuracy: {prediction['model_accuracy']:.2%}")
        print(f"   Prediction confidence: {prediction['prediction_confidence']:.2%}")
        
        # Show sample predictions
        for i, pred in enumerate(prediction['predictions'][:3]):
            print(f"   {i+1}. {pred['timestamp'].strftime('%H:%M')}: {pred['predicted_value']:.3f}s (confidence: {pred['confidence']:.2%})")
    else:
        print(f"❌ Prediction failed: {prediction['error']}")
    
    print("\n4. Optimizing resource allocation...")
    
    # Optimize resource allocation
    allocations = await engine.optimize_resource_allocation('workflow_engine')
    print(f"✅ Optimized allocation for {len(allocations)} components")
    
    for comp_id, allocation in allocations.items():
        print(f"   {comp_id}: {allocation.resource_type.value} = {allocation.allocated_amount:.1f} units")
        print(f"   Strategy: {allocation.allocation_strategy}, Priority: {allocation.priority.value}")
    
    print("\n5. System status and health metrics...")
    
    status = engine.get_system_status()
    print(f"✅ Engine status: {status['engine_status']}")
    print(f"✅ Active components: {status['performance_monitoring']['active_components']}")
    print(f"✅ Total metrics: {status['performance_monitoring']['total_metrics']}")
    print(f"✅ Active recommendations: {status['optimization']['active_recommendations']}")
    print(f"✅ Active models: {status['prediction']['active_models']}")
    print(f"✅ Model accuracy: {status['prediction']['model_accuracy']:.2%}")
    
    if 'overall_health' in status['system_health']:
        print(f"✅ Overall health: {status['system_health']['overall_health']:.2%}")
    
    print("\n6. Starting optimization loop...")
    
    # Start optimization loop for a short demo
    await engine.start_optimization_loop()
    print("✅ Optimization loop started")
    
    # Let it run for a few cycles
    await asyncio.sleep(25)
    
    # Stop optimization loop
    await engine.stop_optimization_loop()
    print("✅ Optimization loop stopped")
    
    print("\n🎉 Real-Time Optimization Engine Demo Complete!")
    return True

if __name__ == "__main__":
    # Run demo
    async def main():
        success = await demo_real_time_optimization_engine()
        print(f"\nDemo completed: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    asyncio.run(main())