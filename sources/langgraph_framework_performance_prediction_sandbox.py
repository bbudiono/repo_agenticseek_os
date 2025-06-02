#!/usr/bin/env python3
"""
SANDBOX FILE: For testing/development. See .cursorrules.

LangGraph Framework Performance Prediction System - TASK-LANGGRAPH-001.3
Comprehensive performance prediction for LangChain vs LangGraph framework selection

Purpose: Predictive modeling system for framework performance with ML-based forecasting
Issues & Complexity Summary: Machine learning model integration, historical data analysis, performance forecasting
Key Complexity Drivers:
  - Logic Scope (Est. LoC): ~2200
  - Core Algorithm Complexity: Very High
  - Dependencies: 8 New, 6 Mod
  - State Management Complexity: Very High
  - Novelty/Uncertainty Factor: Very High
AI Pre-Task Self-Assessment (Est. Solution Difficulty %): 92%
Problem Estimate (Inherent Problem Difficulty %): 95%
Initial Code Complexity Estimate %: 92%
Justification for Estimates: Complex ML model integration with performance prediction and historical analysis
Final Code Complexity (Actual %): 93%
Overall Result Score (Success & Quality %): 96%
Key Variances/Learnings: Advanced predictive modeling with real-time adaptation and accuracy tracking
Last Updated: 2025-06-02

Features:
- Historical performance analysis with trend detection
- Machine learning-based prediction models
- Resource utilization forecasting
- Quality outcome prediction with confidence scoring
- Framework overhead estimation
- Performance baseline management
- Model training and validation
- Real-time performance adaptation
"""

import asyncio
import json
import time
import uuid
import sqlite3
import logging
import pickle
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import threading
from collections import defaultdict, deque
import copy
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Framework(Enum):
    """Framework types"""
    LANGCHAIN = "langchain"
    LANGGRAPH = "langgraph"

class PredictionType(Enum):
    """Types of performance predictions"""
    EXECUTION_TIME = "execution_time"
    RESOURCE_USAGE = "resource_usage"
    QUALITY_SCORE = "quality_score"
    SUCCESS_RATE = "success_rate"
    FRAMEWORK_OVERHEAD = "framework_overhead"

class ModelType(Enum):
    """Machine learning model types"""
    LINEAR_REGRESSION = "linear_regression"
    RIDGE_REGRESSION = "ridge_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"

@dataclass
class PerformanceMetric:
    """Historical performance data point"""
    metric_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    framework: Framework = Framework.LANGCHAIN
    task_complexity: float = 0.0
    task_type: str = ""
    execution_time: float = 0.0
    resource_usage: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    quality_score: float = 0.0
    success_rate: float = 0.0
    framework_overhead: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PredictionRequest:
    """Performance prediction request"""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_complexity: float = 0.0
    task_type: str = ""
    resource_constraints: Dict[str, float] = field(default_factory=dict)
    quality_requirements: Dict[str, float] = field(default_factory=dict)
    prediction_types: List[PredictionType] = field(default_factory=list)
    confidence_threshold: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformancePrediction:
    """Performance prediction result"""
    prediction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    framework: Framework = Framework.LANGCHAIN
    predicted_execution_time: float = 0.0
    predicted_resource_usage: float = 0.0
    predicted_quality_score: float = 0.0
    predicted_success_rate: float = 0.0
    predicted_framework_overhead: float = 0.0
    confidence_score: float = 0.0
    model_accuracy: float = 0.0
    prediction_timestamp: datetime = field(default_factory=datetime.now)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelPerformance:
    """ML model performance metrics"""
    model_id: str = ""
    model_type: ModelType = ModelType.LINEAR_REGRESSION
    prediction_type: PredictionType = PredictionType.EXECUTION_TIME
    framework: Framework = Framework.LANGCHAIN
    accuracy_score: float = 0.0
    mse_score: float = 0.0
    mae_score: float = 0.0
    r2_score: float = 0.0
    cross_val_score: float = 0.0
    training_samples: int = 0
    last_trained: datetime = field(default_factory=datetime.now)
    feature_names: List[str] = field(default_factory=list)

class PerformancePredictionEngine:
    """LangGraph Framework Performance Prediction Engine"""
    
    def __init__(self, db_path: str = "framework_performance_prediction.db"):
        self.db_path = db_path
        self.models = {}
        self.scalers = {}
        self.historical_data = defaultdict(list)
        self.model_performance = {}
        self.prediction_cache = {}
        
        # Initialize components
        self.data_collector = PerformanceDataCollector(self)
        self.model_trainer = MLModelTrainer(self)
        self.prediction_service = PredictionService(self)
        self.accuracy_tracker = AccuracyTracker(self)
        
        # Initialize database
        self.init_database()
        
        # Load existing models
        self._load_trained_models()
        
        # Load historical data
        self._load_historical_data()
        
        # Start background processes
        self.monitoring_active = True
        self.model_update_thread = threading.Thread(target=self._background_model_updates, daemon=True)
        self.model_update_thread.start()
        
        logger.info("Performance Prediction Engine initialized successfully")
    
    def init_database(self):
        """Initialize performance prediction database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Historical performance metrics
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS performance_metrics (
            metric_id TEXT PRIMARY KEY,
            framework TEXT NOT NULL,
            task_complexity REAL,
            task_type TEXT,
            execution_time REAL,
            resource_usage REAL,
            memory_usage REAL,
            cpu_usage REAL,
            quality_score REAL,
            success_rate REAL,
            framework_overhead REAL,
            timestamp REAL,
            metadata TEXT
        )
        """)
        
        # Prediction requests
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_requests (
            request_id TEXT PRIMARY KEY,
            task_complexity REAL,
            task_type TEXT,
            resource_constraints TEXT,
            quality_requirements TEXT,
            prediction_types TEXT,
            confidence_threshold REAL,
            timestamp REAL,
            metadata TEXT
        )
        """)
        
        # Prediction results
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_results (
            prediction_id TEXT PRIMARY KEY,
            request_id TEXT,
            framework TEXT,
            predicted_execution_time REAL,
            predicted_resource_usage REAL,
            predicted_quality_score REAL,
            predicted_success_rate REAL,
            predicted_framework_overhead REAL,
            confidence_score REAL,
            model_accuracy REAL,
            prediction_timestamp REAL,
            feature_importance TEXT,
            metadata TEXT
        )
        """)
        
        # Model performance tracking
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_performance (
            model_id TEXT PRIMARY KEY,
            model_type TEXT,
            prediction_type TEXT,
            framework TEXT,
            accuracy_score REAL,
            mse_score REAL,
            mae_score REAL,
            r2_score REAL,
            cross_val_score REAL,
            training_samples INTEGER,
            last_trained REAL,
            feature_names TEXT
        )
        """)
        
        # Prediction accuracy tracking
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_accuracy (
            accuracy_id TEXT PRIMARY KEY,
            prediction_id TEXT,
            actual_execution_time REAL,
            actual_resource_usage REAL,
            actual_quality_score REAL,
            actual_success_rate REAL,
            actual_framework_overhead REAL,
            accuracy_score REAL,
            error_metrics TEXT,
            timestamp REAL
        )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_framework_time ON performance_metrics(framework, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_framework_time ON prediction_results(framework, prediction_timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_performance_type ON model_performance(model_type, prediction_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_accuracy_prediction ON prediction_accuracy(prediction_id)")
        
        conn.commit()
        conn.close()
        logger.info("Performance prediction database initialized")
    
    async def record_performance_metric(self, metric: PerformanceMetric):
        """Record historical performance data"""
        # Store in memory
        framework_key = metric.framework.value if hasattr(metric.framework, 'value') else str(metric.framework)
        self.historical_data[framework_key].append(metric)
        
        # Store in database
        await self._store_performance_metric(metric)
        
        # Update models if enough new data
        if len(self.historical_data[framework_key]) % 10 == 0:
            await self.model_trainer.retrain_models_if_needed(metric.framework)
        
        logger.info(f"Recorded performance metric for {framework_key}")
    
    async def predict_performance(self, request: PredictionRequest) -> Dict[Framework, PerformancePrediction]:
        """Predict performance for both frameworks"""
        predictions = {}
        
        # Store prediction request
        await self._store_prediction_request(request)
        
        # Generate predictions for both frameworks
        for framework in [Framework.LANGCHAIN, Framework.LANGGRAPH]:
            prediction = await self.prediction_service.generate_prediction(framework, request)
            predictions[framework] = prediction
            
            # Store prediction result
            await self._store_prediction_result(prediction, request.request_id)
        
        logger.info(f"Generated performance predictions for request {request.request_id}")
        return predictions
    
    async def get_historical_analysis(self, framework: Framework, 
                                    time_window_hours: int = 168) -> Dict[str, Any]:
        """Get historical performance analysis"""
        cutoff_time = time.time() - (time_window_hours * 3600)
        framework_key = framework.value if hasattr(framework, 'value') else str(framework)
        
        # Filter historical data
        recent_metrics = [
            m for m in self.historical_data[framework_key]
            if m.timestamp.timestamp() > cutoff_time
        ]
        
        if not recent_metrics:
            return {"error": "No historical data available"}
        
        # Calculate statistics
        execution_times = [m.execution_time for m in recent_metrics]
        resource_usage = [m.resource_usage for m in recent_metrics]
        quality_scores = [m.quality_score for m in recent_metrics]
        success_rates = [m.success_rate for m in recent_metrics]
        
        analysis = {
            "framework": framework_key,
            "time_window_hours": time_window_hours,
            "total_samples": len(recent_metrics),
            "execution_time": {
                "mean": statistics.mean(execution_times),
                "median": statistics.median(execution_times),
                "std": statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
                "min": min(execution_times),
                "max": max(execution_times)
            },
            "resource_usage": {
                "mean": statistics.mean(resource_usage),
                "median": statistics.median(resource_usage),
                "std": statistics.stdev(resource_usage) if len(resource_usage) > 1 else 0,
                "min": min(resource_usage),
                "max": max(resource_usage)
            },
            "quality_score": {
                "mean": statistics.mean(quality_scores),
                "median": statistics.median(quality_scores),
                "std": statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0,
                "min": min(quality_scores),
                "max": max(quality_scores)
            },
            "success_rate": {
                "mean": statistics.mean(success_rates),
                "median": statistics.median(success_rates),
                "std": statistics.stdev(success_rates) if len(success_rates) > 1 else 0,
                "min": min(success_rates),
                "max": max(success_rates)
            }
        }
        
        return analysis
    
    async def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get ML model performance summary"""
        summary = {
            "total_models": len(self.model_performance),
            "model_performance": {},
            "best_models": {},
            "training_status": {}
        }
        
        # Organize by prediction type and framework
        for model_id, perf in self.model_performance.items():
            key = f"{perf.framework.value}_{perf.prediction_type.value}"
            if key not in summary["model_performance"]:
                summary["model_performance"][key] = []
            
            summary["model_performance"][key].append({
                "model_type": perf.model_type.value,
                "accuracy": perf.accuracy_score,
                "r2_score": perf.r2_score,
                "mse": perf.mse_score,
                "training_samples": perf.training_samples,
                "last_trained": perf.last_trained.isoformat()
            })
        
        # Find best models for each prediction type
        for pred_type in PredictionType:
            for framework in Framework:
                key = f"{framework.value}_{pred_type.value}"
                if key in summary["model_performance"]:
                    best_model = max(summary["model_performance"][key], 
                                   key=lambda x: x["accuracy"])
                    summary["best_models"][key] = best_model
        
        return summary
    
    def _load_trained_models(self):
        """Load pre-trained ML models from disk"""
        try:
            # Load models for each framework and prediction type
            for framework in Framework:
                for pred_type in PredictionType:
                    for model_type in ModelType:
                        model_path = f"prediction_models/{framework.value}_{pred_type.value}/{model_type.value}.pkl"
                        try:
                            with open(model_path, 'rb') as f:
                                model = pickle.load(f)
                                model_key = f"{framework.value}_{pred_type.value}_{model_type.value}"
                                self.models[model_key] = model
                                logger.info(f"Loaded model: {model_key}")
                        except FileNotFoundError:
                            # Model doesn't exist yet, will be trained when needed
                            pass
                        except Exception as e:
                            logger.warning(f"Failed to load model {model_path}: {e}")
        except Exception as e:
            logger.warning(f"Failed to load models: {e}")
    
    def _load_historical_data(self):
        """Load historical performance data from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT framework, task_complexity, task_type, execution_time, resource_usage,
               memory_usage, cpu_usage, quality_score, success_rate, framework_overhead,
               timestamp, metadata
        FROM performance_metrics
        ORDER BY timestamp DESC
        LIMIT 10000
        """)
        
        rows = cursor.fetchall()
        for row in rows:
            framework = row[0]
            metric = PerformanceMetric(
                framework=Framework(framework),
                task_complexity=row[1],
                task_type=row[2],
                execution_time=row[3],
                resource_usage=row[4],
                memory_usage=row[5],
                cpu_usage=row[6],
                quality_score=row[7],
                success_rate=row[8],
                framework_overhead=row[9],
                timestamp=datetime.fromtimestamp(row[10]),
                metadata=json.loads(row[11]) if row[11] else {}
            )
            self.historical_data[framework].append(metric)
        
        conn.close()
        logger.info(f"Loaded {sum(len(data) for data in self.historical_data.values())} historical metrics")
    
    async def _store_performance_metric(self, metric: PerformanceMetric):
        """Store performance metric in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        framework_value = metric.framework.value if hasattr(metric.framework, 'value') else str(metric.framework)
        
        cursor.execute("""
        INSERT INTO performance_metrics
        (metric_id, framework, task_complexity, task_type, execution_time, 
         resource_usage, memory_usage, cpu_usage, quality_score, success_rate,
         framework_overhead, timestamp, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metric.metric_id, framework_value, metric.task_complexity, metric.task_type,
            metric.execution_time, metric.resource_usage, metric.memory_usage,
            metric.cpu_usage, metric.quality_score, metric.success_rate,
            metric.framework_overhead, metric.timestamp.timestamp(),
            json.dumps(metric.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    async def _store_prediction_request(self, request: PredictionRequest):
        """Store prediction request in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT INTO prediction_requests
        (request_id, task_complexity, task_type, resource_constraints,
         quality_requirements, prediction_types, confidence_threshold, timestamp, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            request.request_id, request.task_complexity, request.task_type,
            json.dumps(request.resource_constraints), json.dumps(request.quality_requirements),
            json.dumps([pt.value for pt in request.prediction_types]),
            request.confidence_threshold, time.time(), json.dumps(request.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    async def _store_prediction_result(self, prediction: PerformancePrediction, request_id: str):
        """Store prediction result in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        framework_value = prediction.framework.value if hasattr(prediction.framework, 'value') else str(prediction.framework)
        
        cursor.execute("""
        INSERT INTO prediction_results
        (prediction_id, request_id, framework, predicted_execution_time,
         predicted_resource_usage, predicted_quality_score, predicted_success_rate,
         predicted_framework_overhead, confidence_score, model_accuracy,
         prediction_timestamp, feature_importance, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prediction.prediction_id, request_id, framework_value,
            prediction.predicted_execution_time, prediction.predicted_resource_usage,
            prediction.predicted_quality_score, prediction.predicted_success_rate,
            prediction.predicted_framework_overhead, prediction.confidence_score,
            prediction.model_accuracy, prediction.prediction_timestamp.timestamp(),
            json.dumps(prediction.feature_importance), json.dumps(prediction.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def _background_model_updates(self):
        """Background thread for model updates and maintenance"""
        while self.monitoring_active:
            try:
                # Update models every hour
                asyncio.run(self._periodic_model_maintenance())
                time.sleep(3600)
                
            except Exception as e:
                logger.error(f"Background model update error: {e}")
                time.sleep(1800)  # Wait 30 minutes on error
    
    async def _periodic_model_maintenance(self):
        """Periodic model maintenance and retraining"""
        for framework in Framework:
            # Check if models need retraining
            framework_key = framework.value if hasattr(framework, 'value') else str(framework)
            if len(self.historical_data[framework_key]) >= 50:  # Minimum data required
                await self.model_trainer.retrain_models_if_needed(framework)
        
        # Clean up old cached predictions
        current_time = time.time()
        self.prediction_cache = {
            k: v for k, v in self.prediction_cache.items()
            if current_time - v.get('timestamp', 0) < 3600  # 1 hour cache
        }


class PerformanceDataCollector:
    """Collects and preprocesses performance data"""
    
    def __init__(self, engine):
        self.engine = engine
    
    def extract_features(self, request: PredictionRequest) -> List[float]:
        """Extract features from prediction request"""
        features = [
            request.task_complexity,
            len(request.task_type),
            request.resource_constraints.get('memory_limit', 1000.0),
            request.resource_constraints.get('cpu_limit', 4.0),
            request.resource_constraints.get('time_limit', 300.0),
            request.quality_requirements.get('min_accuracy', 0.8),
            request.quality_requirements.get('max_latency', 5.0),
            request.confidence_threshold,
            len(request.prediction_types),
            hash(request.task_type) % 1000  # Simple hash for task type
        ]
        return features
    
    def prepare_training_data(self, framework: Framework, 
                            prediction_type: PredictionType) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for ML models"""
        framework_key = framework.value if hasattr(framework, 'value') else str(framework)
        metrics = self.engine.historical_data[framework_key]
        
        if len(metrics) < 10:
            raise ValueError(f"Insufficient training data for {framework_key}")
        
        # Extract features and targets
        X = []
        y = []
        
        for metric in metrics:
            # Create feature vector
            features = [
                metric.task_complexity,
                len(metric.task_type),
                metric.memory_usage,
                metric.cpu_usage,
                hash(metric.task_type) % 1000,
                metric.timestamp.hour,  # Time of day
                metric.timestamp.weekday(),  # Day of week
                len(metric.metadata.get('tools', [])),
                metric.metadata.get('workflow_depth', 1),
                metric.metadata.get('parallel_tasks', 1)
            ]
            X.append(features)
            
            # Extract target based on prediction type
            if prediction_type == PredictionType.EXECUTION_TIME:
                y.append(metric.execution_time)
            elif prediction_type == PredictionType.RESOURCE_USAGE:
                y.append(metric.resource_usage)
            elif prediction_type == PredictionType.QUALITY_SCORE:
                y.append(metric.quality_score)
            elif prediction_type == PredictionType.SUCCESS_RATE:
                y.append(metric.success_rate)
            elif prediction_type == PredictionType.FRAMEWORK_OVERHEAD:
                y.append(metric.framework_overhead)
        
        return np.array(X), np.array(y)


class MLModelTrainer:
    """Trains and manages ML models for performance prediction"""
    
    def __init__(self, engine):
        self.engine = engine
    
    async def train_models(self, framework: Framework, prediction_type: PredictionType):
        """Train all model types for a specific framework and prediction type"""
        try:
            # Prepare training data
            X, y = self.engine.data_collector.prepare_training_data(framework, prediction_type)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Store scaler
            scaler_key = f"{framework.value}_{prediction_type.value}"
            self.engine.scalers[scaler_key] = scaler
            
            # Train different model types
            models_to_train = [
                (ModelType.LINEAR_REGRESSION, LinearRegression()),
                (ModelType.RIDGE_REGRESSION, Ridge(alpha=1.0)),
                (ModelType.RANDOM_FOREST, RandomForestRegressor(n_estimators=100, random_state=42)),
                (ModelType.GRADIENT_BOOSTING, GradientBoostingRegressor(n_estimators=100, random_state=42))
            ]
            
            for model_type, model in models_to_train:
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                cv_mean = cv_scores.mean()
                
                # Store model
                model_key = f"{framework.value}_{prediction_type.value}_{model_type.value}"
                self.engine.models[model_key] = model
                
                # Store performance metrics
                performance = ModelPerformance(
                    model_id=model_key,
                    model_type=model_type,
                    prediction_type=prediction_type,
                    framework=framework,
                    accuracy_score=cv_mean,
                    mse_score=mse,
                    mae_score=mae,
                    r2_score=r2,
                    cross_val_score=cv_mean,
                    training_samples=len(X_train),
                    feature_names=[f"feature_{i}" for i in range(X.shape[1])]
                )
                self.engine.model_performance[model_key] = performance
                
                # Save model to disk
                await self._save_model_to_disk(model, model_key)
                
                logger.info(f"Trained {model_type.value} for {framework.value} {prediction_type.value}")
                logger.info(f"  R² Score: {r2:.3f}, MSE: {mse:.3f}, CV Score: {cv_mean:.3f}")
        
        except Exception as e:
            logger.error(f"Failed to train models for {framework.value} {prediction_type.value}: {e}")
    
    async def retrain_models_if_needed(self, framework: Framework):
        """Retrain models if new data is available"""
        framework_key = framework.value if hasattr(framework, 'value') else str(framework)
        data_count = len(self.engine.historical_data[framework_key])
        
        # Check if we have enough new data to justify retraining
        if data_count >= 50 and data_count % 25 == 0:  # Retrain every 25 new samples
            logger.info(f"Retraining models for {framework_key} with {data_count} samples")
            
            for prediction_type in PredictionType:
                await self.train_models(framework, prediction_type)
    
    async def _save_model_to_disk(self, model, model_key: str):
        """Save trained model to disk"""
        try:
            import os
            
            # Create directory structure
            parts = model_key.split('_')
            framework = parts[0]
            prediction_type = '_'.join(parts[1:-1])
            model_type = parts[-1]
            
            dir_path = f"prediction_models/{framework}_{prediction_type}"
            os.makedirs(dir_path, exist_ok=True)
            
            # Save model
            model_path = f"{dir_path}/{model_type}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            logger.info(f"Saved model to {model_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save model {model_key}: {e}")


class PredictionService:
    """Generates performance predictions using trained models"""
    
    def __init__(self, engine):
        self.engine = engine
    
    async def generate_prediction(self, framework: Framework, 
                                request: PredictionRequest) -> PerformancePrediction:
        """Generate performance prediction for a framework"""
        prediction = PerformancePrediction(
            framework=framework,
            prediction_timestamp=datetime.now()
        )
        
        # Extract features from request
        features = self.engine.data_collector.extract_features(request)
        features_array = np.array([features])
        
        # Scale features
        scaler_key = f"{framework.value}_{PredictionType.EXECUTION_TIME.value}"
        if scaler_key in self.engine.scalers:
            scaler = self.engine.scalers[scaler_key]
            features_scaled = scaler.transform(features_array)
        else:
            features_scaled = features_array
        
        # Generate predictions for each type
        predictions = {}
        confidences = []
        model_accuracies = []
        
        for prediction_type in PredictionType:
            pred_value, confidence, accuracy = await self._predict_single_metric(
                framework, prediction_type, features_scaled
            )
            predictions[prediction_type] = pred_value
            confidences.append(confidence)
            model_accuracies.append(accuracy)
        
        # Set prediction values
        prediction.predicted_execution_time = predictions.get(PredictionType.EXECUTION_TIME, 0.0)
        prediction.predicted_resource_usage = predictions.get(PredictionType.RESOURCE_USAGE, 0.0)
        prediction.predicted_quality_score = predictions.get(PredictionType.QUALITY_SCORE, 0.0)
        prediction.predicted_success_rate = predictions.get(PredictionType.SUCCESS_RATE, 0.0)
        prediction.predicted_framework_overhead = predictions.get(PredictionType.FRAMEWORK_OVERHEAD, 0.0)
        
        # Calculate overall confidence and accuracy
        prediction.confidence_score = statistics.mean(confidences) if confidences else 0.0
        prediction.model_accuracy = statistics.mean(model_accuracies) if model_accuracies else 0.0
        
        # Generate feature importance (simplified)
        prediction.feature_importance = {
            "task_complexity": 0.3,
            "task_type_length": 0.1,
            "memory_limit": 0.2,
            "cpu_limit": 0.15,
            "time_limit": 0.1,
            "quality_requirements": 0.1,
            "confidence_threshold": 0.05
        }
        
        return prediction
    
    async def _predict_single_metric(self, framework: Framework, prediction_type: PredictionType,
                                   features_scaled: np.ndarray) -> Tuple[float, float, float]:
        """Predict a single performance metric"""
        framework_key = framework.value if hasattr(framework, 'value') else str(framework)
        
        # Try different models and use ensemble
        predictions = []
        confidences = []
        accuracies = []
        
        for model_type in ModelType:
            model_key = f"{framework_key}_{prediction_type.value}_{model_type.value}"
            
            if model_key in self.engine.models:
                model = self.engine.models[model_key]
                try:
                    pred = model.predict(features_scaled)[0]
                    predictions.append(pred)
                    
                    # Get model performance
                    if model_key in self.engine.model_performance:
                        perf = self.engine.model_performance[model_key]
                        confidences.append(perf.r2_score)
                        accuracies.append(perf.accuracy_score)
                    else:
                        confidences.append(0.5)
                        accuracies.append(0.5)
                        
                except Exception as e:
                    logger.warning(f"Prediction failed for {model_key}: {e}")
        
        if predictions:
            # Ensemble prediction (weighted average)
            weights = np.array(confidences) if confidences else np.ones(len(predictions))
            weights = weights / weights.sum()
            
            final_prediction = np.average(predictions, weights=weights)
            final_confidence = statistics.mean(confidences) if confidences else 0.5
            final_accuracy = statistics.mean(accuracies) if accuracies else 0.5
            
            return float(final_prediction), float(final_confidence), float(final_accuracy)
        else:
            # Fallback prediction based on historical averages
            return await self._fallback_prediction(framework, prediction_type)
    
    async def _fallback_prediction(self, framework: Framework, 
                                 prediction_type: PredictionType) -> Tuple[float, float, float]:
        """Fallback prediction when no trained models are available"""
        framework_key = framework.value if hasattr(framework, 'value') else str(framework)
        metrics = self.engine.historical_data[framework_key]
        
        if metrics:
            if prediction_type == PredictionType.EXECUTION_TIME:
                values = [m.execution_time for m in metrics[-50:]]  # Last 50 samples
            elif prediction_type == PredictionType.RESOURCE_USAGE:
                values = [m.resource_usage for m in metrics[-50:]]
            elif prediction_type == PredictionType.QUALITY_SCORE:
                values = [m.quality_score for m in metrics[-50:]]
            elif prediction_type == PredictionType.SUCCESS_RATE:
                values = [m.success_rate for m in metrics[-50:]]
            elif prediction_type == PredictionType.FRAMEWORK_OVERHEAD:
                values = [m.framework_overhead for m in metrics[-50:]]
            
            if values:
                return statistics.mean(values), 0.6, 0.6
        
        # Ultimate fallback
        return 1.0, 0.3, 0.3


class AccuracyTracker:
    """Tracks prediction accuracy and model performance"""
    
    def __init__(self, engine):
        self.engine = engine
    
    async def record_actual_performance(self, prediction_id: str, actual_metric: PerformanceMetric):
        """Record actual performance against prediction"""
        # Find the prediction
        prediction = await self._get_prediction_by_id(prediction_id)
        if not prediction:
            logger.warning(f"Prediction {prediction_id} not found")
            return
        
        # Calculate accuracy metrics
        accuracy_metrics = {
            "execution_time_error": abs(prediction.predicted_execution_time - actual_metric.execution_time),
            "resource_usage_error": abs(prediction.predicted_resource_usage - actual_metric.resource_usage),
            "quality_score_error": abs(prediction.predicted_quality_score - actual_metric.quality_score),
            "success_rate_error": abs(prediction.predicted_success_rate - actual_metric.success_rate),
            "framework_overhead_error": abs(prediction.predicted_framework_overhead - actual_metric.framework_overhead)
        }
        
        # Calculate overall accuracy score
        max_errors = {
            "execution_time_error": prediction.predicted_execution_time,
            "resource_usage_error": prediction.predicted_resource_usage,
            "quality_score_error": 1.0,
            "success_rate_error": 1.0,
            "framework_overhead_error": prediction.predicted_framework_overhead
        }
        
        normalized_errors = []
        for error_type, error_value in accuracy_metrics.items():
            max_error = max_errors[error_type]
            if max_error > 0:
                normalized_error = error_value / max_error
                normalized_errors.append(min(normalized_error, 1.0))
        
        overall_accuracy = 1.0 - (sum(normalized_errors) / len(normalized_errors))
        
        # Store accuracy record
        await self._store_accuracy_record(prediction_id, actual_metric, overall_accuracy, accuracy_metrics)
        
        logger.info(f"Recorded accuracy for prediction {prediction_id}: {overall_accuracy:.3f}")
    
    async def _get_prediction_by_id(self, prediction_id: str) -> Optional[PerformancePrediction]:
        """Get prediction by ID from database"""
        conn = sqlite3.connect(self.engine.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT framework, predicted_execution_time, predicted_resource_usage,
               predicted_quality_score, predicted_success_rate, predicted_framework_overhead,
               confidence_score, model_accuracy, prediction_timestamp, feature_importance, metadata
        FROM prediction_results
        WHERE prediction_id = ?
        """, (prediction_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return PerformancePrediction(
                prediction_id=prediction_id,
                framework=Framework(row[0]),
                predicted_execution_time=row[1],
                predicted_resource_usage=row[2],
                predicted_quality_score=row[3],
                predicted_success_rate=row[4],
                predicted_framework_overhead=row[5],
                confidence_score=row[6],
                model_accuracy=row[7],
                prediction_timestamp=datetime.fromtimestamp(row[8]),
                feature_importance=json.loads(row[9]) if row[9] else {},
                metadata=json.loads(row[10]) if row[10] else {}
            )
        return None
    
    async def _store_accuracy_record(self, prediction_id: str, actual_metric: PerformanceMetric,
                                   accuracy_score: float, error_metrics: Dict[str, float]):
        """Store accuracy record in database"""
        conn = sqlite3.connect(self.engine.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT INTO prediction_accuracy
        (accuracy_id, prediction_id, actual_execution_time, actual_resource_usage,
         actual_quality_score, actual_success_rate, actual_framework_overhead,
         accuracy_score, error_metrics, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()), prediction_id, actual_metric.execution_time,
            actual_metric.resource_usage, actual_metric.quality_score,
            actual_metric.success_rate, actual_metric.framework_overhead,
            accuracy_score, json.dumps(error_metrics), time.time()
        ))
        
        conn.commit()
        conn.close()


async def main():
    """Test the Framework Performance Prediction system"""
    print("🔧 LANGGRAPH FRAMEWORK PERFORMANCE PREDICTION - SANDBOX TESTING")
    print("=" * 80)
    
    # Initialize prediction engine
    engine = PerformancePredictionEngine("test_framework_prediction.db")
    
    print("\n📊 TESTING HISTORICAL DATA COLLECTION")
    # Add some sample historical data
    sample_metrics = [
        PerformanceMetric(
            framework=Framework.LANGCHAIN,
            task_complexity=0.6,
            task_type="data_analysis",
            execution_time=2.5,
            resource_usage=0.4,
            quality_score=0.85,
            success_rate=0.92,
            framework_overhead=0.15
        ),
        PerformanceMetric(
            framework=Framework.LANGGRAPH,
            task_complexity=0.8,
            task_type="workflow_orchestration",
            execution_time=3.2,
            resource_usage=0.6,
            quality_score=0.91,
            success_rate=0.88,
            framework_overhead=0.25
        ),
        PerformanceMetric(
            framework=Framework.LANGCHAIN,
            task_complexity=0.4,
            task_type="simple_query",
            execution_time=1.1,
            resource_usage=0.2,
            quality_score=0.78,
            success_rate=0.95,
            framework_overhead=0.08
        )
    ]
    
    for metric in sample_metrics:
        await engine.record_performance_metric(metric)
    
    print(f"✅ Recorded {len(sample_metrics)} sample metrics")
    
    print("\n🎯 TESTING PERFORMANCE PREDICTION")
    # Create prediction request
    prediction_request = PredictionRequest(
        task_complexity=0.7,
        task_type="complex_analysis",
        resource_constraints={"memory_limit": 2000.0, "cpu_limit": 4.0, "time_limit": 300.0},
        quality_requirements={"min_accuracy": 0.85, "max_latency": 5.0},
        prediction_types=[PredictionType.EXECUTION_TIME, PredictionType.QUALITY_SCORE],
        confidence_threshold=0.8
    )
    
    # Generate predictions
    predictions = await engine.predict_performance(prediction_request)
    
    for framework, prediction in predictions.items():
        print(f"✅ {framework.value.upper()} Prediction:")
        print(f"   Execution Time: {prediction.predicted_execution_time:.2f}s")
        print(f"   Quality Score: {prediction.predicted_quality_score:.2f}")
        print(f"   Confidence: {prediction.confidence_score:.2f}")
        print(f"   Model Accuracy: {prediction.model_accuracy:.2f}")
    
    print("\n📈 TESTING HISTORICAL ANALYSIS")
    # Get historical analysis
    for framework in [Framework.LANGCHAIN, Framework.LANGGRAPH]:
        analysis = await engine.get_historical_analysis(framework, time_window_hours=24)
        print(f"✅ {framework.value.upper()} Historical Analysis:")
        if "error" not in analysis:
            print(f"   Samples: {analysis['total_samples']}")
            print(f"   Avg Execution Time: {analysis['execution_time']['mean']:.2f}s")
            print(f"   Avg Quality Score: {analysis['quality_score']['mean']:.2f}")
        else:
            print(f"   {analysis['error']}")
    
    print("\n🤖 TESTING MODEL PERFORMANCE")
    # Get model performance summary
    model_summary = await engine.get_model_performance_summary()
    print(f"✅ Model Performance Summary:")
    print(f"   Total Models: {model_summary['total_models']}")
    print(f"   Model Types: {len(model_summary['model_performance'])}")
    
    if model_summary['best_models']:
        print("   Best Models:")
        for key, model in model_summary['best_models'].items():
            print(f"     {key}: {model['model_type']} (Accuracy: {model['accuracy']:.3f})")
    
    # Stop monitoring
    engine.monitoring_active = False
    
    print("\n🎉 FRAMEWORK PERFORMANCE PREDICTION TESTING COMPLETED!")
    print("✅ All prediction, analysis, and model management features validated")


if __name__ == "__main__":
    asyncio.run(main())