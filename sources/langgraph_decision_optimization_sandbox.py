#!/usr/bin/env python3
"""
LangGraph Decision Optimization Sandbox Implementation
Implements machine learning-based decision optimization with continuous learning and A/B testing.

* Purpose: ML-based framework selection optimization with continuous learning capabilities
* Issues & Complexity Summary: Complex ML pipeline with real-time feedback and model updating
* Key Complexity Drivers:
  - Logic Scope (Est. LoC): ~2000
  - Core Algorithm Complexity: High (ML algorithms, statistical testing)
  - Dependencies: 8 (sklearn, scipy, numpy, sqlite3, asyncio, threading, json, datetime)
  - State Management Complexity: High (model states, A/B tests, feedback loops)
  - Novelty/Uncertainty Factor: Medium (ML optimization patterns)
* AI Pre-Task Self-Assessment (Est. Solution Difficulty %): 85%
* Problem Estimate (Inherent Problem Difficulty %): 80%
* Initial Code Complexity Estimate %: 85%
* Justification for Estimates: Complex ML pipeline with real-time optimization and statistical validation
* Final Code Complexity (Actual %): TBD
* Overall Result Score (Success & Quality %): TBD
* Key Variances/Learnings: TBD
* Last Updated: 2025-06-04
"""

import asyncio
import sqlite3
import json
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import statistics
import random
import math

# ML and Statistical Libraries
try:
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import StandardScaler
    import scipy.stats as stats
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    # Fallback implementations for basic functionality
    np = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DecisionStrategy(Enum):
    """Decision strategy types for A/B testing"""
    CURRENT_MODEL = "current_model"
    IMPROVED_MODEL = "improved_model"
    RANDOM_BASELINE = "random_baseline"
    EXPERT_RULES = "expert_rules"
    HYBRID_APPROACH = "hybrid_approach"

class ModelType(Enum):
    """Machine learning model types"""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    LOGISTIC_REGRESSION = "logistic_regression"
    ENSEMBLE = "ensemble"

@dataclass
class DecisionRecord:
    """Record of a framework selection decision"""
    decision_id: str
    timestamp: datetime
    task_complexity: float
    framework_selected: str
    strategy_used: str
    confidence_score: float
    actual_performance: Optional[float] = None
    execution_time: Optional[float] = None
    success_rate: Optional[float] = None
    user_satisfaction: Optional[float] = None
    ab_test_group: Optional[str] = None

@dataclass
class ABTestConfiguration:
    """Configuration for A/B testing"""
    test_id: str
    test_name: str
    control_strategy: DecisionStrategy
    treatment_strategy: DecisionStrategy
    traffic_split: float  # Percentage for treatment group
    start_time: datetime
    end_time: datetime
    min_sample_size: int
    significance_level: float
    power: float
    is_active: bool = True

@dataclass
class PerformanceFeedback:
    """Performance feedback for decision learning"""
    decision_id: str
    feedback_type: str
    feedback_value: float
    timestamp: datetime
    source: str
    confidence: float

class DecisionLearningEngine:
    """Machine learning engine for continuous decision optimization"""
    
    def __init__(self, db_path: str = "decision_optimization.db"):
        self.db_path = db_path
        self.models = {}
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        self.model_performance = {}
        self.feature_importance = {}
        self.learning_rate = 0.1
        self.min_training_samples = 50
        self.setup_database()
        self.load_models()
        
    def setup_database(self):
        """Initialize database tables for decision optimization"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Decision records table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS decision_records (
                        decision_id TEXT PRIMARY KEY,
                        timestamp TEXT,
                        task_complexity REAL,
                        framework_selected TEXT,
                        strategy_used TEXT,
                        confidence_score REAL,
                        actual_performance REAL,
                        execution_time REAL,
                        success_rate REAL,
                        user_satisfaction REAL,
                        ab_test_group TEXT
                    )
                """)
                
                # Model performance table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS model_performance (
                        model_id TEXT PRIMARY KEY,
                        model_type TEXT,
                        accuracy REAL,
                        precision REAL,
                        recall REAL,
                        f1_score REAL,
                        training_time REAL,
                        last_updated TEXT,
                        training_samples INTEGER
                    )
                """)
                
                # A/B test results table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ab_test_results (
                        test_id TEXT,
                        decision_id TEXT,
                        strategy TEXT,
                        performance_metric REAL,
                        timestamp TEXT,
                        PRIMARY KEY (test_id, decision_id)
                    )
                """)
                
                # Performance feedback table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_feedback (
                        feedback_id TEXT PRIMARY KEY,
                        decision_id TEXT,
                        feedback_type TEXT,
                        feedback_value REAL,
                        timestamp TEXT,
                        source TEXT,
                        confidence REAL
                    )
                """)
                
                conn.commit()
                logger.info("Decision optimization database initialized successfully")
                
        except Exception as e:
            logger.error(f"Database setup error: {e}")
            raise
    
    def load_models(self):
        """Load pre-trained models from storage"""
        try:
            if not ML_AVAILABLE:
                logger.warning("ML libraries not available, using fallback models")
                self._initialize_fallback_models()
                return
                
            # Initialize models
            self.models = {
                ModelType.RANDOM_FOREST: RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                ),
                ModelType.GRADIENT_BOOSTING: GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                ),
                ModelType.LOGISTIC_REGRESSION: LogisticRegression(
                    random_state=42,
                    max_iter=1000
                )
            }
            
            # Try to load existing trained models
            self._load_model_states()
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Model loading error: {e}")
            self._initialize_fallback_models()
    
    def _initialize_fallback_models(self):
        """Initialize simple fallback models when ML libraries unavailable"""
        self.models = {
            ModelType.RANDOM_FOREST: self._create_fallback_model("random_forest"),
            ModelType.GRADIENT_BOOSTING: self._create_fallback_model("gradient_boosting"),
            ModelType.LOGISTIC_REGRESSION: self._create_fallback_model("logistic_regression")
        }
    
    def _create_fallback_model(self, model_type: str):
        """Create a simple fallback model"""
        class FallbackModel:
            def __init__(self, model_type):
                self.model_type = model_type
                self.accuracy = 0.75  # Default accuracy
                
            def predict(self, X):
                # Simple rule-based prediction
                predictions = []
                for features in X:
                    complexity = features[0] if len(features) > 0 else 0.5
                    # Simple rule: LangGraph for complex tasks, LangChain for simple
                    prediction = 1 if complexity > 0.6 else 0
                    predictions.append(prediction)
                return predictions
                
            def predict_proba(self, X):
                # Simple probability estimation
                predictions = self.predict(X)
                probabilities = []
                for pred in predictions:
                    if pred == 1:
                        probabilities.append([0.3, 0.7])
                    else:
                        probabilities.append([0.7, 0.3])
                return probabilities
                
            def fit(self, X, y):
                # Mock training
                pass
                
        return FallbackModel(model_type)
    
    def _load_model_states(self):
        """Load model states from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM model_performance")
                model_data = cursor.fetchall()
                
                for row in model_data:
                    model_id, model_type, accuracy, precision, recall, f1, training_time, last_updated, samples = row
                    self.model_performance[model_type] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'training_time': training_time,
                        'last_updated': last_updated,
                        'training_samples': samples
                    }
                    
        except Exception as e:
            logger.error(f"Model state loading error: {e}")
    
    def extract_features(self, task_data: Dict) -> List[float]:
        """Extract features for ML model prediction"""
        try:
            features = [
                task_data.get('complexity_score', 0.5),
                task_data.get('resource_requirements', 0.5),
                task_data.get('agent_count', 1) / 10.0,  # Normalize
                task_data.get('workflow_complexity', 0.5),
                task_data.get('state_management_complexity', 0.5),
                task_data.get('memory_requirements', 0.5),
                task_data.get('performance_priority', 0.5),
                task_data.get('quality_priority', 0.5),
                task_data.get('user_tier', 1) / 3.0,  # Normalize tier (1-3)
                task_data.get('historical_performance', 0.5)
            ]
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return [0.5] * 10  # Default features
    
    def predict_optimal_framework(self, task_data: Dict, model_type: ModelType = ModelType.ENSEMBLE) -> Tuple[str, float]:
        """Predict optimal framework using ML models"""
        try:
            features = self.extract_features(task_data)
            
            if model_type == ModelType.ENSEMBLE:
                return self._ensemble_prediction(features)
            else:
                model = self.models.get(model_type)
                if model is None:
                    return "langchain", 0.5
                    
                if ML_AVAILABLE and hasattr(model, 'predict_proba'):
                    # Scale features if using real ML models
                    if self.scaler is not None:
                        features = self.scaler.transform([features])[0]
                    
                    probabilities = model.predict_proba([features])[0]
                    prediction = model.predict([features])[0]
                else:
                    # Fallback model
                    probabilities = model.predict_proba([features])[0]
                    prediction = model.predict([features])[0]
                
                framework = "langgraph" if prediction == 1 else "langchain"
                confidence = max(probabilities)
                
                return framework, confidence
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return "langchain", 0.5
    
    def _ensemble_prediction(self, features: List[float]) -> Tuple[str, float]:
        """Ensemble prediction using multiple models"""
        try:
            predictions = []
            confidences = []
            
            for model_type, model in self.models.items():
                if model is None:
                    continue
                    
                try:
                    if ML_AVAILABLE and hasattr(model, 'predict_proba'):
                        if self.scaler is not None:
                            scaled_features = self.scaler.transform([features])[0]
                        else:
                            scaled_features = features
                        
                        probabilities = model.predict_proba([scaled_features])[0]
                        prediction = model.predict([scaled_features])[0]
                    else:
                        probabilities = model.predict_proba([features])[0]
                        prediction = model.predict([features])[0]
                    
                    predictions.append(prediction)
                    confidences.append(max(probabilities))
                    
                except Exception as e:
                    logger.error(f"Model {model_type} prediction error: {e}")
                    continue
            
            if not predictions:
                return "langchain", 0.5
            
            # Majority vote with confidence weighting
            weighted_vote = sum(pred * conf for pred, conf in zip(predictions, confidences))
            total_confidence = sum(confidences)
            
            if total_confidence == 0:
                return "langchain", 0.5
                
            average_prediction = weighted_vote / total_confidence
            framework = "langgraph" if average_prediction > 0.5 else "langchain"
            confidence = statistics.mean(confidences)
            
            return framework, confidence
            
        except Exception as e:
            logger.error(f"Ensemble prediction error: {e}")
            return "langchain", 0.5
    
    def record_decision(self, decision: DecisionRecord):
        """Record a framework selection decision"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO decision_records 
                    (decision_id, timestamp, task_complexity, framework_selected, 
                     strategy_used, confidence_score, actual_performance, 
                     execution_time, success_rate, user_satisfaction, ab_test_group)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    decision.decision_id,
                    decision.timestamp.isoformat(),
                    decision.task_complexity,
                    decision.framework_selected,
                    decision.strategy_used,
                    decision.confidence_score,
                    decision.actual_performance,
                    decision.execution_time,
                    decision.success_rate,
                    decision.user_satisfaction,
                    decision.ab_test_group
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Decision recording error: {e}")
    
    def update_model_with_feedback(self, feedback_data: List[PerformanceFeedback]):
        """Update models based on performance feedback"""
        try:
            if not feedback_data:
                return
                
            # Collect training data
            training_data = self._prepare_training_data(feedback_data)
            
            if len(training_data['features']) < self.min_training_samples:
                logger.info(f"Insufficient training data: {len(training_data['features'])} samples")
                return
            
            # Update each model
            for model_type, model in self.models.items():
                if model is None:
                    continue
                    
                try:
                    self._train_model(model, training_data, model_type)
                except Exception as e:
                    logger.error(f"Model {model_type} training error: {e}")
            
            logger.info(f"Models updated with {len(feedback_data)} feedback samples")
            
        except Exception as e:
            logger.error(f"Model update error: {e}")
    
    def _prepare_training_data(self, feedback_data: List[PerformanceFeedback]) -> Dict:
        """Prepare training data from feedback"""
        try:
            features = []
            labels = []
            
            # Get decision records for feedback
            decision_ids = [fb.decision_id for fb in feedback_data]
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                placeholders = ','.join('?' * len(decision_ids))
                cursor.execute(f"""
                    SELECT decision_id, task_complexity, framework_selected, 
                           actual_performance, success_rate
                    FROM decision_records 
                    WHERE decision_id IN ({placeholders})
                """, decision_ids)
                
                decision_data = cursor.fetchall()
            
            # Create feature vectors and labels
            for row in decision_data:
                decision_id, complexity, framework, performance, success_rate = row
                
                # Find corresponding feedback
                feedback = next((fb for fb in feedback_data if fb.decision_id == decision_id), None)
                if feedback is None:
                    continue
                
                # Create feature vector (simplified)
                feature_vector = [
                    complexity,
                    performance or 0.5,
                    success_rate or 0.5,
                    feedback.feedback_value,
                    feedback.confidence
                ]
                
                # Label: 1 for langgraph, 0 for langchain
                label = 1 if framework == "langgraph" else 0
                
                features.append(feature_vector)
                labels.append(label)
            
            return {
                'features': features,
                'labels': labels
            }
            
        except Exception as e:
            logger.error(f"Training data preparation error: {e}")
            return {'features': [], 'labels': []}
    
    def _train_model(self, model, training_data: Dict, model_type: ModelType):
        """Train a specific model with new data"""
        try:
            features = training_data['features']
            labels = training_data['labels']
            
            if not features or not labels:
                return
            
            start_time = time.time()
            
            if ML_AVAILABLE and hasattr(model, 'fit'):
                # Scale features for real ML models
                if self.scaler is not None:
                    features = self.scaler.fit_transform(features)
                
                # Split data for validation
                X_train, X_test, y_train, y_test = train_test_split(
                    features, labels, test_size=0.2, random_state=42
                )
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, features, labels, cv=5)
                
            else:
                # Fallback model training (mock)
                model.fit(features, labels)
                accuracy = 0.75 + random.uniform(-0.1, 0.1)  # Mock accuracy
                cv_scores = [accuracy] * 5
            
            training_time = time.time() - start_time
            
            # Store model performance
            self._store_model_performance(
                model_type, accuracy, cv_scores, training_time, len(features)
            )
            
            logger.info(f"Model {model_type.value} trained: accuracy={accuracy:.3f}, time={training_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Model training error: {e}")
    
    def _store_model_performance(self, model_type: ModelType, accuracy: float, 
                                cv_scores: List[float], training_time: float, 
                                sample_count: int):
        """Store model performance metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO model_performance 
                    (model_id, model_type, accuracy, precision, recall, f1_score,
                     training_time, last_updated, training_samples)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(uuid.uuid4()),
                    model_type.value,
                    accuracy,
                    statistics.mean(cv_scores),  # Use as precision proxy
                    accuracy * 0.95,  # Mock recall
                    accuracy * 0.97,  # Mock F1 score
                    training_time,
                    datetime.now().isoformat(),
                    sample_count
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Model performance storage error: {e}")

class ABTestingFramework:
    """A/B testing framework for decision optimization validation"""
    
    def __init__(self, db_path: str = "decision_optimization.db"):
        self.db_path = db_path
        self.active_tests = {}
        self.test_results = {}
        
    def create_ab_test(self, test_config: ABTestConfiguration) -> bool:
        """Create a new A/B test"""
        try:
            # Validate configuration
            if not self._validate_test_config(test_config):
                return False
            
            # Store test configuration
            self._store_test_config(test_config)
            
            # Activate test
            self.active_tests[test_config.test_id] = test_config
            
            logger.info(f"A/B test created: {test_config.test_id}")
            return True
            
        except Exception as e:
            logger.error(f"A/B test creation error: {e}")
            return False
    
    def _validate_test_config(self, config: ABTestConfiguration) -> bool:
        """Validate A/B test configuration"""
        try:
            # Check traffic split
            if not 0.0 <= config.traffic_split <= 1.0:
                logger.error(f"Invalid traffic split: {config.traffic_split}")
                return False
            
            # Check time range
            if config.end_time <= config.start_time:
                logger.error("End time must be after start time")
                return False
            
            # Check sample size
            if config.min_sample_size < 10:
                logger.error("Minimum sample size too small")
                return False
            
            # Check significance level
            if not 0.0 < config.significance_level < 1.0:
                logger.error(f"Invalid significance level: {config.significance_level}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Test config validation error: {e}")
            return False
    
    def _store_test_config(self, config: ABTestConfiguration):
        """Store A/B test configuration in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create table if not exists
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ab_test_configs (
                        test_id TEXT PRIMARY KEY,
                        test_name TEXT,
                        control_strategy TEXT,
                        treatment_strategy TEXT,
                        traffic_split REAL,
                        start_time TEXT,
                        end_time TEXT,
                        min_sample_size INTEGER,
                        significance_level REAL,
                        power REAL,
                        is_active BOOLEAN
                    )
                """)
                
                cursor.execute("""
                    INSERT OR REPLACE INTO ab_test_configs 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    config.test_id,
                    config.test_name,
                    config.control_strategy.value,
                    config.treatment_strategy.value,
                    config.traffic_split,
                    config.start_time.isoformat(),
                    config.end_time.isoformat(),
                    config.min_sample_size,
                    config.significance_level,
                    config.power,
                    config.is_active
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Test config storage error: {e}")
    
    def assign_test_group(self, decision_id: str) -> Optional[str]:
        """Assign decision to A/B test group"""
        try:
            for test_id, config in self.active_tests.items():
                if not config.is_active:
                    continue
                    
                # Check if test is within time range
                now = datetime.now()
                if not (config.start_time <= now <= config.end_time):
                    continue
                
                # Random assignment based on traffic split
                if random.random() < config.traffic_split:
                    return f"{test_id}_treatment"
                else:
                    return f"{test_id}_control"
            
            return None
            
        except Exception as e:
            logger.error(f"Test group assignment error: {e}")
            return None
    
    def record_test_result(self, test_id: str, decision_id: str, 
                          strategy: str, performance_metric: float):
        """Record A/B test result"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO ab_test_results 
                    (test_id, decision_id, strategy, performance_metric, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    test_id,
                    decision_id,
                    strategy,
                    performance_metric,
                    datetime.now().isoformat()
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Test result recording error: {e}")
    
    def analyze_test_results(self, test_id: str) -> Dict:
        """Analyze A/B test results with statistical significance"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT strategy, performance_metric 
                    FROM ab_test_results 
                    WHERE test_id = ?
                """, (test_id,))
                
                results = cursor.fetchall()
            
            if not results:
                return {"error": "No results found"}
            
            # Separate control and treatment groups
            control_metrics = [metric for strategy, metric in results if 'control' in strategy]
            treatment_metrics = [metric for strategy, metric in results if 'treatment' in strategy]
            
            if not control_metrics or not treatment_metrics:
                return {"error": "Insufficient data for both groups"}
            
            # Statistical analysis
            analysis = self._perform_statistical_analysis(control_metrics, treatment_metrics)
            
            # Add sample sizes
            analysis['control_sample_size'] = len(control_metrics)
            analysis['treatment_sample_size'] = len(treatment_metrics)
            analysis['total_sample_size'] = len(results)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Test result analysis error: {e}")
            return {"error": str(e)}
    
    def _perform_statistical_analysis(self, control: List[float], 
                                    treatment: List[float]) -> Dict:
        """Perform statistical analysis on A/B test results"""
        try:
            # Basic statistics
            control_mean = statistics.mean(control)
            treatment_mean = statistics.mean(treatment)
            control_std = statistics.stdev(control) if len(control) > 1 else 0
            treatment_std = statistics.stdev(treatment) if len(treatment) > 1 else 0
            
            # Effect size (Cohen's d)
            pooled_std = math.sqrt(((len(control) - 1) * control_std**2 + 
                                   (len(treatment) - 1) * treatment_std**2) / 
                                   (len(control) + len(treatment) - 2))
            
            effect_size = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
            
            # T-test for statistical significance
            if len(control) > 1 and len(treatment) > 1:
                t_stat, p_value = stats.ttest_ind(treatment, control)
            else:
                t_stat, p_value = 0, 1.0
            
            # Confidence interval for difference in means
            se = math.sqrt(control_std**2/len(control) + treatment_std**2/len(treatment))
            ci_95 = stats.t.interval(0.95, len(control) + len(treatment) - 2, 
                                   loc=treatment_mean - control_mean, scale=se)
            
            return {
                'control_mean': control_mean,
                'treatment_mean': treatment_mean,
                'control_std': control_std,
                'treatment_std': treatment_std,
                'effect_size': effect_size,
                'p_value': p_value,
                't_statistic': t_stat,
                'confidence_interval_95': ci_95,
                'is_significant': p_value < 0.05,
                'improvement_percentage': ((treatment_mean - control_mean) / control_mean * 100) if control_mean > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Statistical analysis error: {e}")
            return {
                'control_mean': statistics.mean(control),
                'treatment_mean': statistics.mean(treatment),
                'error': str(e)
            }

class PerformanceFeedbackSystem:
    """System for collecting and processing performance feedback"""
    
    def __init__(self, db_path: str = "decision_optimization.db"):
        self.db_path = db_path
        self.feedback_buffer = []
        self.feedback_processors = {}
        self.feedback_lock = threading.Lock()
        
    def collect_feedback(self, feedback: PerformanceFeedback):
        """Collect performance feedback"""
        try:
            with self.feedback_lock:
                self.feedback_buffer.append(feedback)
                
                # Store in database
                self._store_feedback(feedback)
                
                # Trigger processing if buffer is full
                if len(self.feedback_buffer) >= 10:
                    self._process_feedback_batch()
                    
        except Exception as e:
            logger.error(f"Feedback collection error: {e}")
    
    def _store_feedback(self, feedback: PerformanceFeedback):
        """Store feedback in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO performance_feedback 
                    (feedback_id, decision_id, feedback_type, feedback_value,
                     timestamp, source, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(uuid.uuid4()),
                    feedback.decision_id,
                    feedback.feedback_type,
                    feedback.feedback_value,
                    feedback.timestamp.isoformat(),
                    feedback.source,
                    feedback.confidence
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Feedback storage error: {e}")
    
    def _process_feedback_batch(self):
        """Process accumulated feedback"""
        try:
            if not self.feedback_buffer:
                return
            
            # Create copy of buffer and clear it
            feedback_batch = self.feedback_buffer.copy()
            self.feedback_buffer.clear()
            
            # Process feedback asynchronously
            threading.Thread(
                target=self._async_feedback_processing,
                args=(feedback_batch,),
                daemon=True
            ).start()
            
        except Exception as e:
            logger.error(f"Feedback batch processing error: {e}")
    
    def _async_feedback_processing(self, feedback_batch: List[PerformanceFeedback]):
        """Asynchronous feedback processing"""
        try:
            # Aggregate feedback by decision
            decision_feedback = {}
            for feedback in feedback_batch:
                if feedback.decision_id not in decision_feedback:
                    decision_feedback[feedback.decision_id] = []
                decision_feedback[feedback.decision_id].append(feedback)
            
            # Process each decision's feedback
            for decision_id, feedbacks in decision_feedback.items():
                self._process_decision_feedback(decision_id, feedbacks)
                
        except Exception as e:
            logger.error(f"Async feedback processing error: {e}")
    
    def _process_decision_feedback(self, decision_id: str, feedbacks: List[PerformanceFeedback]):
        """Process feedback for a specific decision"""
        try:
            # Calculate aggregated feedback metrics
            performance_scores = [fb.feedback_value for fb in feedbacks if fb.feedback_type == 'performance']
            satisfaction_scores = [fb.feedback_value for fb in feedbacks if fb.feedback_type == 'satisfaction']
            
            aggregated_performance = statistics.mean(performance_scores) if performance_scores else None
            aggregated_satisfaction = statistics.mean(satisfaction_scores) if satisfaction_scores else None
            
            # Update decision record with feedback
            self._update_decision_with_feedback(decision_id, aggregated_performance, aggregated_satisfaction)
            
        except Exception as e:
            logger.error(f"Decision feedback processing error: {e}")
    
    def _update_decision_with_feedback(self, decision_id: str, 
                                     performance: Optional[float], 
                                     satisfaction: Optional[float]):
        """Update decision record with aggregated feedback"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                update_fields = []
                update_values = []
                
                if performance is not None:
                    update_fields.append("actual_performance = ?")
                    update_values.append(performance)
                
                if satisfaction is not None:
                    update_fields.append("user_satisfaction = ?")
                    update_values.append(satisfaction)
                
                if update_fields:
                    update_values.append(decision_id)
                    cursor.execute(f"""
                        UPDATE decision_records 
                        SET {', '.join(update_fields)}
                        WHERE decision_id = ?
                    """, update_values)
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Decision update error: {e}")
    
    def get_feedback_summary(self, time_window_hours: int = 24) -> Dict:
        """Get feedback summary for a time window"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT feedback_type, AVG(feedback_value), COUNT(*)
                    FROM performance_feedback 
                    WHERE timestamp > ?
                    GROUP BY feedback_type
                """, (cutoff_time.isoformat(),))
                
                results = cursor.fetchall()
            
            summary = {}
            for feedback_type, avg_value, count in results:
                summary[feedback_type] = {
                    'average_value': avg_value,
                    'sample_count': count
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Feedback summary error: {e}")
            return {}

class DecisionOptimizationOrchestrator:
    """Main orchestrator for decision optimization system"""
    
    def __init__(self, db_path: str = "decision_optimization.db"):
        self.db_path = db_path
        self.learning_engine = DecisionLearningEngine(db_path)
        self.ab_testing = ABTestingFramework(db_path)
        self.feedback_system = PerformanceFeedbackSystem(db_path)
        self.optimization_metrics = {}
        self.is_running = False
        self.optimization_thread = None
        
    def start_optimization(self):
        """Start the decision optimization system"""
        try:
            if self.is_running:
                logger.warning("Optimization system already running")
                return
            
            self.is_running = True
            
            # Start background optimization thread
            self.optimization_thread = threading.Thread(
                target=self._optimization_loop,
                daemon=True
            )
            self.optimization_thread.start()
            
            logger.info("Decision optimization system started")
            
        except Exception as e:
            logger.error(f"Optimization start error: {e}")
            self.is_running = False
    
    def stop_optimization(self):
        """Stop the decision optimization system"""
        try:
            self.is_running = False
            
            if self.optimization_thread and self.optimization_thread.is_alive():
                self.optimization_thread.join(timeout=5.0)
            
            logger.info("Decision optimization system stopped")
            
        except Exception as e:
            logger.error(f"Optimization stop error: {e}")
    
    def _optimization_loop(self):
        """Main optimization loop"""
        try:
            while self.is_running:
                try:
                    # Collect and process feedback
                    self._collect_recent_feedback()
                    
                    # Update models with new data
                    self._update_models()
                    
                    # Analyze A/B test results
                    self._analyze_active_tests()
                    
                    # Calculate optimization metrics
                    self._calculate_optimization_metrics()
                    
                    # Sleep before next iteration
                    time.sleep(300)  # 5 minutes
                    
                except Exception as e:
                    logger.error(f"Optimization loop error: {e}")
                    time.sleep(60)  # Shorter sleep on error
                    
        except Exception as e:
            logger.error(f"Optimization loop fatal error: {e}")
    
    def _collect_recent_feedback(self):
        """Collect recent feedback for processing"""
        try:
            # Get recent decisions without feedback
            cutoff_time = datetime.now() - timedelta(hours=1)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT decision_id, framework_selected, confidence_score
                    FROM decision_records 
                    WHERE timestamp > ? AND actual_performance IS NULL
                """, (cutoff_time.isoformat(),))
                
                recent_decisions = cursor.fetchall()
            
            # Generate simulated feedback for demonstration
            for decision_id, framework, confidence in recent_decisions:
                # Simulate performance feedback based on framework and confidence
                performance_score = confidence * (0.8 + random.uniform(0, 0.4))
                satisfaction_score = performance_score * (0.9 + random.uniform(0, 0.2))
                
                # Create feedback
                performance_feedback = PerformanceFeedback(
                    decision_id=decision_id,
                    feedback_type="performance",
                    feedback_value=performance_score,
                    timestamp=datetime.now(),
                    source="system_simulation",
                    confidence=0.8
                )
                
                satisfaction_feedback = PerformanceFeedback(
                    decision_id=decision_id,
                    feedback_type="satisfaction",
                    feedback_value=satisfaction_score,
                    timestamp=datetime.now(),
                    source="system_simulation",
                    confidence=0.7
                )
                
                self.feedback_system.collect_feedback(performance_feedback)
                self.feedback_system.collect_feedback(satisfaction_feedback)
                
        except Exception as e:
            logger.error(f"Recent feedback collection error: {e}")
    
    def _update_models(self):
        """Update ML models with recent feedback"""
        try:
            # Get recent feedback
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT decision_id, feedback_type, feedback_value, timestamp, source, confidence
                    FROM performance_feedback 
                    WHERE timestamp > ?
                """, (cutoff_time.isoformat(),))
                
                feedback_data = cursor.fetchall()
            
            # Convert to PerformanceFeedback objects
            feedback_objects = []
            for row in feedback_data:
                decision_id, fb_type, fb_value, timestamp_str, source, confidence = row
                feedback = PerformanceFeedback(
                    decision_id=decision_id,
                    feedback_type=fb_type,
                    feedback_value=fb_value,
                    timestamp=datetime.fromisoformat(timestamp_str),
                    source=source,
                    confidence=confidence
                )
                feedback_objects.append(feedback)
            
            # Update models
            if feedback_objects:
                self.learning_engine.update_model_with_feedback(feedback_objects)
                
        except Exception as e:
            logger.error(f"Model update error: {e}")
    
    def _analyze_active_tests(self):
        """Analyze results of active A/B tests"""
        try:
            for test_id in list(self.ab_testing.active_tests.keys()):
                results = self.ab_testing.analyze_test_results(test_id)
                
                if 'error' not in results:
                    # Store analysis results
                    self.optimization_metrics[f"ab_test_{test_id}"] = results
                    
                    # Check if test should be concluded
                    if (results.get('total_sample_size', 0) >= 100 and 
                        results.get('is_significant', False)):
                        logger.info(f"A/B test {test_id} shows significant results")
                        
        except Exception as e:
            logger.error(f"A/B test analysis error: {e}")
    
    def _calculate_optimization_metrics(self):
        """Calculate optimization performance metrics"""
        try:
            # Calculate decision accuracy improvement
            accuracy_improvement = self._calculate_accuracy_improvement()
            
            # Calculate suboptimal decision reduction
            suboptimal_reduction = self._calculate_suboptimal_reduction()
            
            # Calculate feedback loop effectiveness
            feedback_effectiveness = self._calculate_feedback_effectiveness()
            
            # Store metrics
            self.optimization_metrics.update({
                'accuracy_improvement_percent': accuracy_improvement,
                'suboptimal_reduction_percent': suboptimal_reduction,
                'feedback_effectiveness_score': feedback_effectiveness,
                'last_updated': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Optimization metrics calculation error: {e}")
    
    def _calculate_accuracy_improvement(self) -> float:
        """Calculate decision accuracy improvement over time"""
        try:
            # Get decisions from last 30 days
            cutoff_time = datetime.now() - timedelta(days=30)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT timestamp, actual_performance, confidence_score
                    FROM decision_records 
                    WHERE timestamp > ? AND actual_performance IS NOT NULL
                    ORDER BY timestamp
                """, (cutoff_time.isoformat(),))
                
                decisions = cursor.fetchall()
            
            if len(decisions) < 10:
                return 0.0
            
            # Split into early and recent periods
            split_point = len(decisions) // 2
            early_decisions = decisions[:split_point]
            recent_decisions = decisions[split_point:]
            
            # Calculate average accuracy for each period
            early_accuracy = statistics.mean([perf for _, perf, _ in early_decisions])
            recent_accuracy = statistics.mean([perf for _, perf, _ in recent_decisions])
            
            # Calculate improvement percentage
            if early_accuracy > 0:
                improvement = ((recent_accuracy - early_accuracy) / early_accuracy) * 100
                return max(0, improvement)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Accuracy improvement calculation error: {e}")
            return 0.0
    
    def _calculate_suboptimal_reduction(self) -> float:
        """Calculate reduction in suboptimal decisions"""
        try:
            # Define suboptimal as performance < 0.6
            threshold = 0.6
            
            # Get decisions from last 30 days
            cutoff_time = datetime.now() - timedelta(days=30)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT timestamp, actual_performance
                    FROM decision_records 
                    WHERE timestamp > ? AND actual_performance IS NOT NULL
                    ORDER BY timestamp
                """, (cutoff_time.isoformat(),))
                
                decisions = cursor.fetchall()
            
            if len(decisions) < 10:
                return 0.0
            
            # Split into early and recent periods
            split_point = len(decisions) // 2
            early_decisions = decisions[:split_point]
            recent_decisions = decisions[split_point:]
            
            # Calculate suboptimal decision rates
            early_suboptimal_rate = sum(1 for _, perf in early_decisions if perf < threshold) / len(early_decisions)
            recent_suboptimal_rate = sum(1 for _, perf in recent_decisions if perf < threshold) / len(recent_decisions)
            
            # Calculate reduction percentage
            if early_suboptimal_rate > 0:
                reduction = ((early_suboptimal_rate - recent_suboptimal_rate) / early_suboptimal_rate) * 100
                return max(0, reduction)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Suboptimal reduction calculation error: {e}")
            return 0.0
    
    def _calculate_feedback_effectiveness(self) -> float:
        """Calculate feedback loop effectiveness"""
        try:
            # Get feedback processing times
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT dr.decision_id, dr.timestamp as decision_time, 
                           pf.timestamp as feedback_time
                    FROM decision_records dr
                    JOIN performance_feedback pf ON dr.decision_id = pf.decision_id
                    WHERE dr.timestamp > ?
                """, (cutoff_time.isoformat(),))
                
                feedback_times = cursor.fetchall()
            
            if not feedback_times:
                return 0.5  # Default score
            
            # Calculate average feedback delay
            total_delay = 0
            count = 0
            
            for decision_id, decision_time_str, feedback_time_str in feedback_times:
                decision_time = datetime.fromisoformat(decision_time_str)
                feedback_time = datetime.fromisoformat(feedback_time_str)
                delay_hours = (feedback_time - decision_time).total_seconds() / 3600
                total_delay += delay_hours
                count += 1
            
            if count == 0:
                return 0.5
            
            average_delay = total_delay / count
            
            # Score based on delay (24 hours = 1.0, 1 hour = 0.95, immediate = 1.0)
            if average_delay <= 1:
                return 1.0
            elif average_delay <= 24:
                return 1.0 - (average_delay - 1) * 0.02  # Linear decrease
            else:
                return max(0.5, 1.0 - average_delay * 0.01)
                
        except Exception as e:
            logger.error(f"Feedback effectiveness calculation error: {e}")
            return 0.5
    
    def get_optimization_status(self) -> Dict:
        """Get current optimization system status"""
        try:
            status = {
                'is_running': self.is_running,
                'learning_engine_status': 'active' if self.learning_engine else 'inactive',
                'ab_testing_status': 'active' if self.ab_testing else 'inactive',
                'feedback_system_status': 'active' if self.feedback_system else 'inactive',
                'active_ab_tests': len(self.ab_testing.active_tests),
                'optimization_metrics': self.optimization_metrics.copy()
            }
            
            # Add model performance summary
            if hasattr(self.learning_engine, 'model_performance'):
                status['model_performance'] = self.learning_engine.model_performance.copy()
            
            return status
            
        except Exception as e:
            logger.error(f"Status retrieval error: {e}")
            return {'error': str(e)}
    
    def create_sample_ab_test(self) -> str:
        """Create a sample A/B test for demonstration"""
        try:
            test_id = f"decision_optimization_test_{int(time.time())}"
            
            test_config = ABTestConfiguration(
                test_id=test_id,
                test_name="Framework Decision Optimization Test",
                control_strategy=DecisionStrategy.CURRENT_MODEL,
                treatment_strategy=DecisionStrategy.IMPROVED_MODEL,
                traffic_split=0.5,
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(days=7),
                min_sample_size=50,
                significance_level=0.05,
                power=0.8
            )
            
            success = self.ab_testing.create_ab_test(test_config)
            return test_id if success else None
            
        except Exception as e:
            logger.error(f"Sample A/B test creation error: {e}")
            return None

# Demo and Testing Functions
def create_demo_optimization_system():
    """Create a demo decision optimization system"""
    try:
        # Initialize system
        orchestrator = DecisionOptimizationOrchestrator("demo_decision_optimization.db")
        
        # Start optimization
        orchestrator.start_optimization()
        
        # Create sample A/B test
        test_id = orchestrator.create_sample_ab_test()
        
        # Generate sample decisions and feedback
        for i in range(20):
            decision_id = f"demo_decision_{i}"
            
            # Create sample task data
            task_data = {
                'complexity_score': random.uniform(0.3, 0.9),
                'resource_requirements': random.uniform(0.2, 0.8),
                'agent_count': random.randint(1, 5),
                'workflow_complexity': random.uniform(0.1, 0.9),
                'state_management_complexity': random.uniform(0.2, 0.8),
                'memory_requirements': random.uniform(0.1, 0.7),
                'performance_priority': random.uniform(0.5, 1.0),
                'quality_priority': random.uniform(0.5, 1.0),
                'user_tier': random.randint(1, 3),
                'historical_performance': random.uniform(0.4, 0.9)
            }
            
            # Get framework prediction
            framework, confidence = orchestrator.learning_engine.predict_optimal_framework(task_data)
            
            # Assign to A/B test group
            ab_group = orchestrator.ab_testing.assign_test_group(decision_id)
            
            # Create decision record
            decision = DecisionRecord(
                decision_id=decision_id,
                timestamp=datetime.now() - timedelta(minutes=random.randint(1, 1440)),
                task_complexity=task_data['complexity_score'],
                framework_selected=framework,
                strategy_used="ml_prediction",
                confidence_score=confidence,
                ab_test_group=ab_group
            )
            
            # Record decision
            orchestrator.learning_engine.record_decision(decision)
            
            # Generate feedback
            performance_score = confidence * (0.7 + random.uniform(0, 0.3))
            satisfaction_score = performance_score * (0.8 + random.uniform(0, 0.4))
            
            performance_feedback = PerformanceFeedback(
                decision_id=decision_id,
                feedback_type="performance",
                feedback_value=performance_score,
                timestamp=datetime.now(),
                source="demo_system",
                confidence=0.8
            )
            
            satisfaction_feedback = PerformanceFeedback(
                decision_id=decision_id,
                feedback_type="satisfaction",
                feedback_value=satisfaction_score,
                timestamp=datetime.now(),
                source="demo_system",
                confidence=0.7
            )
            
            orchestrator.feedback_system.collect_feedback(performance_feedback)
            orchestrator.feedback_system.collect_feedback(satisfaction_feedback)
            
            # Record A/B test result if applicable
            if ab_group and test_id:
                strategy = "treatment" if "treatment" in ab_group else "control"
                orchestrator.ab_testing.record_test_result(test_id, decision_id, strategy, performance_score)
        
        # Wait for processing
        time.sleep(2)
        
        return orchestrator
        
    except Exception as e:
        logger.error(f"Demo system creation error: {e}")
        return None

if __name__ == "__main__":
    print("🤖 LangGraph Decision Optimization Sandbox")
    print("=" * 60)
    
    try:
        # Create and run demo
        demo_system = create_demo_optimization_system()
        
        if demo_system:
            print("✅ Demo system created successfully")
            
            # Display status
            status = demo_system.get_optimization_status()
            print(f"\n📊 System Status:")
            print(f"Running: {status['is_running']}")
            print(f"Active A/B Tests: {status['active_ab_tests']}")
            
            if 'optimization_metrics' in status:
                metrics = status['optimization_metrics']
                print(f"\n📈 Optimization Metrics:")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        print(f"{key}: {value:.2f}")
                    else:
                        print(f"{key}: {value}")
            
            # Run for demo period
            print("\n⏱️  Running optimization for 10 seconds...")
            time.sleep(10)
            
            # Final status
            final_status = demo_system.get_optimization_status()
            print(f"\n🎯 Final Status: {final_status['is_running']}")
            
            # Stop system
            demo_system.stop_optimization()
            print("✅ Demo completed successfully")
            
        else:
            print("❌ Failed to create demo system")
            
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo error: {e}")