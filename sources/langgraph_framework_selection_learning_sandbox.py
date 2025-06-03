#!/usr/bin/env python3
"""
LangGraph Framework Selection Learning Sandbox Implementation
Implements adaptive framework selection algorithms with performance-based learning and context-awareness.

* Purpose: Adaptive framework selection with pattern recognition and automated parameter tuning
* Issues & Complexity Summary: Complex learning algorithms with context awareness and convergence requirements
* Key Complexity Drivers:
  - Logic Scope (Est. LoC): ~2500
  - Core Algorithm Complexity: High (adaptive learning, pattern recognition, parameter tuning)
  - Dependencies: 10 (sklearn, numpy, scipy, sqlite3, asyncio, threading, json, datetime, collections, statistics)
  - State Management Complexity: High (learning states, patterns, parameters, contexts)
  - Novelty/Uncertainty Factor: Medium-High (adaptive learning with convergence guarantees)
* AI Pre-Task Self-Assessment (Est. Solution Difficulty %): 88%
* Problem Estimate (Inherent Problem Difficulty %): 85%
* Initial Code Complexity Estimate %: 88%
* Justification for Estimates: Complex adaptive learning with multiple optimization objectives and convergence requirements
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
import statistics
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import uuid
import random

# Machine Learning and Statistical Libraries
try:
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    from sklearn.model_selection import GridSearchCV
    import scipy.stats as stats
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    np = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LearningStrategy(Enum):
    """Learning strategy types"""
    PERFORMANCE_BASED = "performance_based"
    CONTEXT_AWARE = "context_aware"
    PATTERN_RECOGNITION = "pattern_recognition"
    HYBRID_ADAPTIVE = "hybrid_adaptive"
    REINFORCEMENT_LEARNING = "reinforcement_learning"

class ContextType(Enum):
    """Context types for awareness"""
    TASK_COMPLEXITY = "task_complexity"
    USER_TIER = "user_tier"
    SYSTEM_LOAD = "system_load"
    TIME_OF_DAY = "time_of_day"
    HISTORICAL_PERFORMANCE = "historical_performance"
    RESOURCE_AVAILABILITY = "resource_availability"

class PatternType(Enum):
    """Pattern types for recognition"""
    TEMPORAL_PATTERN = "temporal_pattern"
    COMPLEXITY_PATTERN = "complexity_pattern"
    PERFORMANCE_PATTERN = "performance_pattern"
    CONTEXT_PATTERN = "context_pattern"
    USER_BEHAVIOR_PATTERN = "user_behavior_pattern"

@dataclass
class SelectionDecision:
    """Framework selection decision with learning context"""
    decision_id: str
    timestamp: datetime
    framework_selected: str
    task_features: Dict[str, float]
    context_features: Dict[str, float]
    confidence_score: float
    strategy_used: str
    performance_outcome: Optional[float] = None
    user_satisfaction: Optional[float] = None
    execution_time: Optional[float] = None
    success_indicator: Optional[bool] = None
    learning_source: Optional[str] = None

@dataclass
class LearningPattern:
    """Identified learning pattern"""
    pattern_id: str
    pattern_type: PatternType
    pattern_description: str
    conditions: Dict[str, Any]
    recommended_framework: str
    confidence: float
    success_rate: float
    sample_count: int
    last_updated: datetime
    effectiveness_score: float

@dataclass
class ContextualRule:
    """Context-aware selection rule"""
    rule_id: str
    rule_name: str
    context_conditions: Dict[str, Any]
    framework_preference: str
    weight: float
    accuracy: float
    usage_count: int
    creation_time: datetime
    last_success: datetime

@dataclass
class ParameterConfiguration:
    """Parameter configuration for tuning"""
    config_id: str
    parameter_name: str
    current_value: float
    optimal_range: Tuple[float, float]
    step_size: float
    accuracy_impact: float
    last_tuned: datetime
    tuning_history: List[Tuple[float, float]]  # (value, accuracy)

class AdaptiveSelectionAlgorithm:
    """Core adaptive algorithm for framework selection learning"""
    
    def __init__(self, db_path: str = "selection_learning.db"):
        self.db_path = db_path
        self.learning_history = deque(maxlen=1000)
        self.pattern_cache = {}
        self.context_weights = {}
        self.performance_baseline = 0.7
        self.learning_rate = 0.01
        self.adaptation_threshold = 0.05
        self.convergence_window = 100
        self.setup_database()
        self.initialize_parameters()
        
    def setup_database(self):
        """Initialize database for selection learning"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Selection decisions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS selection_decisions (
                        decision_id TEXT PRIMARY KEY,
                        timestamp TEXT,
                        framework_selected TEXT,
                        task_features TEXT,
                        context_features TEXT,
                        confidence_score REAL,
                        strategy_used TEXT,
                        performance_outcome REAL,
                        user_satisfaction REAL,
                        execution_time REAL,
                        success_indicator BOOLEAN,
                        learning_source TEXT
                    )
                """)
                
                # Learning patterns table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS learning_patterns (
                        pattern_id TEXT PRIMARY KEY,
                        pattern_type TEXT,
                        pattern_description TEXT,
                        conditions TEXT,
                        recommended_framework TEXT,
                        confidence REAL,
                        success_rate REAL,
                        sample_count INTEGER,
                        last_updated TEXT,
                        effectiveness_score REAL
                    )
                """)
                
                # Contextual rules table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS contextual_rules (
                        rule_id TEXT PRIMARY KEY,
                        rule_name TEXT,
                        context_conditions TEXT,
                        framework_preference TEXT,
                        weight REAL,
                        accuracy REAL,
                        usage_count INTEGER,
                        creation_time TEXT,
                        last_success TEXT
                    )
                """)
                
                # Parameter configurations table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS parameter_configurations (
                        config_id TEXT PRIMARY KEY,
                        parameter_name TEXT,
                        current_value REAL,
                        optimal_range TEXT,
                        step_size REAL,
                        accuracy_impact REAL,
                        last_tuned TEXT,
                        tuning_history TEXT
                    )
                """)
                
                # Learning metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS learning_metrics (
                        metric_id TEXT PRIMARY KEY,
                        timestamp TEXT,
                        accuracy_30_day REAL,
                        error_reduction_percent REAL,
                        pattern_count INTEGER,
                        convergence_rate REAL,
                        parameter_stability REAL
                    )
                """)
                
                conn.commit()
                logger.info("Selection learning database initialized successfully")
                
        except Exception as e:
            logger.error(f"Database setup error: {e}")
            raise
    
    def initialize_parameters(self):
        """Initialize learning parameters"""
        try:
            # Default parameter configurations
            default_params = [
                ("learning_rate", 0.01, (0.001, 0.1), 0.005),
                ("adaptation_threshold", 0.05, (0.01, 0.2), 0.01),
                ("context_weight_decay", 0.95, (0.8, 0.99), 0.02),
                ("pattern_confidence_threshold", 0.7, (0.5, 0.9), 0.05),
                ("convergence_patience", 20, (10, 50), 5)
            ]
            
            for param_name, default_value, optimal_range, step_size in default_params:
                self._initialize_parameter(param_name, default_value, optimal_range, step_size)
            
            # Initialize context weights
            self.context_weights = {
                ContextType.TASK_COMPLEXITY: 0.3,
                ContextType.USER_TIER: 0.2,
                ContextType.SYSTEM_LOAD: 0.15,
                ContextType.TIME_OF_DAY: 0.1,
                ContextType.HISTORICAL_PERFORMANCE: 0.15,
                ContextType.RESOURCE_AVAILABILITY: 0.1
            }
            
            logger.info("Learning parameters initialized successfully")
            
        except Exception as e:
            logger.error(f"Parameter initialization error: {e}")
    
    def _initialize_parameter(self, name: str, value: float, optimal_range: Tuple[float, float], step_size: float):
        """Initialize a single parameter configuration"""
        try:
            config = ParameterConfiguration(
                config_id=str(uuid.uuid4()),
                parameter_name=name,
                current_value=value,
                optimal_range=optimal_range,
                step_size=step_size,
                accuracy_impact=0.0,
                last_tuned=datetime.now(),
                tuning_history=[]
            )
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO parameter_configurations 
                    (config_id, parameter_name, current_value, optimal_range, step_size,
                     accuracy_impact, last_tuned, tuning_history)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    config.config_id,
                    config.parameter_name,
                    config.current_value,
                    json.dumps(config.optimal_range),
                    config.step_size,
                    config.accuracy_impact,
                    config.last_tuned.isoformat(),
                    json.dumps(config.tuning_history)
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Parameter initialization error for {name}: {e}")
    
    def select_framework_adaptive(self, task_features: Dict[str, float], 
                                 context_features: Dict[str, float]) -> Tuple[str, float, str]:
        """Adaptively select framework based on learned patterns"""
        try:
            # Calculate context-aware scores
            scores = self._calculate_adaptive_scores(task_features, context_features)
            
            # Apply learned patterns
            pattern_adjustments = self._apply_learned_patterns(task_features, context_features)
            
            # Combine scores with pattern adjustments
            final_scores = {}
            for framework in scores:
                final_scores[framework] = scores[framework] + pattern_adjustments.get(framework, 0)
            
            # Select best framework
            best_framework = max(final_scores, key=final_scores.get)
            confidence = final_scores[best_framework]
            strategy = "adaptive_learning"
            
            # Record decision for learning
            decision = SelectionDecision(
                decision_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                framework_selected=best_framework,
                task_features=task_features,
                context_features=context_features,
                confidence_score=confidence,
                strategy_used=strategy,
                learning_source="adaptive_algorithm"
            )
            
            self._record_decision(decision)
            
            return best_framework, confidence, strategy
            
        except Exception as e:
            logger.error(f"Adaptive selection error: {e}")
            return "langchain", 0.5, "fallback"
    
    def _calculate_adaptive_scores(self, task_features: Dict[str, float], 
                                  context_features: Dict[str, float]) -> Dict[str, float]:
        """Calculate adaptive scores based on features and context"""
        try:
            scores = {"langchain": 0.5, "langgraph": 0.5}
            
            # Task complexity influence
            complexity = task_features.get('complexity_score', 0.5)
            if complexity > 0.7:
                scores["langgraph"] += 0.2
            else:
                scores["langchain"] += 0.1
            
            # Context-aware adjustments
            for context_type, weight in self.context_weights.items():
                context_value = context_features.get(context_type.value, 0.5)
                
                if context_type == ContextType.SYSTEM_LOAD:
                    if context_value > 0.8:  # High load
                        scores["langchain"] += weight * 0.3
                    else:
                        scores["langgraph"] += weight * 0.2
                
                elif context_type == ContextType.USER_TIER:
                    if context_value > 0.7:  # Premium users
                        scores["langgraph"] += weight * 0.4
                    else:
                        scores["langchain"] += weight * 0.2
                
                elif context_type == ContextType.HISTORICAL_PERFORMANCE:
                    # Favor historically better performing framework
                    if context_value > 0.6:
                        scores["langgraph"] += weight * context_value
                    else:
                        scores["langchain"] += weight * (1 - context_value)
            
            # Normalize scores
            total = sum(scores.values())
            if total > 0:
                scores = {k: v / total for k, v in scores.items()}
            
            return scores
            
        except Exception as e:
            logger.error(f"Adaptive scoring error: {e}")
            return {"langchain": 0.5, "langgraph": 0.5}
    
    def _apply_learned_patterns(self, task_features: Dict[str, float], 
                               context_features: Dict[str, float]) -> Dict[str, float]:
        """Apply learned patterns to adjust framework scores"""
        try:
            adjustments = {"langchain": 0.0, "langgraph": 0.0}
            
            # Get relevant patterns
            patterns = self._get_relevant_patterns(task_features, context_features)
            
            for pattern in patterns:
                if pattern.effectiveness_score > 0.7:  # High effectiveness threshold
                    framework = pattern.recommended_framework
                    adjustment = pattern.confidence * pattern.effectiveness_score * 0.3
                    adjustments[framework] += adjustment
            
            return adjustments
            
        except Exception as e:
            logger.error(f"Pattern application error: {e}")
            return {"langchain": 0.0, "langgraph": 0.0}
    
    def _get_relevant_patterns(self, task_features: Dict[str, float], 
                              context_features: Dict[str, float]) -> List[LearningPattern]:
        """Get patterns relevant to current features"""
        try:
            patterns = []
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM learning_patterns 
                    WHERE effectiveness_score > 0.5
                    ORDER BY effectiveness_score DESC
                    LIMIT 10
                """)
                
                rows = cursor.fetchall()
                
                for row in rows:
                    pattern = LearningPattern(
                        pattern_id=row[0],
                        pattern_type=PatternType(row[1]),
                        pattern_description=row[2],
                        conditions=json.loads(row[3]),
                        recommended_framework=row[4],
                        confidence=row[5],
                        success_rate=row[6],
                        sample_count=row[7],
                        last_updated=datetime.fromisoformat(row[8]),
                        effectiveness_score=row[9]
                    )
                    
                    # Check if pattern conditions match current features
                    if self._pattern_matches_features(pattern, task_features, context_features):
                        patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern retrieval error: {e}")
            return []
    
    def _pattern_matches_features(self, pattern: LearningPattern, 
                                 task_features: Dict[str, float], 
                                 context_features: Dict[str, float]) -> bool:
        """Check if pattern conditions match current features"""
        try:
            conditions = pattern.conditions
            
            # Check task feature conditions
            for feature, value_range in conditions.get('task_features', {}).items():
                if feature in task_features:
                    feature_value = task_features[feature]
                    if not (value_range[0] <= feature_value <= value_range[1]):
                        return False
            
            # Check context feature conditions
            for feature, value_range in conditions.get('context_features', {}).items():
                if feature in context_features:
                    feature_value = context_features[feature]
                    if not (value_range[0] <= feature_value <= value_range[1]):
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Pattern matching error: {e}")
            return False
    
    def _record_decision(self, decision: SelectionDecision):
        """Record selection decision for learning"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO selection_decisions 
                    (decision_id, timestamp, framework_selected, task_features, context_features,
                     confidence_score, strategy_used, performance_outcome, user_satisfaction,
                     execution_time, success_indicator, learning_source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    decision.decision_id,
                    decision.timestamp.isoformat(),
                    decision.framework_selected,
                    json.dumps(decision.task_features),
                    json.dumps(decision.context_features),
                    decision.confidence_score,
                    decision.strategy_used,
                    decision.performance_outcome,
                    decision.user_satisfaction,
                    decision.execution_time,
                    decision.success_indicator,
                    decision.learning_source
                ))
                conn.commit()
                
            # Add to learning history
            self.learning_history.append(decision)
            
        except Exception as e:
            logger.error(f"Decision recording error: {e}")
    
    def update_decision_outcome(self, decision_id: str, performance_outcome: float, 
                               user_satisfaction: float, execution_time: float, 
                               success_indicator: bool):
        """Update decision with performance outcome"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE selection_decisions 
                    SET performance_outcome = ?, user_satisfaction = ?, 
                        execution_time = ?, success_indicator = ?
                    WHERE decision_id = ?
                """, (performance_outcome, user_satisfaction, execution_time, 
                      success_indicator, decision_id))
                conn.commit()
            
            # Trigger learning update
            self._trigger_learning_update(decision_id)
            
        except Exception as e:
            logger.error(f"Decision outcome update error: {e}")
    
    def _trigger_learning_update(self, decision_id: str):
        """Trigger learning process based on updated decision"""
        try:
            # Get the updated decision
            decision = self._get_decision(decision_id)
            if decision is None:
                return
            
            # Update pattern recognition
            self._update_pattern_recognition(decision)
            
            # Update context weights
            self._update_context_weights(decision)
            
            # Check for parameter tuning
            if len(self.learning_history) % 20 == 0:  # Every 20 decisions
                self._trigger_parameter_tuning()
            
        except Exception as e:
            logger.error(f"Learning update trigger error: {e}")
    
    def _get_decision(self, decision_id: str) -> Optional[SelectionDecision]:
        """Get decision by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM selection_decisions WHERE decision_id = ?", (decision_id,))
                row = cursor.fetchone()
                
                if row:
                    return SelectionDecision(
                        decision_id=row[0],
                        timestamp=datetime.fromisoformat(row[1]),
                        framework_selected=row[2],
                        task_features=json.loads(row[3]),
                        context_features=json.loads(row[4]),
                        confidence_score=row[5],
                        strategy_used=row[6],
                        performance_outcome=row[7],
                        user_satisfaction=row[8],
                        execution_time=row[9],
                        success_indicator=row[10],
                        learning_source=row[11]
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Decision retrieval error: {e}")
            return None

class PerformanceBasedLearning:
    """Performance-based learning component"""
    
    def __init__(self, db_path: str = "selection_learning.db"):
        self.db_path = db_path
        self.performance_history = defaultdict(list)
        self.baseline_accuracy = 0.7
        self.improvement_threshold = 0.15  # 15% improvement target
        
    def analyze_performance_trends(self, time_window_days: int = 30) -> Dict[str, float]:
        """Analyze performance trends over time window"""
        try:
            cutoff_time = datetime.now() - timedelta(days=time_window_days)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT framework_selected, performance_outcome, timestamp
                    FROM selection_decisions 
                    WHERE timestamp > ? AND performance_outcome IS NOT NULL
                    ORDER BY timestamp
                """, (cutoff_time.isoformat(),))
                
                results = cursor.fetchall()
            
            # Group by framework and calculate trends
            framework_performance = defaultdict(list)
            for framework, performance, timestamp_str in results:
                timestamp = datetime.fromisoformat(timestamp_str)
                framework_performance[framework].append((timestamp, performance))
            
            # Calculate improvement metrics
            trends = {}
            for framework, performance_data in framework_performance.items():
                if len(performance_data) >= 10:  # Minimum sample size
                    improvement = self._calculate_improvement_rate(performance_data)
                    trends[framework] = improvement
            
            return trends
            
        except Exception as e:
            logger.error(f"Performance trend analysis error: {e}")
            return {}
    
    def _calculate_improvement_rate(self, performance_data: List[Tuple[datetime, float]]) -> float:
        """Calculate improvement rate from performance data"""
        try:
            if len(performance_data) < 2:
                return 0.0
            
            # Sort by timestamp
            performance_data.sort(key=lambda x: x[0])
            
            # Split into early and recent periods
            split_point = len(performance_data) // 2
            early_performance = [p[1] for p in performance_data[:split_point]]
            recent_performance = [p[1] for p in performance_data[split_point:]]
            
            early_avg = statistics.mean(early_performance)
            recent_avg = statistics.mean(recent_performance)
            
            if early_avg > 0:
                improvement_rate = (recent_avg - early_avg) / early_avg
                return improvement_rate
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Improvement rate calculation error: {e}")
            return 0.0
    
    def update_performance_model(self, decisions: List[SelectionDecision]):
        """Update performance model based on recent decisions"""
        try:
            if not decisions:
                return
            
            # Group decisions by framework
            framework_data = defaultdict(list)
            for decision in decisions:
                if decision.performance_outcome is not None:
                    framework_data[decision.framework_selected].append(decision)
            
            # Update performance baselines
            for framework, framework_decisions in framework_data.items():
                performances = [d.performance_outcome for d in framework_decisions]
                avg_performance = statistics.mean(performances)
                
                # Update framework-specific baseline
                self.performance_history[framework].extend(performances)
                
                # Keep only recent history (last 200 decisions)
                if len(self.performance_history[framework]) > 200:
                    self.performance_history[framework] = self.performance_history[framework][-200:]
            
        except Exception as e:
            logger.error(f"Performance model update error: {e}")

class ContextAwareLearning:
    """Context-aware learning component"""
    
    def __init__(self, db_path: str = "selection_learning.db"):
        self.db_path = db_path
        self.context_rules = {}
        self.error_reduction_target = 0.25  # 25% error reduction target
        
    def learn_contextual_patterns(self, decisions: List[SelectionDecision]) -> List[ContextualRule]:
        """Learn contextual patterns from decisions"""
        try:
            # Group decisions by context similarity
            context_groups = self._group_by_context_similarity(decisions)
            
            rules = []
            for group_id, group_decisions in context_groups.items():
                if len(group_decisions) >= 10:  # Minimum sample size
                    rule = self._extract_contextual_rule(group_decisions)
                    if rule and rule.accuracy > 0.7:  # Quality threshold
                        rules.append(rule)
                        self._store_contextual_rule(rule)
            
            return rules
            
        except Exception as e:
            logger.error(f"Contextual pattern learning error: {e}")
            return []
    
    def _group_by_context_similarity(self, decisions: List[SelectionDecision]) -> Dict[str, List[SelectionDecision]]:
        """Group decisions by context similarity"""
        try:
            # Use clustering if ML available, otherwise simple grouping
            if ML_AVAILABLE and len(decisions) >= 20:
                return self._cluster_by_context(decisions)
            else:
                return self._simple_context_grouping(decisions)
                
        except Exception as e:
            logger.error(f"Context grouping error: {e}")
            return {"default": decisions}
    
    def _cluster_by_context(self, decisions: List[SelectionDecision]) -> Dict[str, List[SelectionDecision]]:
        """Cluster decisions by context using ML"""
        try:
            # Extract context feature vectors
            feature_vectors = []
            for decision in decisions:
                vector = []
                for context_type in ContextType:
                    value = decision.context_features.get(context_type.value, 0.5)
                    vector.append(value)
                feature_vectors.append(vector)
            
            # Perform clustering
            n_clusters = min(5, len(decisions) // 10)  # Dynamic cluster count
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(feature_vectors)
            
            # Group by cluster
            groups = defaultdict(list)
            for i, decision in enumerate(decisions):
                cluster_id = f"cluster_{cluster_labels[i]}"
                groups[cluster_id].append(decision)
            
            return dict(groups)
            
        except Exception as e:
            logger.error(f"Context clustering error: {e}")
            return self._simple_context_grouping(decisions)
    
    def _simple_context_grouping(self, decisions: List[SelectionDecision]) -> Dict[str, List[SelectionDecision]]:
        """Simple context grouping fallback"""
        try:
            groups = defaultdict(list)
            
            for decision in decisions:
                # Group by primary context characteristics
                complexity = decision.task_features.get('complexity_score', 0.5)
                user_tier = decision.context_features.get('user_tier', 0.5)
                system_load = decision.context_features.get('system_load', 0.5)
                
                # Create group key based on ranges
                complexity_range = "high" if complexity > 0.7 else "medium" if complexity > 0.4 else "low"
                tier_range = "premium" if user_tier > 0.7 else "standard"
                load_range = "high" if system_load > 0.7 else "normal"
                
                group_key = f"{complexity_range}_{tier_range}_{load_range}"
                groups[group_key].append(decision)
            
            return dict(groups)
            
        except Exception as e:
            logger.error(f"Simple context grouping error: {e}")
            return {"default": decisions}
    
    def _extract_contextual_rule(self, decisions: List[SelectionDecision]) -> Optional[ContextualRule]:
        """Extract contextual rule from grouped decisions"""
        try:
            if not decisions:
                return None
            
            # Analyze framework preferences
            framework_counts = defaultdict(int)
            successful_decisions = []
            
            for decision in decisions:
                framework_counts[decision.framework_selected] += 1
                if decision.success_indicator is True:
                    successful_decisions.append(decision)
            
            # Find preferred framework
            preferred_framework = max(framework_counts, key=framework_counts.get)
            
            # Calculate accuracy
            total_decisions = len(decisions)
            successful_preferred = sum(1 for d in successful_decisions if d.framework_selected == preferred_framework)
            accuracy = successful_preferred / total_decisions if total_decisions > 0 else 0
            
            # Extract context conditions
            context_conditions = self._extract_context_conditions(decisions)
            
            # Create rule
            rule = ContextualRule(
                rule_id=str(uuid.uuid4()),
                rule_name=f"Context_Rule_{preferred_framework}_{int(accuracy*100)}",
                context_conditions=context_conditions,
                framework_preference=preferred_framework,
                weight=accuracy,
                accuracy=accuracy,
                usage_count=total_decisions,
                creation_time=datetime.now(),
                last_success=datetime.now() if successful_decisions else datetime.now()
            )
            
            return rule
            
        except Exception as e:
            logger.error(f"Contextual rule extraction error: {e}")
            return None
    
    def _extract_context_conditions(self, decisions: List[SelectionDecision]) -> Dict[str, Any]:
        """Extract context conditions from decisions"""
        try:
            conditions = {}
            
            # Analyze context feature ranges
            context_features = defaultdict(list)
            for decision in decisions:
                for feature, value in decision.context_features.items():
                    context_features[feature].append(value)
            
            # Define conditions as value ranges
            for feature, values in context_features.items():
                if values:
                    min_val = min(values)
                    max_val = max(values)
                    mean_val = statistics.mean(values)
                    std_val = statistics.stdev(values) if len(values) > 1 else 0
                    
                    # Create condition range
                    lower_bound = max(0, mean_val - std_val)
                    upper_bound = min(1, mean_val + std_val)
                    
                    conditions[feature] = {
                        'range': [lower_bound, upper_bound],
                        'mean': mean_val,
                        'std': std_val
                    }
            
            return conditions
            
        except Exception as e:
            logger.error(f"Context condition extraction error: {e}")
            return {}
    
    def _store_contextual_rule(self, rule: ContextualRule):
        """Store contextual rule in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO contextual_rules 
                    (rule_id, rule_name, context_conditions, framework_preference,
                     weight, accuracy, usage_count, creation_time, last_success)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    rule.rule_id,
                    rule.rule_name,
                    json.dumps(rule.context_conditions),
                    rule.framework_preference,
                    rule.weight,
                    rule.accuracy,
                    rule.usage_count,
                    rule.creation_time.isoformat(),
                    rule.last_success.isoformat()
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Contextual rule storage error: {e}")

class PatternRecognitionEngine:
    """Pattern recognition for optimal selection rules"""
    
    def __init__(self, db_path: str = "selection_learning.db"):
        self.db_path = db_path
        self.pattern_cache = {}
        self.recognition_threshold = 0.7
        
    def identify_patterns(self, decisions: List[SelectionDecision]) -> List[LearningPattern]:
        """Identify patterns in selection decisions"""
        try:
            patterns = []
            
            # Identify different types of patterns
            temporal_patterns = self._identify_temporal_patterns(decisions)
            complexity_patterns = self._identify_complexity_patterns(decisions)
            performance_patterns = self._identify_performance_patterns(decisions)
            
            patterns.extend(temporal_patterns)
            patterns.extend(complexity_patterns)
            patterns.extend(performance_patterns)
            
            # Store patterns
            for pattern in patterns:
                self._store_pattern(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern identification error: {e}")
            return []
    
    def _identify_temporal_patterns(self, decisions: List[SelectionDecision]) -> List[LearningPattern]:
        """Identify temporal patterns in decisions"""
        try:
            patterns = []
            
            # Group by time of day
            hour_groups = defaultdict(list)
            for decision in decisions:
                hour = decision.timestamp.hour
                hour_groups[hour].append(decision)
            
            # Analyze each hour group
            for hour, hour_decisions in hour_groups.items():
                if len(hour_decisions) >= 5:  # Minimum sample size
                    pattern = self._analyze_temporal_group(hour, hour_decisions)
                    if pattern and pattern.effectiveness_score > self.recognition_threshold:
                        patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Temporal pattern identification error: {e}")
            return []
    
    def _analyze_temporal_group(self, hour: int, decisions: List[SelectionDecision]) -> Optional[LearningPattern]:
        """Analyze temporal group for patterns"""
        try:
            # Framework distribution
            framework_counts = defaultdict(int)
            successful_decisions = []
            
            for decision in decisions:
                framework_counts[decision.framework_selected] += 1
                if decision.performance_outcome and decision.performance_outcome > 0.7:
                    successful_decisions.append(decision)
            
            if not framework_counts:
                return None
            
            # Find dominant framework
            dominant_framework = max(framework_counts, key=framework_counts.get)
            success_rate = len(successful_decisions) / len(decisions)
            
            # Create pattern
            pattern = LearningPattern(
                pattern_id=str(uuid.uuid4()),
                pattern_type=PatternType.TEMPORAL_PATTERN,
                pattern_description=f"Hour {hour}: {dominant_framework} performs better",
                conditions={
                    'time_range': [hour, hour + 1],
                    'min_samples': 5
                },
                recommended_framework=dominant_framework,
                confidence=success_rate,
                success_rate=success_rate,
                sample_count=len(decisions),
                last_updated=datetime.now(),
                effectiveness_score=success_rate
            )
            
            return pattern
            
        except Exception as e:
            logger.error(f"Temporal group analysis error: {e}")
            return None
    
    def _identify_complexity_patterns(self, decisions: List[SelectionDecision]) -> List[LearningPattern]:
        """Identify complexity-based patterns"""
        try:
            patterns = []
            
            # Group by complexity ranges
            complexity_groups = {
                'low': [],
                'medium': [],
                'high': []
            }
            
            for decision in decisions:
                complexity = decision.task_features.get('complexity_score', 0.5)
                if complexity < 0.4:
                    complexity_groups['low'].append(decision)
                elif complexity < 0.7:
                    complexity_groups['medium'].append(decision)
                else:
                    complexity_groups['high'].append(decision)
            
            # Analyze each complexity group
            for complexity_level, group_decisions in complexity_groups.items():
                if len(group_decisions) >= 10:  # Minimum sample size
                    pattern = self._analyze_complexity_group(complexity_level, group_decisions)
                    if pattern and pattern.effectiveness_score > self.recognition_threshold:
                        patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Complexity pattern identification error: {e}")
            return []
    
    def _analyze_complexity_group(self, complexity_level: str, decisions: List[SelectionDecision]) -> Optional[LearningPattern]:
        """Analyze complexity group for patterns"""
        try:
            # Framework performance analysis
            framework_performance = defaultdict(list)
            
            for decision in decisions:
                if decision.performance_outcome is not None:
                    framework_performance[decision.framework_selected].append(decision.performance_outcome)
            
            if not framework_performance:
                return None
            
            # Find best performing framework
            framework_averages = {}
            for framework, performances in framework_performance.items():
                if len(performances) >= 3:  # Minimum samples
                    framework_averages[framework] = statistics.mean(performances)
            
            if not framework_averages:
                return None
            
            best_framework = max(framework_averages, key=framework_averages.get)
            effectiveness = framework_averages[best_framework]
            
            # Define complexity range
            range_map = {
                'low': [0.0, 0.4],
                'medium': [0.4, 0.7],
                'high': [0.7, 1.0]
            }
            
            pattern = LearningPattern(
                pattern_id=str(uuid.uuid4()),
                pattern_type=PatternType.COMPLEXITY_PATTERN,
                pattern_description=f"{complexity_level.title()} complexity: {best_framework} optimal",
                conditions={
                    'complexity_range': range_map[complexity_level],
                    'min_samples': 10
                },
                recommended_framework=best_framework,
                confidence=effectiveness,
                success_rate=effectiveness,
                sample_count=len(decisions),
                last_updated=datetime.now(),
                effectiveness_score=effectiveness
            )
            
            return pattern
            
        except Exception as e:
            logger.error(f"Complexity group analysis error: {e}")
            return None
    
    def _identify_performance_patterns(self, decisions: List[SelectionDecision]) -> List[LearningPattern]:
        """Identify performance-based patterns"""
        try:
            patterns = []
            
            # Group by performance outcomes
            performance_groups = {
                'high_performance': [],
                'medium_performance': [],
                'low_performance': []
            }
            
            for decision in decisions:
                if decision.performance_outcome is not None:
                    perf = decision.performance_outcome
                    if perf > 0.8:
                        performance_groups['high_performance'].append(decision)
                    elif perf > 0.6:
                        performance_groups['medium_performance'].append(decision)
                    else:
                        performance_groups['low_performance'].append(decision)
            
            # Analyze high performance group for common characteristics
            high_perf_decisions = performance_groups['high_performance']
            if len(high_perf_decisions) >= 10:
                pattern = self._analyze_high_performance_characteristics(high_perf_decisions)
                if pattern and pattern.effectiveness_score > self.recognition_threshold:
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Performance pattern identification error: {e}")
            return []
    
    def _analyze_high_performance_characteristics(self, decisions: List[SelectionDecision]) -> Optional[LearningPattern]:
        """Analyze characteristics of high-performing decisions"""
        try:
            # Find common framework
            framework_counts = defaultdict(int)
            for decision in decisions:
                framework_counts[decision.framework_selected] += 1
            
            if not framework_counts:
                return None
            
            dominant_framework = max(framework_counts, key=framework_counts.get)
            
            # Analyze common task and context features
            task_features = defaultdict(list)
            context_features = defaultdict(list)
            
            for decision in decisions:
                if decision.framework_selected == dominant_framework:
                    for feature, value in decision.task_features.items():
                        task_features[feature].append(value)
                    for feature, value in decision.context_features.items():
                        context_features[feature].append(value)
            
            # Create conditions based on feature ranges
            conditions = {'task_features': {}, 'context_features': {}}
            
            for feature, values in task_features.items():
                if values and len(values) >= 3:
                    mean_val = statistics.mean(values)
                    std_val = statistics.stdev(values) if len(values) > 1 else 0.1
                    conditions['task_features'][feature] = [
                        max(0, mean_val - std_val),
                        min(1, mean_val + std_val)
                    ]
            
            for feature, values in context_features.items():
                if values and len(values) >= 3:
                    mean_val = statistics.mean(values)
                    std_val = statistics.stdev(values) if len(values) > 1 else 0.1
                    conditions['context_features'][feature] = [
                        max(0, mean_val - std_val),
                        min(1, mean_val + std_val)
                    ]
            
            pattern = LearningPattern(
                pattern_id=str(uuid.uuid4()),
                pattern_type=PatternType.PERFORMANCE_PATTERN,
                pattern_description=f"High performance pattern: {dominant_framework}",
                conditions=conditions,
                recommended_framework=dominant_framework,
                confidence=0.9,  # High confidence for high-performance patterns
                success_rate=1.0,  # By definition, these are successful
                sample_count=len(decisions),
                last_updated=datetime.now(),
                effectiveness_score=0.9
            )
            
            return pattern
            
        except Exception as e:
            logger.error(f"High performance analysis error: {e}")
            return None
    
    def _store_pattern(self, pattern: LearningPattern):
        """Store learning pattern in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO learning_patterns 
                    (pattern_id, pattern_type, pattern_description, conditions,
                     recommended_framework, confidence, success_rate, sample_count,
                     last_updated, effectiveness_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern.pattern_id,
                    pattern.pattern_type.value,
                    pattern.pattern_description,
                    json.dumps(pattern.conditions),
                    pattern.recommended_framework,
                    pattern.confidence,
                    pattern.success_rate,
                    pattern.sample_count,
                    pattern.last_updated.isoformat(),
                    pattern.effectiveness_score
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Pattern storage error: {e}")

class AutomatedParameterTuning:
    """Automated parameter tuning system"""
    
    def __init__(self, db_path: str = "selection_learning.db"):
        self.db_path = db_path
        self.tuning_history = {}
        self.accuracy_target = 0.9  # 90% accuracy target
        self.tuning_patience = 5
        
    def tune_parameters(self, current_accuracy: float) -> Dict[str, float]:
        """Tune parameters to maintain target accuracy"""
        try:
            if current_accuracy >= self.accuracy_target:
                return {}  # No tuning needed
            
            # Get current parameter configurations
            current_params = self._get_current_parameters()
            
            # Identify parameters to tune
            tuning_candidates = self._identify_tuning_candidates(current_params, current_accuracy)
            
            # Perform tuning
            tuned_params = {}
            for param_name, config in tuning_candidates.items():
                new_value = self._tune_single_parameter(param_name, config, current_accuracy)
                if new_value is not None:
                    tuned_params[param_name] = new_value
                    self._update_parameter_value(param_name, new_value, current_accuracy)
            
            return tuned_params
            
        except Exception as e:
            logger.error(f"Parameter tuning error: {e}")
            return {}
    
    def _get_current_parameters(self) -> Dict[str, ParameterConfiguration]:
        """Get current parameter configurations"""
        try:
            params = {}
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM parameter_configurations")
                rows = cursor.fetchall()
                
                for row in rows:
                    config = ParameterConfiguration(
                        config_id=row[0],
                        parameter_name=row[1],
                        current_value=row[2],
                        optimal_range=json.loads(row[3]),
                        step_size=row[4],
                        accuracy_impact=row[5],
                        last_tuned=datetime.fromisoformat(row[6]),
                        tuning_history=json.loads(row[7])
                    )
                    params[config.parameter_name] = config
            
            return params
            
        except Exception as e:
            logger.error(f"Parameter retrieval error: {e}")
            return {}
    
    def _identify_tuning_candidates(self, params: Dict[str, ParameterConfiguration], 
                                   current_accuracy: float) -> Dict[str, ParameterConfiguration]:
        """Identify parameters that need tuning"""
        try:
            candidates = {}
            
            for param_name, config in params.items():
                # Check if parameter has impact on accuracy
                if abs(config.accuracy_impact) > 0.01:  # Meaningful impact threshold
                    # Check if parameter hasn't been tuned recently
                    time_since_tuning = datetime.now() - config.last_tuned
                    if time_since_tuning.total_seconds() > 3600:  # 1 hour minimum
                        candidates[param_name] = config
            
            return candidates
            
        except Exception as e:
            logger.error(f"Tuning candidate identification error: {e}")
            return {}
    
    def _tune_single_parameter(self, param_name: str, config: ParameterConfiguration, 
                              current_accuracy: float) -> Optional[float]:
        """Tune a single parameter"""
        try:
            current_value = config.current_value
            step_size = config.step_size
            optimal_range = config.optimal_range
            
            # Determine tuning direction based on accuracy impact
            if config.accuracy_impact > 0:
                # Positive impact: increase if accuracy is low
                if current_accuracy < self.accuracy_target:
                    new_value = current_value + step_size
                else:
                    return None  # No tuning needed
            else:
                # Negative impact: decrease if accuracy is low
                if current_accuracy < self.accuracy_target:
                    new_value = current_value - step_size
                else:
                    return None  # No tuning needed
            
            # Ensure value stays within optimal range
            new_value = max(optimal_range[0], min(optimal_range[1], new_value))
            
            # Check if this is a meaningful change
            if abs(new_value - current_value) < step_size * 0.1:
                return None
            
            return new_value
            
        except Exception as e:
            logger.error(f"Single parameter tuning error for {param_name}: {e}")
            return None
    
    def _update_parameter_value(self, param_name: str, new_value: float, accuracy: float):
        """Update parameter value and history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get current configuration
                cursor.execute("SELECT * FROM parameter_configurations WHERE parameter_name = ?", (param_name,))
                row = cursor.fetchone()
                
                if row:
                    tuning_history = json.loads(row[7])
                    tuning_history.append((new_value, accuracy))
                    
                    # Keep only recent history
                    if len(tuning_history) > 50:
                        tuning_history = tuning_history[-50:]
                    
                    # Calculate accuracy impact
                    accuracy_impact = self._calculate_accuracy_impact(tuning_history)
                    
                    # Update parameter
                    cursor.execute("""
                        UPDATE parameter_configurations 
                        SET current_value = ?, accuracy_impact = ?, last_tuned = ?, tuning_history = ?
                        WHERE parameter_name = ?
                    """, (new_value, accuracy_impact, datetime.now().isoformat(), 
                          json.dumps(tuning_history), param_name))
                    conn.commit()
                    
                    logger.info(f"Parameter {param_name} tuned to {new_value}")
                    
        except Exception as e:
            logger.error(f"Parameter update error for {param_name}: {e}")
    
    def _calculate_accuracy_impact(self, tuning_history: List[Tuple[float, float]]) -> float:
        """Calculate accuracy impact of parameter changes"""
        try:
            if len(tuning_history) < 2:
                return 0.0
            
            # Calculate correlation between parameter value and accuracy
            values = [entry[0] for entry in tuning_history]
            accuracies = [entry[1] for entry in tuning_history]
            
            if len(set(values)) < 2:  # No variation in values
                return 0.0
            
            # Simple correlation calculation
            mean_value = statistics.mean(values)
            mean_accuracy = statistics.mean(accuracies)
            
            numerator = sum((v - mean_value) * (a - mean_accuracy) for v, a in zip(values, accuracies))
            denominator_v = sum((v - mean_value) ** 2 for v in values)
            denominator_a = sum((a - mean_accuracy) ** 2 for a in accuracies)
            
            if denominator_v == 0 or denominator_a == 0:
                return 0.0
            
            correlation = numerator / math.sqrt(denominator_v * denominator_a)
            return correlation
            
        except Exception as e:
            logger.error(f"Accuracy impact calculation error: {e}")
            return 0.0

class FrameworkSelectionLearningOrchestrator:
    """Main orchestrator for framework selection learning"""
    
    def __init__(self, db_path: str = "selection_learning.db"):
        self.db_path = db_path
        self.adaptive_algorithm = AdaptiveSelectionAlgorithm(db_path)
        self.performance_learning = PerformanceBasedLearning(db_path)
        self.context_learning = ContextAwareLearning(db_path)
        self.pattern_recognition = PatternRecognitionEngine(db_path)
        self.parameter_tuning = AutomatedParameterTuning(db_path)
        self.learning_metrics = {}
        self.is_learning = False
        self.learning_thread = None
        
    def start_learning(self):
        """Start the framework selection learning system"""
        try:
            if self.is_learning:
                logger.warning("Learning system already running")
                return
            
            self.is_learning = True
            
            # Start background learning thread
            self.learning_thread = threading.Thread(
                target=self._learning_loop,
                daemon=True
            )
            self.learning_thread.start()
            
            logger.info("Framework selection learning system started")
            
        except Exception as e:
            logger.error(f"Learning start error: {e}")
            self.is_learning = False
    
    def stop_learning(self):
        """Stop the framework selection learning system"""
        try:
            self.is_learning = False
            
            if self.learning_thread and self.learning_thread.is_alive():
                self.learning_thread.join(timeout=5.0)
            
            logger.info("Framework selection learning system stopped")
            
        except Exception as e:
            logger.error(f"Learning stop error: {e}")
    
    def select_framework_with_learning(self, task_features: Dict[str, float], 
                                     context_features: Dict[str, float]) -> Tuple[str, float, str]:
        """Select framework using learning-enhanced algorithms"""
        try:
            # Use adaptive algorithm for selection
            framework, confidence, strategy = self.adaptive_algorithm.select_framework_adaptive(
                task_features, context_features
            )
            
            return framework, confidence, strategy
            
        except Exception as e:
            logger.error(f"Learning-enhanced selection error: {e}")
            return "langchain", 0.5, "fallback"
    
    def record_selection_outcome(self, decision_id: str, performance_outcome: float, 
                               user_satisfaction: float, execution_time: float, 
                               success_indicator: bool):
        """Record selection outcome for learning"""
        try:
            self.adaptive_algorithm.update_decision_outcome(
                decision_id, performance_outcome, user_satisfaction, 
                execution_time, success_indicator
            )
            
        except Exception as e:
            logger.error(f"Outcome recording error: {e}")
    
    def _learning_loop(self):
        """Main learning loop"""
        try:
            while self.is_learning:
                try:
                    # Get recent decisions
                    recent_decisions = self._get_recent_decisions()
                    
                    if len(recent_decisions) >= 20:  # Minimum sample size
                        # Update performance learning
                        self.performance_learning.update_performance_model(recent_decisions)
                        
                        # Learn contextual patterns
                        self.context_learning.learn_contextual_patterns(recent_decisions)
                        
                        # Identify patterns
                        self.pattern_recognition.identify_patterns(recent_decisions)
                        
                        # Calculate current accuracy
                        current_accuracy = self._calculate_current_accuracy(recent_decisions)
                        
                        # Tune parameters if needed
                        if current_accuracy < 0.9:
                            self.parameter_tuning.tune_parameters(current_accuracy)
                        
                        # Update learning metrics
                        self._update_learning_metrics(recent_decisions)
                    
                    # Sleep before next iteration
                    time.sleep(600)  # 10 minutes
                    
                except Exception as e:
                    logger.error(f"Learning loop iteration error: {e}")
                    time.sleep(120)  # Shorter sleep on error
                    
        except Exception as e:
            logger.error(f"Learning loop fatal error: {e}")
    
    def _get_recent_decisions(self) -> List[SelectionDecision]:
        """Get recent selection decisions"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=24)
            decisions = []
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM selection_decisions 
                    WHERE timestamp > ?
                    ORDER BY timestamp DESC
                    LIMIT 200
                """, (cutoff_time.isoformat(),))
                
                rows = cursor.fetchall()
                
                for row in rows:
                    decision = SelectionDecision(
                        decision_id=row[0],
                        timestamp=datetime.fromisoformat(row[1]),
                        framework_selected=row[2],
                        task_features=json.loads(row[3]),
                        context_features=json.loads(row[4]),
                        confidence_score=row[5],
                        strategy_used=row[6],
                        performance_outcome=row[7],
                        user_satisfaction=row[8],
                        execution_time=row[9],
                        success_indicator=row[10],
                        learning_source=row[11]
                    )
                    decisions.append(decision)
            
            return decisions
            
        except Exception as e:
            logger.error(f"Recent decisions retrieval error: {e}")
            return []
    
    def _calculate_current_accuracy(self, decisions: List[SelectionDecision]) -> float:
        """Calculate current selection accuracy"""
        try:
            successful_decisions = [d for d in decisions if d.success_indicator is True]
            total_decisions = len([d for d in decisions if d.success_indicator is not None])
            
            if total_decisions == 0:
                return 0.7  # Default baseline
            
            accuracy = len(successful_decisions) / total_decisions
            return accuracy
            
        except Exception as e:
            logger.error(f"Accuracy calculation error: {e}")
            return 0.7
    
    def _update_learning_metrics(self, decisions: List[SelectionDecision]):
        """Update learning performance metrics"""
        try:
            # Calculate 30-day accuracy improvement
            accuracy_30_day = self._calculate_30_day_improvement(decisions)
            
            # Calculate error reduction
            error_reduction = self._calculate_error_reduction(decisions)
            
            # Count patterns
            pattern_count = self._count_active_patterns()
            
            # Calculate convergence rate
            convergence_rate = self._calculate_convergence_rate(decisions)
            
            # Parameter stability
            parameter_stability = self._calculate_parameter_stability()
            
            # Store metrics
            self.learning_metrics = {
                'accuracy_30_day_improvement': accuracy_30_day,
                'error_reduction_percent': error_reduction,
                'pattern_count': pattern_count,
                'convergence_rate': convergence_rate,
                'parameter_stability': parameter_stability,
                'last_updated': datetime.now().isoformat()
            }
            
            # Store in database
            self._store_learning_metrics()
            
        except Exception as e:
            logger.error(f"Learning metrics update error: {e}")
    
    def _calculate_30_day_improvement(self, decisions: List[SelectionDecision]) -> float:
        """Calculate accuracy improvement over 30 days"""
        try:
            cutoff_time = datetime.now() - timedelta(days=30)
            historical_decisions = [d for d in decisions if d.timestamp > cutoff_time]
            
            if len(historical_decisions) < 20:
                return 0.0
            
            # Calculate trends using performance learning
            trends = self.performance_learning.analyze_performance_trends(30)
            
            # Average improvement across frameworks
            if trends:
                avg_improvement = statistics.mean(trends.values())
                return avg_improvement * 100  # Convert to percentage
            
            return 0.0
            
        except Exception as e:
            logger.error(f"30-day improvement calculation error: {e}")
            return 0.0
    
    def _calculate_error_reduction(self, decisions: List[SelectionDecision]) -> float:
        """Calculate error reduction percentage"""
        try:
            # Compare error rates between different time periods
            cutoff_time = datetime.now() - timedelta(days=15)
            
            early_decisions = [d for d in decisions if d.timestamp < cutoff_time and d.success_indicator is not None]
            recent_decisions = [d for d in decisions if d.timestamp >= cutoff_time and d.success_indicator is not None]
            
            if not early_decisions or not recent_decisions:
                return 0.0
            
            early_error_rate = 1 - (sum(1 for d in early_decisions if d.success_indicator) / len(early_decisions))
            recent_error_rate = 1 - (sum(1 for d in recent_decisions if d.success_indicator) / len(recent_decisions))
            
            if early_error_rate > 0:
                error_reduction = ((early_error_rate - recent_error_rate) / early_error_rate) * 100
                return max(0, error_reduction)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error reduction calculation error: {e}")
            return 0.0
    
    def _count_active_patterns(self) -> int:
        """Count active learning patterns"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM learning_patterns WHERE effectiveness_score > 0.7")
                count = cursor.fetchone()[0]
                return count
                
        except Exception as e:
            logger.error(f"Pattern counting error: {e}")
            return 0
    
    def _calculate_convergence_rate(self, decisions: List[SelectionDecision]) -> float:
        """Calculate learning convergence rate"""
        try:
            if len(decisions) < 50:
                return 0.0
            
            # Analyze accuracy stability over recent decisions
            recent_100 = decisions[-100:] if len(decisions) >= 100 else decisions
            
            # Calculate accuracy for sliding windows
            window_size = 20
            accuracies = []
            
            for i in range(0, len(recent_100) - window_size + 1, 5):
                window_decisions = recent_100[i:i + window_size]
                successful = sum(1 for d in window_decisions if d.success_indicator is True)
                total = sum(1 for d in window_decisions if d.success_indicator is not None)
                
                if total > 0:
                    accuracy = successful / total
                    accuracies.append(accuracy)
            
            if len(accuracies) < 3:
                return 0.0
            
            # Calculate convergence (lower variance = higher convergence)
            variance = statistics.variance(accuracies) if len(accuracies) > 1 else 1.0
            convergence_rate = max(0, 1 - variance)
            
            return convergence_rate
            
        except Exception as e:
            logger.error(f"Convergence rate calculation error: {e}")
            return 0.0
    
    def _calculate_parameter_stability(self) -> float:
        """Calculate parameter stability score"""
        try:
            params = self.parameter_tuning._get_current_parameters()
            
            if not params:
                return 1.0  # Perfect stability if no parameters
            
            stability_scores = []
            
            for param_name, config in params.items():
                if len(config.tuning_history) >= 3:
                    # Calculate stability based on recent tuning frequency
                    recent_tuning = [entry for entry in config.tuning_history[-10:]]
                    
                    if len(recent_tuning) >= 2:
                        values = [entry[0] for entry in recent_tuning]
                        value_variance = statistics.variance(values)
                        stability = max(0, 1 - value_variance)
                        stability_scores.append(stability)
            
            if stability_scores:
                return statistics.mean(stability_scores)
            
            return 1.0
            
        except Exception as e:
            logger.error(f"Parameter stability calculation error: {e}")
            return 1.0
    
    def _store_learning_metrics(self):
        """Store learning metrics in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO learning_metrics 
                    (metric_id, timestamp, accuracy_30_day, error_reduction_percent,
                     pattern_count, convergence_rate, parameter_stability)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(uuid.uuid4()),
                    datetime.now().isoformat(),
                    self.learning_metrics.get('accuracy_30_day_improvement', 0),
                    self.learning_metrics.get('error_reduction_percent', 0),
                    self.learning_metrics.get('pattern_count', 0),
                    self.learning_metrics.get('convergence_rate', 0),
                    self.learning_metrics.get('parameter_stability', 0)
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Learning metrics storage error: {e}")
    
    def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning system status"""
        try:
            status = {
                'is_learning': self.is_learning,
                'adaptive_algorithm_status': 'active' if self.adaptive_algorithm else 'inactive',
                'performance_learning_status': 'active' if self.performance_learning else 'inactive',
                'context_learning_status': 'active' if self.context_learning else 'inactive',
                'pattern_recognition_status': 'active' if self.pattern_recognition else 'inactive',
                'parameter_tuning_status': 'active' if self.parameter_tuning else 'inactive',
                'learning_metrics': self.learning_metrics.copy(),
                'recent_decision_count': len(self.adaptive_algorithm.learning_history)
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Learning status retrieval error: {e}")
            return {'error': str(e)}

# Demo and Testing Functions
def create_demo_selection_learning_system():
    """Create a demo framework selection learning system"""
    try:
        # Initialize system
        orchestrator = FrameworkSelectionLearningOrchestrator("demo_selection_learning.db")
        
        # Start learning
        orchestrator.start_learning()
        
        # Generate sample decisions with learning progression
        for i in range(100):
            # Create sample task and context features
            task_features = {
                'complexity_score': random.uniform(0.2, 0.9),
                'resource_requirements': random.uniform(0.3, 0.8),
                'agent_count': random.randint(1, 5),
                'workflow_complexity': random.uniform(0.1, 0.9)
            }
            
            context_features = {
                'user_tier': random.uniform(0.3, 1.0),
                'system_load': random.uniform(0.2, 0.9),
                'time_of_day': random.uniform(0.0, 1.0),
                'historical_performance': random.uniform(0.4, 0.9)
            }
            
            # Select framework
            framework, confidence, strategy = orchestrator.select_framework_with_learning(
                task_features, context_features
            )
            
            # Simulate performance outcome (with learning improvement)
            base_performance = 0.6
            learning_bonus = min(0.3, i * 0.003)  # Gradual improvement
            complexity_factor = 1 - (task_features['complexity_score'] * 0.2)
            
            performance_outcome = (base_performance + learning_bonus) * complexity_factor
            performance_outcome += random.uniform(-0.1, 0.1)  # Add noise
            performance_outcome = max(0.3, min(1.0, performance_outcome))
            
            user_satisfaction = performance_outcome * (0.9 + random.uniform(0, 0.2))
            user_satisfaction = max(0.3, min(1.0, user_satisfaction))
            
            execution_time = random.uniform(0.5, 3.0)
            success_indicator = performance_outcome > 0.65
            
            # Record outcome
            decision_id = f"demo_decision_{i}"
            orchestrator.record_selection_outcome(
                decision_id, performance_outcome, user_satisfaction, 
                execution_time, success_indicator
            )
        
        # Wait for learning processing
        time.sleep(3)
        
        return orchestrator
        
    except Exception as e:
        logger.error(f"Demo system creation error: {e}")
        return None

if __name__ == "__main__":
    print("🧠 LangGraph Framework Selection Learning Sandbox")
    print("=" * 70)
    
    try:
        # Create and run demo
        demo_system = create_demo_selection_learning_system()
        
        if demo_system:
            print("✅ Demo system created successfully")
            
            # Display status
            status = demo_system.get_learning_status()
            print(f"\n📊 Learning System Status:")
            print(f"Learning Active: {status['is_learning']}")
            print(f"Recent Decisions: {status['recent_decision_count']}")
            
            if 'learning_metrics' in status:
                metrics = status['learning_metrics']
                print(f"\n📈 Learning Metrics:")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        print(f"{key}: {value:.2f}")
                    else:
                        print(f"{key}: {value}")
            
            # Test adaptive selection
            print(f"\n🧪 Testing Adaptive Selection:")
            test_task = {
                'complexity_score': 0.8,
                'resource_requirements': 0.7,
                'agent_count': 3,
                'workflow_complexity': 0.9
            }
            
            test_context = {
                'user_tier': 0.9,
                'system_load': 0.4,
                'time_of_day': 0.5,
                'historical_performance': 0.8
            }
            
            framework, confidence, strategy = demo_system.select_framework_with_learning(
                test_task, test_context
            )
            
            print(f"Selected Framework: {framework}")
            print(f"Confidence: {confidence:.3f}")
            print(f"Strategy: {strategy}")
            
            # Run for demo period
            print("\n⏱️  Running learning system for 10 seconds...")
            time.sleep(10)
            
            # Final status
            final_status = demo_system.get_learning_status()
            print(f"\n🎯 Final Learning Status: {final_status['is_learning']}")
            
            # Stop system
            demo_system.stop_learning()
            print("✅ Demo completed successfully")
            
        else:
            print("❌ Failed to create demo system")
            
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo error: {e}")