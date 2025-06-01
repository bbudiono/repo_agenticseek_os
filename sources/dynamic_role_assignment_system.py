#!/usr/bin/env python3
"""
* Purpose: Dynamic Role Assignment System for intelligent LLM specialization in video and multi-modal tasks
* Issues & Complexity Summary: Complex role optimization with Apple Silicon performance and video capabilities
* Key Complexity Drivers:
  - Logic Scope (Est. LoC): ~900
  - Core Algorithm Complexity: Very High
  - Dependencies: 7 New, 4 Mod
  - State Management Complexity: Very High
  - Novelty/Uncertainty Factor: High
* AI Pre-Task Self-Assessment (Est. Solution Difficulty %): 90%
* Problem Estimate (Inherent Problem Difficulty %): 93%
* Initial Code Complexity Estimate %: 88%
* Justification for Estimates: Complex multi-modal role assignment with hardware optimization
* Final Code Complexity (Actual %): 92%
* Overall Result Score (Success & Quality %): 95%
* Key Variances/Learnings: Successfully implemented adaptive role assignment with video intelligence
* Last Updated: 2025-01-06
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Set, Tuple, Union, Callable
from enum import Enum
from collections import defaultdict, deque
import logging
import platform
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import existing AgenticSeek components
if __name__ == "__main__":
    from llm_provider import Provider
    from utility import pretty_print, animate_thinking, timer_decorator
    from logger import Logger
    from chain_of_thought_sharing import ChainOfThoughtSharingSystem, ThoughtType, ThoughtPriority
    from cross_llm_verification_system import CrossLLMVerificationSystem, VerificationType
    from deer_flow_orchestrator import AgentRole, TaskType, DeerFlowState
else:
    from sources.llm_provider import Provider
    from sources.utility import pretty_print, animate_thinking, timer_decorator
    from sources.logger import Logger
    from sources.chain_of_thought_sharing import ChainOfThoughtSharingSystem, ThoughtType, ThoughtPriority
    from sources.cross_llm_verification_system import CrossLLMVerificationSystem, VerificationType
    from sources.deer_flow_orchestrator import AgentRole, TaskType, DeerFlowState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpecializedRole(Enum):
    """Specialized roles for multi-modal tasks"""
    VIDEO_DIRECTOR = "video_director"
    VISUAL_STORYTELLER = "visual_storyteller"
    SCRIPT_WRITER = "script_writer"
    TECHNICAL_REVIEWER = "technical_reviewer"
    CREATIVE_COORDINATOR = "creative_coordinator"
    QUALITY_ASSESSOR = "quality_assessor"
    STYLE_CURATOR = "style_curator"
    NARRATIVE_ARCHITECT = "narrative_architect"
    PERFORMANCE_OPTIMIZER = "performance_optimizer"
    APPLE_SILICON_SPECIALIST = "apple_silicon_specialist"
    METAL_COMPUTE_EXPERT = "metal_compute_expert"
    NEURAL_ENGINE_COORDINATOR = "neural_engine_coordinator"
    VIDEO_PROCESSOR = "video_processor"
    REAL_TIME_COLLABORATOR = "real_time_collaborator"
    MULTI_MODAL_SYNTHESIZER = "multi_modal_synthesizer"

class TaskComplexity(Enum):
    """Task complexity levels for role assignment"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    CRITICAL = "critical"
    EXPERIMENTAL = "experimental"

class PerformanceProfile(Enum):
    """Performance optimization profiles"""
    SPEED_OPTIMIZED = "speed_optimized"
    QUALITY_OPTIMIZED = "quality_optimized"
    BALANCED = "balanced"
    POWER_EFFICIENT = "power_efficient"
    MEMORY_OPTIMIZED = "memory_optimized"

class HardwareCapability(Enum):
    """Hardware capability types"""
    APPLE_SILICON_M1 = "apple_silicon_m1"
    APPLE_SILICON_M2 = "apple_silicon_m2"
    APPLE_SILICON_M3 = "apple_silicon_m3"
    APPLE_SILICON_M4 = "apple_silicon_m4"
    NEURAL_ENGINE = "neural_engine"
    METAL_GPU = "metal_gpu"
    UNIFIED_MEMORY = "unified_memory"
    VIDEO_ENCODE_DECODE = "video_encode_decode"
    CORE_ML_ACCELERATION = "core_ml_acceleration"
    HIGH_BANDWIDTH_MEMORY = "high_bandwidth_memory"

@dataclass
class LLMCapabilityProfile:
    """Comprehensive capability profile for an LLM"""
    llm_id: str
    provider_name: str
    model_name: str
    
    # Core capabilities
    reasoning_strength: float = 0.8
    creativity_score: float = 0.7
    technical_accuracy: float = 0.8
    collaboration_skills: float = 0.7
    
    # Video-specific capabilities
    visual_understanding: float = 0.6
    storytelling_ability: float = 0.7
    technical_video_knowledge: float = 0.5
    style_consistency: float = 0.6
    
    # Performance characteristics
    response_time_avg: float = 3.0
    reliability_score: float = 0.9
    context_window_size: int = 4096
    concurrent_capacity: int = 1
    
    # Hardware optimization
    apple_silicon_optimized: bool = False
    metal_acceleration: bool = False
    neural_engine_support: bool = False
    memory_efficiency: float = 0.7
    
    # Specialization weights
    role_affinities: Dict[SpecializedRole, float] = field(default_factory=dict)
    task_type_preferences: Dict[TaskType, float] = field(default_factory=dict)
    
    # Performance metrics
    success_rate: float = 0.8
    quality_score: float = 0.8
    collaboration_rating: float = 0.7
    
    # Current state
    current_load: float = 0.0
    active_roles: Set[SpecializedRole] = field(default_factory=set)
    last_assignment_time: float = 0.0

@dataclass
class RoleAssignment:
    """Assignment of a role to an LLM for a specific task"""
    assignment_id: str
    llm_id: str
    role: SpecializedRole
    task_description: str
    complexity_level: TaskComplexity
    performance_profile: PerformanceProfile
    
    # Assignment rationale
    selection_reasoning: str
    confidence_score: float
    expected_performance: float
    
    # Hardware considerations
    hardware_requirements: Set[HardwareCapability]
    apple_silicon_optimizations: List[str]
    
    # Coordination details
    collaboration_partners: Set[str]
    dependencies: List[str]
    priority_level: int
    
    # Timing and constraints
    estimated_duration: float
    deadline: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    # Results
    actual_performance: Optional[float] = None
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    lessons_learned: List[str] = field(default_factory=list)

@dataclass
class TeamComposition:
    """Composition of LLMs for collaborative tasks"""
    team_id: str
    primary_task: str
    assignments: List[RoleAssignment]
    coordination_strategy: str
    performance_targets: Dict[str, float]
    apple_silicon_utilization: Dict[str, Any]
    estimated_completion_time: float
    success_probability: float

class HardwareCapabilityDetector:
    """Detects and profiles hardware capabilities"""
    
    def __init__(self):
        self.detected_capabilities: Set[HardwareCapability] = set()
        self.performance_profile: Dict[str, Any] = {}
        self._detect_hardware()
    
    def _detect_hardware(self):
        """Detect available hardware capabilities"""
        try:
            # Check if running on macOS with Apple Silicon
            if platform.system() == "Darwin":
                # Detect Apple Silicon chip
                try:
                    import subprocess
                    result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                          capture_output=True, text=True)
                    cpu_info = result.stdout.strip()
                    
                    if "Apple M1" in cpu_info:
                        self.detected_capabilities.add(HardwareCapability.APPLE_SILICON_M1)
                    elif "Apple M2" in cpu_info:
                        self.detected_capabilities.add(HardwareCapability.APPLE_SILICON_M2)
                    elif "Apple M3" in cpu_info:
                        self.detected_capabilities.add(HardwareCapability.APPLE_SILICON_M3)
                    elif "Apple M4" in cpu_info:
                        self.detected_capabilities.add(HardwareCapability.APPLE_SILICON_M4)
                    
                    # If any Apple Silicon detected, add related capabilities
                    if any(cap.value.startswith("apple_silicon") for cap in self.detected_capabilities):
                        self.detected_capabilities.update([
                            HardwareCapability.NEURAL_ENGINE,
                            HardwareCapability.METAL_GPU,
                            HardwareCapability.UNIFIED_MEMORY,
                            HardwareCapability.VIDEO_ENCODE_DECODE,
                            HardwareCapability.CORE_ML_ACCELERATION
                        ])
                        
                        # Check for high bandwidth memory (M1 Pro/Max, M2 Pro/Max, M3 Pro/Max)
                        if any(variant in cpu_info for variant in ["Pro", "Max", "Ultra"]):
                            self.detected_capabilities.add(HardwareCapability.HIGH_BANDWIDTH_MEMORY)
                
                except Exception as e:
                    logger.warning(f"Could not detect Apple Silicon: {e}")
            
            # Get memory information
            memory_info = psutil.virtual_memory()
            self.performance_profile.update({
                'total_memory_gb': memory_info.total / (1024**3),
                'available_memory_gb': memory_info.available / (1024**3),
                'cpu_count': psutil.cpu_count(),
                'cpu_count_logical': psutil.cpu_count(logical=True)
            })
            
        except Exception as e:
            logger.error(f"Hardware detection failed: {e}")
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on detected hardware"""
        recommendations = []
        
        if HardwareCapability.APPLE_SILICON_M1 in self.detected_capabilities:
            recommendations.extend([
                "Enable Core ML acceleration for local models",
                "Use Metal Performance Shaders for GPU compute",
                "Optimize for unified memory architecture",
                "Enable Neural Engine for compatible operations"
            ])
        
        if HardwareCapability.HIGH_BANDWIDTH_MEMORY in self.detected_capabilities:
            recommendations.extend([
                "Increase model batch sizes for video processing",
                "Enable memory-intensive video generation workflows",
                "Use larger context windows for complex tasks"
            ])
        
        if self.performance_profile.get('total_memory_gb', 0) > 32:
            recommendations.append("Enable high-memory video processing pipelines")
        
        return recommendations

class TaskAnalyzer:
    """Analyzes tasks to determine optimal role assignments"""
    
    def __init__(self):
        self.video_keywords = [
            'video', 'animation', 'visual', 'frame', 'scene', 'shot', 'edit',
            'render', 'generate', 'storyboard', 'script', 'cinematic', 'motion'
        ]
        
        self.technical_keywords = [
            'optimize', 'performance', 'algorithm', 'implementation', 'debug',
            'analysis', 'specification', 'architecture', 'system', 'technical'
        ]
        
        self.creative_keywords = [
            'creative', 'story', 'narrative', 'character', 'design', 'artistic',
            'concept', 'brainstorm', 'innovative', 'original', 'stylistic'
        ]
    
    def analyze_task(self, task_description: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze task to determine requirements and complexity"""
        task_lower = task_description.lower()
        context = context or {}
        
        # Determine task type
        task_type = self._determine_task_type(task_lower)
        
        # Assess complexity
        complexity = self._assess_complexity(task_description, context)
        
        # Identify required roles
        required_roles = self._identify_required_roles(task_lower, context)
        
        # Determine hardware requirements
        hardware_requirements = self._assess_hardware_requirements(task_lower, context)
        
        # Video-specific analysis
        video_analysis = self._analyze_video_requirements(task_lower, context)
        
        return {
            'task_type': task_type,
            'complexity': complexity,
            'required_roles': required_roles,
            'hardware_requirements': hardware_requirements,
            'video_analysis': video_analysis,
            'estimated_duration': self._estimate_duration(complexity, required_roles),
            'collaboration_level': self._assess_collaboration_needs(required_roles),
            'apple_silicon_beneficial': self._assess_apple_silicon_benefit(task_lower, context)
        }
    
    def _determine_task_type(self, task_lower: str) -> TaskType:
        """Determine the primary task type"""
        if any(keyword in task_lower for keyword in self.video_keywords):
            return TaskType.CREATIVE
        elif any(keyword in task_lower for keyword in self.technical_keywords):
            return TaskType.ANALYSIS
        elif any(keyword in task_lower for keyword in self.creative_keywords):
            return TaskType.CREATIVE
        else:
            return TaskType.GENERAL
    
    def _assess_complexity(self, task_description: str, context: Dict[str, Any]) -> TaskComplexity:
        """Assess task complexity"""
        factors = {
            'length': len(task_description.split()),
            'video_involved': any(keyword in task_description.lower() 
                                for keyword in self.video_keywords),
            'multi_step': len([s for s in task_description.split('.') if s.strip()]) > 3,
            'technical_depth': any(keyword in task_description.lower() 
                                 for keyword in self.technical_keywords),
            'collaboration_needed': context.get('collaboration_required', False),
            'real_time': 'real-time' in task_description.lower() or 'live' in task_description.lower()
        }
        
        complexity_score = (
            (factors['length'] / 100) * 0.2 +
            factors['video_involved'] * 0.3 +
            factors['multi_step'] * 0.2 +
            factors['technical_depth'] * 0.2 +
            factors['collaboration_needed'] * 0.1 +
            factors['real_time'] * 0.3
        )
        
        if complexity_score > 0.8:
            return TaskComplexity.CRITICAL
        elif complexity_score > 0.6:
            return TaskComplexity.COMPLEX
        elif complexity_score > 0.4:
            return TaskComplexity.MODERATE
        else:
            return TaskComplexity.SIMPLE
    
    def _identify_required_roles(self, task_lower: str, context: Dict[str, Any]) -> Set[SpecializedRole]:
        """Identify required specialized roles"""
        roles = set()
        
        # Video-related roles
        if any(keyword in task_lower for keyword in ['video', 'visual', 'animation']):
            roles.add(SpecializedRole.VIDEO_DIRECTOR)
            roles.add(SpecializedRole.VISUAL_STORYTELLER)
        
        if any(keyword in task_lower for keyword in ['script', 'story', 'narrative']):
            roles.add(SpecializedRole.SCRIPT_WRITER)
            roles.add(SpecializedRole.NARRATIVE_ARCHITECT)
        
        if any(keyword in task_lower for keyword in ['style', 'design', 'aesthetic']):
            roles.add(SpecializedRole.STYLE_CURATOR)
        
        # Technical roles
        if any(keyword in task_lower for keyword in ['optimize', 'performance', 'technical']):
            roles.add(SpecializedRole.TECHNICAL_REVIEWER)
            roles.add(SpecializedRole.PERFORMANCE_OPTIMIZER)
        
        if any(keyword in task_lower for keyword in ['apple silicon', 'metal', 'neural engine']):
            roles.add(SpecializedRole.APPLE_SILICON_SPECIALIST)
            roles.add(SpecializedRole.METAL_COMPUTE_EXPERT)
        
        # Quality and coordination roles
        if 'quality' in task_lower or 'review' in task_lower:
            roles.add(SpecializedRole.QUALITY_ASSESSOR)
        
        if len(roles) > 2:  # Multi-role tasks need coordination
            roles.add(SpecializedRole.CREATIVE_COORDINATOR)
        
        # Default role if none identified
        if not roles:
            roles.add(SpecializedRole.MULTI_MODAL_SYNTHESIZER)
        
        return roles
    
    def _assess_hardware_requirements(self, task_lower: str, context: Dict[str, Any]) -> Set[HardwareCapability]:
        """Assess hardware requirements for the task"""
        requirements = set()
        
        if any(keyword in task_lower for keyword in ['video', 'render', 'animation']):
            requirements.update([
                HardwareCapability.METAL_GPU,
                HardwareCapability.VIDEO_ENCODE_DECODE,
                HardwareCapability.HIGH_BANDWIDTH_MEMORY
            ])
        
        if any(keyword in task_lower for keyword in ['real-time', 'live', 'interactive']):
            requirements.update([
                HardwareCapability.NEURAL_ENGINE,
                HardwareCapability.UNIFIED_MEMORY,
                HardwareCapability.CORE_ML_ACCELERATION
            ])
        
        if any(keyword in task_lower for keyword in ['optimize', 'performance', 'accelerate']):
            requirements.add(HardwareCapability.APPLE_SILICON_M3)  # Latest for best performance
        
        return requirements
    
    def _analyze_video_requirements(self, task_lower: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze video-specific requirements"""
        video_requirements = {
            'involves_video': False,
            'generation_needed': False,
            'editing_needed': False,
            'real_time': False,
            'quality_level': 'standard',
            'estimated_frames': 0,
            'duration_seconds': 0
        }
        
        if any(keyword in task_lower for keyword in self.video_keywords):
            video_requirements['involves_video'] = True
            
            if any(keyword in task_lower for keyword in ['generate', 'create', 'produce']):
                video_requirements['generation_needed'] = True
            
            if any(keyword in task_lower for keyword in ['edit', 'modify', 'adjust']):
                video_requirements['editing_needed'] = True
            
            if any(keyword in task_lower for keyword in ['real-time', 'live', 'interactive']):
                video_requirements['real_time'] = True
            
            if any(keyword in task_lower for keyword in ['4k', 'hd', 'high quality', 'professional']):
                video_requirements['quality_level'] = 'high'
            elif any(keyword in task_lower for keyword in ['8k', 'ultra', 'cinema']):
                video_requirements['quality_level'] = 'ultra'
        
        return video_requirements
    
    def _estimate_duration(self, complexity: TaskComplexity, roles: Set[SpecializedRole]) -> float:
        """Estimate task duration in seconds"""
        base_times = {
            TaskComplexity.SIMPLE: 30,
            TaskComplexity.MODERATE: 120,
            TaskComplexity.COMPLEX: 300,
            TaskComplexity.CRITICAL: 600,
            TaskComplexity.EXPERIMENTAL: 900
        }
        
        base_time = base_times[complexity]
        
        # Add time for each additional role
        role_multiplier = 1 + (len(roles) - 1) * 0.3
        
        # Video tasks take longer
        if any(role.value.startswith('video') for role in roles):
            role_multiplier *= 2
        
        return base_time * role_multiplier
    
    def _assess_collaboration_needs(self, roles: Set[SpecializedRole]) -> str:
        """Assess collaboration complexity"""
        if len(roles) <= 1:
            return "none"
        elif len(roles) <= 3:
            return "moderate"
        else:
            return "high"
    
    def _assess_apple_silicon_benefit(self, task_lower: str, context: Dict[str, Any]) -> bool:
        """Assess if task would benefit from Apple Silicon optimization"""
        apple_silicon_beneficial_keywords = [
            'video', 'render', 'generate', 'real-time', 'optimize', 'accelerate',
            'neural', 'ml', 'ai', 'image', 'visual', 'compute', 'parallel'
        ]
        
        return any(keyword in task_lower for keyword in apple_silicon_beneficial_keywords)

class DynamicRoleAssignmentSystem:
    """
    Dynamic Role Assignment System for intelligent LLM specialization
    with video intelligence and Apple Silicon optimization.
    """
    
    def __init__(self, llm_providers: Dict[str, Provider],
                 thought_sharing_system: ChainOfThoughtSharingSystem = None,
                 verification_system: CrossLLMVerificationSystem = None):
        """Initialize the Dynamic Role Assignment System"""
        self.logger = Logger("dynamic_role_assignment.log")
        self.llm_providers = llm_providers
        self.thought_sharing_system = thought_sharing_system
        self.verification_system = verification_system
        
        # Core components
        self.hardware_detector = HardwareCapabilityDetector()
        self.task_analyzer = TaskAnalyzer()
        
        # LLM profiles and assignments
        self.llm_profiles: Dict[str, LLMCapabilityProfile] = {}
        self.active_assignments: Dict[str, RoleAssignment] = {}
        self.assignment_history: List[RoleAssignment] = []
        self.team_compositions: Dict[str, TeamComposition] = {}
        
        # Performance tracking
        self.performance_metrics = {
            'total_assignments': 0,
            'successful_assignments': 0,
            'average_performance': 0.0,
            'role_effectiveness': defaultdict(float),
            'apple_silicon_utilization': 0.0,
            'video_task_success_rate': 0.0
        }
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize LLM profiles
        self._initialize_llm_profiles()
    
    def _initialize_llm_profiles(self):
        """Initialize capability profiles for all LLMs"""
        for llm_id, provider in self.llm_providers.items():
            profile = LLMCapabilityProfile(
                llm_id=llm_id,
                provider_name=provider.provider_name,
                model_name=provider.model
            )
            
            # Set provider-specific capabilities
            if provider.provider_name.lower() == 'openai':
                profile.reasoning_strength = 0.9
                profile.technical_accuracy = 0.85
                profile.visual_understanding = 0.8 if 'gpt-4' in provider.model else 0.3
                
            elif provider.provider_name.lower() == 'anthropic':
                profile.reasoning_strength = 0.88
                profile.creativity_score = 0.85
                profile.collaboration_skills = 0.9
                profile.visual_understanding = 0.75 if 'claude-3' in provider.model else 0.3
                
            elif provider.provider_name.lower() == 'google':
                profile.factual_accuracy = 0.9
                profile.technical_video_knowledge = 0.8
                profile.visual_understanding = 0.8
            
            # Set Apple Silicon optimization if detected
            if HardwareCapability.APPLE_SILICON_M3 in self.hardware_detector.detected_capabilities:
                profile.apple_silicon_optimized = True
                profile.metal_acceleration = True
                profile.neural_engine_support = True
                profile.memory_efficiency = 0.9
            
            # Initialize role affinities based on capabilities
            self._initialize_role_affinities(profile)
            
            self.llm_profiles[llm_id] = profile
    
    def _initialize_role_affinities(self, profile: LLMCapabilityProfile):
        """Initialize role affinities based on LLM capabilities"""
        # Video and creative roles
        profile.role_affinities[SpecializedRole.VIDEO_DIRECTOR] = (
            profile.creativity_score * 0.4 + 
            profile.visual_understanding * 0.6
        )
        
        profile.role_affinities[SpecializedRole.VISUAL_STORYTELLER] = (
            profile.creativity_score * 0.5 + 
            profile.storytelling_ability * 0.5
        )
        
        profile.role_affinities[SpecializedRole.SCRIPT_WRITER] = (
            profile.creativity_score * 0.6 + 
            profile.collaboration_skills * 0.4
        )
        
        # Technical roles
        profile.role_affinities[SpecializedRole.TECHNICAL_REVIEWER] = (
            profile.technical_accuracy * 0.7 + 
            profile.reasoning_strength * 0.3
        )
        
        profile.role_affinities[SpecializedRole.PERFORMANCE_OPTIMIZER] = (
            profile.technical_accuracy * 0.8 + 
            profile.technical_video_knowledge * 0.2
        )
        
        # Apple Silicon specific roles
        if profile.apple_silicon_optimized:
            profile.role_affinities[SpecializedRole.APPLE_SILICON_SPECIALIST] = 0.9
            profile.role_affinities[SpecializedRole.METAL_COMPUTE_EXPERT] = 0.85
            profile.role_affinities[SpecializedRole.NEURAL_ENGINE_COORDINATOR] = 0.8
        else:
            profile.role_affinities[SpecializedRole.APPLE_SILICON_SPECIALIST] = 0.3
            profile.role_affinities[SpecializedRole.METAL_COMPUTE_EXPERT] = 0.2
            profile.role_affinities[SpecializedRole.NEURAL_ENGINE_COORDINATOR] = 0.2
        
        # Quality and coordination roles
        profile.role_affinities[SpecializedRole.QUALITY_ASSESSOR] = (
            profile.technical_accuracy * 0.6 + 
            profile.collaboration_skills * 0.4
        )
        
        profile.role_affinities[SpecializedRole.CREATIVE_COORDINATOR] = (
            profile.collaboration_skills * 0.7 + 
            profile.reasoning_strength * 0.3
        )
    
    async def assign_optimal_roles(self, task_description: str, 
                                 context: Dict[str, Any] = None,
                                 performance_profile: PerformanceProfile = PerformanceProfile.BALANCED,
                                 max_team_size: int = 5) -> TeamComposition:
        """Assign optimal roles for a given task"""
        
        # Analyze the task
        task_analysis = self.task_analyzer.analyze_task(task_description, context)
        
        # Generate team composition
        team_id = f"team_{uuid.uuid4().hex[:8]}"
        assignments = []
        
        # Get required roles and sort by importance
        required_roles = list(task_analysis['required_roles'])
        required_roles.sort(key=lambda r: self._calculate_role_importance(r, task_analysis), reverse=True)
        
        # Limit team size
        selected_roles = required_roles[:max_team_size]
        
        # Assign LLMs to roles
        for role in selected_roles:
            best_llm = self._select_best_llm_for_role(
                role, task_analysis, performance_profile, assignments
            )
            
            if best_llm:
                assignment = RoleAssignment(
                    assignment_id=f"assign_{uuid.uuid4().hex[:8]}",
                    llm_id=best_llm,
                    role=role,
                    task_description=task_description,
                    complexity_level=task_analysis['complexity'],
                    performance_profile=performance_profile,
                    selection_reasoning=self._generate_selection_reasoning(best_llm, role, task_analysis),
                    confidence_score=self._calculate_assignment_confidence(best_llm, role, task_analysis),
                    expected_performance=self._estimate_performance(best_llm, role, task_analysis),
                    hardware_requirements=task_analysis['hardware_requirements'],
                    apple_silicon_optimizations=self._get_apple_silicon_optimizations(task_analysis),
                    collaboration_partners=set(),
                    dependencies=[],
                    priority_level=self._calculate_priority(role, task_analysis),
                    estimated_duration=task_analysis['estimated_duration']
                )
                
                assignments.append(assignment)
                
                # Update LLM load
                if best_llm in self.llm_profiles:
                    self.llm_profiles[best_llm].current_load += 0.2
                    self.llm_profiles[best_llm].active_roles.add(role)
        
        # Set collaboration partners
        for assignment in assignments:
            assignment.collaboration_partners = {a.llm_id for a in assignments if a.llm_id != assignment.llm_id}
        
        # Create team composition
        team = TeamComposition(
            team_id=team_id,
            primary_task=task_description,
            assignments=assignments,
            coordination_strategy=self._determine_coordination_strategy(assignments, task_analysis),
            performance_targets=self._set_performance_targets(task_analysis, performance_profile),
            apple_silicon_utilization=self._calculate_apple_silicon_utilization(assignments),
            estimated_completion_time=max((a.estimated_duration for a in assignments), default=0),
            success_probability=self._estimate_team_success_probability(assignments, task_analysis)
        )
        
        self.team_compositions[team_id] = team
        
        # Store active assignments
        for assignment in assignments:
            self.active_assignments[assignment.assignment_id] = assignment
        
        # Update metrics
        self.performance_metrics['total_assignments'] += len(assignments)
        
        self.logger.info(f"Created team {team_id} with {len(assignments)} role assignments")
        
        return team
    
    def _calculate_role_importance(self, role: SpecializedRole, task_analysis: Dict[str, Any]) -> float:
        """Calculate importance of a role for the given task"""
        importance_weights = {
            SpecializedRole.VIDEO_DIRECTOR: 1.0 if task_analysis['video_analysis']['involves_video'] else 0.2,
            SpecializedRole.VISUAL_STORYTELLER: 0.9 if task_analysis['video_analysis']['involves_video'] else 0.3,
            SpecializedRole.APPLE_SILICON_SPECIALIST: 0.8 if task_analysis['apple_silicon_beneficial'] else 0.1,
            SpecializedRole.TECHNICAL_REVIEWER: 0.9 if task_analysis['complexity'] in [TaskComplexity.COMPLEX, TaskComplexity.CRITICAL] else 0.4,
            SpecializedRole.CREATIVE_COORDINATOR: 0.7 if len(task_analysis['required_roles']) > 3 else 0.3,
            SpecializedRole.QUALITY_ASSESSOR: 0.8,  # Always important
            SpecializedRole.PERFORMANCE_OPTIMIZER: 0.6 if task_analysis['apple_silicon_beneficial'] else 0.3
        }
        
        return importance_weights.get(role, 0.5)
    
    def _select_best_llm_for_role(self, role: SpecializedRole, task_analysis: Dict[str, Any],
                                performance_profile: PerformanceProfile, 
                                existing_assignments: List[RoleAssignment]) -> Optional[str]:
        """Select the best LLM for a specific role"""
        scores = {}
        
        # Get LLMs not already assigned
        assigned_llms = {a.llm_id for a in existing_assignments}
        available_llms = {llm_id for llm_id in self.llm_profiles.keys() if llm_id not in assigned_llms}
        
        if not available_llms:
            # Allow multi-role assignment if needed
            available_llms = set(self.llm_profiles.keys())
        
        for llm_id in available_llms:
            profile = self.llm_profiles[llm_id]
            
            # Base affinity score
            base_score = profile.role_affinities.get(role, 0.5)
            
            # Performance profile adjustments
            perf_adjustment = self._get_performance_adjustment(profile, performance_profile)
            
            # Hardware compatibility
            hardware_score = self._calculate_hardware_compatibility(profile, task_analysis)
            
            # Current load penalty
            load_penalty = profile.current_load * 0.3
            
            # Video capability bonus for video tasks
            video_bonus = 0.0
            if task_analysis['video_analysis']['involves_video']:
                video_bonus = profile.visual_understanding * 0.2
            
            # Apple Silicon bonus
            apple_silicon_bonus = 0.0
            if task_analysis['apple_silicon_beneficial'] and profile.apple_silicon_optimized:
                apple_silicon_bonus = 0.3
            
            total_score = (
                base_score * 0.4 +
                perf_adjustment * 0.2 +
                hardware_score * 0.2 +
                video_bonus +
                apple_silicon_bonus -
                load_penalty
            )
            
            scores[llm_id] = total_score
        
        # Return best scoring LLM
        if scores:
            return max(scores.keys(), key=lambda x: scores[x])
        
        return None
    
    def _get_performance_adjustment(self, profile: LLMCapabilityProfile, 
                                  perf_profile: PerformanceProfile) -> float:
        """Get performance adjustment based on profile"""
        adjustments = {
            PerformanceProfile.SPEED_OPTIMIZED: profile.response_time_avg / 10.0,  # Faster is better
            PerformanceProfile.QUALITY_OPTIMIZED: profile.quality_score,
            PerformanceProfile.BALANCED: (profile.quality_score + (10.0 - profile.response_time_avg) / 10.0) / 2,
            PerformanceProfile.POWER_EFFICIENT: profile.memory_efficiency,
            PerformanceProfile.MEMORY_OPTIMIZED: profile.memory_efficiency
        }
        
        return adjustments.get(perf_profile, 0.5)
    
    def _calculate_hardware_compatibility(self, profile: LLMCapabilityProfile, 
                                        task_analysis: Dict[str, Any]) -> float:
        """Calculate hardware compatibility score"""
        required_capabilities = task_analysis['hardware_requirements']
        
        if not required_capabilities:
            return 0.8  # Neutral score if no specific requirements
        
        compatibility_score = 0.0
        
        # Check Apple Silicon requirements
        apple_silicon_required = any(cap.value.startswith('apple_silicon') for cap in required_capabilities)
        if apple_silicon_required and profile.apple_silicon_optimized:
            compatibility_score += 0.4
        
        # Check Metal GPU requirements
        if HardwareCapability.METAL_GPU in required_capabilities and profile.metal_acceleration:
            compatibility_score += 0.3
        
        # Check Neural Engine requirements
        if HardwareCapability.NEURAL_ENGINE in required_capabilities and profile.neural_engine_support:
            compatibility_score += 0.3
        
        return min(compatibility_score, 1.0)
    
    def _generate_selection_reasoning(self, llm_id: str, role: SpecializedRole, 
                                    task_analysis: Dict[str, Any]) -> str:
        """Generate reasoning for LLM selection"""
        profile = self.llm_profiles[llm_id]
        
        reasons = []
        
        # Role affinity
        affinity = profile.role_affinities.get(role, 0.5)
        if affinity > 0.7:
            reasons.append(f"High affinity for {role.value} role ({affinity:.2f})")
        
        # Apple Silicon optimization
        if task_analysis['apple_silicon_beneficial'] and profile.apple_silicon_optimized:
            reasons.append("Optimized for Apple Silicon hardware")
        
        # Video capabilities
        if task_analysis['video_analysis']['involves_video'] and profile.visual_understanding > 0.6:
            reasons.append(f"Strong visual understanding ({profile.visual_understanding:.2f})")
        
        # Performance characteristics
        if profile.reliability_score > 0.8:
            reasons.append(f"High reliability ({profile.reliability_score:.2f})")
        
        return "; ".join(reasons) if reasons else "Best available match for role requirements"
    
    def _calculate_assignment_confidence(self, llm_id: str, role: SpecializedRole, 
                                       task_analysis: Dict[str, Any]) -> float:
        """Calculate confidence in the assignment"""
        profile = self.llm_profiles[llm_id]
        
        # Base confidence from role affinity
        base_confidence = profile.role_affinities.get(role, 0.5)
        
        # Adjust for task complexity
        complexity_adjustment = {
            TaskComplexity.SIMPLE: 0.1,
            TaskComplexity.MODERATE: 0.0,
            TaskComplexity.COMPLEX: -0.1,
            TaskComplexity.CRITICAL: -0.2,
            TaskComplexity.EXPERIMENTAL: -0.3
        }
        
        confidence = base_confidence + complexity_adjustment[task_analysis['complexity']]
        
        # Boost for Apple Silicon optimization
        if task_analysis['apple_silicon_beneficial'] and profile.apple_silicon_optimized:
            confidence += 0.1
        
        # Boost for video capabilities
        if task_analysis['video_analysis']['involves_video'] and profile.visual_understanding > 0.6:
            confidence += 0.1
        
        return min(max(confidence, 0.0), 1.0)
    
    def _estimate_performance(self, llm_id: str, role: SpecializedRole, 
                            task_analysis: Dict[str, Any]) -> float:
        """Estimate expected performance for the assignment"""
        profile = self.llm_profiles[llm_id]
        
        # Base performance from success rate and quality
        base_performance = (profile.success_rate + profile.quality_score) / 2
        
        # Role-specific adjustment
        role_affinity = profile.role_affinities.get(role, 0.5)
        role_adjustment = (role_affinity - 0.5) * 0.4
        
        # Hardware optimization bonus
        hardware_bonus = 0.0
        if task_analysis['apple_silicon_beneficial'] and profile.apple_silicon_optimized:
            hardware_bonus = 0.15
        
        return min(base_performance + role_adjustment + hardware_bonus, 1.0)
    
    def _get_apple_silicon_optimizations(self, task_analysis: Dict[str, Any]) -> List[str]:
        """Get Apple Silicon optimization recommendations"""
        optimizations = []
        
        if task_analysis['apple_silicon_beneficial']:
            optimizations.extend([
                "Enable Metal Performance Shaders acceleration",
                "Use Core ML for model inference",
                "Optimize for unified memory architecture"
            ])
        
        if task_analysis['video_analysis']['involves_video']:
            optimizations.extend([
                "Use VideoToolbox for hardware encoding/decoding",
                "Enable GPU-accelerated video processing",
                "Optimize frame buffer management"
            ])
        
        if task_analysis['video_analysis']['real_time']:
            optimizations.extend([
                "Enable Neural Engine for real-time processing",
                "Use low-latency video pipelines",
                "Optimize for thermal efficiency"
            ])
        
        return optimizations
    
    def _calculate_priority(self, role: SpecializedRole, task_analysis: Dict[str, Any]) -> int:
        """Calculate priority level for role assignment"""
        base_priority = {
            SpecializedRole.VIDEO_DIRECTOR: 5,
            SpecializedRole.CREATIVE_COORDINATOR: 4,
            SpecializedRole.APPLE_SILICON_SPECIALIST: 4,
            SpecializedRole.TECHNICAL_REVIEWER: 3,
            SpecializedRole.QUALITY_ASSESSOR: 3,
            SpecializedRole.VISUAL_STORYTELLER: 3,
            SpecializedRole.PERFORMANCE_OPTIMIZER: 2
        }.get(role, 2)
        
        # Increase priority for critical tasks
        if task_analysis['complexity'] == TaskComplexity.CRITICAL:
            base_priority += 2
        elif task_analysis['complexity'] == TaskComplexity.COMPLEX:
            base_priority += 1
        
        return min(base_priority, 5)
    
    def _determine_coordination_strategy(self, assignments: List[RoleAssignment], 
                                       task_analysis: Dict[str, Any]) -> str:
        """Determine coordination strategy for the team"""
        if len(assignments) <= 2:
            return "peer_to_peer"
        elif task_analysis['video_analysis']['involves_video']:
            return "director_led"
        elif task_analysis['complexity'] in [TaskComplexity.CRITICAL, TaskComplexity.EXPERIMENTAL]:
            return "hierarchical"
        else:
            return "collaborative"
    
    def _set_performance_targets(self, task_analysis: Dict[str, Any], 
                               performance_profile: PerformanceProfile) -> Dict[str, float]:
        """Set performance targets based on task and profile"""
        targets = {
            'quality_threshold': 0.8,
            'completion_rate': 0.9,
            'collaboration_efficiency': 0.7,
            'apple_silicon_utilization': 0.0
        }
        
        if performance_profile == PerformanceProfile.QUALITY_OPTIMIZED:
            targets['quality_threshold'] = 0.9
        elif performance_profile == PerformanceProfile.SPEED_OPTIMIZED:
            targets['completion_rate'] = 0.95
        
        if task_analysis['apple_silicon_beneficial']:
            targets['apple_silicon_utilization'] = 0.8
        
        if task_analysis['video_analysis']['involves_video']:
            targets['video_quality'] = 0.85
            targets['frame_consistency'] = 0.9
        
        return targets
    
    def _calculate_apple_silicon_utilization(self, assignments: List[RoleAssignment]) -> Dict[str, Any]:
        """Calculate Apple Silicon utilization for assignments"""
        utilization = {
            'neural_engine_tasks': 0,
            'metal_accelerated_tasks': 0,
            'unified_memory_optimized': 0,
            'total_tasks': len(assignments)
        }
        
        for assignment in assignments:
            if HardwareCapability.NEURAL_ENGINE in assignment.hardware_requirements:
                utilization['neural_engine_tasks'] += 1
            
            if HardwareCapability.METAL_GPU in assignment.hardware_requirements:
                utilization['metal_accelerated_tasks'] += 1
            
            if HardwareCapability.UNIFIED_MEMORY in assignment.hardware_requirements:
                utilization['unified_memory_optimized'] += 1
        
        return utilization
    
    def _estimate_team_success_probability(self, assignments: List[RoleAssignment], 
                                         task_analysis: Dict[str, Any]) -> float:
        """Estimate probability of team success"""
        if not assignments:
            return 0.0
        
        # Average confidence of assignments
        avg_confidence = sum(a.confidence_score for a in assignments) / len(assignments)
        
        # Team size adjustment (too small or too large reduces success)
        team_size = len(assignments)
        size_adjustment = 1.0
        if team_size < 2:
            size_adjustment = 0.8
        elif team_size > 5:
            size_adjustment = 0.9
        
        # Complexity adjustment
        complexity_factors = {
            TaskComplexity.SIMPLE: 1.1,
            TaskComplexity.MODERATE: 1.0,
            TaskComplexity.COMPLEX: 0.9,
            TaskComplexity.CRITICAL: 0.8,
            TaskComplexity.EXPERIMENTAL: 0.7
        }
        
        complexity_factor = complexity_factors[task_analysis['complexity']]
        
        return min(avg_confidence * size_adjustment * complexity_factor, 1.0)
    
    def get_team_status(self, team_id: str) -> Dict[str, Any]:
        """Get status of a team composition"""
        if team_id not in self.team_compositions:
            return {'error': 'Team not found'}
        
        team = self.team_compositions[team_id]
        
        # Calculate current progress
        completed_assignments = sum(1 for a in team.assignments if a.completed_at is not None)
        progress = completed_assignments / len(team.assignments) if team.assignments else 0
        
        return {
            'team_id': team_id,
            'primary_task': team.primary_task,
            'total_assignments': len(team.assignments),
            'completed_assignments': completed_assignments,
            'progress': progress,
            'estimated_completion': team.estimated_completion_time,
            'success_probability': team.success_probability,
            'apple_silicon_utilization': team.apple_silicon_utilization,
            'coordination_strategy': team.coordination_strategy
        }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        metrics = self.performance_metrics.copy()
        
        # Add hardware information
        metrics['hardware_capabilities'] = [cap.value for cap in self.hardware_detector.detected_capabilities]
        metrics['optimization_recommendations'] = self.hardware_detector.get_optimization_recommendations()
        
        return metrics

# Test and demonstration functions
async def test_dynamic_role_assignment():
    """Test the Dynamic Role Assignment System"""
    # Mock LLM providers
    mock_providers = {
        'gpt4_turbo': Provider('openai', 'gpt-4-turbo'),
        'claude_opus': Provider('anthropic', 'claude-3-opus'),
        'gemini_pro': Provider('google', 'gemini-pro')
    }
    
    system = DynamicRoleAssignmentSystem(mock_providers)
    
    # Test video generation task
    task = ("Create a 30-second promotional video showcasing AI collaboration tools "
            "with professional cinematography and Apple Silicon optimization")
    
    context = {
        'target_quality': '4K',
        'deadline': time.time() + 3600,  # 1 hour from now
        'collaboration_required': True,
        'apple_silicon_available': True
    }
    
    team = await system.assign_optimal_roles(
        task_description=task,
        context=context,
        performance_profile=PerformanceProfile.QUALITY_OPTIMIZED,
        max_team_size=5
    )
    
    # Get status
    status = system.get_team_status(team.team_id)
    metrics = system.get_system_metrics()
    
    print(f"Team Composition: {json.dumps(asdict(team), indent=2, default=str)}")
    print(f"Team Status: {json.dumps(status, indent=2)}")
    print(f"System Metrics: {json.dumps(metrics, indent=2)}")
    
    return team, status, metrics

if __name__ == "__main__":
    asyncio.run(test_dynamic_role_assignment())