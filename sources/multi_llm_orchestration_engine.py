#!/usr/bin/env python3
"""
Multi-LLM Orchestration Engine (MLACS-001)
===================================================

* Purpose: Core orchestration system for coordinating multiple LLM instances in collaborative tasks
* Issues & Complexity Summary: Complex multi-agent coordination with dynamic assignment and state management
* Key Complexity Drivers:
  - Logic Scope (Est. LoC): ~1200
  - Core Algorithm Complexity: High
  - Dependencies: 5 New, 2 Mod
  - State Management Complexity: Very High
  - Novelty/Uncertainty Factor: High
* AI Pre-Task Self-Assessment (Est. Solution Difficulty %): 85%
* Problem Estimate (Inherent Problem Difficulty %): 90%
* Initial Code Complexity Estimate %: 85%
* Justification for Estimates: Sophisticated multi-LLM coordination with real-time state management
* Final Code Complexity (Actual %): TBD
* Overall Result Score (Success & Quality %): TBD
* Key Variances/Learnings: TBD
* Last Updated: 2025-06-01
"""

import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable, Set
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor

# AgenticSeek integration
try:
    from sources.utility import pretty_print, animate_thinking
    from sources.llm_provider import Provider
except ImportError:
    # Fallback for testing
    def pretty_print(msg, color="info"):
        print(f"[{color.upper()}] {msg}")
    def animate_thinking(msg, color="info"):
        print(f"[{color.upper()}] {msg}")


class LLMCapability(Enum):
    """LLM capability categories for dynamic assignment"""
    REASONING = "reasoning"
    CREATIVITY = "creativity"
    ANALYSIS = "analysis"
    CODING = "coding"
    FACTUAL_LOOKUP = "factual_lookup"
    SYNTHESIS = "synthesis"
    VERIFICATION = "verification"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    CONVERSATION = "conversation"


class TaskComplexity(Enum):
    """Task complexity levels for assignment optimization"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"
    EXPERT = "expert"


class CollaborationMode(Enum):
    """Available collaboration modes"""
    MASTER_SLAVE = "master_slave"
    PEER_TO_PEER = "peer_to_peer"
    HYBRID = "hybrid"
    COMPETITIVE = "competitive"


class LLMInstanceStatus(Enum):
    """Status of individual LLM instances"""
    IDLE = "idle"
    BUSY = "busy"
    THINKING = "thinking"
    COMMUNICATING = "communicating"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class LLMInstance:
    """Represents a single LLM instance in the coordination system"""
    instance_id: str
    provider: str
    model: str
    capabilities: Set[LLMCapability]
    max_tokens: int = 4000
    cost_per_token: float = 0.0001
    response_time_avg: float = 2.0
    quality_score: float = 0.8
    status: LLMInstanceStatus = LLMInstanceStatus.IDLE
    current_task_id: Optional[str] = None
    provider_instance: Optional[Any] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_used: int = 0
    
    def get_success_rate(self) -> float:
        """Calculate success rate for this instance"""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    def get_efficiency_score(self) -> float:
        """Calculate efficiency score based on speed, quality, and cost"""
        speed_score = max(0, 1 - (self.response_time_avg / 10.0))  # Normalize to 10s max
        cost_score = max(0, 1 - (self.cost_per_token / 0.001))     # Normalize to $0.001 max
        return (speed_score + self.quality_score + cost_score + self.get_success_rate()) / 4


@dataclass
class TaskRequest:
    """Represents a task request for LLM coordination"""
    task_id: str
    user_query: str
    required_capabilities: List[LLMCapability]
    estimated_complexity: TaskComplexity
    collaboration_mode: CollaborationMode
    max_response_time: float = 30.0
    priority: int = 1  # 1=low, 5=critical
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class CoordinationSession:
    """Represents an active coordination session"""
    session_id: str
    task_request: TaskRequest
    assigned_instances: List[LLMInstance]
    master_instance: Optional[LLMInstance] = None
    collaboration_mode: CollaborationMode = CollaborationMode.PEER_TO_PEER
    status: str = "initializing"
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    messages: List[Dict[str, Any]] = field(default_factory=list)
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    final_result: Optional[Dict[str, Any]] = None
    quality_metrics: Dict[str, float] = field(default_factory=dict)


class TaskComplexityAnalyzer:
    """Analyzes task complexity and required capabilities"""
    
    def __init__(self):
        self.complexity_indicators = {
            TaskComplexity.SIMPLE: {
                "keywords": ["what", "when", "where", "simple", "basic", "quick"],
                "max_words": 20,
                "capabilities": [LLMCapability.FACTUAL_LOOKUP, LLMCapability.CONVERSATION]
            },
            TaskComplexity.MODERATE: {
                "keywords": ["explain", "describe", "compare", "analyze", "how"],
                "max_words": 100,
                "capabilities": [LLMCapability.ANALYSIS, LLMCapability.REASONING, LLMCapability.SYNTHESIS]
            },
            TaskComplexity.COMPLEX: {
                "keywords": ["create", "design", "strategy", "complex", "multiple", "comprehensive"],
                "max_words": 300,
                "capabilities": [LLMCapability.CREATIVITY, LLMCapability.REASONING, LLMCapability.SYNTHESIS]
            },
            TaskComplexity.VERY_COMPLEX: {
                "keywords": ["research", "investigate", "multi-step", "detailed", "thorough"],
                "max_words": 500,
                "capabilities": [LLMCapability.ANALYSIS, LLMCapability.REASONING, LLMCapability.VERIFICATION]
            },
            TaskComplexity.EXPERT: {
                "keywords": ["expert", "professional", "advanced", "technical", "specialized"],
                "max_words": 1000,
                "capabilities": [LLMCapability.CODING, LLMCapability.ANALYSIS, LLMCapability.VERIFICATION]
            }
        }
    
    def analyze_task(self, query: str, context: Dict[str, Any] = None) -> TaskRequest:
        """Analyze a user query and determine complexity and requirements"""
        query_lower = query.lower()
        word_count = len(query.split())
        
        # Determine complexity
        complexity_scores = {}
        for complexity, indicators in self.complexity_indicators.items():
            score = 0
            
            # Keyword matching
            keyword_matches = sum(1 for keyword in indicators["keywords"] if keyword in query_lower)
            score += keyword_matches * 0.3
            
            # Word count consideration
            if word_count <= indicators["max_words"]:
                score += 0.4
            
            complexity_scores[complexity] = score
        
        estimated_complexity = max(complexity_scores.items(), key=lambda x: x[1])[0]
        
        # Determine required capabilities
        required_capabilities = []
        
        # Add capabilities based on query content
        if any(word in query_lower for word in ["code", "program", "script", "function"]):
            required_capabilities.append(LLMCapability.CODING)
        
        if any(word in query_lower for word in ["create", "generate", "write", "compose"]):
            required_capabilities.append(LLMCapability.CREATIVITY)
        
        if any(word in query_lower for word in ["analyze", "examine", "study", "investigate"]):
            required_capabilities.append(LLMCapability.ANALYSIS)
        
        if any(word in query_lower for word in ["verify", "check", "confirm", "validate"]):
            required_capabilities.append(LLMCapability.VERIFICATION)
        
        if any(word in query_lower for word in ["summarize", "brief", "overview", "summary"]):
            required_capabilities.append(LLMCapability.SUMMARIZATION)
        
        if any(word in query_lower for word in ["translate", "translation", "language"]):
            required_capabilities.append(LLMCapability.TRANSLATION)
        
        # Default capabilities for estimated complexity
        if not required_capabilities:
            required_capabilities = self.complexity_indicators[estimated_complexity]["capabilities"]
        
        # Determine collaboration mode
        collaboration_mode = CollaborationMode.PEER_TO_PEER
        if estimated_complexity in [TaskComplexity.VERY_COMPLEX, TaskComplexity.EXPERT]:
            collaboration_mode = CollaborationMode.MASTER_SLAVE
        elif word_count > 200:
            collaboration_mode = CollaborationMode.HYBRID
        
        task_id = str(uuid.uuid4())
        
        return TaskRequest(
            task_id=task_id,
            user_query=query,
            required_capabilities=required_capabilities,
            estimated_complexity=estimated_complexity,
            collaboration_mode=collaboration_mode,
            context=context or {},
            metadata={
                "word_count": word_count,
                "complexity_scores": {k.value: v for k, v in complexity_scores.items()}
            }
        )


class LLMInstanceManager:
    """Manages LLM instances and their lifecycle"""
    
    def __init__(self):
        self.instances: Dict[str, LLMInstance] = {}
        self.provider_configs = {
            "openai": {
                "models": ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"],
                "capabilities": [LLMCapability.REASONING, LLMCapability.CREATIVITY, LLMCapability.CODING],
                "cost_per_token": 0.0001
            },
            "anthropic": {
                "models": ["claude-3-sonnet", "claude-3-opus", "claude-3-haiku"],
                "capabilities": [LLMCapability.REASONING, LLMCapability.ANALYSIS, LLMCapability.SYNTHESIS],
                "cost_per_token": 0.00015
            },
            "google": {
                "models": ["gemini-pro", "gemini-ultra"],
                "capabilities": [LLMCapability.FACTUAL_LOOKUP, LLMCapability.ANALYSIS, LLMCapability.TRANSLATION],
                "cost_per_token": 0.00008
            },
            "ollama": {
                "models": ["llama2", "mistral", "codellama"],
                "capabilities": [LLMCapability.CODING, LLMCapability.CONVERSATION, LLMCapability.REASONING],
                "cost_per_token": 0.0  # Local models
            }
        }
    
    async def register_instance(self, provider: str, model: str, custom_config: Dict = None) -> str:
        """Register a new LLM instance"""
        instance_id = f"{provider}_{model}_{str(uuid.uuid4())[:8]}"
        
        config = self.provider_configs.get(provider, {})
        capabilities = set(config.get("capabilities", [LLMCapability.CONVERSATION]))
        
        if custom_config:
            capabilities.update(custom_config.get("capabilities", []))
        
        try:
            # Initialize provider instance
            provider_instance = Provider(provider, model)
            
            instance = LLMInstance(
                instance_id=instance_id,
                provider=provider,
                model=model,
                capabilities=capabilities,
                cost_per_token=config.get("cost_per_token", 0.0001),
                provider_instance=provider_instance
            )
            
            self.instances[instance_id] = instance
            pretty_print(f"Registered LLM instance: {instance_id}", color="success")
            
            return instance_id
            
        except Exception as e:
            pretty_print(f"Failed to register instance {instance_id}: {str(e)}", color="failure")
            raise
    
    def get_available_instances(self, capabilities: List[LLMCapability] = None) -> List[LLMInstance]:
        """Get available instances, optionally filtered by capabilities"""
        available = [
            instance for instance in self.instances.values()
            if instance.status == LLMInstanceStatus.IDLE
        ]
        
        if capabilities:
            capability_set = set(capabilities)
            available = [
                instance for instance in available
                if capability_set.intersection(instance.capabilities)
            ]
        
        # Sort by efficiency score
        available.sort(key=lambda x: x.get_efficiency_score(), reverse=True)
        
        return available
    
    def get_instance_stats(self) -> Dict[str, Any]:
        """Get statistics about all instances"""
        total = len(self.instances)
        idle = sum(1 for i in self.instances.values() if i.status == LLMInstanceStatus.IDLE)
        busy = sum(1 for i in self.instances.values() if i.status == LLMInstanceStatus.BUSY)
        error = sum(1 for i in self.instances.values() if i.status == LLMInstanceStatus.ERROR)
        
        return {
            "total_instances": total,
            "idle": idle,
            "busy": busy,
            "error": error,
            "providers": list(set(i.provider for i in self.instances.values())),
            "average_efficiency": sum(i.get_efficiency_score() for i in self.instances.values()) / max(total, 1)
        }


class MultiLLMOrchestrationEngine:
    """Main orchestration engine for coordinating multiple LLMs"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.task_analyzer = TaskComplexityAnalyzer()
        self.instance_manager = LLMInstanceManager()
        self.active_sessions: Dict[str, CoordinationSession] = {}
        self.session_history: List[CoordinationSession] = []
        
        # Performance tracking
        self.metrics = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "average_response_time": 0.0,
            "total_tokens_used": 0,
            "total_cost": 0.0,
            "start_time": datetime.now()
        }
        
        # Executor for concurrent operations
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        pretty_print("Multi-LLM Orchestration Engine initialized", color="success")
    
    async def initialize_default_instances(self) -> Dict[str, str]:
        """Initialize default LLM instances for testing"""
        default_configs = [
            {"provider": "ollama", "model": "llama2"},
            {"provider": "test", "model": "test-model-1"},  # Fallback for testing
            {"provider": "test", "model": "test-model-2"}   # Fallback for testing
        ]
        
        registered_instances = {}
        
        for config in default_configs:
            try:
                instance_id = await self.instance_manager.register_instance(
                    config["provider"], 
                    config["model"]
                )
                registered_instances[f"{config['provider']}_{config['model']}"] = instance_id
            except Exception as e:
                pretty_print(f"Failed to register {config['provider']}_{config['model']}: {str(e)}", color="warning")
        
        return registered_instances
    
    def analyze_task_requirements(self, query: str, context: Dict[str, Any] = None) -> TaskRequest:
        """Analyze a user query and determine task requirements"""
        return self.task_analyzer.analyze_task(query, context)
    
    def select_optimal_instances(
        self, 
        task_request: TaskRequest, 
        max_instances: int = 3
    ) -> List[LLMInstance]:
        """Select optimal LLM instances for a task"""
        available_instances = self.instance_manager.get_available_instances(
            task_request.required_capabilities
        )
        
        if not available_instances:
            raise ValueError("No available instances for task requirements")
        
        # Selection algorithm based on task complexity
        if task_request.estimated_complexity == TaskComplexity.SIMPLE:
            # Use single best instance
            selected = available_instances[:1]
        elif task_request.estimated_complexity in [TaskComplexity.MODERATE, TaskComplexity.COMPLEX]:
            # Use 2-3 instances for collaboration
            selected = available_instances[:min(max_instances, 3)]
        else:
            # Use all available instances for expert tasks
            selected = available_instances[:max_instances]
        
        # Ensure we have at least one instance
        if not selected and available_instances:
            selected = [available_instances[0]]
        
        return selected
    
    async def create_coordination_session(self, task_request: TaskRequest) -> CoordinationSession:
        """Create a new coordination session"""
        selected_instances = self.select_optimal_instances(task_request)
        
        if not selected_instances:
            raise ValueError("No suitable instances available for task")
        
        # Determine master instance for master-slave mode
        master_instance = None
        if task_request.collaboration_mode == CollaborationMode.MASTER_SLAVE:
            master_instance = max(selected_instances, key=lambda x: x.quality_score)
        
        session = CoordinationSession(
            session_id=str(uuid.uuid4()),
            task_request=task_request,
            assigned_instances=selected_instances,
            master_instance=master_instance,
            collaboration_mode=task_request.collaboration_mode
        )
        
        # Mark instances as busy
        for instance in selected_instances:
            instance.status = LLMInstanceStatus.BUSY
            instance.current_task_id = task_request.task_id
        
        self.active_sessions[session.session_id] = session
        
        pretty_print(
            f"Created coordination session {session.session_id} with {len(selected_instances)} instances",
            color="info"
        )
        
        return session
    
    async def coordinate_llm_responses(self, session: CoordinationSession) -> Dict[str, Any]:
        """Coordinate responses from multiple LLM instances"""
        session.status = "coordinating"
        start_time = time.time()
        
        try:
            if session.collaboration_mode == CollaborationMode.MASTER_SLAVE:
                result = await self._coordinate_master_slave(session)
            elif session.collaboration_mode == CollaborationMode.PEER_TO_PEER:
                result = await self._coordinate_peer_to_peer(session)
            else:
                result = await self._coordinate_hybrid(session)
            
            # Calculate metrics
            end_time = time.time()
            response_time = end_time - start_time
            
            session.end_time = datetime.now()
            session.status = "completed"
            session.final_result = result
            session.quality_metrics = {
                "response_time": response_time,
                "participating_instances": len(session.assigned_instances),
                "collaboration_mode": session.collaboration_mode.value
            }
            
            # Update global metrics
            self.metrics["total_tasks"] += 1
            self.metrics["successful_tasks"] += 1
            self.metrics["average_response_time"] = (
                (self.metrics["average_response_time"] * (self.metrics["total_tasks"] - 1) + response_time) 
                / self.metrics["total_tasks"]
            )
            
            return result
            
        except Exception as e:
            session.status = "failed"
            session.end_time = datetime.now()
            self.metrics["total_tasks"] += 1
            self.metrics["failed_tasks"] += 1
            
            pretty_print(f"Coordination failed for session {session.session_id}: {str(e)}", color="failure")
            raise
        
        finally:
            # Release instances
            for instance in session.assigned_instances:
                instance.status = LLMInstanceStatus.IDLE
                instance.current_task_id = None
                instance.last_used = datetime.now()
            
            # Move to history
            self.session_history.append(session)
            if session.session_id in self.active_sessions:
                del self.active_sessions[session.session_id]
    
    async def _coordinate_master_slave(self, session: CoordinationSession) -> Dict[str, Any]:
        """Coordinate using master-slave architecture"""
        master = session.master_instance
        slaves = [i for i in session.assigned_instances if i != master]
        
        pretty_print(f"Master-slave coordination: {master.instance_id} managing {len(slaves)} slaves", color="info")
        
        # Master analyzes and delegates sub-tasks
        master_prompt = f"""
        You are the master coordinator managing multiple AI assistants to solve this task:
        {session.task_request.user_query}
        
        You have {len(slaves)} assistant(s) available. Please:
        1. Break down the task into sub-tasks if needed
        2. Provide your initial analysis
        3. Specify what you need from other assistants (if any)
        
        Respond in JSON format with:
        {{
            "analysis": "your analysis",
            "subtasks": ["subtask1", "subtask2", ...],
            "delegation_needed": true/false,
            "response": "your response if no delegation needed"
        }}
        """
        
        # Get master's initial response
        master_response = await self._get_llm_response(master, master_prompt, session.session_id)
        
        try:
            master_analysis = json.loads(master_response)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "response": master_response,
                "coordination_mode": "master_slave",
                "instances_used": [master.instance_id],
                "notes": "Master provided direct response (JSON parsing failed)"
            }
        
        # If delegation is needed, coordinate with slaves
        if master_analysis.get("delegation_needed", False) and slaves:
            slave_responses = {}
            
            for i, slave in enumerate(slaves):
                if i < len(master_analysis.get("subtasks", [])):
                    subtask = master_analysis["subtasks"][i]
                    slave_prompt = f"""
                    As part of a coordinated response, please handle this specific subtask:
                    {subtask}
                    
                    Context from the original query:
                    {session.task_request.user_query}
                    
                    Provide a focused response for your assigned subtask.
                    """
                    
                    slave_response = await self._get_llm_response(slave, slave_prompt, session.session_id)
                    slave_responses[slave.instance_id] = {
                        "subtask": subtask,
                        "response": slave_response
                    }
            
            # Master synthesizes final response
            synthesis_prompt = f"""
            Based on the responses from your team, provide a comprehensive final answer:
            
            Original query: {session.task_request.user_query}
            Your analysis: {master_analysis.get('analysis', '')}
            
            Team responses:
            {json.dumps(slave_responses, indent=2)}
            
            Synthesize these into a coherent, comprehensive response.
            """
            
            final_response = await self._get_llm_response(master, synthesis_prompt, session.session_id)
            
            return {
                "response": final_response,
                "coordination_mode": "master_slave",
                "master_analysis": master_analysis,
                "slave_contributions": slave_responses,
                "instances_used": [i.instance_id for i in session.assigned_instances]
            }
        
        else:
            # Master handles the task independently
            return {
                "response": master_analysis.get("response", master_response),
                "coordination_mode": "master_slave",
                "instances_used": [master.instance_id],
                "master_analysis": master_analysis
            }
    
    async def _coordinate_peer_to_peer(self, session: CoordinationSession) -> Dict[str, Any]:
        """Coordinate using peer-to-peer architecture"""
        instances = session.assigned_instances
        pretty_print(f"Peer-to-peer coordination with {len(instances)} instances", color="info")
        
        # Get initial responses from all peers
        peer_responses = {}
        tasks = []
        
        for instance in instances:
            prompt = f"""
            You are collaborating with other AI assistants to answer this query:
            {session.task_request.user_query}
            
            Provide your perspective and analysis. If this is a complex task, focus on your strengths:
            - Analysis and reasoning
            - Creative solutions
            - Factual information
            - Verification and validation
            
            Your response will be combined with others to create a comprehensive answer.
            """
            
            task = self._get_llm_response(instance, prompt, session.session_id)
            tasks.append((instance, task))
        
        # Collect all responses
        for instance, task in tasks:
            try:
                response = await task
                peer_responses[instance.instance_id] = response
            except Exception as e:
                pretty_print(f"Error from instance {instance.instance_id}: {str(e)}", color="warning")
                peer_responses[instance.instance_id] = f"Error: {str(e)}"
        
        # Use the best-performing instance to synthesize responses
        best_instance = max(instances, key=lambda x: x.get_efficiency_score())
        
        synthesis_prompt = f"""
        You are synthesizing responses from multiple AI perspectives on this query:
        {session.task_request.user_query}
        
        Here are the different perspectives:
        {json.dumps(peer_responses, indent=2)}
        
        Create a comprehensive, well-structured response that:
        1. Incorporates the best insights from each perspective
        2. Resolves any contradictions
        3. Provides a complete answer to the original query
        4. Maintains coherence and flow
        
        Focus on creating the most accurate and helpful response possible.
        """
        
        synthesized_response = await self._get_llm_response(best_instance, synthesis_prompt, session.session_id)
        
        return {
            "response": synthesized_response,
            "coordination_mode": "peer_to_peer",
            "peer_contributions": peer_responses,
            "synthesizer": best_instance.instance_id,
            "instances_used": [i.instance_id for i in instances]
        }
    
    async def _coordinate_hybrid(self, session: CoordinationSession) -> Dict[str, Any]:
        """Coordinate using hybrid approach"""
        # For now, default to peer-to-peer
        # TODO: Implement more sophisticated hybrid coordination
        return await self._coordinate_peer_to_peer(session)
    
    async def _get_llm_response(
        self, 
        instance: LLMInstance, 
        prompt: str, 
        session_id: str
    ) -> str:
        """Get response from a specific LLM instance"""
        instance.status = LLMInstanceStatus.THINKING
        instance.total_requests += 1
        
        start_time = time.time()
        
        try:
            # For now, simulate response since we need provider integration
            # TODO: Replace with actual provider calls
            await asyncio.sleep(0.5)  # Simulate processing time
            
            response = f"Response from {instance.instance_id}: Processed '{prompt[:50]}...'"
            
            # Update instance metrics
            response_time = time.time() - start_time
            instance.response_time_avg = (
                (instance.response_time_avg * (instance.total_requests - 1) + response_time) 
                / instance.total_requests
            )
            instance.successful_requests += 1
            
            return response
            
        except Exception as e:
            instance.failed_requests += 1
            instance.status = LLMInstanceStatus.ERROR
            raise e
        
        finally:
            instance.status = LLMInstanceStatus.BUSY  # Will be set to IDLE by coordinator
    
    async def process_query(
        self, 
        query: str, 
        context: Dict[str, Any] = None,
        collaboration_mode: CollaborationMode = None
    ) -> Dict[str, Any]:
        """Main entry point for processing a query with LLM coordination"""
        pretty_print(f"Processing query: {query[:100]}...", color="info")
        
        # Analyze task requirements
        task_request = self.analyze_task_requirements(query, context)
        
        # Override collaboration mode if specified
        if collaboration_mode:
            task_request.collaboration_mode = collaboration_mode
        
        # Create coordination session
        session = await self.create_coordination_session(task_request)
        
        # Coordinate responses
        result = await self.coordinate_llm_responses(session)
        
        pretty_print(f"Query processed successfully in session {session.session_id}", color="success")
        
        return result
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "orchestration_engine": {
                "active_sessions": len(self.active_sessions),
                "total_sessions": len(self.session_history),
                "metrics": self.metrics
            },
            "instance_manager": self.instance_manager.get_instance_stats(),
            "uptime": str(datetime.now() - self.metrics["start_time"])
        }
    
    async def shutdown(self):
        """Gracefully shutdown the orchestration engine"""
        # Cancel active sessions
        for session in self.active_sessions.values():
            session.status = "cancelled"
            for instance in session.assigned_instances:
                instance.status = LLMInstanceStatus.IDLE
                instance.current_task_id = None
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        pretty_print("Multi-LLM Orchestration Engine shutdown complete", color="info")


# Example usage and testing functions
async def test_orchestration_engine():
    """Test the orchestration engine with sample queries"""
    engine = MultiLLMOrchestrationEngine()
    
    # Initialize default instances
    await engine.initialize_default_instances()
    
    # Test queries of varying complexity
    test_queries = [
        "What is the capital of France?",  # Simple
        "Explain the concept of machine learning and provide examples",  # Moderate
        "Design a comprehensive marketing strategy for a new tech startup",  # Complex
        "Research and analyze the economic impact of artificial intelligence on employment"  # Expert
    ]
    
    for query in test_queries:
        try:
            result = await engine.process_query(query)
            pretty_print(f"Query: {query}", color="info")
            pretty_print(f"Result: {result['response'][:200]}...", color="success")
            pretty_print(f"Mode: {result['coordination_mode']}, Instances: {len(result['instances_used'])}", color="status")
            print("-" * 80)
        except Exception as e:
            pretty_print(f"Error processing query '{query}': {str(e)}", color="failure")
    
    # Print system status
    status = engine.get_system_status()
    pretty_print(f"System Status: {json.dumps(status, indent=2)}", color="info")
    
    await engine.shutdown()


if __name__ == "__main__":
    asyncio.run(test_orchestration_engine())