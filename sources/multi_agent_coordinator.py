#!/usr/bin/env python3
"""
* Purpose: Multi-agent coordination system with peer review and consensus mechanisms
* Issues & Complexity Summary: Complex agent orchestration with concurrent execution and consensus building
* Key Complexity Drivers:
  - Logic Scope (Est. LoC): ~400
  - Core Algorithm Complexity: High
  - Dependencies: 5 New, 2 Mod
  - State Management Complexity: High
  - Novelty/Uncertainty Factor: Medium
* AI Pre-Task Self-Assessment (Est. Solution Difficulty %): 85%
* Problem Estimate (Inherent Problem Difficulty %): 90%
* Initial Code Complexity Estimate %: 85%
* Justification for Estimates: Complex agent coordination, concurrent execution, and consensus mechanisms
* Final Code Complexity (Actual %): TBD
* Overall Result Score (Success & Quality %): TBD
* Key Variances/Learnings: TBD
* Last Updated: 2025-01-06
"""

import asyncio
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid

if __name__ == "__main__":
    from agents.agent import Agent
    from utility import pretty_print
else:
    from sources.agents.agent import Agent
    from sources.utility import pretty_print

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentRole(Enum):
    """Agent role definitions for specialization"""
    BROWSER = "browser"
    CODER = "coder" 
    PLANNER = "planner"
    GENERAL = "general"
    FILE_MANAGER = "file_manager"
    REVIEWER = "reviewer"
    VALIDATOR = "validator"
    MANAGER = "manager"

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class ExecutionStatus(Enum):
    """Execution status tracking"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REVIEWED = "reviewed"
    VALIDATED = "validated"

@dataclass
class AgentResult:
    """Result from agent execution"""
    agent_id: str
    agent_role: AgentRole
    content: str
    confidence_score: float  # 0.0 - 1.0
    execution_time: float
    metadata: Dict[str, Any]
    timestamp: float
    status: ExecutionStatus = ExecutionStatus.COMPLETED

@dataclass
class PeerReview:
    """Peer review of agent result"""
    reviewer_id: str
    reviewer_role: AgentRole
    target_result_id: str
    review_score: float  # 0.0 - 1.0
    review_comments: str
    suggested_improvements: List[str]
    validation_passed: bool
    timestamp: float

@dataclass
class ConsensusResult:
    """Final consensus result from multiple agents"""
    primary_result: AgentResult
    peer_reviews: List[PeerReview]
    consensus_score: float
    final_content: str
    confidence_level: float
    execution_metadata: Dict[str, Any]
    total_processing_time: float

class MultiAgentCoordinator:
    """
    Production-ready multi-agent coordination system with peer review and consensus.
    Implements CrewAI-inspired patterns for robust agent orchestration.
    """
    
    def __init__(self, max_concurrent_agents: int = 3, enable_peer_review: bool = True):
        self.max_concurrent_agents = max_concurrent_agents
        self.enable_peer_review = enable_peer_review
        self.agents: Dict[AgentRole, Agent] = {}
        self.active_executions: Dict[str, Dict] = {}
        self.execution_history: List[ConsensusResult] = []
        
        # Agent specialization mapping
        self.agent_specializations = {
            AgentRole.BROWSER: ["web_search", "browsing", "scraping", "research"],
            AgentRole.CODER: ["coding", "programming", "debug", "development"],
            AgentRole.PLANNER: ["planning", "organization", "scheduling", "strategy"],
            AgentRole.GENERAL: ["conversation", "general", "questions", "help"],
            AgentRole.FILE_MANAGER: ["files", "documents", "storage", "management"],
            AgentRole.REVIEWER: ["review", "validation", "quality", "assessment"],
            AgentRole.VALIDATOR: ["validation", "verification", "testing", "checking"],
            AgentRole.MANAGER: ["coordination", "management", "oversight", "orchestration"]
        }
        
        # Performance thresholds
        self.confidence_threshold = 0.7
        self.consensus_threshold = 0.8
        self.max_execution_time = 30.0  # seconds
        
        logger.info(f"MultiAgentCoordinator initialized - max_concurrent: {max_concurrent_agents}, peer_review: {enable_peer_review}")

    def register_agent(self, role: AgentRole, agent: Agent) -> None:
        """Register an agent with a specific role"""
        self.agents[role] = agent
        logger.info(f"Registered agent: {role.value}")

    def select_primary_agent(self, query: str, task_type: str = None) -> AgentRole:
        """
        Select the most appropriate primary agent for the task
        Uses keyword matching and task type analysis
        """
        query_lower = query.lower()
        task_keywords = query_lower.split()
        
        # Score each agent based on specialization match
        agent_scores = {}
        for role, specializations in self.agent_specializations.items():
            if role not in self.agents:
                continue
                
            score = 0
            for keyword in task_keywords:
                for spec in specializations:
                    if keyword in spec or spec in keyword:
                        score += 1
            
            # Boost score for exact task type match
            if task_type and task_type.lower() in specializations:
                score += 5
                
            agent_scores[role] = score
        
        # Select agent with highest score, default to GENERAL
        if not agent_scores:
            return AgentRole.GENERAL
            
        selected_role = max(agent_scores, key=agent_scores.get)
        logger.info(f"Selected primary agent: {selected_role.value} (score: {agent_scores[selected_role]})")
        return selected_role

    def select_peer_reviewers(self, primary_role: AgentRole, task_complexity: str = "medium") -> List[AgentRole]:
        """
        Select appropriate peer reviewers based on task and primary agent
        """
        if not self.enable_peer_review:
            return []
            
        available_reviewers = [role for role in self.agents.keys() if role != primary_role]
        
        # For high complexity tasks, include specialized reviewers
        if task_complexity == "high":
            preferred_reviewers = [AgentRole.REVIEWER, AgentRole.VALIDATOR, AgentRole.MANAGER]
            reviewers = [role for role in preferred_reviewers if role in available_reviewers]
            
            # Add one domain expert if available
            if primary_role in [AgentRole.CODER, AgentRole.BROWSER]:
                domain_experts = [AgentRole.CODER, AgentRole.BROWSER]
                for expert in domain_experts:
                    if expert in available_reviewers and expert not in reviewers:
                        reviewers.append(expert)
                        break
        else:
            # For medium/low complexity, just use general reviewer
            reviewers = [role for role in [AgentRole.REVIEWER, AgentRole.GENERAL] if role in available_reviewers][:2]
        
        return reviewers[:2]  # Limit to 2 reviewers for performance

    async def execute_with_peer_review(self, query: str, task_type: str = None, priority: TaskPriority = TaskPriority.MEDIUM) -> ConsensusResult:
        """
        Main execution method with peer review and consensus building
        """
        execution_id = str(uuid.uuid4())
        start_time = time.time()
        
        logger.info(f"Starting multi-agent execution: {execution_id}")
        
        try:
            # Phase 1: Select primary agent and execute
            primary_role = self.select_primary_agent(query, task_type)
            primary_result = await self._execute_primary_agent(primary_role, query, execution_id)
            
            if not self.enable_peer_review:
                return ConsensusResult(
                    primary_result=primary_result,
                    peer_reviews=[],
                    consensus_score=primary_result.confidence_score,
                    final_content=primary_result.content,
                    confidence_level=primary_result.confidence_score,
                    execution_metadata={"execution_id": execution_id},
                    total_processing_time=time.time() - start_time
                )
            
            # Phase 2: Peer review (if enabled and confidence below threshold)
            peer_reviews = []
            if primary_result.confidence_score < self.confidence_threshold:
                peer_reviewers = self.select_peer_reviewers(primary_role, "high" if priority.value >= 3 else "medium")
                peer_reviews = await self._execute_peer_reviews(peer_reviewers, primary_result, query)
            
            # Phase 3: Build consensus
            consensus = self._build_consensus(primary_result, peer_reviews)
            consensus.total_processing_time = time.time() - start_time
            
            # Store in execution history
            self.execution_history.append(consensus)
            
            logger.info(f"Multi-agent execution completed: {execution_id} in {consensus.total_processing_time:.2f}s")
            return consensus
            
        except Exception as e:
            logger.error(f"Multi-agent execution failed: {execution_id} - {str(e)}")
            raise

    async def _execute_primary_agent(self, role: AgentRole, query: str, execution_id: str) -> AgentResult:
        """Execute primary agent task"""
        if role not in self.agents:
            raise ValueError(f"Agent {role.value} not registered")
        
        agent = self.agents[role]
        start_time = time.time()
        
        try:
            # Execute agent task
            response = await asyncio.wait_for(
                self._run_agent_task(agent, query),
                timeout=self.max_execution_time
            )
            
            execution_time = time.time() - start_time
            
            # Calculate confidence based on response quality and execution time
            confidence = self._calculate_confidence(response, execution_time)
            
            result = AgentResult(
                agent_id=f"{role.value}_{execution_id}",
                agent_role=role,
                content=response,
                confidence_score=confidence,
                execution_time=execution_time,
                metadata={"query": query, "execution_id": execution_id},
                timestamp=time.time(),
                status=ExecutionStatus.COMPLETED
            )
            
            logger.info(f"Primary agent {role.value} completed in {execution_time:.2f}s with confidence {confidence:.2f}")
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Primary agent {role.value} timed out")
            raise
        except Exception as e:
            logger.error(f"Primary agent {role.value} failed: {str(e)}")
            raise

    async def _execute_peer_reviews(self, reviewer_roles: List[AgentRole], primary_result: AgentResult, original_query: str) -> List[PeerReview]:
        """Execute peer reviews concurrently"""
        if not reviewer_roles:
            return []
        
        review_tasks = []
        for role in reviewer_roles:
            if role in self.agents:
                task = self._execute_single_review(role, primary_result, original_query)
                review_tasks.append(task)
        
        if not review_tasks:
            return []
        
        # Execute reviews concurrently
        reviews = await asyncio.gather(*review_tasks, return_exceptions=True)
        
        # Filter out exceptions and return valid reviews
        valid_reviews = [review for review in reviews if isinstance(review, PeerReview)]
        logger.info(f"Completed {len(valid_reviews)} peer reviews")
        
        return valid_reviews

    async def _execute_single_review(self, reviewer_role: AgentRole, primary_result: AgentResult, original_query: str) -> PeerReview:
        """Execute a single peer review"""
        agent = self.agents[reviewer_role]
        
        # Create review prompt
        review_prompt = f"""
        Please review the following response to the query: "{original_query}"
        
        Agent Response: {primary_result.content}
        Agent Confidence: {primary_result.confidence_score:.2f}
        
        Provide:
        1. A score from 0.0 to 1.0 for response quality
        2. Comments on accuracy, completeness, and usefulness
        3. Specific suggestions for improvement
        4. Whether this response passes validation (yes/no)
        
        Format your response as JSON with keys: score, comments, improvements, validation_passed
        """
        
        try:
            response = await asyncio.wait_for(
                self._run_agent_task(agent, review_prompt),
                timeout=15.0  # Shorter timeout for reviews
            )
            
            # Parse review response
            review_data = self._parse_review_response(response)
            
            return PeerReview(
                reviewer_id=f"{reviewer_role.value}_{uuid.uuid4()}",
                reviewer_role=reviewer_role,
                target_result_id=primary_result.agent_id,
                review_score=review_data.get("score", 0.5),
                review_comments=review_data.get("comments", ""),
                suggested_improvements=review_data.get("improvements", []),
                validation_passed=review_data.get("validation_passed", True),
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Review by {reviewer_role.value} failed: {str(e)}")
            # Return neutral review on failure
            return PeerReview(
                reviewer_id=f"{reviewer_role.value}_{uuid.uuid4()}",
                reviewer_role=reviewer_role,
                target_result_id=primary_result.agent_id,
                review_score=0.5,
                review_comments=f"Review failed: {str(e)}",
                suggested_improvements=[],
                validation_passed=True,
                timestamp=time.time()
            )

    def _build_consensus(self, primary_result: AgentResult, peer_reviews: List[PeerReview]) -> ConsensusResult:
        """Build consensus from primary result and peer reviews"""
        if not peer_reviews:
            return ConsensusResult(
                primary_result=primary_result,
                peer_reviews=[],
                consensus_score=primary_result.confidence_score,
                final_content=primary_result.content,
                confidence_level=primary_result.confidence_score,
                execution_metadata={"consensus_method": "single_agent"},
                total_processing_time=0.0
            )
        
        # Calculate weighted consensus score
        review_scores = [review.review_score for review in peer_reviews]
        avg_review_score = sum(review_scores) / len(review_scores) if review_scores else 0.5
        
        # Weight primary result higher, but consider peer feedback
        consensus_score = (primary_result.confidence_score * 0.6) + (avg_review_score * 0.4)
        
        # Check if validation passed from all reviewers
        all_validations_passed = all(review.validation_passed for review in peer_reviews)
        
        # Final confidence based on consensus and validation
        final_confidence = consensus_score if all_validations_passed else consensus_score * 0.8
        
        # Combine improvements from reviews
        all_improvements = []
        for review in peer_reviews:
            all_improvements.extend(review.suggested_improvements)
        
        # Build final content (could be enhanced with actual improvements)
        final_content = primary_result.content
        if all_improvements and consensus_score < self.consensus_threshold:
            final_content += f"\n\nSuggested improvements: {'; '.join(all_improvements[:3])}"
        
        return ConsensusResult(
            primary_result=primary_result,
            peer_reviews=peer_reviews,
            consensus_score=consensus_score,
            final_content=final_content,
            confidence_level=final_confidence,
            execution_metadata={
                "consensus_method": "peer_review",
                "review_count": len(peer_reviews),
                "validation_passed": all_validations_passed
            },
            total_processing_time=0.0
        )

    async def _run_agent_task(self, agent: Agent, query: str) -> str:
        """Run agent task with proper error handling"""
        try:
            # Assume agent has an async process method
            if hasattr(agent, 'process_async'):
                return await agent.process_async(query)
            else:
                # Fallback to sync method in thread pool
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, agent.process, query)
        except Exception as e:
            logger.error(f"Agent task failed: {str(e)}")
            return f"Error: {str(e)}"

    def _calculate_confidence(self, response: str, execution_time: float) -> float:
        """Calculate confidence score based on response quality and timing"""
        base_confidence = 0.7
        
        # Penalize very fast responses (likely errors) or very slow ones
        time_factor = 1.0
        if execution_time < 1.0:
            time_factor = 0.8
        elif execution_time > 20.0:
            time_factor = 0.9
        
        # Boost confidence for longer, more detailed responses
        length_factor = min(1.2, 0.8 + (len(response) / 1000))
        
        # Penalize error responses
        error_factor = 0.3 if "error" in response.lower() else 1.0
        
        confidence = min(1.0, base_confidence * time_factor * length_factor * error_factor)
        return round(confidence, 2)

    def _parse_review_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON review response with fallback"""
        try:
            # Try to extract JSON from response
            if "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
                return json.loads(json_str)
        except:
            pass
        
        # Fallback parsing
        score = 0.5
        comments = response[:200] if response else "No review provided"
        improvements = []
        validation_passed = True
        
        # Simple heuristics
        if "good" in response.lower() or "excellent" in response.lower():
            score = 0.8
        elif "poor" in response.lower() or "bad" in response.lower():
            score = 0.3
        
        if "improve" in response.lower():
            improvements = ["General improvements suggested"]
        
        return {
            "score": score,
            "comments": comments,
            "improvements": improvements,
            "validation_passed": validation_passed
        }

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.execution_history:
            return {"total_executions": 0}
        
        total_executions = len(self.execution_history)
        avg_processing_time = sum(result.total_processing_time for result in self.execution_history) / total_executions
        avg_confidence = sum(result.confidence_level for result in self.execution_history) / total_executions
        
        return {
            "total_executions": total_executions,
            "average_processing_time": round(avg_processing_time, 2),
            "average_confidence": round(avg_confidence, 2),
            "peer_review_usage": sum(1 for result in self.execution_history if result.peer_reviews) / total_executions
        }

# Example usage and testing
async def main():
    """Example usage of MultiAgentCoordinator"""
    coordinator = MultiAgentCoordinator(enable_peer_review=True)
    
    # This would normally be replaced with real agent instances
    class MockAgent(Agent):
        def __init__(self, name):
            super().__init__()
            self.name = name
        
        def process(self, query):
            return f"Mock response from {self.name} for: {query}"
    
    # Register mock agents
    coordinator.register_agent(AgentRole.GENERAL, MockAgent("GeneralAgent"))
    coordinator.register_agent(AgentRole.BROWSER, MockAgent("BrowserAgent"))
    coordinator.register_agent(AgentRole.REVIEWER, MockAgent("ReviewerAgent"))
    
    # Test execution
    result = await coordinator.execute_with_peer_review(
        "What is the weather today?",
        task_type="general",
        priority=TaskPriority.MEDIUM
    )
    
    print(f"Final result: {result.final_content}")
    print(f"Confidence: {result.confidence_level}")
    print(f"Processing time: {result.total_processing_time:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())