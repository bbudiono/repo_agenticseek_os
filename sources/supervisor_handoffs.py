#!/usr/bin/env python3
"""
* Purpose: Supervisor + handoffs pattern implementation for DeerFlow-style agent coordination
* Issues & Complexity Summary: Complex supervisor pattern with dynamic handoffs and workflow orchestration
* Key Complexity Drivers:
  - Logic Scope (Est. LoC): ~700
  - Core Algorithm Complexity: Very High
  - Dependencies: 6 New, 4 Mod
  - State Management Complexity: Very High
  - Novelty/Uncertainty Factor: Medium
* AI Pre-Task Self-Assessment (Est. Solution Difficulty %): 88%
* Problem Estimate (Inherent Problem Difficulty %): 92%
* Initial Code Complexity Estimate %: 88%
* Justification for Estimates: Complex supervisor patterns with dynamic handoffs and state management
* Final Code Complexity (Actual %): TBD
* Overall Result Score (Success & Quality %): TBD
* Key Variances/Learnings: TBD
* Last Updated: 2025-01-06
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from enum import Enum
from abc import ABC, abstractmethod
import logging
from contextlib import asynccontextmanager

if __name__ == "__main__":
    from deer_flow_orchestrator import DeerFlowState, TaskType, AgentRole, DeerFlowAgent, AgentOutput
    from specialized_agents import SpecializedAgentFactory
    from utility import pretty_print
else:
    from sources.deer_flow_orchestrator import DeerFlowState, TaskType, AgentRole, DeerFlowAgent, AgentOutput
    from sources.specialized_agents import SpecializedAgentFactory
    from sources.utility import pretty_print

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HandoffTrigger(Enum):
    """Triggers for agent handoffs"""
    TASK_COMPLETION = "task_completion"
    SKILL_REQUIREMENT = "skill_requirement"
    QUALITY_THRESHOLD = "quality_threshold"
    TIMEOUT = "timeout"
    ERROR_RECOVERY = "error_recovery"
    USER_INTERVENTION = "user_intervention"
    SUPERVISOR_DECISION = "supervisor_decision"

class SupervisorDecision(Enum):
    """Supervisor decision types"""
    CONTINUE = "continue"
    HANDOFF = "handoff"
    PARALLEL_EXECUTION = "parallel_execution"
    RETRY = "retry"
    ESCALATE = "escalate"
    TERMINATE = "terminate"

class WorkflowStage(Enum):
    """Workflow execution stages"""
    INITIALIZATION = "initialization"
    COORDINATION = "coordination"
    EXECUTION = "execution"
    REVIEW = "review"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"
    COMPLETION = "completion"
    ERROR_HANDLING = "error_handling"

@dataclass
class HandoffRequest:
    """Request for agent handoff"""
    from_agent: AgentRole
    to_agent: AgentRole
    trigger: HandoffTrigger
    context: Dict[str, Any]
    priority: int
    reasoning: str
    timestamp: float
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class SupervisorDecisionRecord:
    """Record of supervisor decision"""
    decision: SupervisorDecision
    reasoning: str
    affected_agents: List[AgentRole]
    context: Dict[str, Any]
    confidence: float
    timestamp: float
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class WorkflowExecution:
    """Workflow execution tracking"""
    workflow_id: str
    current_stage: WorkflowStage
    active_agents: List[AgentRole]
    completed_stages: List[WorkflowStage]
    handoff_history: List[HandoffRequest]
    supervisor_decisions: List[SupervisorDecisionRecord]
    performance_metrics: Dict[str, Any]
    start_time: float
    last_update: float

class SupervisorAgent(DeerFlowAgent):
    """
    Supervisor agent implementing hierarchical coordination and handoff management
    """
    
    def __init__(self):
        super().__init__(AgentRole.COORDINATOR, "Supervisor")
        self.agent_factory = SpecializedAgentFactory()
        self.active_workflows: Dict[str, WorkflowExecution] = {}
        self.agent_capabilities = self._initialize_agent_capabilities()
        self.handoff_rules = self._initialize_handoff_rules()
        self.quality_thresholds = {
            "confidence_minimum": 0.7,
            "execution_time_maximum": 60.0,
            "retry_limit": 3,
            "escalation_threshold": 0.5
        }
        
    def _initialize_agent_capabilities(self) -> Dict[AgentRole, Dict[str, Any]]:
        """Initialize agent capability matrix"""
        return {
            AgentRole.COORDINATOR: {
                "skills": ["task_routing", "coordination", "decision_making"],
                "complexity_level": 4,
                "execution_time": "fast",
                "quality_score": 0.9,
                "resource_usage": "low"
            },
            AgentRole.RESEARCHER: {
                "skills": ["web_search", "data_gathering", "source_validation"],
                "complexity_level": 3,
                "execution_time": "medium",
                "quality_score": 0.8,
                "resource_usage": "medium"
            },
            AgentRole.CODER: {
                "skills": ["code_analysis", "execution", "debugging", "security_assessment"],
                "complexity_level": 4,
                "execution_time": "medium",
                "quality_score": 0.85,
                "resource_usage": "medium"
            },
            AgentRole.SYNTHESIZER: {
                "skills": ["content_generation", "synthesis", "report_creation"],
                "complexity_level": 3,
                "execution_time": "fast",
                "quality_score": 0.9,
                "resource_usage": "low"
            },
            AgentRole.PLANNER: {
                "skills": ["planning", "strategy", "resource_allocation"],
                "complexity_level": 3,
                "execution_time": "fast",
                "quality_score": 0.8,
                "resource_usage": "low"
            }
        }
    
    def _initialize_handoff_rules(self) -> Dict[Tuple[AgentRole, AgentRole], Dict[str, Any]]:
        """Initialize handoff rules between agents"""
        return {
            # Coordinator handoffs
            (AgentRole.COORDINATOR, AgentRole.PLANNER): {
                "triggers": [HandoffTrigger.TASK_COMPLETION, HandoffTrigger.SKILL_REQUIREMENT],
                "conditions": ["complex_task", "planning_required"],
                "priority": 2
            },
            (AgentRole.COORDINATOR, AgentRole.RESEARCHER): {
                "triggers": [HandoffTrigger.SKILL_REQUIREMENT],
                "conditions": ["research_task", "information_gathering"],
                "priority": 3
            },
            
            # Planner handoffs
            (AgentRole.PLANNER, AgentRole.RESEARCHER): {
                "triggers": [HandoffTrigger.TASK_COMPLETION, HandoffTrigger.SKILL_REQUIREMENT],
                "conditions": ["research_phase", "data_collection"],
                "priority": 2
            },
            (AgentRole.PLANNER, AgentRole.CODER): {
                "triggers": [HandoffTrigger.SKILL_REQUIREMENT],
                "conditions": ["code_analysis", "technical_execution"],
                "priority": 2
            },
            
            # Research to synthesis
            (AgentRole.RESEARCHER, AgentRole.SYNTHESIZER): {
                "triggers": [HandoffTrigger.TASK_COMPLETION],
                "conditions": ["research_complete", "synthesis_required"],
                "priority": 1
            },
            
            # Code to synthesis
            (AgentRole.CODER, AgentRole.SYNTHESIZER): {
                "triggers": [HandoffTrigger.TASK_COMPLETION],
                "conditions": ["code_analysis_complete", "synthesis_required"],
                "priority": 1
            },
            
            # Error recovery handoffs
            (AgentRole.RESEARCHER, AgentRole.COORDINATOR): {
                "triggers": [HandoffTrigger.ERROR_RECOVERY, HandoffTrigger.QUALITY_THRESHOLD],
                "conditions": ["execution_failure", "low_confidence"],
                "priority": 4
            },
            (AgentRole.CODER, AgentRole.COORDINATOR): {
                "triggers": [HandoffTrigger.ERROR_RECOVERY, HandoffTrigger.TIMEOUT],
                "conditions": ["execution_failure", "timeout_exceeded"],
                "priority": 4
            }
        }
    
    async def execute(self, state: DeerFlowState) -> AgentOutput:
        """Execute supervisor coordination"""
        start_time = time.time()
        workflow_id = state["metadata"].get("workflow_id", str(uuid.uuid4()))
        
        # Initialize workflow tracking
        workflow = self._initialize_workflow(workflow_id, state)
        self.active_workflows[workflow_id] = workflow
        
        # Execute supervised workflow
        supervision_result = await self._execute_supervised_workflow(workflow, state)
        
        # Update workflow completion
        workflow.current_stage = WorkflowStage.COMPLETION
        workflow.last_update = time.time()
        
        return AgentOutput(
            agent_role=self.role,
            content=f"Supervised workflow completed with {len(workflow.handoff_history)} handoffs",
            confidence=supervision_result["confidence"],
            metadata={
                "workflow_id": workflow_id,
                "stages_completed": len(workflow.completed_stages),
                "handoffs_executed": len(workflow.handoff_history),
                "supervisor_decisions": len(workflow.supervisor_decisions),
                "performance_metrics": workflow.performance_metrics
            },
            timestamp=time.time(),
            execution_time=time.time() - start_time
        )
    
    def _initialize_workflow(self, workflow_id: str, state: DeerFlowState) -> WorkflowExecution:
        """Initialize workflow execution tracking"""
        return WorkflowExecution(
            workflow_id=workflow_id,
            current_stage=WorkflowStage.INITIALIZATION,
            active_agents=[],
            completed_stages=[],
            handoff_history=[],
            supervisor_decisions=[],
            performance_metrics={
                "total_execution_time": 0.0,
                "agent_utilization": {},
                "quality_scores": [],
                "handoff_efficiency": 0.0
            },
            start_time=time.time(),
            last_update=time.time()
        )
    
    async def _execute_supervised_workflow(self, workflow: WorkflowExecution, state: DeerFlowState) -> Dict[str, Any]:
        """Execute workflow with supervisor oversight"""
        try:
            # Stage 1: Coordination
            await self._execute_stage(workflow, WorkflowStage.COORDINATION, state)
            
            # Stage 2: Execution (with potential handoffs)
            await self._execute_stage(workflow, WorkflowStage.EXECUTION, state)
            
            # Stage 3: Review and validation
            await self._execute_stage(workflow, WorkflowStage.REVIEW, state)
            
            # Stage 4: Synthesis
            await self._execute_stage(workflow, WorkflowStage.SYNTHESIS, state)
            
            # Calculate final results
            final_confidence = self._calculate_workflow_confidence(workflow)
            
            return {
                "success": True,
                "confidence": final_confidence,
                "stages_completed": len(workflow.completed_stages),
                "total_time": time.time() - workflow.start_time
            }
            
        except Exception as e:
            logger.error(f"Supervised workflow failed: {str(e)}")
            workflow.current_stage = WorkflowStage.ERROR_HANDLING
            await self._handle_workflow_error(workflow, state, str(e))
            
            return {
                "success": False,
                "confidence": 0.3,
                "error": str(e),
                "recovery_attempted": True
            }
    
    async def _execute_stage(self, workflow: WorkflowExecution, stage: WorkflowStage, state: DeerFlowState):
        """Execute a specific workflow stage with supervision"""
        workflow.current_stage = stage
        workflow.last_update = time.time()
        
        logger.info(f"Executing stage: {stage.value}")
        
        # Determine appropriate agent for stage
        agent_role = self._select_agent_for_stage(stage, state)
        workflow.active_agents = [agent_role]
        
        # Execute with supervision
        stage_result = await self._execute_agent_with_supervision(workflow, agent_role, state)
        
        # Evaluate results and decide on handoffs
        supervisor_decision = await self._evaluate_stage_results(workflow, stage, stage_result, state)
        
        # Execute handoffs if needed
        if supervisor_decision.decision == SupervisorDecision.HANDOFF:
            await self._execute_handoff(workflow, supervisor_decision, state)
        elif supervisor_decision.decision == SupervisorDecision.PARALLEL_EXECUTION:
            await self._execute_parallel_agents(workflow, supervisor_decision.affected_agents, state)
        elif supervisor_decision.decision == SupervisorDecision.RETRY:
            await self._retry_stage_execution(workflow, stage, state)
        
        # Mark stage as completed
        workflow.completed_stages.append(stage)
        
        # Update performance metrics
        self._update_performance_metrics(workflow, stage_result)
    
    def _select_agent_for_stage(self, stage: WorkflowStage, state: DeerFlowState) -> AgentRole:
        """Select appropriate agent for workflow stage"""
        task_type = state.get("task_type", TaskType.GENERAL_QUERY)
        
        stage_agent_map = {
            WorkflowStage.COORDINATION: AgentRole.COORDINATOR,
            WorkflowStage.EXECUTION: self._select_execution_agent(task_type),
            WorkflowStage.REVIEW: AgentRole.COORDINATOR,
            WorkflowStage.SYNTHESIS: AgentRole.SYNTHESIZER,
            WorkflowStage.VALIDATION: AgentRole.COORDINATOR
        }
        
        return stage_agent_map.get(stage, AgentRole.COORDINATOR)
    
    def _select_execution_agent(self, task_type: TaskType) -> AgentRole:
        """Select execution agent based on task type"""
        execution_map = {
            TaskType.RESEARCH: AgentRole.RESEARCHER,
            TaskType.CODE_ANALYSIS: AgentRole.CODER,
            TaskType.WEB_CRAWLING: AgentRole.RESEARCHER,
            TaskType.DATA_PROCESSING: AgentRole.CODER,
            TaskType.REPORT_GENERATION: AgentRole.SYNTHESIZER,
            TaskType.GENERAL_QUERY: AgentRole.SYNTHESIZER
        }
        return execution_map.get(task_type, AgentRole.SYNTHESIZER)
    
    async def _execute_agent_with_supervision(
        self, 
        workflow: WorkflowExecution, 
        agent_role: AgentRole, 
        state: DeerFlowState
    ) -> Dict[str, Any]:
        """Execute agent with supervisor oversight"""
        start_time = time.time()
        
        try:
            # Create agent
            agent = self.agent_factory.create_agent(agent_role)
            
            # Execute with timeout
            result = await asyncio.wait_for(
                agent.execute(state),
                timeout=self.quality_thresholds["execution_time_maximum"]
            )
            
            execution_time = time.time() - start_time
            
            # Update workflow metrics
            if agent_role.value not in workflow.performance_metrics["agent_utilization"]:
                workflow.performance_metrics["agent_utilization"][agent_role.value] = 0
            workflow.performance_metrics["agent_utilization"][agent_role.value] += execution_time
            
            return {
                "success": True,
                "result": result,
                "execution_time": execution_time,
                "agent_role": agent_role
            }
            
        except asyncio.TimeoutError:
            logger.warning(f"Agent {agent_role.value} timed out")
            return {
                "success": False,
                "error": "timeout",
                "execution_time": time.time() - start_time,
                "agent_role": agent_role
            }
        except Exception as e:
            logger.error(f"Agent {agent_role.value} execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time,
                "agent_role": agent_role
            }
    
    async def _evaluate_stage_results(
        self, 
        workflow: WorkflowExecution, 
        stage: WorkflowStage, 
        stage_result: Dict[str, Any], 
        state: DeerFlowState
    ) -> SupervisorDecisionRecord:
        """Evaluate stage results and make supervisor decision"""
        
        if not stage_result["success"]:
            # Handle failure
            if stage_result.get("error") == "timeout":
                decision = SupervisorDecision.RETRY
                reasoning = f"Agent timed out during {stage.value}, retrying with different approach"
            else:
                decision = SupervisorDecision.ESCALATE
                reasoning = f"Agent failed during {stage.value}, escalating for error recovery"
            
            affected_agents = [stage_result["agent_role"]]
            confidence = 0.3
            
        else:
            # Evaluate success quality
            result = stage_result["result"]
            confidence = result.confidence
            
            if confidence >= self.quality_thresholds["confidence_minimum"]:
                decision = SupervisorDecision.CONTINUE
                reasoning = f"Stage {stage.value} completed successfully with confidence {confidence:.2f}"
                affected_agents = []
            else:
                # Low confidence - consider handoff
                potential_handoff = self._identify_potential_handoff(
                    stage_result["agent_role"], confidence, state
                )
                
                if potential_handoff:
                    decision = SupervisorDecision.HANDOFF
                    reasoning = f"Low confidence {confidence:.2f}, handing off to {potential_handoff.value}"
                    affected_agents = [stage_result["agent_role"], potential_handoff]
                else:
                    decision = SupervisorDecision.CONTINUE
                    reasoning = f"Proceeding with confidence {confidence:.2f}, no better alternative"
                    affected_agents = []
        
        supervisor_decision = SupervisorDecisionRecord(
            decision=decision,
            reasoning=reasoning,
            affected_agents=affected_agents,
            context={
                "stage": stage.value,
                "stage_result": stage_result,
                "workflow_id": workflow.workflow_id
            },
            confidence=confidence,
            timestamp=time.time()
        )
        
        workflow.supervisor_decisions.append(supervisor_decision)
        return supervisor_decision
    
    def _identify_potential_handoff(
        self, 
        current_agent: AgentRole, 
        confidence: float, 
        state: DeerFlowState
    ) -> Optional[AgentRole]:
        """Identify potential handoff target for low confidence results"""
        
        # Check handoff rules
        for (from_agent, to_agent), rule in self.handoff_rules.items():
            if from_agent == current_agent:
                # Check if handoff conditions are met
                if HandoffTrigger.QUALITY_THRESHOLD in rule["triggers"]:
                    # Check if target agent has better capabilities for this task
                    target_capabilities = self.agent_capabilities.get(to_agent, {})
                    if target_capabilities.get("quality_score", 0) > confidence:
                        return to_agent
        
        return None
    
    async def _execute_handoff(
        self, 
        workflow: WorkflowExecution, 
        supervisor_decision: SupervisorDecisionRecord, 
        state: DeerFlowState
    ):
        """Execute agent handoff"""
        if len(supervisor_decision.affected_agents) < 2:
            return
        
        from_agent = supervisor_decision.affected_agents[0]
        to_agent = supervisor_decision.affected_agents[1]
        
        handoff_request = HandoffRequest(
            from_agent=from_agent,
            to_agent=to_agent,
            trigger=HandoffTrigger.SUPERVISOR_DECISION,
            context=supervisor_decision.context,
            priority=2,
            reasoning=supervisor_decision.reasoning,
            timestamp=time.time()
        )
        
        workflow.handoff_history.append(handoff_request)
        
        logger.info(f"Executing handoff: {from_agent.value} → {to_agent.value}")
        
        # Execute target agent
        handoff_result = await self._execute_agent_with_supervision(workflow, to_agent, state)
        
        # Update state with handoff result
        if handoff_result["success"]:
            result = handoff_result["result"]
            state["agent_outputs"][to_agent.value] = asdict(result)
            logger.info(f"Handoff successful: {to_agent.value} confidence {result.confidence:.2f}")
        else:
            logger.warning(f"Handoff failed: {to_agent.value} - {handoff_result.get('error', 'unknown')}")
    
    async def _execute_parallel_agents(
        self, 
        workflow: WorkflowExecution, 
        agent_roles: List[AgentRole], 
        state: DeerFlowState
    ):
        """Execute multiple agents in parallel"""
        logger.info(f"Executing parallel agents: {[role.value for role in agent_roles]}")
        
        # Create tasks for parallel execution
        tasks = []
        for role in agent_roles:
            task = self._execute_agent_with_supervision(workflow, role, state)
            tasks.append(task)
        
        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            role = agent_roles[i]
            if isinstance(result, Exception):
                logger.error(f"Parallel agent {role.value} failed: {str(result)}")
            elif result["success"]:
                agent_result = result["result"]
                state["agent_outputs"][role.value] = asdict(agent_result)
                logger.info(f"Parallel agent {role.value} completed with confidence {agent_result.confidence:.2f}")
    
    async def _retry_stage_execution(
        self, 
        workflow: WorkflowExecution, 
        stage: WorkflowStage, 
        state: DeerFlowState
    ):
        """Retry stage execution with different approach"""
        logger.info(f"Retrying stage: {stage.value}")
        
        # Select alternative agent if available
        current_agent = workflow.active_agents[0] if workflow.active_agents else AgentRole.COORDINATOR
        alternative_agent = self._select_alternative_agent(current_agent, state)
        
        if alternative_agent and alternative_agent != current_agent:
            retry_result = await self._execute_agent_with_supervision(workflow, alternative_agent, state)
            
            if retry_result["success"]:
                result = retry_result["result"]
                state["agent_outputs"][alternative_agent.value] = asdict(result)
                logger.info(f"Retry successful with {alternative_agent.value}")
            else:
                logger.warning(f"Retry with {alternative_agent.value} also failed")
        else:
            logger.warning("No alternative agent available for retry")
    
    def _select_alternative_agent(self, current_agent: AgentRole, state: DeerFlowState) -> Optional[AgentRole]:
        """Select alternative agent for retry"""
        task_type = state.get("task_type", TaskType.GENERAL_QUERY)
        
        # Define alternative agents for each role
        alternatives = {
            AgentRole.RESEARCHER: [AgentRole.SYNTHESIZER, AgentRole.COORDINATOR],
            AgentRole.CODER: [AgentRole.SYNTHESIZER, AgentRole.COORDINATOR],
            AgentRole.SYNTHESIZER: [AgentRole.COORDINATOR],
            AgentRole.COORDINATOR: [AgentRole.SYNTHESIZER],
            AgentRole.PLANNER: [AgentRole.COORDINATOR, AgentRole.SYNTHESIZER]
        }
        
        potential_alternatives = alternatives.get(current_agent, [])
        return potential_alternatives[0] if potential_alternatives else None
    
    async def _handle_workflow_error(self, workflow: WorkflowExecution, state: DeerFlowState, error: str):
        """Handle workflow-level errors"""
        logger.error(f"Handling workflow error: {error}")
        
        # Create error recovery decision
        recovery_decision = SupervisorDecisionRecord(
            decision=SupervisorDecision.ESCALATE,
            reasoning=f"Workflow error requiring escalation: {error}",
            affected_agents=[AgentRole.COORDINATOR],
            context={"error": error, "workflow_id": workflow.workflow_id},
            confidence=0.2,
            timestamp=time.time()
        )
        
        workflow.supervisor_decisions.append(recovery_decision)
        
        # Attempt basic recovery
        try:
            recovery_agent = self.agent_factory.create_agent(AgentRole.SYNTHESIZER)
            recovery_result = await recovery_agent.execute(state)
            
            state["agent_outputs"]["error_recovery"] = asdict(recovery_result)
            logger.info("Error recovery attempt completed")
            
        except Exception as recovery_error:
            logger.error(f"Error recovery failed: {str(recovery_error)}")
    
    def _update_performance_metrics(self, workflow: WorkflowExecution, stage_result: Dict[str, Any]):
        """Update workflow performance metrics"""
        if stage_result["success"]:
            result = stage_result["result"]
            workflow.performance_metrics["quality_scores"].append(result.confidence)
        
        workflow.performance_metrics["total_execution_time"] += stage_result["execution_time"]
        
        # Calculate handoff efficiency
        if workflow.handoff_history:
            successful_handoffs = sum(1 for handoff in workflow.handoff_history 
                                    if handoff.trigger != HandoffTrigger.ERROR_RECOVERY)
            workflow.performance_metrics["handoff_efficiency"] = successful_handoffs / len(workflow.handoff_history)
    
    def _calculate_workflow_confidence(self, workflow: WorkflowExecution) -> float:
        """Calculate overall workflow confidence"""
        quality_scores = workflow.performance_metrics.get("quality_scores", [])
        if not quality_scores:
            return 0.5
        
        # Weight recent scores higher
        weighted_scores = []
        for i, score in enumerate(quality_scores):
            weight = 1.0 + (i * 0.1)  # Later scores get higher weight
            weighted_scores.extend([score] * int(weight * 10))
        
        return sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0.5
    
    def can_handle(self, task_type: TaskType) -> bool:
        return True  # Supervisor handles all task types
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific workflow"""
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            return None
        
        return {
            "workflow_id": workflow_id,
            "current_stage": workflow.current_stage.value,
            "active_agents": [agent.value for agent in workflow.active_agents],
            "completed_stages": [stage.value for stage in workflow.completed_stages],
            "handoffs_count": len(workflow.handoff_history),
            "decisions_count": len(workflow.supervisor_decisions),
            "performance_metrics": workflow.performance_metrics,
            "execution_time": time.time() - workflow.start_time
        }
    
    def get_all_workflows_status(self) -> Dict[str, Any]:
        """Get status of all active workflows"""
        return {
            "total_workflows": len(self.active_workflows),
            "workflows": {wf_id: self.get_workflow_status(wf_id) for wf_id in self.active_workflows.keys()}
        }

# Example usage and testing
async def main():
    """Test supervisor and handoffs pattern"""
    supervisor = SupervisorAgent()
    
    # Create test state
    test_state: DeerFlowState = {
        "messages": [],
        "user_query": "Research AI developments and analyze code performance with comprehensive review",
        "task_type": TaskType.RESEARCH,
        "current_step": "initialized",
        "agent_outputs": {},
        "research_results": [],
        "code_analysis": {},
        "synthesis_result": "",
        "validation_status": False,
        "final_report": "",
        "metadata": {"workflow_id": str(uuid.uuid4())},
        "checkpoints": [],
        "error_logs": [],
        "execution_time": 0.0,
        "confidence_scores": {}
    }
    
    # Execute supervised workflow
    print("Testing supervisor + handoffs pattern...")
    result = await supervisor.execute(test_state)
    
    print(f"Supervisor result: {result.content}")
    print(f"Confidence: {result.confidence}")
    print(f"Metadata: {json.dumps(result.metadata, indent=2)}")
    
    # Show workflow status
    workflow_id = test_state["metadata"]["workflow_id"]
    status = supervisor.get_workflow_status(workflow_id)
    print(f"\nWorkflow status:")
    print(json.dumps(status, indent=2))

if __name__ == "__main__":
    asyncio.run(main())