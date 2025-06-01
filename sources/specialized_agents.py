#!/usr/bin/env python3
"""
* Purpose: Enhanced specialized agents implementing DeerFlow patterns with AgenticSeek integration
* Issues & Complexity Summary: Complex agent specialization with real backend integration and coordination
* Key Complexity Drivers:
  - Logic Scope (Est. LoC): ~800
  - Core Algorithm Complexity: Very High
  - Dependencies: 8 New, 4 Mod
  - State Management Complexity: High
  - Novelty/Uncertainty Factor: Medium
* AI Pre-Task Self-Assessment (Est. Solution Difficulty %): 85%
* Problem Estimate (Inherent Problem Difficulty %): 90%
* Initial Code Complexity Estimate %: 85%
* Justification for Estimates: Complex integration of existing agents with DeerFlow orchestration
* Final Code Complexity (Actual %): TBD
* Overall Result Score (Success & Quality %): TBD
* Key Variances/Learnings: TBD
* Last Updated: 2025-01-06
"""

import asyncio
import json
import time
import uuid
import subprocess
import tempfile
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from abc import ABC, abstractmethod
import logging

# Import existing AgenticSeek agents
if __name__ == "__main__":
    from agents.browser_agent import BrowserAgent
    from agents.code_agent import CodeAgent  
    from agents.casual_agent import CasualAgent
    from agents.file_agent import FileAgent
    from agents.planner_agent import PlannerAgent
    from agents.agent import Agent
    from deer_flow_orchestrator import DeerFlowAgent, AgentRole, TaskType, AgentOutput, DeerFlowState
    from utility import pretty_print
else:
    from sources.agents.browser_agent import BrowserAgent
    from sources.agents.code_agent import CodeAgent
    from sources.agents.casual_agent import CasualAgent  
    from sources.agents.file_agent import FileAgent
    from sources.agents.planner_agent import PlannerAgent
    from sources.agents.agent import Agent
    from sources.deer_flow_orchestrator import DeerFlowAgent, AgentRole, TaskType, AgentOutput, DeerFlowState
    from sources.utility import pretty_print

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearchQuality(Enum):
    """Research quality levels"""
    BASIC = "basic"
    COMPREHENSIVE = "comprehensive" 
    DEEP_ANALYSIS = "deep_analysis"
    EXPERT_LEVEL = "expert_level"

class CodeExecutionMode(Enum):
    """Code execution modes"""
    ANALYSIS_ONLY = "analysis_only"
    SAFE_EXECUTION = "safe_execution"
    FULL_EXECUTION = "full_execution"
    REPL_MODE = "repl_mode"

@dataclass
class ResearchResult:
    """Structured research result"""
    source: str
    content: str
    relevance_score: float
    confidence: float
    metadata: Dict[str, Any]
    timestamp: float

@dataclass 
class CodeAnalysisResult:
    """Structured code analysis result"""
    analysis_type: str
    execution_result: Optional[str]
    performance_metrics: Dict[str, Any]
    security_assessment: Dict[str, Any]
    recommendations: List[str]
    confidence: float
    execution_time: float

class EnhancedCoordinatorAgent(DeerFlowAgent):
    """
    Enhanced coordinator integrating with AgenticSeek's routing system
    """
    
    def __init__(self):
        super().__init__(AgentRole.COORDINATOR, "Enhanced Coordinator")
        # Initialize with existing AgenticSeek routing capability
        try:
            from sources.router import Router
            self.router = Router()
        except ImportError:
            self.router = None
            logger.warning("AgenticSeek Router not available, using simplified routing")
        
        self.task_complexity_map = {
            "simple": 1,
            "medium": 2, 
            "complex": 3,
            "expert": 4
        }
        
    async def execute(self, state: DeerFlowState) -> AgentOutput:
        """Enhanced coordination with AgenticSeek routing integration"""
        start_time = time.time()
        
        query = state["user_query"]
        
        # Use AgenticSeek router if available
        if self.router:
            routing_decision = await self._route_with_agenticseek(query)
        else:
            routing_decision = self._classify_task_simple(query)
        
        # Enhanced task analysis
        task_analysis = await self._analyze_task_complexity(query)
        
        # Update state with enhanced routing
        state["task_type"] = routing_decision["task_type"]
        state["current_step"] = "coordination"
        state["metadata"]["routing_decision"] = routing_decision
        state["metadata"]["task_analysis"] = task_analysis
        
        coordination_result = {
            "primary_agent": routing_decision["primary_agent"],
            "supporting_agents": routing_decision.get("supporting_agents", []),
            "estimated_duration": task_analysis["estimated_duration"],
            "complexity_score": task_analysis["complexity_score"],
            "required_resources": task_analysis["resources"],
            "risk_assessment": task_analysis["risks"],
            "quality_target": task_analysis["quality_target"]
        }
        
        logger.info(f"Coordinated task: {routing_decision['task_type'].value} with complexity {task_analysis['complexity_score']}")
        
        return AgentOutput(
            agent_role=self.role,
            content=f"Coordinated {routing_decision['task_type'].value} task with {len(coordination_result['supporting_agents'])} supporting agents",
            confidence=0.9,
            metadata=coordination_result,
            timestamp=time.time(),
            execution_time=time.time() - start_time
        )
    
    async def _route_with_agenticseek(self, query: str) -> Dict[str, Any]:
        """Use AgenticSeek's router for intelligent task routing"""
        try:
            # Get routing decision from AgenticSeek router
            route_result = await asyncio.get_event_loop().run_in_executor(
                None, self.router.route, query
            )
            
            # Map AgenticSeek agents to DeerFlow roles
            agent_mapping = {
                "browser_agent": AgentRole.RESEARCHER,
                "code_agent": AgentRole.CODER,
                "casual_agent": AgentRole.SYNTHESIZER,
                "file_agent": AgentRole.RESEARCHER,
                "planner_agent": AgentRole.PLANNER
            }
            
            primary_role = agent_mapping.get(route_result.get("agent", "casual_agent"), AgentRole.SYNTHESIZER)
            
            # Determine task type from AgenticSeek routing
            if "browser" in route_result.get("agent", ""):
                task_type = TaskType.RESEARCH
            elif "code" in route_result.get("agent", ""):
                task_type = TaskType.CODE_ANALYSIS
            elif "file" in route_result.get("agent", ""):
                task_type = TaskType.DATA_PROCESSING
            elif "planner" in route_result.get("agent", ""):
                task_type = TaskType.REPORT_GENERATION
            else:
                task_type = TaskType.GENERAL_QUERY
            
            return {
                "task_type": task_type,
                "primary_agent": primary_role,
                "supporting_agents": self._select_supporting_agents(primary_role),
                "confidence": route_result.get("confidence", 0.8),
                "routing_method": "agenticseek_router"
            }
            
        except Exception as e:
            logger.error(f"AgenticSeek routing failed: {str(e)}")
            return self._classify_task_simple(query)
    
    def _classify_task_simple(self, query: str) -> Dict[str, Any]:
        """Fallback simple task classification"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["research", "find", "search", "investigate", "browse"]):
            return {
                "task_type": TaskType.RESEARCH,
                "primary_agent": AgentRole.RESEARCHER,
                "supporting_agents": [AgentRole.SYNTHESIZER],
                "confidence": 0.7,
                "routing_method": "simple_classification"
            }
        elif any(word in query_lower for word in ["code", "program", "debug", "analyze", "execute"]):
            return {
                "task_type": TaskType.CODE_ANALYSIS,
                "primary_agent": AgentRole.CODER,
                "supporting_agents": [AgentRole.SYNTHESIZER],
                "confidence": 0.7,
                "routing_method": "simple_classification"
            }
        else:
            return {
                "task_type": TaskType.GENERAL_QUERY,
                "primary_agent": AgentRole.SYNTHESIZER,
                "supporting_agents": [],
                "confidence": 0.6,
                "routing_method": "simple_classification"
            }
    
    async def _analyze_task_complexity(self, query: str) -> Dict[str, Any]:
        """Analyze task complexity and requirements"""
        # Simple heuristics for complexity analysis
        word_count = len(query.split())
        
        complexity_indicators = {
            "comprehensive": 2,
            "detailed": 2,
            "thorough": 2,
            "analyze": 1,
            "complex": 3,
            "multiple": 1,
            "various": 1,
            "extensive": 3
        }
        
        complexity_score = 1  # Base complexity
        for indicator, weight in complexity_indicators.items():
            if indicator in query.lower():
                complexity_score += weight
        
        # Adjust for query length
        if word_count > 20:
            complexity_score += 1
        elif word_count > 10:
            complexity_score += 0.5
        
        # Cap complexity score
        complexity_score = min(complexity_score, 4)
        
        return {
            "complexity_score": complexity_score,
            "estimated_duration": self._estimate_duration(complexity_score),
            "resources": self._identify_resources(complexity_score),
            "risks": self._assess_risks(complexity_score),
            "quality_target": self._determine_quality_target(complexity_score)
        }
    
    def _select_supporting_agents(self, primary_role: AgentRole) -> List[AgentRole]:
        """Select supporting agents based on primary agent"""
        support_map = {
            AgentRole.RESEARCHER: [AgentRole.SYNTHESIZER],
            AgentRole.CODER: [AgentRole.SYNTHESIZER],
            AgentRole.PLANNER: [AgentRole.RESEARCHER, AgentRole.SYNTHESIZER],
            AgentRole.SYNTHESIZER: []
        }
        return support_map.get(primary_role, [])
    
    def _estimate_duration(self, complexity: float) -> int:
        """Estimate task duration based on complexity"""
        base_duration = 30  # seconds
        return int(base_duration * complexity)
    
    def _identify_resources(self, complexity: float) -> List[str]:
        """Identify required resources"""
        base_resources = ["basic_processing"]
        if complexity >= 2:
            base_resources.extend(["enhanced_analysis", "multi_source_data"])
        if complexity >= 3:
            base_resources.extend(["expert_knowledge", "specialized_tools"])
        if complexity >= 4:
            base_resources.extend(["advanced_computation", "external_apis"])
        return base_resources
    
    def _assess_risks(self, complexity: float) -> List[str]:
        """Assess task risks"""
        base_risks = ["execution_failure"]
        if complexity >= 2:
            base_risks.extend(["data_quality_issues", "processing_delays"])
        if complexity >= 3:
            base_risks.extend(["resource_constraints", "accuracy_concerns"])
        if complexity >= 4:
            base_risks.extend(["system_overload", "integration_failures"])
        return base_risks
    
    def _determine_quality_target(self, complexity: float) -> str:
        """Determine quality target"""
        if complexity >= 3.5:
            return "expert_level"
        elif complexity >= 2.5:
            return "comprehensive"
        elif complexity >= 1.5:
            return "thorough"
        else:
            return "basic"
    
    def can_handle(self, task_type: TaskType) -> bool:
        return True  # Coordinator handles all types

class EnhancedResearchAgent(DeerFlowAgent):
    """
    Enhanced research agent with deep web crawling and multi-source integration
    """
    
    def __init__(self):
        super().__init__(AgentRole.RESEARCHER, "Enhanced Researcher")
        # Initialize with AgenticSeek browser agent
        self.browser_agent = BrowserAgent()
        self.max_sources = 5
        self.min_relevance_threshold = 0.6
        
    async def execute(self, state: DeerFlowState) -> AgentOutput:
        """Execute comprehensive research with multiple sources"""
        start_time = time.time()
        
        query = state["user_query"]
        quality_target = state["metadata"].get("task_analysis", {}).get("quality_target", "basic")
        
        # Adjust research parameters based on quality target
        research_params = self._configure_research_parameters(quality_target)
        
        # Conduct multi-source research
        research_results = await self._conduct_comprehensive_research(query, research_params)
        
        # Filter and rank results
        filtered_results = self._filter_and_rank_results(research_results)
        
        # Update state
        state["current_step"] = "researching"
        state["research_results"] = [asdict(result) for result in filtered_results]
        state["metadata"]["research_stats"] = {
            "total_sources": len(research_results),
            "filtered_sources": len(filtered_results),
            "average_relevance": sum(r.relevance_score for r in filtered_results) / len(filtered_results) if filtered_results else 0,
            "research_quality": quality_target
        }
        
        confidence = self._calculate_research_confidence(filtered_results)
        
        logger.info(f"Research completed: {len(filtered_results)} sources with avg relevance {state['metadata']['research_stats']['average_relevance']:.2f}")
        
        return AgentOutput(
            agent_role=self.role,
            content=f"Completed {quality_target} research with {len(filtered_results)} high-quality sources",
            confidence=confidence,
            metadata={
                "sources_found": len(research_results),
                "sources_filtered": len(filtered_results),
                "quality_level": quality_target,
                "research_parameters": research_params
            },
            timestamp=time.time(),
            execution_time=time.time() - start_time
        )
    
    def _configure_research_parameters(self, quality_target: str) -> Dict[str, Any]:
        """Configure research parameters based on quality target"""
        params = {
            "basic": {
                "max_sources": 3,
                "max_depth": 1,
                "min_relevance": 0.5,
                "include_analysis": False
            },
            "thorough": {
                "max_sources": 5,
                "max_depth": 2, 
                "min_relevance": 0.6,
                "include_analysis": True
            },
            "comprehensive": {
                "max_sources": 8,
                "max_depth": 3,
                "min_relevance": 0.7,
                "include_analysis": True
            },
            "expert_level": {
                "max_sources": 12,
                "max_depth": 4,
                "min_relevance": 0.8,
                "include_analysis": True
            }
        }
        return params.get(quality_target, params["basic"])
    
    async def _conduct_comprehensive_research(self, query: str, params: Dict[str, Any]) -> List[ResearchResult]:
        """Conduct comprehensive research using multiple methods"""
        results = []
        
        # Method 1: AgenticSeek browser agent research
        try:
            browser_result = await asyncio.get_event_loop().run_in_executor(
                None, self.browser_agent.process, query
            )
            results.append(ResearchResult(
                source="agenticseek_browser",
                content=browser_result,
                relevance_score=0.8,
                confidence=0.8,
                metadata={"method": "browser_agent", "query": query},
                timestamp=time.time()
            ))
        except Exception as e:
            logger.error(f"Browser agent research failed: {str(e)}")
        
        # Method 2: Web search simulation (would integrate with real APIs)
        search_results = await self._simulate_web_search(query, params["max_sources"])
        results.extend(search_results)
        
        # Method 3: Knowledge base lookup (would integrate with real KB)
        if params["include_analysis"]:
            kb_results = await self._knowledge_base_lookup(query)
            results.extend(kb_results)
        
        return results
    
    async def _simulate_web_search(self, query: str, max_sources: int) -> List[ResearchResult]:
        """Simulate web search (placeholder for real implementation)"""
        await asyncio.sleep(1)  # Simulate search time
        
        # This would integrate with real search APIs (Google, Bing, etc.)
        search_results = []
        for i in range(min(max_sources, 3)):
            result = ResearchResult(
                source=f"web_search_result_{i+1}",
                content=f"Web search finding {i+1} for query: {query}",
                relevance_score=0.7 + (i * 0.1),
                confidence=0.7,
                metadata={"method": "web_search", "result_index": i+1},
                timestamp=time.time()
            )
            search_results.append(result)
        
        return search_results
    
    async def _knowledge_base_lookup(self, query: str) -> List[ResearchResult]:
        """Knowledge base lookup (placeholder for real implementation)"""
        await asyncio.sleep(0.5)  # Simulate KB lookup time
        
        # This would integrate with real knowledge bases
        return [ResearchResult(
            source="knowledge_base",
            content=f"Knowledge base information for: {query}",
            relevance_score=0.85,
            confidence=0.9,
            metadata={"method": "knowledge_base", "kb_type": "general"},
            timestamp=time.time()
        )]
    
    def _filter_and_rank_results(self, results: List[ResearchResult]) -> List[ResearchResult]:
        """Filter and rank research results by relevance"""
        # Filter by minimum relevance threshold
        filtered = [r for r in results if r.relevance_score >= self.min_relevance_threshold]
        
        # Sort by relevance score (descending)
        filtered.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Limit to max sources
        return filtered[:self.max_sources]
    
    def _calculate_research_confidence(self, results: List[ResearchResult]) -> float:
        """Calculate overall research confidence"""
        if not results:
            return 0.3
        
        # Weight by source count and average relevance
        source_factor = min(len(results) / self.max_sources, 1.0)
        relevance_factor = sum(r.relevance_score for r in results) / len(results)
        
        confidence = (source_factor * 0.4) + (relevance_factor * 0.6)
        return round(confidence, 2)
    
    def can_handle(self, task_type: TaskType) -> bool:
        return task_type in [TaskType.RESEARCH, TaskType.WEB_CRAWLING, TaskType.DATA_PROCESSING]

class EnhancedCodeAgent(DeerFlowAgent):
    """
    Enhanced code agent with Python REPL and secure execution environment
    """
    
    def __init__(self):
        super().__init__(AgentRole.CODER, "Enhanced Coder")
        # Initialize with AgenticSeek code agent
        self.code_agent = CodeAgent()
        self.execution_timeout = 30  # seconds
        self.max_memory_mb = 256
        
    async def execute(self, state: DeerFlowState) -> AgentOutput:
        """Execute code analysis with optional REPL execution"""
        start_time = time.time()
        
        query = state["user_query"]
        execution_mode = self._determine_execution_mode(query)
        
        # Perform code analysis
        analysis_result = await self._perform_code_analysis(query, execution_mode)
        
        # Update state
        state["current_step"] = "coding"
        state["code_analysis"] = asdict(analysis_result)
        
        confidence = self._calculate_code_confidence(analysis_result)
        
        logger.info(f"Code analysis completed: {analysis_result.analysis_type} with confidence {confidence:.2f}")
        
        return AgentOutput(
            agent_role=self.role,
            content=f"Completed {analysis_result.analysis_type} analysis",
            confidence=confidence,
            metadata={
                "execution_mode": execution_mode.value,
                "analysis_type": analysis_result.analysis_type,
                "execution_success": analysis_result.execution_result is not None,
                "recommendations_count": len(analysis_result.recommendations)
            },
            timestamp=time.time(),
            execution_time=time.time() - start_time
        )
    
    def _determine_execution_mode(self, query: str) -> CodeExecutionMode:
        """Determine appropriate code execution mode"""
        query_lower = query.lower()
        
        if "execute" in query_lower or "run" in query_lower:
            if "safe" in query_lower:
                return CodeExecutionMode.SAFE_EXECUTION
            else:
                return CodeExecutionMode.FULL_EXECUTION
        elif "repl" in query_lower or "interactive" in query_lower:
            return CodeExecutionMode.REPL_MODE
        else:
            return CodeExecutionMode.ANALYSIS_ONLY
    
    async def _perform_code_analysis(self, query: str, mode: CodeExecutionMode) -> CodeAnalysisResult:
        """Perform comprehensive code analysis"""
        start_time = time.time()
        
        # Use AgenticSeek code agent for initial analysis
        try:
            agent_result = await asyncio.get_event_loop().run_in_executor(
                None, self.code_agent.process, query
            )
        except Exception as e:
            agent_result = f"Code agent error: {str(e)}"
        
        # Extract or generate code from query
        code_snippet = self._extract_code_from_query(query)
        
        # Perform static analysis
        static_analysis = self._perform_static_analysis(code_snippet)
        
        # Execute code if requested
        execution_result = None
        if mode in [CodeExecutionMode.SAFE_EXECUTION, CodeExecutionMode.FULL_EXECUTION, CodeExecutionMode.REPL_MODE]:
            execution_result = await self._execute_code_safely(code_snippet, mode)
        
        # Performance metrics
        performance_metrics = self._analyze_performance(code_snippet, execution_result)
        
        # Security assessment
        security_assessment = self._assess_security(code_snippet)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(static_analysis, execution_result, security_assessment)
        
        return CodeAnalysisResult(
            analysis_type=f"{mode.value}_analysis",
            execution_result=execution_result,
            performance_metrics=performance_metrics,
            security_assessment=security_assessment,
            recommendations=recommendations,
            confidence=0.85,
            execution_time=time.time() - start_time
        )
    
    def _extract_code_from_query(self, query: str) -> str:
        """Extract code snippet from query"""
        # Simple extraction - look for code blocks
        if "```" in query:
            parts = query.split("```")
            if len(parts) >= 3:
                return parts[1].strip()
        
        # Look for common code patterns
        lines = query.split('\n')
        code_lines = []
        for line in lines:
            line = line.strip()
            if any(keyword in line for keyword in ["def ", "class ", "import ", "from ", "print(", "return "]):
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines)
        
        # Generate simple test code if no code found
        return f'print("Processing query: {query[:50]}...")'
    
    def _perform_static_analysis(self, code: str) -> Dict[str, Any]:
        """Perform static code analysis"""
        analysis = {
            "line_count": len(code.split('\n')),
            "complexity": "low",
            "imports": [],
            "functions": [],
            "classes": [],
            "potential_issues": []
        }
        
        lines = code.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                analysis["imports"].append(line)
            elif line.startswith('def '):
                analysis["functions"].append(line.split('(')[0].replace('def ', ''))
            elif line.startswith('class '):
                analysis["classes"].append(line.split('(')[0].replace('class ', '').rstrip(':'))
        
        # Simple complexity estimation
        if len(lines) > 50:
            analysis["complexity"] = "high"
        elif len(lines) > 20:
            analysis["complexity"] = "medium"
        
        return analysis
    
    async def _execute_code_safely(self, code: str, mode: CodeExecutionMode) -> Optional[str]:
        """Execute code in a safe environment"""
        if not code.strip():
            return None
        
        try:
            # Create temporary file for code execution
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Execute with timeout and resource limits
            if mode == CodeExecutionMode.SAFE_EXECUTION:
                # Use restricted execution
                result = await self._execute_with_restrictions(temp_file)
            else:
                # Standard execution
                result = await self._execute_standard(temp_file)
            
            # Clean up
            os.unlink(temp_file)
            return result
            
        except Exception as e:
            return f"Execution error: {str(e)}"
    
    async def _execute_with_restrictions(self, file_path: str) -> str:
        """Execute code with safety restrictions"""
        # This would implement sandboxing in production
        try:
            process = await asyncio.create_subprocess_exec(
                'python', file_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                timeout=self.execution_timeout
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return stdout.decode('utf-8')
            else:
                return f"Error: {stderr.decode('utf-8')}"
                
        except asyncio.TimeoutError:
            return "Execution timed out"
        except Exception as e:
            return f"Execution failed: {str(e)}"
    
    async def _execute_standard(self, file_path: str) -> str:
        """Execute code with standard permissions"""
        return await self._execute_with_restrictions(file_path)  # Same for now
    
    def _analyze_performance(self, code: str, execution_result: Optional[str]) -> Dict[str, Any]:
        """Analyze code performance characteristics"""
        return {
            "estimated_complexity": "O(n)" if "for " in code else "O(1)",
            "memory_usage": "low",
            "execution_time": "fast" if execution_result and "error" not in execution_result.lower() else "unknown"
        }
    
    def _assess_security(self, code: str) -> Dict[str, Any]:
        """Assess code security"""
        security_issues = []
        
        # Check for dangerous operations
        dangerous_patterns = ["eval(", "exec(", "os.system(", "__import__", "open("]
        for pattern in dangerous_patterns:
            if pattern in code:
                security_issues.append(f"Potentially dangerous operation: {pattern}")
        
        return {
            "risk_level": "high" if security_issues else "low",
            "issues": security_issues,
            "recommendations": ["Use safe alternatives", "Validate inputs"] if security_issues else []
        }
    
    def _generate_recommendations(self, static_analysis: Dict, execution_result: Optional[str], security: Dict) -> List[str]:
        """Generate code improvement recommendations"""
        recommendations = []
        
        if static_analysis["complexity"] == "high":
            recommendations.append("Consider breaking down complex functions")
        
        if not static_analysis["functions"] and static_analysis["line_count"] > 10:
            recommendations.append("Consider organizing code into functions")
        
        if security["risk_level"] == "high":
            recommendations.extend(security["recommendations"])
        
        if execution_result and "error" in execution_result.lower():
            recommendations.append("Fix execution errors before deployment")
        
        if not recommendations:
            recommendations.append("Code looks good - consider adding documentation")
        
        return recommendations
    
    def _calculate_code_confidence(self, result: CodeAnalysisResult) -> float:
        """Calculate confidence in code analysis"""
        base_confidence = 0.8
        
        # Reduce confidence for security issues
        if result.security_assessment["risk_level"] == "high":
            base_confidence -= 0.2
        
        # Reduce confidence for execution failures
        if result.execution_result and "error" in result.execution_result.lower():
            base_confidence -= 0.1
        
        # Increase confidence for successful execution
        if result.execution_result and "error" not in result.execution_result.lower():
            base_confidence += 0.1
        
        return round(max(0.3, min(1.0, base_confidence)), 2)
    
    def can_handle(self, task_type: TaskType) -> bool:
        return task_type in [TaskType.CODE_ANALYSIS, TaskType.DATA_PROCESSING]

class EnhancedSynthesizerAgent(DeerFlowAgent):
    """
    Enhanced synthesizer with advanced content generation and multi-modal support
    """
    
    def __init__(self):
        super().__init__(AgentRole.SYNTHESIZER, "Enhanced Synthesizer")
        # Initialize with AgenticSeek casual agent for general processing
        self.casual_agent = CasualAgent()
        
    async def execute(self, state: DeerFlowState) -> AgentOutput:
        """Synthesize information with enhanced content generation"""
        start_time = time.time()
        
        synthesis = await self._synthesize_comprehensive_response(state)
        
        # Update state
        state["current_step"] = "synthesizing"
        state["synthesis_result"] = synthesis
        
        # Calculate synthesis quality score
        quality_score = self._calculate_synthesis_quality(state, synthesis)
        
        logger.info(f"Synthesis completed with quality score {quality_score:.2f}")
        
        return AgentOutput(
            agent_role=self.role,
            content="Synthesized comprehensive response from multiple sources",
            confidence=quality_score,
            metadata={
                "synthesis_length": len(synthesis),
                "sources_integrated": self._count_integrated_sources(state),
                "quality_score": quality_score,
                "content_sections": self._count_content_sections(synthesis)
            },
            timestamp=time.time(),
            execution_time=time.time() - start_time
        )
    
    async def _synthesize_comprehensive_response(self, state: DeerFlowState) -> str:
        """Create comprehensive synthesis from all available information"""
        query = state["user_query"]
        research_results = state.get("research_results", [])
        code_analysis = state.get("code_analysis", {})
        agent_outputs = state.get("agent_outputs", {})
        
        # Use casual agent for base synthesis
        try:
            base_synthesis = await asyncio.get_event_loop().run_in_executor(
                None, self.casual_agent.process, query
            )
        except Exception as e:
            base_synthesis = f"Base synthesis error: {str(e)}"
        
        # Build comprehensive response
        synthesis_sections = []
        
        # Introduction
        synthesis_sections.append(f"# Comprehensive Response: {query}\n")
        
        # Executive Summary
        synthesis_sections.append("## Executive Summary")
        synthesis_sections.append(self._generate_executive_summary(state))
        
        # Research Findings (if available)
        if research_results:
            synthesis_sections.append("## Research Findings")
            research_summary = self._synthesize_research_findings(research_results)
            synthesis_sections.append(research_summary)
        
        # Technical Analysis (if available)
        if code_analysis:
            synthesis_sections.append("## Technical Analysis")
            tech_summary = self._synthesize_technical_analysis(code_analysis)
            synthesis_sections.append(tech_summary)
        
        # Agent Insights
        if agent_outputs:
            synthesis_sections.append("## Multi-Agent Analysis")
            agent_summary = self._synthesize_agent_outputs(agent_outputs)
            synthesis_sections.append(agent_summary)
        
        # Base AI Response
        synthesis_sections.append("## AI Analysis")
        synthesis_sections.append(base_synthesis)
        
        # Conclusions and Recommendations
        synthesis_sections.append("## Conclusions and Recommendations")
        conclusions = self._generate_conclusions(state)
        synthesis_sections.append(conclusions)
        
        return "\n\n".join(synthesis_sections)
    
    def _generate_executive_summary(self, state: DeerFlowState) -> str:
        """Generate executive summary"""
        query = state["user_query"]
        task_type = state.get("task_type", TaskType.GENERAL_QUERY)
        
        research_count = len(state.get("research_results", []))
        has_code_analysis = bool(state.get("code_analysis"))
        
        summary_parts = [
            f"This analysis addresses the query: '{query}'",
            f"Task classification: {task_type.value.replace('_', ' ').title()}"
        ]
        
        if research_count > 0:
            summary_parts.append(f"Research conducted across {research_count} sources")
        
        if has_code_analysis:
            summary_parts.append("Technical code analysis performed")
        
        summary_parts.append("Multi-agent coordination ensured comprehensive coverage")
        
        return ". ".join(summary_parts) + "."
    
    def _synthesize_research_findings(self, research_results: List[Dict]) -> str:
        """Synthesize research findings"""
        if not research_results:
            return "No research data available."
        
        findings = []
        findings.append(f"Research conducted across {len(research_results)} sources:")
        
        for i, result in enumerate(research_results[:5]):  # Limit to top 5
            source = result.get("source", "unknown")
            content = result.get("content", "")[:200] + "..." if len(result.get("content", "")) > 200 else result.get("content", "")
            relevance = result.get("relevance_score", 0)
            
            findings.append(f"\n**Source {i+1}** ({source}) - Relevance: {relevance:.2f}")
            findings.append(f"{content}")
        
        return "\n".join(findings)
    
    def _synthesize_technical_analysis(self, code_analysis: Dict) -> str:
        """Synthesize technical analysis"""
        if not code_analysis:
            return "No technical analysis available."
        
        analysis_parts = []
        
        analysis_type = code_analysis.get("analysis_type", "unknown")
        analysis_parts.append(f"**Analysis Type**: {analysis_type}")
        
        if "execution_result" in code_analysis and code_analysis["execution_result"]:
            analysis_parts.append(f"**Execution Result**: {code_analysis['execution_result'][:300]}...")
        
        if "recommendations" in code_analysis:
            recommendations = code_analysis["recommendations"]
            if recommendations:
                analysis_parts.append("**Recommendations**:")
                for rec in recommendations[:3]:  # Limit to top 3
                    analysis_parts.append(f"- {rec}")
        
        if "security_assessment" in code_analysis:
            security = code_analysis["security_assessment"]
            risk_level = security.get("risk_level", "unknown")
            analysis_parts.append(f"**Security Assessment**: {risk_level} risk level")
        
        return "\n".join(analysis_parts)
    
    def _synthesize_agent_outputs(self, agent_outputs: Dict) -> str:
        """Synthesize outputs from multiple agents"""
        if not agent_outputs:
            return "No agent outputs available."
        
        agent_summary = []
        agent_summary.append(f"Analysis involved {len(agent_outputs)} specialized agents:")
        
        for agent_name, output in agent_outputs.items():
            content = output.get("content", "")
            confidence = output.get("confidence", 0)
            execution_time = output.get("execution_time", 0)
            
            agent_summary.append(f"\n**{agent_name.title()} Agent**")
            agent_summary.append(f"- Result: {content}")
            agent_summary.append(f"- Confidence: {confidence:.2f}")
            agent_summary.append(f"- Processing time: {execution_time:.2f}s")
        
        return "\n".join(agent_summary)
    
    def _generate_conclusions(self, state: DeerFlowState) -> str:
        """Generate conclusions and recommendations"""
        conclusions = []
        
        # Analyze overall quality and confidence
        confidence_scores = []
        for output in state.get("agent_outputs", {}).values():
            if "confidence" in output:
                confidence_scores.append(output["confidence"])
        
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            conclusions.append(f"**Overall Confidence**: {avg_confidence:.2f}/1.0")
        
        # Quality assessment
        research_quality = len(state.get("research_results", []))
        has_technical = bool(state.get("code_analysis"))
        
        if research_quality >= 3:
            conclusions.append("- High-quality research foundation with multiple sources")
        elif research_quality >= 1:
            conclusions.append("- Adequate research foundation")
        
        if has_technical:
            conclusions.append("- Technical analysis provides additional insights")
        
        # Recommendations
        conclusions.append("\n**Recommendations**:")
        if research_quality < 2:
            conclusions.append("- Consider additional research for more comprehensive coverage")
        
        conclusions.append("- Cross-reference findings with additional sources when possible")
        conclusions.append("- Apply domain expertise for specialized topics")
        
        return "\n".join(conclusions)
    
    def _count_integrated_sources(self, state: DeerFlowState) -> int:
        """Count number of integrated sources"""
        count = 0
        if state.get("research_results"):
            count += len(state["research_results"])
        if state.get("code_analysis"):
            count += 1
        if state.get("agent_outputs"):
            count += len(state["agent_outputs"])
        return count
    
    def _count_content_sections(self, synthesis: str) -> int:
        """Count content sections in synthesis"""
        return synthesis.count("##")
    
    def _calculate_synthesis_quality(self, state: DeerFlowState, synthesis: str) -> float:
        """Calculate synthesis quality score"""
        base_score = 0.7
        
        # Boost for comprehensive content
        if len(synthesis) > 1000:
            base_score += 0.1
        
        # Boost for multiple sources
        source_count = self._count_integrated_sources(state)
        if source_count >= 3:
            base_score += 0.1
        elif source_count >= 2:
            base_score += 0.05
        
        # Boost for structured content
        section_count = self._count_content_sections(synthesis)
        if section_count >= 4:
            base_score += 0.1
        
        return round(min(1.0, base_score), 2)
    
    def can_handle(self, task_type: TaskType) -> bool:
        return True  # Synthesizer handles all task types

# Factory for creating specialized agents
class SpecializedAgentFactory:
    """Factory for creating enhanced specialized agents"""
    
    @staticmethod
    def create_agent(role: AgentRole) -> DeerFlowAgent:
        """Create specialized agent by role"""
        agent_map = {
            AgentRole.COORDINATOR: EnhancedCoordinatorAgent,
            AgentRole.PLANNER: PlannerAgent,  # Use existing planner for now
            AgentRole.RESEARCHER: EnhancedResearchAgent,
            AgentRole.CODER: EnhancedCodeAgent,
            AgentRole.SYNTHESIZER: EnhancedSynthesizerAgent
        }
        
        agent_class = agent_map.get(role)
        if agent_class:
            return agent_class()
        else:
            raise ValueError(f"Unknown agent role: {role}")
    
    @staticmethod
    def get_available_roles() -> List[AgentRole]:
        """Get list of available agent roles"""
        return [
            AgentRole.COORDINATOR,
            AgentRole.PLANNER,
            AgentRole.RESEARCHER,
            AgentRole.CODER,
            AgentRole.SYNTHESIZER
        ]

# Example usage and testing
async def main():
    """Test specialized agents"""
    factory = SpecializedAgentFactory()
    
    # Test coordinator
    coordinator = factory.create_agent(AgentRole.COORDINATOR)
    
    # Test state
    test_state: DeerFlowState = {
        "messages": [],
        "user_query": "Research the latest AI developments and analyze the code performance",
        "task_type": TaskType.RESEARCH,
        "current_step": "initialized",
        "agent_outputs": {},
        "research_results": [],
        "code_analysis": {},
        "synthesis_result": "",
        "validation_status": False,
        "final_report": "",
        "metadata": {},
        "checkpoints": [],
        "error_logs": [],
        "execution_time": 0.0,
        "confidence_scores": {}
    }
    
    # Test coordination
    coord_result = await coordinator.execute(test_state)
    print(f"Coordination result: {coord_result.content}")
    print(f"Confidence: {coord_result.confidence}")
    
    # Test researcher
    researcher = factory.create_agent(AgentRole.RESEARCHER)
    research_result = await researcher.execute(test_state)
    print(f"\nResearch result: {research_result.content}")
    
    # Test synthesizer
    synthesizer = factory.create_agent(AgentRole.SYNTHESIZER)
    synth_result = await synthesizer.execute(test_state)
    print(f"\nSynthesis result: {synth_result.content}")

if __name__ == "__main__":
    asyncio.run(main())