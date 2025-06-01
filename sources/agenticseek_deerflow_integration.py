#!/usr/bin/env python3
"""
* Purpose: Comprehensive AgenticSeek-DeerFlow integration with voice-first multi-agent coordination
* Issues & Complexity Summary: Complex integration of voice processing, multi-agent coordination, and state management
* Key Complexity Drivers:
  - Logic Scope (Est. LoC): ~500
  - Core Algorithm Complexity: Very High
  - Dependencies: 10 New, 5 Mod
  - State Management Complexity: Very High
  - Novelty/Uncertainty Factor: High
* AI Pre-Task Self-Assessment (Est. Solution Difficulty %): 92%
* Problem Estimate (Inherent Problem Difficulty %): 95%
* Initial Code Complexity Estimate %: 90%
* Justification for Estimates: Complete integration of voice, multi-agent, and state systems
* Final Code Complexity (Actual %): TBD
* Overall Result Score (Success & Quality %): TBD
* Key Variances/Learnings: TBD
* Last Updated: 2025-01-06
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union, Callable
from enum import Enum
import logging
from contextlib import asynccontextmanager

# Import DeerFlow components
if __name__ == "__main__":
    from deer_flow_orchestrator import DeerFlowOrchestrator, DeerFlowState, TaskType, AgentRole
    from specialized_agents import SpecializedAgentFactory, EnhancedCoordinatorAgent, EnhancedResearchAgent, EnhancedCodeAgent, EnhancedSynthesizerAgent
    from multi_agent_coordinator import MultiAgentCoordinator, AgentResult, ConsensusResult
    from speech_to_text import AudioTranscriber, AudioRecorder
    from text_to_speech import Speech
    from interaction import Interaction
    from utility import pretty_print
else:
    from sources.deer_flow_orchestrator import DeerFlowOrchestrator, DeerFlowState, TaskType, AgentRole
    from sources.specialized_agents import SpecializedAgentFactory, EnhancedCoordinatorAgent, EnhancedResearchAgent, EnhancedCodeAgent, EnhancedSynthesizerAgent
    from sources.multi_agent_coordinator import MultiAgentCoordinator, AgentResult, ConsensusResult
    from sources.speech_to_text import AudioTranscriber, AudioRecorder
    from sources.text_to_speech import Speech
    from sources.interaction import Interaction
    from sources.utility import pretty_print

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceInteractionMode(Enum):
    """Voice interaction modes"""
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    IDLE = "idle"

class ProcessingStatus(Enum):
    """Processing status for voice commands"""
    INITIALIZING = "initializing"
    COORDINATING = "coordinating"
    RESEARCHING = "researching"
    ANALYZING = "analyzing"
    SYNTHESIZING = "synthesizing"
    COMPLETING = "completing"
    ERROR = "error"

@dataclass
class VoiceInteractionState:
    """State for voice interaction management"""
    mode: VoiceInteractionMode
    is_active: bool
    current_transcript: str
    processing_status: ProcessingStatus
    agent_status: Dict[str, str]
    confidence_level: float
    execution_time: float
    last_response: str
    session_id: str

@dataclass
class IntegratedResponse:
    """Comprehensive response from integrated system"""
    content: str
    spoken_summary: str
    voice_optimized_summary: str
    confidence: float
    processing_time: float
    sources_used: List[str]
    agents_involved: List[str]
    metadata: Dict[str, Any]
    session_id: str

class AgenticSeekDeerFlowIntegration:
    """
    Comprehensive integration of AgenticSeek voice capabilities with DeerFlow multi-agent orchestration
    """
    
    def __init__(self, enable_voice: bool = True, enable_peer_review: bool = True):
        self.enable_voice = enable_voice
        self.enable_peer_review = enable_peer_review
        
        # Initialize core components
        self.deer_flow_orchestrator = DeerFlowOrchestrator(enable_checkpointing=True)
        self.multi_agent_coordinator = MultiAgentCoordinator(enable_peer_review=enable_peer_review)
        self.agent_factory = SpecializedAgentFactory()
        
        # Voice components (if enabled)
        if enable_voice:
            self.audio_transcriber = None
            self.audio_recorder = None
            self.speech_synthesizer = Speech(enable=True, language="en", voice_idx=6)
            self.interaction_manager = Interaction()
        
        # Integration state
        self.voice_state = VoiceInteractionState(
            mode=VoiceInteractionMode.IDLE,
            is_active=False,
            current_transcript="",
            processing_status=ProcessingStatus.INITIALIZING,
            agent_status={},
            confidence_level=0.0,
            execution_time=0.0,
            last_response="",
            session_id=str(uuid.uuid4())
        )
        
        # Register specialized agents with coordinator
        self._register_specialized_agents()
        
        # Session management
        self.active_sessions: Dict[str, Dict] = {}
        self.processing_queue = asyncio.Queue()
        
        logger.info(f"AgenticSeek-DeerFlow Integration initialized - Voice: {enable_voice}, Peer Review: {enable_peer_review}")
    
    def _register_specialized_agents(self):
        """Register enhanced specialized agents with the multi-agent coordinator"""
        try:
            # Create and register enhanced agents
            roles_to_register = [
                AgentRole.COORDINATOR,
                AgentRole.RESEARCHER, 
                AgentRole.CODER,
                AgentRole.SYNTHESIZER
            ]
            
            for role in roles_to_register:
                try:
                    agent = self.agent_factory.create_agent(role)
                    # Convert to multi-agent coordinator format (simplified)
                    self.multi_agent_coordinator.register_agent(role, agent)
                    logger.info(f"Registered enhanced {role.value} agent")
                except Exception as e:
                    logger.error(f"Failed to register {role.value} agent: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Agent registration failed: {str(e)}")
    
    async def start_voice_interaction(self, ai_name: str = "agenticseek") -> bool:
        """Start voice interaction system"""
        if not self.enable_voice:
            logger.warning("Voice interaction not enabled")
            return False
        
        try:
            # Initialize voice components
            self.audio_recorder = AudioRecorder(verbose=True)
            self.audio_transcriber = AudioTranscriber(ai_name=ai_name, verbose=True)
            
            # Start audio processing
            self.audio_recorder.start()
            self.audio_transcriber.start()
            
            # Update state
            self.voice_state.mode = VoiceInteractionMode.LISTENING
            self.voice_state.is_active = True
            
            logger.info("Voice interaction started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start voice interaction: {str(e)}")
            return False
    
    async def stop_voice_interaction(self):
        """Stop voice interaction system"""
        if self.audio_recorder:
            self.audio_recorder.join()
        if self.audio_transcriber:
            self.audio_transcriber.join()
        
        self.voice_state.mode = VoiceInteractionMode.IDLE
        self.voice_state.is_active = False
        
        logger.info("Voice interaction stopped")
    
    async def process_voice_command(self, timeout: float = 60.0) -> Optional[IntegratedResponse]:
        """Process voice command with full multi-agent coordination"""
        if not self.voice_state.is_active:
            return None
        
        try:
            # Wait for voice input
            transcript = await self._wait_for_voice_input(timeout)
            if not transcript:
                return None
            
            # Update state
            self.voice_state.current_transcript = transcript
            self.voice_state.mode = VoiceInteractionMode.PROCESSING
            self.voice_state.processing_status = ProcessingStatus.COORDINATING
            
            # Process with integrated system
            response = await self._process_integrated_command(transcript)
            
            # Speak response if voice enabled
            if self.enable_voice and response:
                await self._speak_response(response.voice_optimized_summary)
            
            # Update state
            self.voice_state.mode = VoiceInteractionMode.IDLE
            self.voice_state.processing_status = ProcessingStatus.COMPLETING
            self.voice_state.last_response = response.content if response else ""
            
            return response
            
        except Exception as e:
            logger.error(f"Voice command processing failed: {str(e)}")
            self.voice_state.processing_status = ProcessingStatus.ERROR
            return None
    
    async def process_text_command(self, query: str) -> IntegratedResponse:
        """Process text command with full multi-agent coordination"""
        return await self._process_integrated_command(query)
    
    async def _process_integrated_command(self, query: str) -> IntegratedResponse:
        """Process command using integrated DeerFlow + Multi-Agent system"""
        session_id = str(uuid.uuid4())
        start_time = time.time()
        
        logger.info(f"Processing integrated command: {query[:100]}...")
        
        try:
            # Phase 1: DeerFlow orchestration for workflow management
            self.voice_state.processing_status = ProcessingStatus.COORDINATING
            deer_flow_result = await self.deer_flow_orchestrator.execute_workflow(
                user_query=query,
                task_type=None  # Let coordinator decide
            )
            
            # Phase 2: Multi-agent peer review (if enabled)
            consensus_result = None
            if self.enable_peer_review:
                self.voice_state.processing_status = ProcessingStatus.ANALYZING
                consensus_result = await self.multi_agent_coordinator.execute_with_peer_review(
                    query=query,
                    task_type=deer_flow_result.get("task_type", TaskType.GENERAL_QUERY).value if deer_flow_result.get("task_type") else "general"
                )
            
            # Phase 3: Synthesis and optimization
            self.voice_state.processing_status = ProcessingStatus.SYNTHESIZING
            integrated_response = await self._synthesize_integrated_response(
                query, deer_flow_result, consensus_result, start_time, session_id
            )
            
            # Update session tracking
            self.active_sessions[session_id] = {
                "query": query,
                "deer_flow_result": deer_flow_result,
                "consensus_result": consensus_result,
                "response": integrated_response,
                "timestamp": time.time()
            }
            
            logger.info(f"Integrated command processed in {integrated_response.processing_time:.2f}s")
            return integrated_response
            
        except Exception as e:
            logger.error(f"Integrated command processing failed: {str(e)}")
            
            # Return error response
            return IntegratedResponse(
                content=f"I encountered an error processing your request: {str(e)}",
                spoken_summary="I'm sorry, I encountered an error processing your request.",
                voice_optimized_summary="Sorry, there was an error.",
                confidence=0.1,
                processing_time=time.time() - start_time,
                sources_used=[],
                agents_involved=[],
                metadata={"error": str(e)},
                session_id=session_id
            )
    
    async def _synthesize_integrated_response(
        self, 
        query: str, 
        deer_flow_result: Dict[str, Any], 
        consensus_result: Optional[ConsensusResult],
        start_time: float,
        session_id: str
    ) -> IntegratedResponse:
        """Synthesize response from DeerFlow and consensus results"""
        
        # Extract key information
        final_report = deer_flow_result.get("final_report", "")
        agent_outputs = deer_flow_result.get("agent_outputs", {})
        confidence_scores = deer_flow_result.get("confidence_scores", {})
        
        # Integrate consensus if available
        if consensus_result:
            final_content = consensus_result.final_content
            overall_confidence = consensus_result.confidence_level
            sources_used = [result.agent_role.value for result in [consensus_result.primary_result]]
            agents_involved = list(set([output.get("agent_role", "unknown") for output in agent_outputs.values()]))
        else:
            final_content = final_report
            overall_confidence = confidence_scores.get("overall", 0.7)
            sources_used = []
            agents_involved = list(agent_outputs.keys())
        
        # Create voice-optimized summaries
        spoken_summary = self._create_spoken_summary(final_content)
        voice_optimized = self._create_voice_optimized_summary(final_content)
        
        # Collect metadata
        metadata = {
            "deer_flow_execution_time": deer_flow_result.get("execution_time", 0),
            "consensus_processing_time": consensus_result.total_processing_time if consensus_result else 0,
            "agent_count": len(agents_involved),
            "checkpoints": len(deer_flow_result.get("checkpoints", [])),
            "task_type": deer_flow_result.get("task_type", TaskType.GENERAL_QUERY).value if deer_flow_result.get("task_type") else "general",
            "has_peer_review": consensus_result is not None
        }
        
        return IntegratedResponse(
            content=final_content,
            spoken_summary=spoken_summary,
            voice_optimized_summary=voice_optimized,
            confidence=overall_confidence,
            processing_time=time.time() - start_time,
            sources_used=sources_used,
            agents_involved=agents_involved,
            metadata=metadata,
            session_id=session_id
        )
    
    def _create_spoken_summary(self, content: str) -> str:
        """Create spoken summary optimized for TTS"""
        if len(content) <= 200:
            return content
        
        # Extract first paragraph or sentence
        sentences = content.split('. ')
        if sentences:
            first_sentence = sentences[0]
            if len(first_sentence) <= 150:
                return first_sentence + "."
        
        # Fallback to truncation
        return content[:150] + "..."
    
    def _create_voice_optimized_summary(self, content: str) -> str:
        """Create ultra-short summary for voice interaction"""
        if len(content) <= 50:
            return content
        
        # Very short summary
        words = content.split()[:8]
        return " ".join(words) + "..."
    
    async def _wait_for_voice_input(self, timeout: float) -> Optional[str]:
        """Wait for voice input with timeout"""
        if not self.audio_transcriber:
            return None
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            transcript = self.audio_transcriber.get_transcript()
            if transcript and transcript.strip():
                return transcript.strip()
            await asyncio.sleep(0.1)
        
        return None
    
    async def _speak_response(self, text: str):
        """Speak response using TTS"""
        if not self.speech_synthesizer:
            return
        
        try:
            self.voice_state.mode = VoiceInteractionMode.SPEAKING
            # Run TTS in executor to avoid blocking
            await asyncio.get_event_loop().run_in_executor(
                None, self.speech_synthesizer.speak, text
            )
            self.voice_state.mode = VoiceInteractionMode.IDLE
        except Exception as e:
            logger.error(f"TTS failed: {str(e)}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        deer_flow_status = self.deer_flow_orchestrator.get_workflow_status()
        coordinator_stats = self.multi_agent_coordinator.get_execution_stats()
        
        return {
            "voice_interaction": {
                "enabled": self.enable_voice,
                "active": self.voice_state.is_active,
                "mode": self.voice_state.mode.value,
                "processing_status": self.voice_state.processing_status.value
            },
            "deer_flow": deer_flow_status,
            "multi_agent_coordinator": coordinator_stats,
            "active_sessions": len(self.active_sessions),
            "peer_review_enabled": self.enable_peer_review,
            "session_id": self.voice_state.session_id
        }
    
    def get_voice_interaction_state(self) -> VoiceInteractionState:
        """Get current voice interaction state"""
        return self.voice_state
    
    async def run_voice_interaction_loop(self, max_iterations: int = 100):
        """Run continuous voice interaction loop"""
        if not self.enable_voice:
            logger.warning("Voice interaction not enabled")
            return
        
        logger.info("Starting voice interaction loop")
        
        # Start voice system
        voice_started = await self.start_voice_interaction()
        if not voice_started:
            logger.error("Failed to start voice interaction")
            return
        
        try:
            iteration = 0
            while iteration < max_iterations and self.voice_state.is_active:
                iteration += 1
                
                logger.info(f"Voice interaction iteration {iteration}")
                
                # Process voice command
                response = await self.process_voice_command(timeout=30.0)
                
                if response:
                    logger.info(f"Processed command with confidence {response.confidence:.2f}")
                    pretty_print(f"Response: {response.voice_optimized_summary}", color="success")
                else:
                    logger.info("No voice command received")
                
                # Short pause between iterations
                await asyncio.sleep(1.0)
                
        except KeyboardInterrupt:
            logger.info("Voice interaction loop interrupted by user")
        except Exception as e:
            logger.error(f"Voice interaction loop error: {str(e)}")
        finally:
            await self.stop_voice_interaction()
            logger.info("Voice interaction loop ended")

# Example usage and testing
async def main():
    """Example usage of integrated system"""
    # Initialize integrated system
    integration = AgenticSeekDeerFlowIntegration(
        enable_voice=False,  # Set to True for voice testing
        enable_peer_review=True
    )
    
    # Test text command processing
    print("Testing integrated text command processing...")
    
    response = await integration.process_text_command(
        "Research the latest developments in AI multi-agent systems and analyze the performance implications"
    )
    
    print(f"\nResponse Content:")
    print(response.content[:500] + "..." if len(response.content) > 500 else response.content)
    print(f"\nSpoken Summary: {response.spoken_summary}")
    print(f"Voice Optimized: {response.voice_optimized_summary}")
    print(f"Confidence: {response.confidence:.2f}")
    print(f"Processing Time: {response.processing_time:.2f}s")
    print(f"Agents Involved: {response.agents_involved}")
    
    # Show system status
    status = integration.get_system_status()
    print(f"\nSystem Status:")
    print(json.dumps(status, indent=2))

if __name__ == "__main__":
    asyncio.run(main())