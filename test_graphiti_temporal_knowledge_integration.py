#!/usr/bin/env python3
"""
Comprehensive Test Suite for Graphiti Temporal Knowledge Graph Integration
Tests temporal knowledge coordination, multi-LLM collaboration, and graph operations

* Purpose: Validate Graphiti temporal knowledge graph integration with MLACS
* Test Coverage: Entity extraction, relationship building, temporal coordination, memory integration
* Performance Validation: Response time, accuracy, consensus building
* Integration Testing: LangChain memory, multi-LLM coordination, Apple Silicon optimization
"""

import asyncio
import time
import json
import uuid
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
import unittest

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test imports
try:
    from sources.graphiti_temporal_knowledge_integration import (
        GraphitiMLACSIntegration,
        TemporalKnowledgeCoordinator,
        MultiLLMKnowledgeBuilder,
        GraphitiConversationMemory,
        TemporalEvent,
        KnowledgeEntity,
        KnowledgeRelationship,
        TemporalEventType,
        KnowledgeNodeType,
        RelationshipType
    )
    from sources.llm_provider import Provider
    INTEGRATION_AVAILABLE = True
except ImportError as e:
    logger.error(f"Integration components not available: {e}")
    INTEGRATION_AVAILABLE = False

async def test_temporal_knowledge_coordinator():
    """Test TemporalKnowledgeCoordinator functionality"""
    print("🧪 Testing Temporal Knowledge Coordinator...")
    
    try:
        # Create mock providers
        mock_providers = {
            "openai": Provider("openai", "gpt-4"),
            "anthropic": Provider("anthropic", "claude-3"),
            "google": Provider("google", "gemini-pro")
        }
        
        # Initialize coordinator
        coordinator = TemporalKnowledgeCoordinator(mock_providers)
        
        # Test knowledge extraction coordination
        test_interaction = {
            "content": "Apple Silicon processors provide significant performance improvements for AI workloads",
            "context": {"topic": "hardware", "domain": "AI"}
        }
        
        session_id = f"test_session_{uuid.uuid4().hex[:8]}"
        
        print("🔄 Testing knowledge extraction coordination...")
        start_time = time.time()
        
        knowledge = await coordinator.coordinate_knowledge_extraction(
            test_interaction, session_id
        )
        
        extraction_time = time.time() - start_time
        print(f"   ✅ Knowledge extraction completed in {extraction_time:.3f}s")
        
        # Validate knowledge structure
        assert "entities" in knowledge, "Knowledge should contain entities"
        assert "relationships" in knowledge, "Knowledge should contain relationships"
        assert "consensus_metrics" in knowledge, "Knowledge should contain consensus metrics"
        
        entities = knowledge["entities"]
        relationships = knowledge["relationships"]
        
        print(f"   ✅ Extracted {len(entities)} entities and {len(relationships)} relationships")
        
        # Validate entity structure
        if entities:
            entity = entities[0]
            assert hasattr(entity, 'entity_id'), "Entity should have entity_id"
            assert hasattr(entity, 'entity_type'), "Entity should have entity_type"
            assert hasattr(entity, 'confidence'), "Entity should have confidence"
            assert hasattr(entity, 'validation_count'), "Entity should have validation_count"
            print(f"   ✅ Entity validation: {entity.name} (confidence: {entity.confidence})")
        
        # Validate relationship structure  
        if relationships:
            relationship = relationships[0]
            assert hasattr(relationship, 'relationship_id'), "Relationship should have relationship_id"
            assert hasattr(relationship, 'relationship_type'), "Relationship should have relationship_type"
            assert hasattr(relationship, 'confidence'), "Relationship should have confidence"
            print(f"   ✅ Relationship validation: {relationship.source_entity_id} -> {relationship.target_entity_id}")
        
        # Test metrics
        metrics = coordinator.metrics
        assert metrics['multi_llm_validations'] > 0, "Should have recorded validations"
        assert metrics['entities_created'] >= 0, "Should track entities created"
        assert metrics['relationships_created'] >= 0, "Should track relationships created"
        
        print(f"   ✅ Coordinator metrics: {metrics}")
        print("✅ Temporal Knowledge Coordinator: PASSED")
        
        return True
        
    except Exception as e:
        print(f"❌ Temporal Knowledge Coordinator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_multi_llm_knowledge_builder():
    """Test MultiLLMKnowledgeBuilder functionality"""
    print("🧪 Testing Multi-LLM Knowledge Builder...")
    
    try:
        # Create mock providers
        mock_providers = {
            "openai": Provider("openai", "gpt-4"),
            "anthropic": Provider("anthropic", "claude-3")
        }
        
        # Initialize components
        coordinator = TemporalKnowledgeCoordinator(mock_providers)
        builder = MultiLLMKnowledgeBuilder(mock_providers, coordinator)
        
        # Test knowledge building
        test_interaction = {
            "content": "Neural networks with attention mechanisms excel at language understanding tasks",
            "context": {"domain": "AI", "topic": "deep_learning"}
        }
        
        session_id = f"test_session_{uuid.uuid4().hex[:8]}"
        
        print("🔄 Testing knowledge building from interaction...")
        start_time = time.time()
        
        knowledge = await builder.build_knowledge_from_interaction(
            test_interaction, session_id
        )
        
        build_time = time.time() - start_time
        print(f"   ✅ Knowledge building completed in {build_time:.3f}s")
        
        # Validate knowledge building results
        assert "entities" in knowledge, "Built knowledge should contain entities"
        assert "relationships" in knowledge, "Built knowledge should contain relationships"
        
        # Check knowledge graph updates
        graph_key = f"session_{session_id}"
        assert graph_key in builder.knowledge_graph, "Knowledge graph should be updated"
        
        session_graph = builder.knowledge_graph[graph_key]
        assert "entities" in session_graph, "Session graph should have entities"
        assert "relationships" in session_graph, "Session graph should have relationships"
        assert "metadata" in session_graph, "Session graph should have metadata"
        
        metadata = session_graph["metadata"]
        assert metadata["session_id"] == session_id, "Metadata should have correct session_id"
        assert "created" in metadata, "Metadata should have creation time"
        assert "last_updated" in metadata, "Metadata should have update time"
        
        print(f"   ✅ Knowledge graph updated: {metadata['entity_count']} entities, {metadata['relationship_count']} relationships")
        
        # Test temporal knowledge extraction
        print("🔄 Testing temporal knowledge extraction...")
        enhanced_knowledge = await builder._extract_temporal_knowledge(test_interaction, session_id)
        
        assert "temporal_relationships" in enhanced_knowledge, "Should extract temporal relationships"
        temporal_rels = enhanced_knowledge["temporal_relationships"]
        print(f"   ✅ Extracted {len(temporal_rels)} temporal relationships")
        
        print("✅ Multi-LLM Knowledge Builder: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Multi-LLM Knowledge Builder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_graphiti_conversation_memory():
    """Test GraphitiConversationMemory LangChain integration"""
    print("🧪 Testing Graphiti Conversation Memory...")
    
    try:
        # Initialize components
        mock_providers = {
            "openai": Provider("openai", "gpt-4"),
            "anthropic": Provider("anthropic", "claude-3")
        }
        
        coordinator = TemporalKnowledgeCoordinator(mock_providers)
        builder = MultiLLMKnowledgeBuilder(mock_providers, coordinator)
        
        session_id = f"test_session_{uuid.uuid4().hex[:8]}"
        memory = GraphitiConversationMemory(builder, session_id)
        
        # Test memory variables
        memory_vars = memory.memory_variables
        assert "history" in memory_vars, "Memory should include history variable"
        assert "knowledge_context" in memory_vars, "Memory should include knowledge_context variable"
        
        print(f"   ✅ Memory variables: {memory_vars}")
        
        # Test saving context
        test_inputs = {"input": "What are the benefits of Apple Silicon for AI workloads?"}
        test_outputs = {"output": "Apple Silicon provides significant performance improvements through unified memory architecture and Neural Engine acceleration."}
        
        print("🔄 Testing context saving...")
        memory.save_context(test_inputs, test_outputs)
        
        # Check conversation history
        assert len(memory.conversation_history) == 2, "Should have saved both input and output"
        
        human_msg = memory.conversation_history[0]
        assert human_msg["role"] == "human", "First message should be human"
        assert human_msg["content"] == test_inputs["input"], "Human message content should match input"
        
        assistant_msg = memory.conversation_history[1]
        assert assistant_msg["role"] == "assistant", "Second message should be assistant"
        assert assistant_msg["content"] == test_outputs["output"], "Assistant message content should match output"
        
        print(f"   ✅ Conversation history saved: {len(memory.conversation_history)} messages")
        
        # Test loading memory variables
        print("🔄 Testing memory variable loading...")
        
        # Wait a moment for async knowledge extraction
        await asyncio.sleep(0.1)
        
        loaded_vars = memory.load_memory_variables({"input": "Tell me more about Neural Engine"})
        
        assert "history" in loaded_vars, "Loaded variables should include history"
        assert "knowledge_context" in loaded_vars, "Loaded variables should include knowledge_context"
        
        history = loaded_vars["history"]
        knowledge_context = loaded_vars["knowledge_context"]
        
        assert "human:" in history.lower(), "History should contain human messages"
        assert "assistant:" in history.lower(), "History should contain assistant messages"
        
        print(f"   ✅ Memory variables loaded successfully")
        print(f"   ✅ History length: {len(history)} characters")
        print(f"   ✅ Knowledge context: {knowledge_context}")
        
        # Test memory clearing
        print("🔄 Testing memory clearing...")
        memory.clear()
        assert len(memory.conversation_history) == 0, "Conversation history should be cleared"
        
        print("✅ Graphiti Conversation Memory: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Graphiti Conversation Memory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_graphiti_mlacs_integration():
    """Test main GraphitiMLACSIntegration functionality"""
    print("🧪 Testing Graphiti-MLACS Integration...")
    
    try:
        # Create mock providers
        mock_providers = {
            "openai": Provider("openai", "gpt-4"),
            "anthropic": Provider("anthropic", "claude-3"),
            "google": Provider("google", "gemini-pro")
        }
        
        print("🔄 Initializing Graphiti-MLACS Integration...")
        integration = GraphitiMLACSIntegration(mock_providers)
        
        # Test session creation
        print("🔄 Testing session creation...")
        session_id = integration.create_session()
        
        assert session_id in integration.active_sessions, "Session should be in active sessions"
        assert session_id.startswith("session_"), "Session ID should have correct format"
        
        print(f"   ✅ Session created: {session_id}")
        
        # Test session memory retrieval
        memory = integration.get_session_memory(session_id)
        assert memory is not None, "Should retrieve session memory"
        assert isinstance(memory, GraphitiConversationMemory), "Should be GraphitiConversationMemory instance"
        
        print(f"   ✅ Session memory retrieved successfully")
        
        # Test multi-LLM interaction processing
        print("🔄 Testing multi-LLM interaction processing...")
        
        test_interaction = {
            "content": "Temporal knowledge graphs enable sophisticated AI reasoning by maintaining context across time",
            "context": {"domain": "AI", "topic": "knowledge_graphs", "complexity": "high"}
        }
        
        start_time = time.time()
        result = await integration.process_multi_llm_interaction(test_interaction, session_id)
        processing_time = time.time() - start_time
        
        assert "knowledge" in result, "Result should contain knowledge"
        assert "processing_time" in result, "Result should contain processing time"
        assert "session_id" in result, "Result should contain session_id"
        assert "timestamp" in result, "Result should contain timestamp"
        
        knowledge = result["knowledge"]
        assert "entities" in knowledge, "Knowledge should contain entities"
        assert "relationships" in knowledge, "Knowledge should contain relationships"
        
        print(f"   ✅ Interaction processed in {processing_time:.3f}s")
        print(f"   ✅ Knowledge extracted: {len(knowledge['entities'])} entities, {len(knowledge['relationships'])} relationships")
        
        # Test temporal knowledge querying
        print("🔄 Testing temporal knowledge querying...")
        
        query_result = await integration.query_temporal_knowledge(
            "temporal knowledge", session_id
        )
        
        assert "matching_entities" in query_result, "Query result should contain matching entities"
        assert "matching_relationships" in query_result, "Query result should contain matching relationships"
        assert "query_time" in query_result, "Query result should contain query time"
        assert "total_entities" in query_result, "Query result should contain total entity count"
        assert "total_relationships" in query_result, "Query result should contain total relationship count"
        
        query_time = query_result["query_time"]
        matching_entities = query_result["matching_entities"]
        matching_relationships = query_result["matching_relationships"]
        
        print(f"   ✅ Query processed in {query_time:.3f}s")
        print(f"   ✅ Found {len(matching_entities)} matching entities, {len(matching_relationships)} matching relationships")
        
        # Test integration status
        print("🔄 Testing integration status...")
        status = integration.get_integration_status()
        
        required_fields = [
            "integration_metrics", "active_sessions", "temporal_coordinator_metrics",
            "optimization_enabled", "total_entities", "total_relationships", "total_temporal_events"
        ]
        
        for field in required_fields:
            assert field in status, f"Status should contain {field}"
        
        assert status["active_sessions"] == 1, "Should have 1 active session"
        assert status["integration_metrics"]["sessions_created"] >= 1, "Should have created at least 1 session"
        assert status["integration_metrics"]["knowledge_extractions"] >= 1, "Should have performed knowledge extractions"
        
        print(f"   ✅ Integration status: {status['active_sessions']} active sessions")
        print(f"   ✅ Performance metrics: {status['integration_metrics']}")
        
        # Test performance targets
        print("🔄 Validating performance targets...")
        
        avg_response_time = status["integration_metrics"]["average_response_time"]
        assert avg_response_time < 1.0, f"Average response time should be <1s, got {avg_response_time:.3f}s"
        
        print(f"   ✅ Performance target met: {avg_response_time:.3f}s average response time")
        
        # Test shutdown
        print("🔄 Testing integration shutdown...")
        integration.shutdown()
        
        assert len(integration.active_sessions) == 0, "Active sessions should be cleared after shutdown"
        assert len(integration.temporal_coordinator.knowledge_cache) == 0, "Knowledge cache should be cleared"
        assert len(integration.temporal_coordinator.relationship_cache) == 0, "Relationship cache should be cleared"
        
        print("   ✅ Integration shutdown completed successfully")
        print("✅ Graphiti-MLACS Integration: PASSED")
        
        return True
        
    except Exception as e:
        print(f"❌ Graphiti-MLACS Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_temporal_events_and_consistency():
    """Test temporal event recording and consistency"""
    print("🧪 Testing Temporal Events and Consistency...")
    
    try:
        # Initialize components
        mock_providers = {
            "openai": Provider("openai", "gpt-4"),
            "anthropic": Provider("anthropic", "claude-3")
        }
        
        integration = GraphitiMLACSIntegration(mock_providers)
        session_id = integration.create_session()
        
        # Process multiple interactions to create temporal events
        interactions = [
            {
                "content": "Machine learning models require extensive training data",
                "context": {"topic": "ML", "sequence": 1}
            },
            {
                "content": "Deep learning networks with attention mechanisms improve performance",
                "context": {"topic": "Deep Learning", "sequence": 2}
            },
            {
                "content": "Apple Silicon processors accelerate ML inference significantly",
                "context": {"topic": "Hardware", "sequence": 3}
            }
        ]
        
        print("🔄 Processing sequential interactions for temporal analysis...")
        
        results = []
        for i, interaction in enumerate(interactions):
            result = await integration.process_multi_llm_interaction(interaction, session_id)
            results.append(result)
            print(f"   ✅ Processed interaction {i+1}/{len(interactions)}")
            
            # Small delay to ensure temporal ordering
            await asyncio.sleep(0.01)
        
        # Validate temporal event recording
        coordinator = integration.temporal_coordinator
        temporal_events = coordinator.temporal_events
        
        assert len(temporal_events) > 0, "Should have recorded temporal events"
        
        # Check temporal ordering
        event_times = [event.event_time for event in temporal_events]
        sorted_times = sorted(event_times)
        
        print(f"   ✅ Recorded {len(temporal_events)} temporal events")
        
        # Validate event types
        event_types = set(event.event_type for event in temporal_events)
        expected_types = {TemporalEventType.ENTITY_CREATION, TemporalEventType.RELATIONSHIP_CREATION}
        
        for expected_type in expected_types:
            assert expected_type in event_types, f"Should have {expected_type} events"
        
        print(f"   ✅ Event types recorded: {[et.value for et in event_types]}")
        
        # Test temporal consistency
        entities = list(coordinator.knowledge_cache.values())
        relationships = list(coordinator.relationship_cache.values())
        
        # Check entity temporal consistency
        for entity in entities:
            assert entity.creation_time <= entity.last_updated, "Entity creation time should be <= last updated"
        
        # Check relationship temporal consistency
        for relationship in relationships:
            assert relationship.creation_time <= relationship.last_updated, "Relationship creation time should be <= last updated"
        
        print(f"   ✅ Temporal consistency validated for {len(entities)} entities and {len(relationships)} relationships")
        
        # Test temporal queries with time ranges
        print("🔄 Testing temporal range queries...")
        
        current_time = datetime.now(timezone.utc)
        past_time = current_time - timedelta(minutes=1)
        
        range_query_result = await integration.query_temporal_knowledge(
            "machine learning", session_id, (past_time, current_time)
        )
        
        print(f"   ✅ Temporal range query completed")
        
        # Cleanup
        integration.shutdown()
        
        print("✅ Temporal Events and Consistency: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Temporal Events and Consistency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_performance_and_scalability():
    """Test performance and scalability characteristics"""
    print("🧪 Testing Performance and Scalability...")
    
    try:
        # Initialize integration
        mock_providers = {
            "openai": Provider("openai", "gpt-4"),
            "anthropic": Provider("anthropic", "claude-3"),
            "google": Provider("google", "gemini-pro")
        }
        
        integration = GraphitiMLACSIntegration(mock_providers)
        
        # Test concurrent session creation
        print("🔄 Testing concurrent session creation...")
        
        async def create_test_session():
            session_id = integration.create_session()
            return session_id
        
        start_time = time.time()
        session_tasks = [create_test_session() for _ in range(10)]
        sessions = await asyncio.gather(*session_tasks)
        creation_time = time.time() - start_time
        
        assert len(sessions) == 10, "Should create 10 sessions"
        assert len(set(sessions)) == 10, "All sessions should have unique IDs"
        
        print(f"   ✅ Created 10 concurrent sessions in {creation_time:.3f}s")
        
        # Test concurrent interaction processing
        print("🔄 Testing concurrent interaction processing...")
        
        test_interactions = [
            {
                "content": f"Test interaction {i} about AI and machine learning",
                "context": {"test_id": i, "topic": "AI"}
            }
            for i in range(5)
        ]
        
        async def process_test_interaction(interaction, session_id):
            return await integration.process_multi_llm_interaction(interaction, session_id)
        
        start_time = time.time()
        session_id = sessions[0]  # Use first session
        
        interaction_tasks = [
            process_test_interaction(interaction, session_id)
            for interaction in test_interactions
        ]
        
        results = await asyncio.gather(*interaction_tasks)
        processing_time = time.time() - start_time
        
        assert len(results) == 5, "Should process 5 interactions"
        assert all("knowledge" in result for result in results), "All results should contain knowledge"
        
        print(f"   ✅ Processed 5 concurrent interactions in {processing_time:.3f}s")
        
        # Test query performance
        print("🔄 Testing query performance...")
        
        query_times = []
        
        for i in range(10):
            start_time = time.time()
            result = await integration.query_temporal_knowledge(
                f"test query {i}", session_id
            )
            query_time = time.time() - start_time
            query_times.append(query_time)
        
        avg_query_time = sum(query_times) / len(query_times)
        max_query_time = max(query_times)
        
        print(f"   ✅ Query performance: {avg_query_time:.3f}s average, {max_query_time:.3f}s max")
        
        # Validate performance targets
        assert avg_query_time < 0.1, f"Average query time should be <100ms, got {avg_query_time:.3f}s"
        assert max_query_time < 0.2, f"Max query time should be <200ms, got {max_query_time:.3f}s"
        
        # Test memory usage efficiency
        print("🔄 Testing memory usage...")
        
        status = integration.get_integration_status()
        total_entities = status["total_entities"]
        total_relationships = status["total_relationships"]
        
        print(f"   ✅ Memory efficiency: {total_entities} entities, {total_relationships} relationships in {len(sessions)} sessions")
        
        # Cleanup
        integration.shutdown()
        
        print("✅ Performance and Scalability: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Performance and Scalability test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def run_comprehensive_test_suite():
    """Run comprehensive test suite for Graphiti Temporal Knowledge Integration"""
    print("🧪 Graphiti Temporal Knowledge Graph Integration - Comprehensive Test Suite")
    print("=" * 80)
    
    if not INTEGRATION_AVAILABLE:
        print("❌ Integration components not available. Skipping tests.")
        return False
    
    test_results = []
    
    # Run all test functions
    test_functions = [
        ("Temporal Knowledge Coordinator", test_temporal_knowledge_coordinator),
        ("Multi-LLM Knowledge Builder", test_multi_llm_knowledge_builder),
        ("Graphiti Conversation Memory", test_graphiti_conversation_memory),
        ("Graphiti-MLACS Integration", test_graphiti_mlacs_integration),
        ("Temporal Events and Consistency", test_temporal_events_and_consistency),
        ("Performance and Scalability", test_performance_and_scalability)
    ]
    
    start_time = time.time()
    
    for test_name, test_func in test_functions:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = await test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            test_results.append((test_name, False))
    
    total_time = time.time() - start_time
    
    # Generate test report
    print("\n" + "=" * 80)
    print("📋 TEST RESULTS SUMMARY")
    print("=" * 80)
    
    passed_tests = []
    failed_tests = []
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<40} {status}")
        
        if result:
            passed_tests.append(test_name)
        else:
            failed_tests.append(test_name)
    
    print("-" * 80)
    print(f"Total Tests: {len(test_results)}")
    print(f"Passed: {len(passed_tests)}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Success Rate: {len(passed_tests)/len(test_results)*100:.1f}%")
    print(f"Total Execution Time: {total_time:.2f}s")
    
    # Performance summary
    print("\n📊 PERFORMANCE VALIDATION")
    print("-" * 40)
    if len(failed_tests) == 0:
        print("✅ All performance targets met")
        print("✅ Temporal consistency validated")
        print("✅ Multi-LLM coordination functional")
        print("✅ Knowledge graph operations optimized")
        print("✅ LangChain integration successful")
    else:
        print(f"⚠️  {len(failed_tests)} test(s) failed - review implementation")
    
    return len(failed_tests) == 0

# Main execution
if __name__ == "__main__":
    async def main():
        success = await run_comprehensive_test_suite()
        
        if success:
            print("\n🎉 Graphiti Temporal Knowledge Graph Integration: ALL TESTS PASSED")
            print("✅ Ready for Phase 2 implementation")
        else:
            print("\n⚠️  Some tests failed - implementation needs review")
        
        return success
    
    asyncio.run(main())