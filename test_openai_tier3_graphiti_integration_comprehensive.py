#!/usr/bin/env python3
"""
Comprehensive Test Suite for OpenAI Tier 3 Graphiti Integration
Tests all aspects of the enhanced Tier 3 storage with Graphiti temporal knowledge graphs

Author: AgenticSeek Development Team  
Date: 2025-01-06
Version: 1.0.0
"""

import asyncio
import pytest
import tempfile
import shutil
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the Tier 3 Graphiti integration
try:
    from sources.openai_tier3_graphiti_integration_sandbox import (
        Tier3GraphitiIntegration,
        EnhancedLongTermPersistentStorage,
        GraphNode,
        GraphRelationship,
        GraphNodeType,
        RelationshipType,
        EmbeddingService
    )
    TIER3_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Tier 3 Graphiti integration not available: {e}")
    TIER3_AVAILABLE = False

class TestTier3GraphitiIntegration:
    """Test suite for Tier 3 Graphiti integration"""
    
    @pytest.fixture
    def temp_storage_path(self):
        """Create temporary storage path for testing"""
        temp_dir = tempfile.mkdtemp(prefix="tier3_test_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    async def tier3_integration(self, temp_storage_path):
        """Create Tier 3 integration instance for testing"""
        if not TIER3_AVAILABLE:
            pytest.skip("Tier 3 Graphiti integration not available")
        
        storage_path = os.path.join(temp_storage_path, "test_tier3")
        integration = Tier3GraphitiIntegration(storage_path)
        yield integration
        
        # Cleanup
        try:
            integration.graph_storage.connection.close()
        except:
            pass
    
    @pytest.fixture
    async def enhanced_storage(self, temp_storage_path):
        """Create enhanced storage instance for testing"""
        if not TIER3_AVAILABLE:
            pytest.skip("Tier 3 Graphiti integration not available")
        
        storage_path = os.path.join(temp_storage_path, "test_enhanced")
        storage = EnhancedLongTermPersistentStorage(storage_path)
        yield storage
        
        # Cleanup
        try:
            storage.connection.close()
            storage.graphiti_integration.graph_storage.connection.close()
        except:
            pass
    
    @pytest.mark.asyncio
    async def test_embedding_service(self):
        """Test embedding service functionality"""
        if not TIER3_AVAILABLE:
            pytest.skip("Tier 3 Graphiti integration not available")
        
        embedding_service = EmbeddingService()
        
        # Test embedding generation
        text = "Python is a programming language"
        embedding = await embedding_service.generate_embedding(text)
        
        assert embedding is not None
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, (int, float)) for x in embedding)
        
        # Test similarity calculation
        text2 = "Python programming language features"
        embedding2 = await embedding_service.generate_embedding(text2)
        
        similarity = embedding_service.calculate_similarity(embedding, embedding2)
        assert 0 <= similarity <= 1
        
        print(f"✅ Embedding service test passed - similarity: {similarity:.3f}")
    
    @pytest.mark.asyncio
    async def test_knowledge_node_creation(self, tier3_integration):
        """Test creating and storing knowledge nodes"""
        if not TIER3_AVAILABLE:
            pytest.skip("Tier 3 Graphiti integration not available")
        
        # Create knowledge node
        node = await tier3_integration.create_knowledge_node(
            content="Machine learning is a subset of artificial intelligence",
            node_type=GraphNodeType.FACT,
            properties={'domain': 'ai', 'confidence': 0.95},
            semantic_tags={'ai', 'machine_learning'},
            source_agent="test_agent"
        )
        
        assert node is not None
        assert node.id is not None
        assert node.content == "Machine learning is a subset of artificial intelligence"
        assert node.node_type == GraphNodeType.FACT
        assert 'ai' in node.semantic_tags
        assert 'test_agent' in node.source_agents
        assert node.embedding is not None
        
        # Verify node was stored
        retrieved_node = await tier3_integration.graph_storage.get_node(node.id)
        assert retrieved_node is not None
        assert retrieved_node.content == node.content
        
        print(f"✅ Knowledge node creation test passed - node ID: {node.id}")
    
    @pytest.mark.asyncio
    async def test_relationship_creation(self, tier3_integration):
        """Test creating relationships between nodes"""
        if not TIER3_AVAILABLE:
            pytest.skip("Tier 3 Graphiti integration not available")
        
        # Create two nodes
        node1 = await tier3_integration.create_knowledge_node(
            content="Neural networks are computational models",
            node_type=GraphNodeType.CONCEPT,
            source_agent="test_agent"
        )
        
        node2 = await tier3_integration.create_knowledge_node(
            content="Deep learning uses neural networks",
            node_type=GraphNodeType.FACT,
            source_agent="test_agent"
        )
        
        # Create relationship
        relationship = await tier3_integration.create_relationship(
            source_id=node1.id,
            target_id=node2.id,
            relationship_type=RelationshipType.ENABLES,
            strength=0.9,
            context={'domain': 'machine_learning'},
            evidence=['Deep learning architectures use neural network structures']
        )
        
        assert relationship is not None
        assert relationship.source_id == node1.id
        assert relationship.target_id == node2.id
        assert relationship.relationship_type == RelationshipType.ENABLES
        assert relationship.strength == 0.9
        
        # Verify relationship was stored
        relationships = await tier3_integration.graph_storage.get_node_relationships(node1.id)
        assert len(relationships) >= 1
        
        print(f"✅ Relationship creation test passed - relationship ID: {relationship.id}")
    
    @pytest.mark.asyncio
    async def test_semantic_search(self, tier3_integration):
        """Test semantic search functionality"""
        if not TIER3_AVAILABLE:
            pytest.skip("Tier 3 Graphiti integration not available")
        
        # Create test knowledge nodes
        test_nodes = [
            ("Python is a programming language", GraphNodeType.FACT),
            ("Machine learning algorithms need data", GraphNodeType.FACT),
            ("Neural networks mimic brain function", GraphNodeType.CONCEPT),
            ("Data science involves statistical analysis", GraphNodeType.FACT),
            ("Artificial intelligence encompasses many techniques", GraphNodeType.CONCEPT)
        ]
        
        created_nodes = []
        for content, node_type in test_nodes:
            node = await tier3_integration.create_knowledge_node(
                content=content,
                node_type=node_type,
                source_agent="test_agent"
            )
            created_nodes.append(node)
        
        # Test search for programming-related content
        results = await tier3_integration.semantic_search(
            query="programming and development",
            max_results=5,
            similarity_threshold=0.1
        )
        
        assert len(results) > 0
        assert all(isinstance(result, tuple) for result in results)
        assert all(len(result) == 2 for result in results)
        
        # Check that results are sorted by similarity
        similarities = [result[1] for result in results]
        assert similarities == sorted(similarities, reverse=True)
        
        print(f"✅ Semantic search test passed - found {len(results)} results")
    
    @pytest.mark.asyncio
    async def test_conversation_knowledge_storage(self, tier3_integration):
        """Test storing knowledge from conversations"""
        if not TIER3_AVAILABLE:
            pytest.skip("Tier 3 Graphiti integration not available")
        
        conversation_content = (
            "Let's discuss machine learning algorithms and their applications. "
            "Neural networks are particularly effective for pattern recognition tasks. "
            "Python programming language is commonly used for implementing these algorithms."
        )
        
        # Store conversation knowledge
        nodes = await tier3_integration.store_conversation_knowledge(
            conversation_content=conversation_content,
            agent_id="conversation_agent",
            session_id="test_session_001"
        )
        
        assert len(nodes) > 0
        
        # First node should be the conversation node
        conversation_node = nodes[0]
        assert conversation_node.node_type == GraphNodeType.CONVERSATION
        assert conversation_node.properties['session_id'] == "test_session_001"
        assert 'conversation_agent' in conversation_node.source_agents
        
        # Other nodes should be concept nodes
        concept_nodes = nodes[1:]
        for node in concept_nodes:
            assert node.node_type == GraphNodeType.CONCEPT
            assert 'conversation_agent' in node.source_agents
        
        print(f"✅ Conversation knowledge storage test passed - stored {len(nodes)} nodes")
    
    @pytest.mark.asyncio
    async def test_cross_session_knowledge(self, tier3_integration):
        """Test cross-session knowledge retrieval"""
        if not TIER3_AVAILABLE:
            pytest.skip("Tier 3 Graphiti integration not available")
        
        # Create knowledge in multiple sessions
        sessions = ["session_001", "session_002", "session_003"]
        
        for session_id in sessions:
            await tier3_integration.store_conversation_knowledge(
                conversation_content=f"Discussion about Python programming in {session_id}",
                agent_id="test_agent",
                session_id=session_id
            )
        
        # Test cross-session retrieval
        cross_session_knowledge = await tier3_integration.get_cross_session_knowledge(
            session_id="session_004",  # New session
            previous_sessions=3
        )
        
        # Should find concepts that appear across sessions
        assert isinstance(cross_session_knowledge, dict)
        
        print(f"✅ Cross-session knowledge test passed - found {len(cross_session_knowledge)} shared concepts")
    
    @pytest.mark.asyncio
    async def test_related_knowledge_retrieval(self, tier3_integration):
        """Test retrieving related knowledge through graph traversal"""
        if not TIER3_AVAILABLE:
            pytest.skip("Tier 3 Graphiti integration not available")
        
        # Create a knowledge network
        central_node = await tier3_integration.create_knowledge_node(
            content="Deep learning is a machine learning technique",
            node_type=GraphNodeType.CONCEPT,
            source_agent="test_agent"
        )
        
        related_nodes = []
        for i, content in enumerate([
            "Neural networks are deep learning architectures",
            "Convolutional neural networks process images",
            "Recurrent neural networks handle sequences",
            "Transformers are attention-based models"
        ]):
            node = await tier3_integration.create_knowledge_node(
                content=content,
                node_type=GraphNodeType.FACT,
                source_agent="test_agent"
            )
            related_nodes.append(node)
            
            # Create relationship to central node
            await tier3_integration.create_relationship(
                source_id=central_node.id,
                target_id=node.id,
                relationship_type=RelationshipType.CONTAINS,
                strength=0.8
            )
        
        # Test related knowledge retrieval
        related_knowledge = await tier3_integration.get_related_knowledge(
            node_id=central_node.id,
            relationship_types=[RelationshipType.CONTAINS]
        )
        
        assert 'source_node' in related_knowledge
        assert 'related_nodes' in related_knowledge
        assert related_knowledge['source_node'].id == central_node.id
        assert len(related_knowledge['related_nodes']) >= len(related_nodes)
        
        print(f"✅ Related knowledge retrieval test passed - found {len(related_knowledge['related_nodes'])} related nodes")
    
    @pytest.mark.asyncio
    async def test_enhanced_storage_integration(self, enhanced_storage):
        """Test the enhanced storage with backward compatibility"""
        if not TIER3_AVAILABLE:
            pytest.skip("Tier 3 Graphiti integration not available")
        
        # Test semantic search through enhanced interface
        search_results = await enhanced_storage.semantic_search_knowledge(
            query="artificial intelligence",
            max_results=5
        )
        
        assert isinstance(search_results, list)
        
        # Test conversation storage through enhanced interface
        conv_nodes = await enhanced_storage.store_conversation_knowledge(
            conversation_content="Discussing AI and machine learning concepts",
            agent_id="enhanced_test_agent",
            session_id="enhanced_test_session"
        )
        
        assert len(conv_nodes) > 0
        
        # Test cross-session context
        cross_session_context = await enhanced_storage.get_cross_session_context(
            session_id="new_session"
        )
        
        assert isinstance(cross_session_context, dict)
        
        print(f"✅ Enhanced storage integration test passed")
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, tier3_integration):
        """Test performance metrics collection"""
        if not TIER3_AVAILABLE:
            pytest.skip("Tier 3 Graphiti integration not available")
        
        # Create some test data
        for i in range(5):
            await tier3_integration.create_knowledge_node(
                content=f"Test fact number {i}",
                node_type=GraphNodeType.FACT,
                source_agent="metrics_test_agent"
            )
        
        # Perform some operations to generate metrics
        await tier3_integration.semantic_search("test", max_results=3)
        
        # Get performance metrics
        metrics = await tier3_integration.get_performance_metrics()
        
        assert 'timestamp' in metrics
        assert 'storage_metrics' in metrics
        assert 'performance_metrics' in metrics
        assert 'cache_hit_rate' in metrics
        
        assert metrics['storage_metrics']['total_nodes'] >= 5
        assert metrics['performance_metrics']['nodes_created'] >= 5
        assert metrics['performance_metrics']['searches_performed'] >= 1
        
        print(f"✅ Performance metrics test passed - {metrics['storage_metrics']['total_nodes']} nodes tracked")
    
    @pytest.mark.asyncio
    async def test_node_caching(self, tier3_integration):
        """Test node caching functionality"""
        if not TIER3_AVAILABLE:
            pytest.skip("Tier 3 Graphiti integration not available")
        
        # Create a node
        node = await tier3_integration.create_knowledge_node(
            content="Cached node for testing",
            node_type=GraphNodeType.FACT,
            source_agent="cache_test_agent"
        )
        
        # Access node multiple times to test caching
        initial_cache_hits = tier3_integration.metrics['cache_hits']
        
        # First access should be cache miss (from database)
        related1 = await tier3_integration.get_related_knowledge(node.id)
        
        # Second access should be cache hit
        related2 = await tier3_integration.get_related_knowledge(node.id)
        
        final_cache_hits = tier3_integration.metrics['cache_hits']
        
        # Should have at least one cache hit
        assert final_cache_hits > initial_cache_hits
        
        print(f"✅ Node caching test passed - cache hits increased by {final_cache_hits - initial_cache_hits}")
    
    @pytest.mark.asyncio
    async def test_graph_node_serialization(self):
        """Test GraphNode serialization and deserialization"""
        if not TIER3_AVAILABLE:
            pytest.skip("Tier 3 Graphiti integration not available")
        
        # Create a complex node
        original_node = GraphNode(
            id=str(uuid.uuid4()),
            node_type=GraphNodeType.ENTITY,
            content="Test entity with complex properties",
            embedding=[0.1, 0.2, 0.3],
            properties={'domain': 'test', 'complexity': 'high'},
            semantic_tags={'test', 'entity', 'complex'},
            confidence=0.85,
            source_agents={'test_agent_1', 'test_agent_2'}
        )
        
        # Serialize to dictionary
        node_dict = original_node.to_dict()
        
        # Deserialize back to node
        reconstructed_node = GraphNode.from_dict(node_dict)
        
        # Verify all fields are preserved
        assert reconstructed_node.id == original_node.id
        assert reconstructed_node.node_type == original_node.node_type
        assert reconstructed_node.content == original_node.content
        assert reconstructed_node.embedding == original_node.embedding
        assert reconstructed_node.properties == original_node.properties
        assert reconstructed_node.semantic_tags == original_node.semantic_tags
        assert reconstructed_node.confidence == original_node.confidence
        assert reconstructed_node.source_agents == original_node.source_agents
        
        print(f"✅ Graph node serialization test passed")
    
    @pytest.mark.asyncio
    async def test_relationship_types_and_traversal(self, tier3_integration):
        """Test different relationship types and graph traversal"""
        if not TIER3_AVAILABLE:
            pytest.skip("Tier 3 Graphiti integration not available")
        
        # Create nodes for testing different relationship types
        ai_node = await tier3_integration.create_knowledge_node(
            content="Artificial Intelligence",
            node_type=GraphNodeType.CONCEPT,
            source_agent="relationship_test_agent"
        )
        
        ml_node = await tier3_integration.create_knowledge_node(
            content="Machine Learning",
            node_type=GraphNodeType.CONCEPT,
            source_agent="relationship_test_agent"
        )
        
        dl_node = await tier3_integration.create_knowledge_node(
            content="Deep Learning",
            node_type=GraphNodeType.CONCEPT,
            source_agent="relationship_test_agent"
        )
        
        # Create different types of relationships
        relationships = [
            (ai_node.id, ml_node.id, RelationshipType.CONTAINS),
            (ml_node.id, dl_node.id, RelationshipType.CONTAINS),
            (ai_node.id, dl_node.id, RelationshipType.RELATED_TO),
        ]
        
        created_relationships = []
        for source_id, target_id, rel_type in relationships:
            rel = await tier3_integration.create_relationship(
                source_id=source_id,
                target_id=target_id,
                relationship_type=rel_type,
                strength=0.9
            )
            created_relationships.append(rel)
        
        # Test traversal with specific relationship types
        contains_related = await tier3_integration.get_related_knowledge(
            node_id=ai_node.id,
            relationship_types=[RelationshipType.CONTAINS]
        )
        
        all_related = await tier3_integration.get_related_knowledge(
            node_id=ai_node.id
        )
        
        # Verify filtering works
        assert len(contains_related['related_nodes']) <= len(all_related['related_nodes'])
        
        print(f"✅ Relationship types and traversal test passed")

async def run_comprehensive_tier3_tests():
    """Run all Tier 3 Graphiti integration tests"""
    print("🧪 Starting Comprehensive Tier 3 Graphiti Integration Tests...")
    
    if not TIER3_AVAILABLE:
        print("❌ Tier 3 Graphiti integration not available - skipping tests")
        return {
            'status': 'skipped',
            'reason': 'Tier 3 Graphiti integration not available'
        }
    
    # Create test instance
    test_suite = TestTier3GraphitiIntegration()
    
    # Create temporary storage
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="tier3_comprehensive_test_")
    
    try:
        # Initialize test components
        storage_path = os.path.join(temp_dir, "comprehensive_test")
        tier3_integration = Tier3GraphitiIntegration(storage_path)
        enhanced_storage = EnhancedLongTermPersistentStorage(storage_path)
        
        test_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_details': []
        }
        
        # List of tests to run
        tests = [
            ('Embedding Service', test_suite.test_embedding_service),
            ('Knowledge Node Creation', lambda: test_suite.test_knowledge_node_creation(tier3_integration)),
            ('Relationship Creation', lambda: test_suite.test_relationship_creation(tier3_integration)),
            ('Semantic Search', lambda: test_suite.test_semantic_search(tier3_integration)),
            ('Conversation Knowledge Storage', lambda: test_suite.test_conversation_knowledge_storage(tier3_integration)),
            ('Cross-Session Knowledge', lambda: test_suite.test_cross_session_knowledge(tier3_integration)),
            ('Related Knowledge Retrieval', lambda: test_suite.test_related_knowledge_retrieval(tier3_integration)),
            ('Enhanced Storage Integration', lambda: test_suite.test_enhanced_storage_integration(enhanced_storage)),
            ('Performance Metrics', lambda: test_suite.test_performance_metrics(tier3_integration)),
            ('Node Caching', lambda: test_suite.test_node_caching(tier3_integration)),
            ('Graph Node Serialization', test_suite.test_graph_node_serialization),
            ('Relationship Types and Traversal', lambda: test_suite.test_relationship_types_and_traversal(tier3_integration)),
        ]
        
        # Run each test
        for test_name, test_func in tests:
            test_results['total_tests'] += 1
            
            try:
                print(f"\n🔍 Running: {test_name}")
                await test_func()
                test_results['passed_tests'] += 1
                test_results['test_details'].append({
                    'name': test_name,
                    'status': 'passed',
                    'error': None
                })
                
            except Exception as e:
                test_results['failed_tests'] += 1
                test_results['test_details'].append({
                    'name': test_name,
                    'status': 'failed',
                    'error': str(e)
                })
                print(f"❌ Test failed: {test_name} - {e}")
        
        # Get final metrics
        final_metrics = await tier3_integration.get_performance_metrics()
        test_results['final_metrics'] = final_metrics
        
        # Calculate success rate
        success_rate = (test_results['passed_tests'] / test_results['total_tests']) * 100
        test_results['success_rate'] = success_rate
        
        # Print summary
        print(f"\n📊 Test Summary:")
        print(f"   Total Tests: {test_results['total_tests']}")
        print(f"   Passed: {test_results['passed_tests']}")
        print(f"   Failed: {test_results['failed_tests']}")
        print(f"   Success Rate: {success_rate:.1f}%")
        
        if test_results['failed_tests'] > 0:
            print(f"\n❌ Failed Tests:")
            for test_detail in test_results['test_details']:
                if test_detail['status'] == 'failed':
                    print(f"   - {test_detail['name']}: {test_detail['error']}")
        
        # Cleanup
        try:
            tier3_integration.graph_storage.connection.close()
            enhanced_storage.connection.close()
            enhanced_storage.graphiti_integration.graph_storage.connection.close()
        except:
            pass
        
        return test_results
        
    finally:
        # Cleanup temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    # Run the comprehensive tests
    result = asyncio.run(run_comprehensive_tier3_tests())
    
    if result['status'] == 'skipped':
        print(f"Tests skipped: {result['reason']}")
        exit(0)
    
    # Exit with appropriate code
    if result['failed_tests'] > 0:
        exit(1)
    else:
        print("🎉 All tests passed!")
        exit(0)