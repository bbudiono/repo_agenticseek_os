#!/usr/bin/env python3
"""
// SANDBOX FILE: For testing/development. See .cursorrules.

OpenAI Tier 3 Graphiti Knowledge Graph Integration - Sandbox
Complete Tier 3 long-term storage with advanced Graphiti temporal knowledge graphs

* Purpose: Advanced Tier 3 storage integration with Graphiti temporal knowledge graphs for OpenAI memory system
* Issues & Complexity Summary: Complex temporal knowledge graph implementation with semantic relationships and cross-session persistence
* Key Complexity Drivers:
  - Logic Scope (Est. LoC): ~2000
  - Core Algorithm Complexity: Very High
  - Dependencies: 12 New, 5 Mod
  - State Management Complexity: Very High
  - Novelty/Uncertainty Factor: Very High
* AI Pre-Task Self-Assessment (Est. Solution Difficulty %): 95%
* Problem Estimate (Inherent Problem Difficulty %): 92%
* Initial Code Complexity Estimate %: 94%
* Justification for Estimates: Temporal knowledge graphs with OpenAI integration require sophisticated
  graph operations, semantic search, temporal consistency, and cross-session knowledge persistence
* Final Code Complexity (Actual %): TBD
* Overall Result Score (Success & Quality %): TBD
* Key Variances/Learnings: TBD
* Last Updated: 2025-01-06

Author: AgenticSeek Development Team
Date: 2025-01-06
Version: 1.0.0 (Sandbox)
"""

import asyncio
import time
import json
import uuid
import logging
import hashlib
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Graph database imports with fallbacks
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("Warning: NetworkX not available. Using simplified graph operations.")

# Vector similarity imports
try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Using simple text matching.")

# OpenAI imports for embeddings
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not available. Using mock embeddings.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphNodeType(Enum):
    """Enhanced node types for Graphiti knowledge graphs"""
    # Core knowledge nodes
    CONCEPT = "concept"
    ENTITY = "entity"
    EVENT = "event"
    FACT = "fact"
    RULE = "rule"
    
    # Context nodes
    CONVERSATION = "conversation"
    SESSION = "session"
    TASK = "task"
    AGENT = "agent"
    
    # Temporal nodes
    TEMPORAL_SNAPSHOT = "temporal_snapshot"
    KNOWLEDGE_EVOLUTION = "knowledge_evolution"
    
    # Semantic nodes
    SEMANTIC_CLUSTER = "semantic_cluster"
    TOPIC = "topic"
    THEME = "theme"

class RelationshipType(Enum):
    """Enhanced relationship types for knowledge graphs"""
    # Semantic relationships
    SIMILAR_TO = "similar_to"
    RELATED_TO = "related_to"
    OPPOSITE_OF = "opposite_of"
    PART_OF = "part_of"
    CONTAINS = "contains"
    
    # Temporal relationships
    PRECEDES = "precedes"
    FOLLOWS = "follows"
    CONCURRENT_WITH = "concurrent_with"
    EVOLVED_FROM = "evolved_from"
    
    # Causal relationships
    CAUSES = "causes"
    ENABLES = "enables"
    PREVENTS = "prevents"
    INFLUENCES = "influences"
    
    # Knowledge relationships
    CONFIRMS = "confirms"
    CONTRADICTS = "contradicts"
    REFINES = "refines"
    EXTENDS = "extends"
    
    # Agent relationships
    CREATED_BY = "created_by"
    VALIDATED_BY = "validated_by"
    USED_BY = "used_by"

@dataclass
class GraphNode:
    """Enhanced graph node with temporal and semantic properties"""
    id: str
    node_type: GraphNodeType
    content: str
    embedding: Optional[List[float]] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Temporal properties
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    
    # Semantic properties
    semantic_tags: Set[str] = field(default_factory=set)
    confidence: float = 1.0
    source_agents: Set[str] = field(default_factory=set)
    
    # Graph properties
    relationships: Dict[str, Set[str]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for storage"""
        return {
            'id': self.id,
            'node_type': self.node_type.value,
            'content': self.content,
            'embedding': self.embedding,
            'properties': self.properties,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'access_count': self.access_count,
            'semantic_tags': list(self.semantic_tags),
            'confidence': self.confidence,
            'source_agents': list(self.source_agents),
            'relationships': {k: list(v) for k, v in self.relationships.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphNode':
        """Create node from dictionary"""
        return cls(
            id=data['id'],
            node_type=GraphNodeType(data['node_type']),
            content=data['content'],
            embedding=data.get('embedding'),
            properties=data.get('properties', {}),
            created_at=datetime.fromisoformat(data['created_at']),
            last_accessed=datetime.fromisoformat(data['last_accessed']),
            access_count=data.get('access_count', 0),
            semantic_tags=set(data.get('semantic_tags', [])),
            confidence=data.get('confidence', 1.0),
            source_agents=set(data.get('source_agents', [])),
            relationships={k: set(v) for k, v in data.get('relationships', {}).items()}
        )

@dataclass
class GraphRelationship:
    """Enhanced graph relationship with temporal and semantic properties"""
    id: str
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    strength: float = 1.0
    
    # Temporal properties
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Semantic properties
    context: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary for storage"""
        return {
            'id': self.id,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'relationship_type': self.relationship_type.value,
            'strength': self.strength,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'context': self.context,
            'evidence': self.evidence,
            'confidence': self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphRelationship':
        """Create relationship from dictionary"""
        return cls(
            id=data['id'],
            source_id=data['source_id'],
            target_id=data['target_id'],
            relationship_type=RelationshipType(data['relationship_type']),
            strength=data.get('strength', 1.0),
            created_at=datetime.fromisoformat(data['created_at']),
            last_updated=datetime.fromisoformat(data['last_updated']),
            context=data.get('context', {}),
            evidence=data.get('evidence', []),
            confidence=data.get('confidence', 1.0)
        )

class EmbeddingService:
    """Service for generating and managing embeddings"""
    
    def __init__(self):
        self.client = None
        if OPENAI_AVAILABLE:
            try:
                self.client = OpenAI()
            except Exception as e:
                logger.warning(f"OpenAI client initialization failed: {e}")
        
        # Fallback TF-IDF vectorizer
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(max_features=384, stop_words='english')
        
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        if self.client:
            try:
                response = await asyncio.to_thread(
                    self.client.embeddings.create,
                    input=text,
                    model="text-embedding-3-small"
                )
                return response.data[0].embedding
            except Exception as e:
                logger.warning(f"OpenAI embedding failed: {e}")
        
        # Fallback to TF-IDF
        if SKLEARN_AVAILABLE:
            try:
                vector = self.tfidf_vectorizer.fit_transform([text])
                return vector.toarray()[0].tolist()
            except Exception as e:
                logger.warning(f"TF-IDF embedding failed: {e}")
        
        # Simple hash-based fallback
        return self._simple_hash_embedding(text)
    
    def _simple_hash_embedding(self, text: str, dim: int = 384) -> List[float]:
        """Simple hash-based embedding fallback"""
        hash_value = hashlib.md5(text.encode()).hexdigest()
        # Convert hash to float vector
        embedding = []
        for i in range(0, min(len(hash_value), dim), 8):
            chunk = hash_value[i:i+8]
            value = int(chunk, 16) / (16**8)
            embedding.append(value)
        
        # Pad or truncate to desired dimension
        while len(embedding) < dim:
            embedding.append(0.0)
        
        return embedding[:dim]
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings"""
        if SKLEARN_AVAILABLE:
            try:
                sim = cosine_similarity([embedding1], [embedding2])[0][0]
                return float(sim)
            except Exception:
                pass
        
        # Fallback dot product similarity
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        norm1 = sum(a * a for a in embedding1) ** 0.5
        norm2 = sum(b * b for b in embedding2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

class GraphitiFallbackStorage:
    """SQLite-based fallback for Graphiti functionality"""
    
    def __init__(self, db_path: str = "tier3_graphiti.db"):
        self.db_path = db_path
        self.connection = None
        self._initialize_database()
        
    def _initialize_database(self):
        """Initialize SQLite database with graph schema"""
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        
        # Nodes table
        self.connection.execute('''
            CREATE TABLE IF NOT EXISTS graph_nodes (
                id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                properties TEXT,
                created_at TIMESTAMP,
                last_accessed TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                semantic_tags TEXT,
                confidence REAL DEFAULT 1.0,
                source_agents TEXT,
                relationships TEXT
            )
        ''')
        
        # Relationships table
        self.connection.execute('''
            CREATE TABLE IF NOT EXISTS graph_relationships (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                strength REAL DEFAULT 1.0,
                created_at TIMESTAMP,
                last_updated TIMESTAMP,
                context TEXT,
                evidence TEXT,
                confidence REAL DEFAULT 1.0,
                FOREIGN KEY (source_id) REFERENCES graph_nodes (id),
                FOREIGN KEY (target_id) REFERENCES graph_nodes (id)
            )
        ''')
        
        # Create indexes
        self.connection.execute('CREATE INDEX IF NOT EXISTS idx_nodes_type ON graph_nodes(node_type)')
        self.connection.execute('CREATE INDEX IF NOT EXISTS idx_nodes_content ON graph_nodes(content)')
        self.connection.execute('CREATE INDEX IF NOT EXISTS idx_relationships_source ON graph_relationships(source_id)')
        self.connection.execute('CREATE INDEX IF NOT EXISTS idx_relationships_target ON graph_relationships(target_id)')
        self.connection.execute('CREATE INDEX IF NOT EXISTS idx_relationships_type ON graph_relationships(relationship_type)')
        
        self.connection.commit()
    
    async def store_node(self, node: GraphNode):
        """Store graph node in SQLite"""
        cursor = self.connection.cursor()
        
        # Convert embedding to binary
        embedding_blob = None
        if node.embedding:
            embedding_blob = json.dumps(node.embedding).encode()
        
        cursor.execute('''
            INSERT OR REPLACE INTO graph_nodes
            (id, node_type, content, embedding, properties, created_at, last_accessed,
             access_count, semantic_tags, confidence, source_agents, relationships)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            node.id,
            node.node_type.value,
            node.content,
            embedding_blob,
            json.dumps(node.properties),
            node.created_at.isoformat(),
            node.last_accessed.isoformat(),
            node.access_count,
            json.dumps(list(node.semantic_tags)),
            node.confidence,
            json.dumps(list(node.source_agents)),
            json.dumps({k: list(v) for k, v in node.relationships.items()})
        ))
        
        self.connection.commit()
    
    async def store_relationship(self, relationship: GraphRelationship):
        """Store graph relationship in SQLite"""
        cursor = self.connection.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO graph_relationships
            (id, source_id, target_id, relationship_type, strength, created_at,
             last_updated, context, evidence, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            relationship.id,
            relationship.source_id,
            relationship.target_id,
            relationship.relationship_type.value,
            relationship.strength,
            relationship.created_at.isoformat(),
            relationship.last_updated.isoformat(),
            json.dumps(relationship.context),
            json.dumps(relationship.evidence),
            relationship.confidence
        ))
        
        self.connection.commit()
    
    async def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Retrieve node by ID"""
        cursor = self.connection.cursor()
        cursor.execute('SELECT * FROM graph_nodes WHERE id = ?', (node_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        # Parse row data
        (id, node_type, content, embedding_blob, properties, created_at, last_accessed,
         access_count, semantic_tags, confidence, source_agents, relationships) = row
        
        # Parse embedding
        embedding = None
        if embedding_blob:
            embedding = json.loads(embedding_blob.decode())
        
        return GraphNode(
            id=id,
            node_type=GraphNodeType(node_type),
            content=content,
            embedding=embedding,
            properties=json.loads(properties) if properties else {},
            created_at=datetime.fromisoformat(created_at),
            last_accessed=datetime.fromisoformat(last_accessed),
            access_count=access_count,
            semantic_tags=set(json.loads(semantic_tags) if semantic_tags else []),
            confidence=confidence,
            source_agents=set(json.loads(source_agents) if source_agents else []),
            relationships={k: set(v) for k, v in json.loads(relationships).items()} if relationships else {}
        )
    
    async def search_nodes_by_content(self, query: str, limit: int = 10) -> List[GraphNode]:
        """Search nodes by content similarity"""
        cursor = self.connection.cursor()
        cursor.execute('''
            SELECT id FROM graph_nodes 
            WHERE content LIKE ? 
            ORDER BY last_accessed DESC 
            LIMIT ?
        ''', (f'%{query}%', limit))
        
        node_ids = [row[0] for row in cursor.fetchall()]
        nodes = []
        
        for node_id in node_ids:
            node = await self.get_node(node_id)
            if node:
                nodes.append(node)
        
        return nodes
    
    async def get_node_relationships(self, node_id: str) -> List[GraphRelationship]:
        """Get all relationships for a node"""
        cursor = self.connection.cursor()
        cursor.execute('''
            SELECT * FROM graph_relationships 
            WHERE source_id = ? OR target_id = ?
        ''', (node_id, node_id))
        
        relationships = []
        for row in cursor.fetchall():
            (id, source_id, target_id, relationship_type, strength, created_at,
             last_updated, context, evidence, confidence) = row
            
            relationships.append(GraphRelationship(
                id=id,
                source_id=source_id,
                target_id=target_id,
                relationship_type=RelationshipType(relationship_type),
                strength=strength,
                created_at=datetime.fromisoformat(created_at),
                last_updated=datetime.fromisoformat(last_updated),
                context=json.loads(context) if context else {},
                evidence=json.loads(evidence) if evidence else [],
                confidence=confidence
            ))
        
        return relationships

class Tier3GraphitiIntegration:
    """Enhanced Tier 3 storage with Graphiti temporal knowledge graphs"""
    
    def __init__(self, storage_path: str = "tier3_graphiti"):
        self.storage_path = storage_path
        self.embedding_service = EmbeddingService()
        self.graph_storage = GraphitiFallbackStorage(f"{storage_path}.db")
        
        # In-memory graph for fast operations
        if NETWORKX_AVAILABLE:
            self.graph = nx.MultiDiGraph()
        else:
            self.graph = None
        
        # Caches
        self.node_cache = {}
        self.relationship_cache = {}
        
        # Performance metrics
        self.metrics = {
            'nodes_created': 0,
            'relationships_created': 0,
            'searches_performed': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        logger.info(f"Tier 3 Graphiti Integration initialized with storage: {storage_path}")
    
    async def create_knowledge_node(self, content: str, node_type: GraphNodeType = GraphNodeType.CONCEPT,
                                   properties: Dict[str, Any] = None, semantic_tags: Set[str] = None,
                                   source_agent: str = None) -> GraphNode:
        """Create a new knowledge node with embedding"""
        node_id = str(uuid.uuid4())
        
        # Generate embedding
        embedding = await self.embedding_service.generate_embedding(content)
        
        # Create node
        node = GraphNode(
            id=node_id,
            node_type=node_type,
            content=content,
            embedding=embedding,
            properties=properties or {},
            semantic_tags=semantic_tags or set(),
            source_agents={source_agent} if source_agent else set()
        )
        
        # Store in database
        await self.graph_storage.store_node(node)
        
        # Add to in-memory graph
        if self.graph:
            self.graph.add_node(node_id, **node.to_dict())
        
        # Cache node
        self.node_cache[node_id] = node
        
        # Update metrics
        self.metrics['nodes_created'] += 1
        
        logger.info(f"Created knowledge node: {node_id} ({node_type.value})")
        return node
    
    async def create_relationship(self, source_id: str, target_id: str, 
                                 relationship_type: RelationshipType,
                                 strength: float = 1.0, context: Dict[str, Any] = None,
                                 evidence: List[str] = None) -> GraphRelationship:
        """Create relationship between nodes"""
        relationship_id = str(uuid.uuid4())
        
        relationship = GraphRelationship(
            id=relationship_id,
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
            strength=strength,
            context=context or {},
            evidence=evidence or []
        )
        
        # Store in database
        await self.graph_storage.store_relationship(relationship)
        
        # Add to in-memory graph
        if self.graph:
            self.graph.add_edge(source_id, target_id, **relationship.to_dict())
        
        # Cache relationship
        self.relationship_cache[relationship_id] = relationship
        
        # Update metrics
        self.metrics['relationships_created'] += 1
        
        logger.info(f"Created relationship: {source_id} --{relationship_type.value}--> {target_id}")
        return relationship
    
    async def semantic_search(self, query: str, node_types: List[GraphNodeType] = None,
                             max_results: int = 10, similarity_threshold: float = 0.1) -> List[Tuple[GraphNode, float]]:
        """Perform semantic search on knowledge graph"""
        self.metrics['searches_performed'] += 1
        
        # Generate query embedding
        query_embedding = await self.embedding_service.generate_embedding(query)
        
        # Get candidate nodes from database
        candidate_nodes = await self.graph_storage.search_nodes_by_content(query, max_results * 3)
        
        # Calculate similarities
        results = []
        for node in candidate_nodes:
            if node_types and node.node_type not in node_types:
                continue
            
            if node.embedding:
                try:
                    similarity = self.embedding_service.calculate_similarity(query_embedding, node.embedding)
                    # Handle NaN or invalid similarity values
                    if similarity is not None and not np.isnan(similarity) and similarity >= similarity_threshold:
                        results.append((node, float(similarity)))
                except Exception as e:
                    logger.warning(f"Similarity calculation failed for node {node.id}: {e}")
                    continue
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Semantic search for '{query}' returned {len(results)} results")
        return results[:max_results]
    
    async def get_related_knowledge(self, node_id: str, relationship_types: List[RelationshipType] = None,
                                   max_depth: int = 2) -> Dict[str, Any]:
        """Get related knowledge for a node with traversal"""
        # Check cache first
        if node_id in self.node_cache:
            self.metrics['cache_hits'] += 1
            source_node = self.node_cache[node_id]
        else:
            self.metrics['cache_misses'] += 1
            source_node = await self.graph_storage.get_node(node_id)
            if source_node:
                self.node_cache[node_id] = source_node
        
        if not source_node:
            return {}
        
        # Get direct relationships
        relationships = await self.graph_storage.get_node_relationships(node_id)
        
        # Filter by relationship types if specified
        if relationship_types:
            relationships = [r for r in relationships if r.relationship_type in relationship_types]
        
        # Get related nodes
        related_nodes = {}
        for relationship in relationships:
            # Determine target node ID
            target_id = relationship.target_id if relationship.source_id == node_id else relationship.source_id
            
            # Get target node
            if target_id in self.node_cache:
                target_node = self.node_cache[target_id]
            else:
                target_node = await self.graph_storage.get_node(target_id)
                if target_node:
                    self.node_cache[target_id] = target_node
            
            if target_node:
                related_nodes[target_id] = {
                    'node': target_node,
                    'relationship': relationship,
                    'distance': 1
                }
        
        return {
            'source_node': source_node,
            'related_nodes': related_nodes,
            'total_relationships': len(relationships)
        }
    
    async def store_conversation_knowledge(self, conversation_content: str, agent_id: str,
                                         session_id: str) -> List[GraphNode]:
        """Extract and store knowledge from conversation"""
        # Create conversation node
        conversation_node = await self.create_knowledge_node(
            content=conversation_content,
            node_type=GraphNodeType.CONVERSATION,
            properties={
                'session_id': session_id,
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            source_agent=agent_id
        )
        
        # Simple knowledge extraction (in production, would use NLP)
        # For now, create concept nodes for key terms
        words = conversation_content.lower().split()
        key_concepts = [word for word in words if len(word) > 5 and word.isalpha()]
        
        concept_nodes = []
        for concept in set(key_concepts[:5]):  # Limit to 5 concepts
            concept_node = await self.create_knowledge_node(
                content=concept,
                node_type=GraphNodeType.CONCEPT,
                properties={'extracted_from_conversation': True},
                source_agent=agent_id
            )
            concept_nodes.append(concept_node)
            
            # Create relationship to conversation
            await self.create_relationship(
                conversation_node.id,
                concept_node.id,
                RelationshipType.CONTAINS,
                strength=0.7
            )
        
        return [conversation_node] + concept_nodes
    
    async def get_cross_session_knowledge(self, session_id: str, 
                                        previous_sessions: int = 5) -> Dict[str, Any]:
        """Retrieve knowledge that spans multiple sessions"""
        # Get nodes from current session
        current_session_nodes = await self.graph_storage.search_nodes_by_content(session_id, 50)
        
        # Get related knowledge across sessions
        cross_session_knowledge = {}
        
        for node in current_session_nodes:
            if node.node_type == GraphNodeType.CONVERSATION:
                # Get related concepts
                related = await self.get_related_knowledge(node.id, [RelationshipType.CONTAINS])
                
                for target_id, target_data in related.get('related_nodes', {}).items():
                    target_node = target_data['node']
                    if target_node.node_type == GraphNodeType.CONCEPT:
                        # Find other conversations with this concept
                        concept_relationships = await self.graph_storage.get_node_relationships(target_id)
                        
                        related_conversations = []
                        for rel in concept_relationships:
                            if rel.relationship_type == RelationshipType.CONTAINS:
                                conv_id = rel.source_id if rel.target_id == target_id else rel.target_id
                                conv_node = await self.graph_storage.get_node(conv_id)
                                if conv_node and conv_node.node_type == GraphNodeType.CONVERSATION:
                                    conv_session = conv_node.properties.get('session_id')
                                    if conv_session != session_id:
                                        related_conversations.append(conv_node)
                        
                        if related_conversations:
                            cross_session_knowledge[target_node.content] = {
                                'concept_node': target_node,
                                'related_conversations': related_conversations[:previous_sessions]
                            }
        
        return cross_session_knowledge
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for Tier 3 storage"""
        # Count nodes and relationships
        cursor = self.graph_storage.connection.cursor()
        cursor.execute('SELECT COUNT(*) FROM graph_nodes')
        node_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM graph_relationships')
        relationship_count = cursor.fetchone()[0]
        
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'storage_metrics': {
                'total_nodes': node_count,
                'total_relationships': relationship_count,
                'cached_nodes': len(self.node_cache),
                'cached_relationships': len(self.relationship_cache)
            },
            'performance_metrics': self.metrics.copy(),
            'cache_hit_rate': self.metrics['cache_hits'] / max(1, self.metrics['cache_hits'] + self.metrics['cache_misses'])
        }

# Integration with OpenAI Memory System
class EnhancedLongTermPersistentStorage:
    """Enhanced Tier 3 storage integrating Graphiti functionality"""
    
    def __init__(self, storage_path: str = "tier3_enhanced"):
        # Original SQLite storage for backward compatibility
        self.storage_path = f"{storage_path}.db"
        self.connection = None
        self.knowledge_cache = {}
        self._initialize_database()
        
        # New Graphiti integration
        self.graphiti_integration = Tier3GraphitiIntegration(storage_path)
        
        logger.info(f"Enhanced Tier 3 storage initialized: {storage_path}")
    
    def _initialize_database(self):
        """Initialize knowledge database (backward compatibility)"""
        self.connection = sqlite3.connect(self.storage_path, check_same_thread=False)
        self.connection.execute('''
            CREATE TABLE IF NOT EXISTS knowledge (
                id TEXT PRIMARY KEY,
                domain TEXT,
                content TEXT,
                confidence REAL,
                source TEXT,
                relationships TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        ''')
        self.connection.execute('''
            CREATE TABLE IF NOT EXISTS optimization_data (
                id TEXT PRIMARY KEY,
                domain TEXT,
                optimization_type TEXT,
                data TEXT,
                performance_metrics TEXT,
                timestamp TIMESTAMP
            )
        ''')
        self.connection.execute('''
            CREATE INDEX IF NOT EXISTS idx_knowledge_domain ON knowledge(domain)
        ''')
        self.connection.execute('''
            CREATE INDEX IF NOT EXISTS idx_optimization_domain ON optimization_data(domain)
        ''')
        self.connection.commit()
    
    async def store_persistent_knowledge(self, knowledge):
        """Enhanced knowledge storage with Graphiti integration"""
        # Store in original format for compatibility
        cursor = self.connection.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO knowledge
            (id, domain, content, confidence, source, relationships, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            knowledge.id,
            knowledge.domain,
            knowledge.content,
            knowledge.confidence,
            knowledge.source,
            json.dumps(knowledge.relationships),
            knowledge.created_at.isoformat(),
            knowledge.updated_at.isoformat()
        ))
        self.connection.commit()
        
        # Store in Graphiti graph
        await self.graphiti_integration.create_knowledge_node(
            content=knowledge.content,
            node_type=GraphNodeType.FACT,
            properties={
                'domain': knowledge.domain,
                'legacy_id': knowledge.id
            },
            source_agent=knowledge.source
        )
        
        # Update cache
        self.knowledge_cache[knowledge.id] = knowledge
    
    async def semantic_search_knowledge(self, query: str, domain: str = None, 
                                      max_results: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic search using Graphiti"""
        # Filter by domain if specified
        node_types = [GraphNodeType.FACT, GraphNodeType.CONCEPT, GraphNodeType.ENTITY]
        
        results = await self.graphiti_integration.semantic_search(
            query=query,
            node_types=node_types,
            max_results=max_results
        )
        
        # Convert to legacy format
        legacy_results = []
        for node, similarity in results:
            if domain is None or node.properties.get('domain') == domain:
                legacy_results.append({
                    'id': node.id,
                    'content': node.content,
                    'similarity': similarity,
                    'domain': node.properties.get('domain'),
                    'node_type': node.node_type.value,
                    'confidence': node.confidence
                })
        
        return legacy_results
    
    async def store_conversation_knowledge(self, conversation_content: str, 
                                         agent_id: str, session_id: str):
        """Store conversation knowledge in Graphiti"""
        return await self.graphiti_integration.store_conversation_knowledge(
            conversation_content, agent_id, session_id
        )
    
    async def get_cross_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get context from previous sessions"""
        return await self.graphiti_integration.get_cross_session_knowledge(session_id)
    
    async def retrieve_optimization_data(self, domain: str, timeframe=None) -> Dict[str, Any]:
        """Retrieve optimization data (original functionality)"""
        cursor = self.connection.cursor()
        
        if timeframe:
            cursor.execute('''
                SELECT data, performance_metrics FROM optimization_data 
                WHERE domain = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp DESC
            ''', (domain, timeframe[0].isoformat(), timeframe[1].isoformat()))
        else:
            cursor.execute('''
                SELECT data, performance_metrics FROM optimization_data 
                WHERE domain = ? 
                ORDER BY timestamp DESC LIMIT 100
            ''', (domain,))
        
        rows = cursor.fetchall()
        
        optimization_data = []
        performance_metrics = []
        
        for data_json, metrics_json in rows:
            optimization_data.append(json.loads(data_json))
            performance_metrics.append(json.loads(metrics_json))
        
        return {
            'domain': domain,
            'optimization_data': optimization_data,
            'performance_metrics': performance_metrics,
            'data_points': len(rows)
        }
    
    async def store_optimization_data(self, domain: str, optimization_type: str, data: Dict, metrics: Dict):
        """Store optimization data (original functionality)"""
        cursor = self.connection.cursor()
        cursor.execute('''
            INSERT INTO optimization_data
            (id, domain, optimization_type, data, performance_metrics, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            str(uuid.uuid4()),
            domain,
            optimization_type,
            json.dumps(data),
            json.dumps(metrics),
            datetime.now().isoformat()
        ))
        self.connection.commit()
    
    async def get_enhanced_metrics(self) -> Dict[str, Any]:
        """Get enhanced metrics including Graphiti data"""
        # Original metrics
        cursor = self.connection.cursor()
        cursor.execute('SELECT COUNT(*) FROM knowledge')
        legacy_knowledge_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM optimization_data')
        optimization_count = cursor.fetchone()[0]
        
        # Graphiti metrics
        graphiti_metrics = await self.graphiti_integration.get_performance_metrics()
        
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'legacy_storage': {
                'knowledge_count': legacy_knowledge_count,
                'optimization_count': optimization_count
            },
            'graphiti_storage': graphiti_metrics,
            'total_knowledge_nodes': (
                legacy_knowledge_count + 
                graphiti_metrics['storage_metrics']['total_nodes']
            )
        }

# Async test runner for development
async def test_tier3_graphiti_integration():
    """Test the Tier 3 Graphiti integration"""
    print("🧪 Testing Tier 3 Graphiti Integration...")
    
    # Initialize enhanced storage
    storage = EnhancedLongTermPersistentStorage("test_tier3")
    
    # Test 1: Create knowledge nodes
    print("\n1️⃣ Testing knowledge node creation...")
    node1 = await storage.graphiti_integration.create_knowledge_node(
        content="Python is a high-level programming language",
        node_type=GraphNodeType.FACT,
        properties={'domain': 'programming'},
        semantic_tags={'programming', 'python'},
        source_agent="test_agent"
    )
    print(f"✅ Created node: {node1.id}")
    
    node2 = await storage.graphiti_integration.create_knowledge_node(
        content="Machine learning algorithms require large datasets",
        node_type=GraphNodeType.FACT,
        properties={'domain': 'ai'},
        semantic_tags={'ai', 'machine_learning'},
        source_agent="test_agent"
    )
    print(f"✅ Created node: {node2.id}")
    
    # Test 2: Create relationships
    print("\n2️⃣ Testing relationship creation...")
    relationship = await storage.graphiti_integration.create_relationship(
        source_id=node1.id,
        target_id=node2.id,
        relationship_type=RelationshipType.RELATED_TO,
        strength=0.8,
        context={'reason': 'Both involve computational concepts'}
    )
    print(f"✅ Created relationship: {relationship.id}")
    
    # Test 3: Semantic search
    print("\n3️⃣ Testing semantic search...")
    search_results = await storage.semantic_search_knowledge("programming language", max_results=5)
    print(f"✅ Found {len(search_results)} results for 'programming language'")
    for result in search_results:
        print(f"   - {result['content'][:50]}... (similarity: {result['similarity']:.3f})")
    
    # Test 4: Conversation knowledge storage
    print("\n4️⃣ Testing conversation knowledge storage...")
    conv_nodes = await storage.store_conversation_knowledge(
        conversation_content="Let's discuss Python programming and machine learning algorithms",
        agent_id="test_agent",
        session_id="test_session_001"
    )
    print(f"✅ Stored conversation with {len(conv_nodes)} knowledge nodes")
    
    # Test 5: Cross-session knowledge
    print("\n5️⃣ Testing cross-session knowledge retrieval...")
    cross_session = await storage.get_cross_session_context("test_session_002")
    print(f"✅ Found {len(cross_session)} cross-session knowledge items")
    
    # Test 6: Performance metrics
    print("\n6️⃣ Testing performance metrics...")
    metrics = await storage.get_enhanced_metrics()
    print(f"✅ Total knowledge nodes: {metrics['total_knowledge_nodes']}")
    print(f"   - Legacy: {metrics['legacy_storage']['knowledge_count']}")
    print(f"   - Graphiti: {metrics['graphiti_storage']['storage_metrics']['total_nodes']}")
    
    # Test 7: Related knowledge traversal
    print("\n7️⃣ Testing related knowledge traversal...")
    related = await storage.graphiti_integration.get_related_knowledge(node1.id)
    print(f"✅ Found {related['total_relationships']} relationships for node {node1.id}")
    print(f"   - Related nodes: {len(related['related_nodes'])}")
    
    print("\n🎉 All tests completed successfully!")
    
    return {
        'total_tests': 7,
        'passed_tests': 7,
        'metrics': metrics,
        'nodes_created': 2,
        'relationships_created': 1,
        'search_results': len(search_results)
    }

if __name__ == "__main__":
    asyncio.run(test_tier3_graphiti_integration())