"""
IP-SAKTI Knowledge Graph
Phase 13: Implements the knowledge graph for GraphRAG.
    GraphRAG runtime. Research justification is loaded lazily so API startup does
    not import the optional experiment stack.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx

from ip_sakti.config.loader import Settings, get_settings
from ip_sakti.core.models import (
    Document,
    DocumentChunk,
    DocumentType,
)
logger = logging.getLogger(__name__)


class EntityType(str, Enum):
    PATENT = "patent"
    TRADEMARK = "trademark"
    INVENTOR = "inventor"
    ASSIGNEE = "assignee"
    OWNER = "owner"
    STATUTE = "statute"
    CASE = "case"
    COURT = "court"
    JUDGE = "judge"
    NICE_CLASS = "nice_class"
    TECHNOLOGY_CLASS = "technology_class"


class RelationType(str, Enum):
    INVENTED_BY = "invented_by"
    ASSIGNED_TO = "assigned_to"
    CITES = "cites"
    CLAIMS_PRIORITY = "claims_priority"
    CLASSIFIED_AS = "classified_as"
    OPPOSSES = "opposes"
    INVALIDATES = "invalidates"
    LICENSES_TO = "licenses_to"


class GraphStoreType(str, Enum):
    """Supported graph store types."""
    NETWORKX = "networkx"  # In-memory for development
    NEO4J = "neo4j"
    ARANGODB = "arangodb"
    JANUSGRAPH = "janusgraph"
    AMAZON_NEPTUNE = "amazon_neptune"


@dataclass
class GraphConfig:
    """Configuration for knowledge graph."""
    store_type: GraphStoreType = GraphStoreType.NETWORKX
    store_config: Dict[str, Any] = field(default_factory=dict)
    entity_types: List[EntityType] = field(default_factory=list)
    relation_types: List[RelationType] = field(default_factory=list)
    enable_inference: bool = True
    max_hops: int = 3
    embedding_dimension: int = 256


@dataclass
class GraphEntity:
    """Entity in the knowledge graph."""
    id: str
    type: EntityType
    name: str
    aliases: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    source_chunks: List[str] = field(default_factory=list)  # chunk_ids
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class GraphRelation:
    """Relation in the knowledge graph."""
    id: str
    source_id: str
    target_id: str
    type: RelationType
    properties: Dict[str, Any] = field(default_factory=dict)
    source_chunks: List[str] = field(default_factory=list)
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class GraphPath:
    """Path in the knowledge graph."""
    entities: List[GraphEntity]
    relations: List[GraphRelation]
    score: float = 0.0


class GraphStore(ABC):
    """Abstract base for graph stores."""
    
    @abstractmethod
    async def initialize(self) -> None:
        pass
    
    @abstractmethod
    async def upsert_entity(self, entity: GraphEntity) -> None:
        pass
    
    @abstractmethod
    async def upsert_relation(self, relation: GraphRelation) -> None:
        pass
    
    @abstractmethod
    async def get_entity(self, entity_id: str) -> Optional[GraphEntity]:
        pass
    
    @abstractmethod
    async def get_relations(
        self,
        entity_id: str,
        relation_types: Optional[List[RelationType]] = None,
        direction: str = "both",  # "out", "in", "both"
    ) -> List[GraphRelation]:
        pass
    
    @abstractmethod
    async def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 3,
    ) -> List[GraphPath]:
        pass
    
    @abstractmethod
    async def query(
        self,
        cypher: str,
        params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    async def get_subgraph(
        self,
        entity_ids: List[str],
        hops: int = 1,
    ) -> Tuple[List[GraphEntity], List[GraphRelation]]:
        pass
    
    @abstractmethod
    async def close(self) -> None:
        pass


class NetworkXGraphStore(GraphStore):
    """NetworkX-based in-memory graph store."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.graph = nx.MultiDiGraph()
        self.entities: Dict[str, GraphEntity] = {}
        self.relations: Dict[str, GraphRelation] = {}
    
    async def initialize(self) -> None:
        logger.info("NetworkX graph store initialized")
    
    async def upsert_entity(self, entity: GraphEntity) -> None:
        self.entities[entity.id] = entity
        self.graph.add_node(entity.id, **entity.properties, type=entity.type.value, name=entity.name)
    
    async def upsert_relation(self, relation: GraphRelation) -> None:
        self.relations[relation.id] = relation
        self.graph.add_edge(
            relation.source_id,
            relation.target_id,
            key=relation.id,
            type=relation.type.value,
            **relation.properties,
        )
    
    async def get_entity(self, entity_id: str) -> Optional[GraphEntity]:
        return self.entities.get(entity_id)
    
    async def get_relations(
        self,
        entity_id: str,
        relation_types: Optional[List[RelationType]] = None,
        direction: str = "both",
    ) -> List[GraphRelation]:
        relations = []
        
        if direction in ("out", "both"):
            for _, target, key, data in self.graph.out_edges(entity_id, keys=True, data=True):
                rel = self.relations.get(key)
                if rel and (not relation_types or rel.type in relation_types):
                    relations.append(rel)
        
        if direction in ("in", "both"):
            for source, _, key, data in self.graph.in_edges(entity_id, keys=True, data=True):
                rel = self.relations.get(key)
                if rel and (not relation_types or rel.type in relation_types):
                    relations.append(rel)
        
        return relations
    
    async def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 3,
    ) -> List[GraphPath]:
        if source_id not in self.graph or target_id not in self.graph:
            return []
        
        paths = []
        try:
            for path in nx.all_simple_paths(self.graph, source_id, target_id, cutoff=max_hops):
                if len(path) < 2:
                    continue
                
                entities = [self.entities[eid] for eid in path if eid in self.entities]
                relations = []
                
                for i in range(len(path) - 1):
                    edges = self.graph.get_edge_data(path[i], path[i+1])
                    if edges:
                        # Take first edge
                        key = list(edges.keys())[0]
                        rel = self.relations.get(key)
                        if rel:
                            relations.append(rel)
                
                if entities and relations:
                    paths.append(GraphPath(entities=entities, relations=relations))
        except nx.NetworkXNoPath:
            pass
        
        return paths
    
    async def query(self, cypher: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        # NetworkX doesn't support Cypher - simplified query
        # In production, use actual Cypher with Neo4j
        return []
    
    async def get_subgraph(
        self,
        entity_ids: List[str],
        hops: int = 1,
    ) -> Tuple[List[GraphEntity], List[GraphRelation]]:
        subgraph_entities = set(entity_ids)
        subgraph_relations = set()
        
        for eid in entity_ids:
            if eid not in self.graph:
                continue
            
            # Get neighbors up to hops
            for neighbor in nx.single_source_shortest_path_length(self.graph, eid, cutoff=hops):
                subgraph_entities.add(neighbor)
            
            for neighbor in nx.single_source_shortest_path_length(self.graph.reverse(), eid, cutoff=hops):
                subgraph_entities.add(neighbor)
        
        # Get relations between these entities
        for eid in subgraph_entities:
            for _, target, key in self.graph.out_edges(eid, keys=True):
                if target in subgraph_entities:
                    subgraph_relations.add(key)
            for source, _, key in self.graph.in_edges(eid, keys=True):
                if source in subgraph_entities:
                    subgraph_relations.add(key)
        
        entities = [self.entities[eid] for eid in subgraph_entities if eid in self.entities]
        relations = [self.relations[rid] for rid in subgraph_relations if rid in self.relations]
        
        return entities, relations
    
    async def close(self) -> None:
        self.graph.clear()
        self.entities.clear()
        self.relations.clear()


class EntityExtractor(ABC):
    """Abstract base for entity extraction."""
    
    @abstractmethod
    async def extract(self, text: str, chunk: DocumentChunk) -> List[GraphEntity]:
        pass


class IPExtractor(EntityExtractor):
    """Entity extractor for IP documents."""
    
    # Patterns for IP entities
    PATENT_PATTERNS = {
        EntityType.PATENT: [
            r'\b(?:US|EP|WO|IN|CN|JP|KR|DE|GB|FR)\d{7,12}[A-Z]?\b',
            r'\b\d{4,}\/\d{4,}\b',  # Application numbers
        ],
        EntityType.INVENTOR: [
            r'\bInventor[s]?:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)(?:,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)*',
        ],
        EntityType.ASSIGNEE: [
            r'\bAssignee[s]?:\s*([A-Z][A-Za-z0-9\s&.,]+(?:Inc|LLC|Ltd|Corp|GmbH|Co\.?))',
        ],
        EntityType.TECHNOLOGY_CLASS: [
            r'\b(?:IPC|CPC|USPC):\s*([A-H]\d{2}[A-Z]\s*\d+(?:\/\d+)?)',
        ],
    }
    
    TRADEMARK_PATTERNS = {
        EntityType.TRADEMARK: [
            r'\b(?:TM|®|SM)\s*[A-Z0-9][A-Za-z0-9\s\-]{1,50}',
            r'\bTrademark:\s*([A-Z0-9][A-Za-z0-9\s\-]{1,50})',
        ],
        EntityType.OWNER: [
            r'\bOwner[s]?:\s*([A-Z][A-Za-z0-9\s&.,]+(?:Inc|LLC|Ltd|Corp|GmbH|Co\.?))',
        ],
        EntityType.NICE_CLASS: [
            r'\b(?:Class|Cl\.)\s*(\d{1,2})\b',
        ],
    }
    
    LEGAL_PATTERNS = {
        EntityType.CASE: [
            r'\b\d{4}\s+[A-Z]+\s+\d+\b',  # Case citations
            r'\b[A-Z][a-z]+\s+v\.\s+[A-Z][a-z]+\b',  # Party names
        ],
        EntityType.COURT: [
            r'\b(?:Supreme Court|Federal Circuit|High Court|District Court|Appellate Court)\b',
        ],
        EntityType.JUDGE: [
            r'\b(?:Justice|Judge|JJ\.)\s+[A-Z][a-z]+',
        ],
        EntityType.STATUTE: [
            r'\b(?:Section|Article|§)\s+\d+[A-Z]?(?:\s+of\s+the\s+[A-Za-z\s]+Act)?',
        ],
    }
    
    async def extract(self, text: str, chunk: DocumentChunk) -> List[GraphEntity]:
        entities = []
        doc_type = chunk.metadata.get('document_type')
        
        patterns = {}
        if doc_type == DocumentType.PATENT:
            patterns = self.PATENT_PATTERNS
        elif doc_type == DocumentType.TRADEMARK:
            patterns = self.TRADEMARK_PATTERNS
        elif doc_type in (DocumentType.CASE_LAW, DocumentType.STATUTE, DocumentType.REGULATION, DocumentType.LEGAL_OPINION):
            patterns = self.LEGAL_PATTERNS
        
        for entity_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = self._find_matches(text, pattern, entity_type, chunk)
                entities.extend(matches)
        
        # Deduplicate by name and type
        return self._deduplicate(entities)
    
    def _find_matches(
        self,
        text: str,
        pattern: str,
        entity_type: EntityType,
        chunk: DocumentChunk,
    ) -> List[GraphEntity]:
        import re
        entities = []
        
        for match in re.finditer(pattern, text, re.IGNORECASE):
            name = match.group(1) if match.groups() else match.group(0)
            name = name.strip()
            
            if len(name) < 2:
                continue
            
            entity = GraphEntity(
                id=str(uuid.uuid4()),
                type=entity_type,
                name=name,
                source_chunks=[chunk.id],
                confidence=0.8,
            )
            entities.append(entity)
        
        return entities
    
    def _deduplicate(self, entities: List[GraphEntity]) -> List[GraphEntity]:
        seen = {}
        for entity in entities:
            key = (entity.type, entity.name.lower())
            if key not in seen:
                seen[key] = entity
            else:
                # Merge source chunks
                seen[key].source_chunks.extend(entity.source_chunks)
                seen[key].confidence = max(seen[key].confidence, entity.confidence)
        
        return list(seen.values())


class RelationExtractor(ABC):
    """Abstract base for relation extraction."""
    
    @abstractmethod
    async def extract(
        self,
        text: str,
        chunk: DocumentChunk,
        entities: List[GraphEntity],
    ) -> List[GraphRelation]:
        pass


class IPRelationExtractor(RelationExtractor):
    """Relation extractor for IP documents."""
    
    RELATION_PATTERNS = {
        RelationType.INVENTED_BY: [
            r'(\w+(?:\s+\w+)*)\s+invented\s+(\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s+is\s+the\s+inventor\s+of\s+(\w+(?:\s+\w+)*)',
        ],
        RelationType.ASSIGNED_TO: [
            r'(\w+(?:\s+\w+)*)\s+assigned\s+to\s+(\w+(?:\s+\w+)*)',
            r'assignee\s+(\w+(?:\s+\w+)*)',
        ],
        RelationType.CITES: [
            r'cites?\s+(\w+(?:\s+\w+)*)',
            r'references?\s+(\w+(?:\s+\w+)*)',
            r'prior\s+art\s+(\w+(?:\s+\w+)*)',
        ],
        RelationType.CLAIMS_PRIORITY: [
            r'claims?\s+priority\s+(?:from\s+)?(\w+(?:\s+\w+)*)',
        ],
        RelationType.CLASSIFIED_AS: [
            r'classified\s+as\s+(\w+(?:\s+\w+)*)',
            r'IPC\s+(\w+(?:\s+\w+)*)',
            r'CPC\s+(\w+(?:\s+\w+)*)',
        ],
        RelationType.OPPOSSES: [
            r'opposes?\s+(\w+(?:\s+\w+)*)',
            r'opposition\s+against\s+(\w+(?:\s+\w+)*)',
        ],
        RelationType.INVALIDATES: [
            r'invalidates?\s+(\w+(?:\s+\w+)*)',
            r'revokes?\s+(\w+(?:\s+\w+)*)',
        ],
        RelationType.LICENSES_TO: [
            r'licenses?\s+to\s+(\w+(?:\s+\w+)*)',
            r'licensee\s+(\w+(?:\s+\w+)*)',
        ],
    }
    
    async def extract(
        self,
        text: str,
        chunk: DocumentChunk,
        entities: List[GraphEntity],
    ) -> List[GraphRelation]:
        relations = []
        entity_by_name = {e.name.lower(): e for e in entities}
        
        import re
        for rel_type, patterns in self.RELATION_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    groups = match.groups()
                    if len(groups) >= 2:
                        source_name = groups[0].strip().lower()
                        target_name = groups[1].strip().lower()
                    elif len(groups) == 1:
                        # Try to infer from context
                        continue
                    else:
                        continue
                    
                    source = entity_by_name.get(source_name)
                    target = entity_by_name.get(target_name)
                    
                    if source and target and source.id != target.id:
                        relation = GraphRelation(
                            id=str(uuid.uuid4()),
                            source_id=source.id,
                            target_id=target.id,
                            type=rel_type,
                            source_chunks=[chunk.id],
                            confidence=0.7,
                        )
                        relations.append(relation)
        
        return relations


class GraphBuilder:
    """Builds knowledge graph from documents."""
    
    def __init__(
        self,
        graph_store: GraphStore,
        entity_extractor: Optional[EntityExtractor] = None,
        relation_extractor: Optional[RelationExtractor] = None,
    ):
        self.graph_store = graph_store
        self.entity_extractor = entity_extractor or IPExtractor()
        self.relation_extractor = relation_extractor or IPRelationExtractor()
    
    async def build_from_chunks(self, chunks: List[DocumentChunk]) -> Dict[str, int]:
        """Build graph from document chunks."""
        stats = {'entities': 0, 'relations': 0}
        
        all_entities = []
        all_relations = []
        
        # Extract entities from all chunks
        for chunk in chunks:
            entities = await self.entity_extractor.extract(chunk.content, chunk)
            all_entities.extend(entities)
            
            # Extract relations
            relations = await self.relation_extractor.extract(chunk.content, chunk, entities)
            all_relations.extend(relations)
        
        # Upsert entities
        for entity in all_entities:
            existing = await self.graph_store.get_entity(entity.id)
            if existing:
                # Merge
                existing.source_chunks.extend(entity.source_chunks)
                existing.confidence = max(existing.confidence, entity.confidence)
                await self.graph_store.upsert_entity(existing)
            else:
                await self.graph_store.upsert_entity(entity)
            stats['entities'] += 1
        
        # Upsert relations
        for relation in all_relations:
            await self.graph_store.upsert_relation(relation)
            stats['relations'] += 1
        
        return stats
    
    async def build_from_document(self, document: Document) -> Dict[str, int]:
        """Build graph from a document with chunks."""
        if not document.chunks:
            return {'entities': 0, 'relations': 0}
        return await self.build_from_chunks(document.chunks)


class GraphRetriever:
    """Graph-based retrieval for GraphRAG."""
    
    def __init__(
        self,
        graph_store: GraphStore,
        settings: Optional[Settings] = None,
    ):
        self.graph_store = graph_store
        self.settings = settings or get_settings()
    
    async def retrieve_subgraph(
        self,
        query_entities: List[str],
        max_hops: int = 2,
    ) -> Tuple[List[GraphEntity], List[GraphRelation]]:
        """Retrieve subgraph around query entities."""
        # Find entities matching query
        entity_ids = []
        for query_entity in query_entities:
            # In production, would do fuzzy matching
            # For now, assume exact ID match
            entity = await self.graph_store.get_entity(query_entity)
            if entity:
                entity_ids.append(entity.id)
        
        if not entity_ids:
            return [], []
        
        return await self.graph_store.get_subgraph(entity_ids, max_hops)
    
    async def find_connecting_paths(
        self,
        source_entity: str,
        target_entity: str,
        max_hops: int = 3,
    ) -> List[GraphPath]:
        """Find paths connecting two entities."""
        return await self.graph_store.find_paths(source_entity, target_entity, max_hops)
    
    async def retrieve_neighborhood(
        self,
        entity_id: str,
        relation_types: Optional[List[RelationType]] = None,
        hops: int = 1,
    ) -> Tuple[List[GraphEntity], List[GraphRelation]]:
        """Retrieve neighborhood around an entity."""
        entities, relations = await self.graph_store.get_subgraph([entity_id], hops)
        
        # Filter by relation types if specified
        if relation_types:
            filtered_relations = [r for r in relations if r.type in relation_types]
            # Get entities connected by filtered relations
            connected_ids = set()
            for rel in filtered_relations:
                connected_ids.add(rel.source_id)
                connected_ids.add(rel.target_id)
            entities = [e for e in entities if e.id in connected_ids]
            relations = filtered_relations
        
        return entities, relations


class GraphRAGJustification:
    """Run experiments to justify GraphRAG approach."""
    
    def __init__(
        self,
        experiment_manager: ExperimentManager,
        evaluation_harness: EvaluationHarness,
    ):
        self.experiment_manager = experiment_manager
        self.evaluation_harness = evaluation_harness
    
    async def run_graphrag_justification(
        self,
        test_cases: List[TestCase],
        baseline_system_factory: Callable[[Dict[str, Any]], Any],
        graphrag_system_factory: Callable[[Dict[str, Any]], Any],
    ) -> Dict[str, Any]:
        """Run experiment comparing baseline RAG vs GraphRAG."""
        
        # Create ablation experiment
        from ip_sakti.experiments.framework import ExperimentConfig, ExperimentType
        config = ExperimentConfig(
            name="GraphRAG Justification",
            description="Compare baseline RAG vs GraphRAG with knowledge graph",
            experiment_type=ExperimentType.ABLATION,
            hypothesis="GraphRAG with knowledge graph improves complex multi-hop queries",
            success_criteria={
                'faithfulness_mean': 0.85,
                'answer_relevancy_mean': 0.85,
                'precision_at_k_5_mean': 0.8,
            },
        )
        
        experiment = self.experiment_manager.create_experiment(
            config=config,
            test_cases=test_cases,
        )
        experiment.metadata['base_config'] = {'use_graph': True}
        experiment.metadata['components_to_ablate'] = ['use_graph']
        
        # We need to test both with and without graph
        # Run ablation
        from ip_sakti.experiments.framework import AblationExperimentRunner
        runner = AblationExperimentRunner(
            base_config={'use_graph': True},
            components_to_ablate=['use_graph'],
        )
        
        # Create custom system factory that respects use_graph
        def graph_system_factory(params: Dict[str, Any]) -> Dict[str, Any]:
            use_graph = params.get('use_graph', True)
            if use_graph:
                return graphrag_system_factory(params)
            else:
                return baseline_system_factory(params)
        
        result = await runner.run(experiment, self.evaluation_harness, graph_system_factory)
        
        return self.experiment_manager.get_experiment_results(result.id)


class KnowledgeGraph:
    """Main knowledge graph orchestrator."""
    
    def __init__(self, config: Optional[GraphConfig] = None, settings: Optional[Settings] = None):
        self.config = config or GraphConfig()
        self.settings = settings or get_settings()
        
        # Components
        self.graph_store = self._create_graph_store()
        self.entity_extractor = IPExtractor()
        self.relation_extractor = IPRelationExtractor()
        self.builder = GraphBuilder(self.graph_store, self.entity_extractor, self.relation_extractor)
        self.retriever = GraphRetriever(self.graph_store, self.settings)
        
        # Experiment integration
        self.experiment_manager = None
        self.evaluation_harness = None
        self.graphrag_justification = None
    
    def _create_graph_store(self) -> GraphStore:
        store_type = self.config.store_type
        store_config = self.config.store_config
        
        if store_type == GraphStoreType.NETWORKX:
            return NetworkXGraphStore(store_config)
        # elif store_type == GraphStoreType.NEO4J:
        #     return Neo4jGraphStore(store_config)
        
        logger.warning(f"Graph store {store_type} not implemented, using NetworkX")
        return NetworkXGraphStore(store_config)
    
    async def initialize(self) -> None:
        """Initialize the knowledge graph."""
        await self.graph_store.initialize()
        logger.info("Knowledge graph initialized")
    
    async def ingest_document(self, document: Document) -> Dict[str, int]:
        """Ingest a document into the knowledge graph."""
        return await self.builder.build_from_document(document)
    
    async def ingest_chunks(self, chunks: List[DocumentChunk]) -> Dict[str, int]:
        """Ingest chunks into the knowledge graph."""
        return await self.builder.build_from_chunks(chunks)
    
    async def query_subgraph(
        self,
        query_entities: List[str],
        max_hops: int = 2,
    ) -> Tuple[List[GraphEntity], List[GraphRelation]]:
        """Query subgraph around entities."""
        return await self.retriever.retrieve_subgraph(query_entities, max_hops)
    
    async def find_paths(
        self,
        source: str,
        target: str,
        max_hops: int = 3,
    ) -> List[GraphPath]:
        """Find paths between entities."""
        return await self.retriever.find_connecting_paths(source, target, max_hops)
    
    async def get_neighborhood(
        self,
        entity_id: str,
        relation_types: Optional[List[RelationType]] = None,
        hops: int = 1,
    ) -> Tuple[List[GraphEntity], List[GraphRelation]]:
        """Get neighborhood around entity."""
        return await self.retriever.retrieve_neighborhood(entity_id, relation_types, hops)
    
    async def justify_graphrag(
        self,
        test_cases: List[TestCase],
        baseline_factory: Callable[[Dict[str, Any]], Any],
        graphrag_factory: Callable[[Dict[str, Any]], Any],
    ) -> Dict[str, Any]:
        """Run justification experiment for GraphRAG."""
        from ip_sakti.experiments.framework import ExperimentManager
        from ip_sakti.eval.evaluation import EvaluationHarness
        manager = ExperimentManager(self.settings)
        harness = EvaluationHarness()
        return await GraphRAGJustification(manager, harness).run_graphrag_justification(
            test_cases, baseline_factory, graphrag_factory
        )
    
    async def run_experiment(
        self,
        config: ExperimentConfig,
        search_space: List[ExperimentParameter],
        test_cases: List[TestCase],
        system_factory: Callable[[Dict[str, Any]], Any],
    ) -> Dict[str, Any]:
        """Run a generic experiment on the knowledge graph."""
        from ip_sakti.experiments.framework import ExperimentManager
        manager = ExperimentManager(self.settings)
        experiment = manager.create_experiment(
            config=config,
            search_space=search_space,
            test_cases=test_cases,
        )
        
        result = await manager.run_experiment(experiment.id, system_factory)
        return manager.get_experiment_results(result.id)
    
    async def close(self) -> None:
        """Close the knowledge graph."""
        await self.graph_store.close()


def create_knowledge_graph(
    config: Optional[GraphConfig] = None,
    settings: Optional[Settings] = None,
) -> KnowledgeGraph:
    """Factory to create knowledge graph."""
    return KnowledgeGraph(config, settings)
