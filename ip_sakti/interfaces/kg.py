"""
Knowledge Graph interfaces for IP-SAKTI.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class GraphNodeType(str, Enum):
    """Types of nodes in the legal knowledge graph."""
    CASE = "case"
    STATUTE = "statute"
    SECTION = "section"
    SUBSECTION = "subsection"
    CITATION = "citation"
    JUDGE = "judge"
    COURT = "court"
    PARTY = "party"
    LEGAL_CONCEPT = "legal_concept"
    AUTHORITY = "authority"
    DATE = "date"
    JURISDICTION = "jurisdiction"


class GraphEdgeType(str, Enum):
    """Types of edges in the legal knowledge graph."""
    CITES = "cites"
    CITED_BY = "cited_by"
    DEFINES = "defines"
    AMENDS = "amends"
    REPEALS = "repeals"
    OVERRULES = "overrules"
    DISTINGUISHES = "distinguishes"
    FOLLOWS = "follows"
    CONCURS = "concurs"
    DISSENTS = "dissents"
    PART_OF = "part_of"
    HAS_SECTION = "has_section"
    HAS_JUDGE = "has_judge"
    HAS_COURT = "has_court"
    HAS_PARTY = "has_party"
    RELATES_TO = "relates_to"
    SIMILAR_TO = "similar_to"


class GraphNode(BaseModel):
    """Node in the knowledge graph."""
    id: str
    type: GraphNodeType
    label: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None


class GraphEdge(BaseModel):
    """Edge in the knowledge graph."""
    id: str
    source: str
    target: str
    type: GraphEdgeType
    properties: Dict[str, Any] = Field(default_factory=dict)
    weight: float = 1.0


class GraphQuery(BaseModel):
    """Query for graph traversal."""
    start_nodes: List[str] = Field(default_factory=list)
    node_types: Optional[List[GraphNodeType]] = None
    edge_types: Optional[List[GraphEdgeType]] = None
    max_depth: int = Field(default=2, ge=1, le=5)
    limit: int = Field(default=50, ge=1, le=500)
    filters: Dict[str, Any] = Field(default_factory=dict)
    return_paths: bool = False


class GraphResult(BaseModel):
    """Result from graph query."""
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
    paths: List[List[str]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IKnowledgeGraph(ABC):
    """Interface for knowledge graph components."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Knowledge graph name."""
        pass
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize graph backend."""
        pass
    
    @abstractmethod
    async def add_node(self, node: GraphNode) -> str:
        """Add node to graph."""
        pass
    
    @abstractmethod
    async def add_nodes(self, nodes: List[GraphNode]) -> List[str]:
        """Batch add nodes."""
        pass
    
    @abstractmethod
    async def add_edge(self, edge: GraphEdge) -> str:
        """Add edge to graph."""
        pass
    
    @abstractmethod
    async def add_edges(self, edges: List[GraphEdge]) -> List[str]:
        """Batch add edges."""
        pass
    
    @abstractmethod
    async def query(self, query: GraphQuery) -> GraphResult:
        """Execute graph query."""
        pass
    
    @abstractmethod
    async def get_neighbors(self, node_id: str, edge_types: Optional[List[GraphEdgeType]] = None, max_depth: int = 1) -> GraphResult:
        """Get neighboring nodes."""
        pass
    
    @abstractmethod
    async def find_path(self, source_id: str, target_id: str, max_depth: int = 4) -> Optional[List[str]]:
        """Find shortest path between nodes."""
        pass
    
    @abstractmethod
    async def get_subgraph(self, node_ids: List[str], max_depth: int = 2) -> GraphResult:
        """Get subgraph around nodes."""
        pass
    
    @abstractmethod
    async def compute_pagerank(self, node_type: Optional[GraphNodeType] = None) -> Dict[str, float]:
        """Compute PageRank scores for authority propagation."""
        pass
    
    @abstractmethod
    async def detect_communities(self, algorithm: str = "leiden") -> Dict[str, int]:
        """Detect communities in graph."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Health check."""
        pass


class KGMetrics(BaseModel):
    """Metrics for knowledge graph evaluation."""
    node_count: int = 0
    edge_count: int = 0
    avg_degree: float = 0.0
    num_communities: int = 0
    query_latency_ms: float = 0.0