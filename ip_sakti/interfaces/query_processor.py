"""
Query processor interfaces for IP-SAKTI.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum
from ip_sakti.interfaces.retriever import QueryIntent, QueryComplexity


class QueryAnalysis(BaseModel):
    """Result of query analysis."""
    intent: QueryIntent
    complexity: QueryComplexity
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    legal_citations: List[str] = Field(default_factory=list)
    acts_referenced: List[str] = Field(default_factory=list)
    sections_referenced: List[str] = Field(default_factory=list)
    dates_referenced: List[str] = Field(default_factory=list)
    jurisdiction_hints: List[str] = Field(default_factory=list)
    requires_multi_hop: bool = False
    requires_comparison: bool = False
    requires_temporal: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryRewrite(BaseModel):
    """Rewritten query variants."""
    original_query: str
    rewritten_queries: List[str]
    hyde_document: Optional[str] = None
    step_back_query: Optional[str] = None
    sub_queries: List[str] = Field(default_factory=list)
    expansion_terms: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryRoute(BaseModel):
    """Routing decision for query."""
    primary_strategy: str  # RetrievalStrategy
    fallback_strategies: List[str] = Field(default_factory=list)
    use_kg: bool = False
    use_graph: bool = False
    use_contextual: bool = False
    requires_reranking: bool = True
    rerank_stages: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IQueryProcessor(ABC):
    """Interface for query processing components."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Query processor name."""
        pass
    
    @abstractmethod
    async def analyze(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> QueryAnalysis:
        """Analyze query for intent, complexity, entities."""
        pass
    
    @abstractmethod
    async def rewrite(self, query: str, analysis: QueryAnalysis) -> QueryRewrite:
        """Rewrite query for better retrieval."""
        pass
    
    @abstractmethod
    async def route(self, query: str, analysis: QueryAnalysis, rewrite: QueryRewrite) -> QueryRoute:
        """Determine retrieval strategy for query."""
        pass
    
    @abstractmethod
    async def process(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Full query processing pipeline: analyze → rewrite → route."""
        pass


class QueryProcessorMetrics(BaseModel):
    """Metrics for query processor evaluation."""
    analysis_latency_ms: float = 0.0
    rewrite_latency_ms: float = 0.0
    route_latency_ms: float = 0.0
    intent_accuracy: float = 0.0
    rewrite_quality: float = 0.0


class IQueryAnalyzer(ABC):
    """Interface for query analysis."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Analyzer name."""
        pass
    
    @abstractmethod
    async def analyze(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> QueryAnalysis:
        """Analyze query for intent, complexity, entities."""
        pass
    
    @abstractmethod
    async def extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities from query."""
        pass
    
    @abstractmethod
    async def decompose(self, query: str) -> List[str]:
        """Decompose complex query into sub-queries."""
        pass


class IQueryRewriter(ABC):
    """Interface for query rewriting."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Rewriter name."""
        pass
    
    @abstractmethod
    async def rewrite(self, query: str, analysis: QueryAnalysis) -> QueryRewrite:
        """Rewrite query for better retrieval."""
        pass
    
    @abstractmethod
    async def expand(self, query: str, num_variants: int = 3) -> List[str]:
        """Expand query into multiple variants."""
        pass


class IQueryRouter(ABC):
    """Interface for query routing."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Router name."""
        pass
    
    @abstractmethod
    async def route(self, query: str, analysis: Optional[QueryAnalysis] = None) -> QueryRoute:
        """Determine retrieval strategy for query."""
        pass
    
    @abstractmethod
    async def should_use_kg(self, query: str) -> bool:
        """Determine if KG retrieval is needed."""
        pass
    
    @abstractmethod
    async def should_use_graph(self, query: str) -> bool:
        """Determine if graph traversal is needed."""
        pass