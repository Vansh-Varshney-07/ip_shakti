"""
Retriever interfaces for IP-SAKTI.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class RetrievalStrategy(str, Enum):
    """Available retrieval strategies."""
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    GRAPH = "graph"
    KG = "kg"
    CONTEXTUAL = "contextual"
    AUTO = "auto"


class QueryIntent(str, Enum):
    """Query intent classification."""
    FACTUAL = "factual"
    COMPARISON = "comparison"
    MULTI_HOP = "multi_hop"
    SUMMARIZATION = "summarization"
    LEGAL_SEARCH = "legal_search"
    PROVISION_LOOKUP = "provision_lookup"
    CASE_LAW = "case_law"
    STATUTORY_INTERPRETATION = "statutory_interpretation"
    UNKNOWN = "unknown"


class QueryComplexity(str, Enum):
    """Query complexity level."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    MULTI_HOP = "multi_hop"


class Chunk(BaseModel):
    """Chunk model for retrieval results."""
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source: Optional[str] = None
    chunk_index: int = 0
    parent_id: Optional[str] = None


class CitationRef(BaseModel):
    """Citation reference."""
    cited_act: str
    cited_section: str
    cited_text: str
    citation_type: str
    hierarchy_path: Optional[Dict[str, Any]] = None
    confidence: float = 1.0


class RetrievalRequest(BaseModel):
    """Request for retrieval."""
    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    strategy: RetrievalStrategy = RetrievalStrategy.AUTO
    intent: Optional[QueryIntent] = None
    complexity: Optional[QueryComplexity] = None
    filters: Dict[str, Any] = Field(default_factory=dict)
    jurisdiction: Optional[str] = None
    authority_tiers: Optional[List[int]] = None
    temporal_range: Optional[Dict[str, str]] = None  # start_date, end_date
    include_metadata: bool = True
    rerank: bool = True
    rerank_top_k: Optional[int] = None
    contextual_embedding: bool = False


class RetrievalResult(BaseModel):
    """Result from retrieval."""
    chunks: List[Chunk]
    scores: List[float]
    strategy_used: RetrievalStrategy
    total_candidates: int
    retrieval_time_ms: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IRetriever(ABC):
    """Interface for retrieval components."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Retriever name."""
        pass
    
    @property
    @abstractmethod
    def supported_strategies(self) -> List[RetrievalStrategy]:
        """List of supported retrieval strategies."""
        pass
    
    @abstractmethod
    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Retrieve relevant chunks for a query."""
        pass
    
    @abstractmethod
    async def retrieve_batch(self, requests: List[RetrievalRequest]) -> List[RetrievalResult]:
        """Batch retrieval for multiple queries."""
        pass
    
    @abstractmethod
    async def get_similar_chunks(self, chunk: Chunk, top_k: int = 5) -> RetrievalResult:
        """Find chunks similar to a given chunk."""
        pass


class IHybridRetriever(IRetriever):
    """Interface for hybrid retrieval combining multiple strategies."""
    
    @abstractmethod
    async def retrieve_with_strategies(
        self,
        query: str,
        strategies: List[RetrievalStrategy],
        k: int = 10,
        weights: Optional[Dict[RetrievalStrategy, float]] = None
    ) -> RetrievalResult:
        """Retrieve using multiple strategies with optional weights."""
        pass
    
    def add_retriever(self, retriever: IRetriever) -> None:
        """Add a retriever to the hybrid."""
        pass


class RetrievalMetrics(BaseModel):
    """Metrics for retrieval evaluation."""
    recall_at_k: Dict[int, float] = Field(default_factory=dict)
    precision_at_k: Dict[int, float] = Field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_k: Dict[int, float] = Field(default_factory=dict)
    latency_ms: float = 0.0
    total_queries: int = 0