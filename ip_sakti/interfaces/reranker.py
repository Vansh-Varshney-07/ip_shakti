"""
Reranker interfaces for IP-SAKTI.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class Chunk(BaseModel):
    """Chunk model for reranking."""
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source: Optional[str] = None
    chunk_index: int = 0
    parent_id: Optional[str] = None


class RerankerModel(str, Enum):
    """Supported reranker models."""
    BGE_RERANKER_V2_M3 = "BAAI/bge-reranker-v2-m3"
    BGE_RERANKER_BASE = "BAAI/bge-reranker-base"
    NV_RERANK_MISTRAL_4B = "nvidia/nv-rerankqa-mistral-4b-v3"
    MS_MARCO_MINILM = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    MS_MARCO_MPNET = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    JINA_RERANKER_V2 = "jinaai/jina-reranker-v2-base-multilingual"
    COHERE_RERANK_3 = "rerank-english-v3.0"
    COHERE_RERANK_MULTILINGUAL = "rerank-multilingual-v3.0"


class RerankStrategy(str, Enum):
    """Reranking strategies."""
    CROSS_ENCODER = "cross_encoder"
    LLM_JUDGE = "llm_judge"
    AUTHORITY_FUSION = "authority_fusion"
    HYBRID = "hybrid"
    NONE = "none"


class RerankRequest(BaseModel):
    """Request for reranking."""
    query: str
    chunks: List[Chunk]
    top_k: int = Field(default=10, ge=1, le=100)
    strategy: RerankStrategy = RerankStrategy.CROSS_ENCODER
    model: Optional[RerankerModel] = None
    authority_scores: Optional[List[float]] = None
    alpha: float = Field(default=0.7, ge=0.0, le=1.0)  # relevance weight
    beta: float = Field(default=0.2, ge=0.0, le=1.0)   # authority weight
    gamma: float = Field(default=0.1, ge=0.0, le=1.0)   # recency weight


class RerankResult(BaseModel):
    """Result from reranking."""
    chunks: List[Chunk]
    relevance_scores: List[float]
    original_indices: List[int]
    strategy_used: RerankStrategy
    rerank_time_ms: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IReranker(ABC):
    """Interface for reranking components."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Reranker name."""
        pass
    
    @property
    @abstractmethod
    def supported_strategies(self) -> List[RerankStrategy]:
        """List of supported reranking strategies."""
        pass
    
    @property
    @abstractmethod
    def default_model(self) -> RerankerModel:
        """Default reranker model."""
        pass
    
    @abstractmethod
    async def rerank(self, request: RerankRequest) -> RerankResult:
        """Rerank chunks based on query relevance."""
        pass
    
    @abstractmethod
    async def rerank_batch(self, requests: List[RerankRequest]) -> List[RerankResult]:
        """Batch reranking."""
        pass
    
    @abstractmethod
    async def score_pairs(self, query: str, texts: List[str]) -> List[float]:
        """Score query-text pairs directly."""
        pass


class RerankerMetrics(BaseModel):
    """Metrics for reranker evaluation."""
    latency_ms: float = 0.0
    throughput_pairs_per_sec: float = 0.0
    ndcg_improvement: float = 0.0
    model_load_time_ms: float = 0.0