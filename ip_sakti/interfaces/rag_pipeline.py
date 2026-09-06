"""
RAG pipeline interface for IP-SAKTI.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncGenerator
from pydantic import BaseModel, Field

from ip_sakti.interfaces.retriever import RetrievalResult
from ip_sakti.interfaces.generator import GenerationResult
from ip_sakti.interfaces.query_processor import QueryAnalysis, QueryRewrite, QueryRoute


class RAGResponse(BaseModel):
    """Response from RAG pipeline."""
    answer: str
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    retrieval_results: List[RetrievalResult] = Field(default_factory=list)
    query_analysis: Optional[QueryAnalysis] = None
    query_rewrite: Optional[QueryRewrite] = None
    query_route: Optional[QueryRoute] = None
    generation_result: Optional[GenerationResult] = None


class IRAGPipeline(ABC):
    """Interface for RAG pipeline."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Pipeline name."""
        pass
    
    @abstractmethod
    async def query(self, query: str, **kwargs) -> RAGResponse:
        """Execute RAG query."""
        pass
    
    @abstractmethod
    async def stream_query(self, query: str, **kwargs) -> AsyncGenerator[str, None]:
        """Stream RAG query response."""
        pass
    
    @abstractmethod
    async def batch_query(self, queries: List[str], **kwargs) -> List[RAGResponse]:
        """Execute multiple RAG queries."""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Health check."""
        pass


class RAGPipelineMetrics(BaseModel):
    """Metrics for RAG pipeline evaluation."""
    avg_latency_ms: float = 0.0
    throughput_queries_per_sec: float = 0.0
    success_rate: float = 1.0
    avg_confidence: float = 0.0