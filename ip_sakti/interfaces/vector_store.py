"""
Vector store interfaces for IP-SAKTI.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class Chunk(BaseModel):
    """Chunk model for vector store."""
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source: Optional[str] = None
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0
    authority_score: float = 0.0
    jurisdiction: Optional[str] = None
    document_type: Optional[str] = None
    embedding: Optional[List[float]] = None


class VectorStoreBackend(str, Enum):
    """Supported vector store backends."""
    QDRANT = "qdrant"
    CHROMA = "chroma"
    FAISS = "faiss"
    IN_MEMORY = "in_memory"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    MILVUS = "milvus"
    AZURE_AI_SEARCH = "azure_ai_search"
    ELASTICSEARCH = "elasticsearch"


class DistanceMetric(str, Enum):
    """Distance metrics for vector search."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


class VectorStoreConfig(BaseModel):
    """Configuration for vector store."""
    backend: VectorStoreBackend
    collection_name: str = "ip_sakti_chunks"
    dimensions: int = 1024
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    host: Optional[str] = None
    port: Optional[int] = None
    api_key: Optional[str] = None
    path: Optional[str] = None
    index_params: Dict[str, Any] = Field(default_factory=dict)
    search_params: Dict[str, Any] = Field(default_factory=dict)


class IndexRequest(BaseModel):
    """Request for indexing chunks."""
    chunks: List[Chunk]
    upsert: bool = True
    batch_size: int = Field(default=100, ge=1, le=1000)


class SearchRequest(BaseModel):
    """Request for vector search."""
    query_vector: List[float]
    top_k: int = Field(default=10, ge=1, le=1000)
    filter: Optional[Dict[str, Any]] = None
    include_vectors: bool = False
    include_metadata: bool = True


class SearchResult(BaseModel):
    """Result from vector search."""
    chunk: Chunk
    score: float
    rank: int


class IVectorStore(ABC):
    """Interface for vector store components."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Vector store name."""
        pass
    
    @property
    @abstractmethod
    def backend(self) -> VectorStoreBackend:
        """Backend type."""
        pass
    
    @abstractmethod
    async def initialize(self, config: VectorStoreConfig) -> None:
        """Initialize vector store."""
        pass
    
    @abstractmethod
    async def index(self, request: IndexRequest) -> Dict[str, Any]:
        """Index chunks into vector store."""
        pass
    
    @abstractmethod
    async def search(self, request: SearchRequest) -> List[SearchResult]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    async def delete(self, chunk_ids: List[str]) -> Dict[str, Any]:
        """Delete chunks by IDs."""
        pass
    
    @abstractmethod
    async def get_by_ids(self, chunk_ids: List[str]) -> List[Chunk]:
        """Get chunks by IDs."""
        pass
    
    @abstractmethod
    async def count(self, filter: Optional[Dict[str, Any]] = None) -> int:
        """Count chunks in collection."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Health check."""
        pass


class VectorStoreMetrics(BaseModel):
    """Metrics for vector store evaluation."""
    index_latency_ms: float = 0.0
    search_latency_ms: float = 0.0
    index_throughput_chunks_per_sec: float = 0.0
    search_throughput_queries_per_sec: float = 0.0
    memory_usage_mb: float = 0.0
    index_size_mb: float = 0.0


class VectorStoreHealth(BaseModel):
    """Health status for vector store."""
    status: str = "healthy"
    backend: str = ""
    collection_count: int = 0
    total_vectors: int = 0
    memory_usage_mb: float = 0.0
    latency_ms: float = 0.0
    last_error: Optional[str] = None