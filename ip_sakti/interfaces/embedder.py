"""
Embedder interfaces for IP-SAKTI.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class EmbeddingModel(str, Enum):
    """Supported embedding models."""
    BGE_M3 = "BAAI/bge-m3"
    BGE_LARGE_EN_V1_5 = "BAAI/bge-large-en-v1.5"
    E5_LARGE_V2 = "intfloat/e5-large-v2"
    E5_MULTILINGUAL = "intfloat/multilingual-e5-large"
    NV_EMBED_V2 = "nvidia/nv-embed-v2"
    COHERE_EMBED_V3 = "cohere/embed-english-v3.0"
    OPENAI_TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    OPENAI_TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    JINA_EMBEDDINGS_V2 = "jinaai/jina-embeddings-v2-base-en"


class EmbeddingRequest(BaseModel):
    """Request for embedding generation."""
    texts: List[str]
    model: Optional[EmbeddingModel] = None
    input_type: str = Field(default="search_document", pattern="^(search_query|search_document|classification|clustering)$")
    truncate: bool = True
    batch_size: int = Field(default=32, ge=1, le=256)
    normalize: bool = True


class EmbeddingResult(BaseModel):
    """Result from embedding generation."""
    embeddings: List[List[float]]
    model: EmbeddingModel
    dimensions: int
    token_count: int
    processing_time_ms: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IEmbedder(ABC):
    """Interface for embedding components."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Embedder name."""
        pass
    
    @property
    @abstractmethod
    def default_model(self) -> EmbeddingModel:
        """Default embedding model."""
        pass
    
    @property
    @abstractmethod
    def supported_models(self) -> List[EmbeddingModel]:
        """List of supported models."""
        pass
    
    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Embedding dimensions for the default model."""
        pass
    
    @abstractmethod
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Generate embeddings for texts."""
        pass
    
    @abstractmethod
    async def embed_single(self, text: str, input_type: str = "search_document") -> List[float]:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding optimized for query."""
        pass
    
    @abstractmethod
    async def embed_document(self, text: str) -> List[float]:
        """Generate embedding optimized for document."""
        pass


class EmbeddingMetrics(BaseModel):
    """Metrics for embedding evaluation."""
    throughput_texts_per_sec: float = 0.0
    latency_ms_per_text: float = 0.0
    memory_usage_mb: float = 0.0
    model_load_time_ms: float = 0.0


class IMultilingualEmbedder(ABC):
    """Interface for multilingual embedding components."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Embedder name."""
        pass
    
    @property
    @abstractmethod
    def supported_languages(self) -> List[str]:
        """Supported languages."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension."""
        pass
    
    @abstractmethod
    async def embed(self, texts: List[str], language: Optional[str] = None) -> List[List[float]]:
        """Generate embeddings for texts with language awareness."""
        pass
    
    @abstractmethod
    async def embed_query(self, query: str, language: Optional[str] = None) -> List[float]:
        """Generate embedding for a single query with language awareness."""
        pass