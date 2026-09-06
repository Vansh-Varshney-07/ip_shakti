"""
Core interfaces for IP Sakti RAG system.
Defines contracts for all major components enabling dependency injection and testing.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncIterator, Union
from dataclasses import dataclass
from enum import Enum


class RetrievalStrategy(Enum):
    """Supported retrieval strategies."""
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    GRAPH = "graph"
    KG = "kg"


@dataclass
class RetrievalResult:
    """Standardized retrieval result."""
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    source_authority: Optional[float] = None
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID


@dataclass
class QueryAnalysis:
    """Analyzed query with intent and decomposition."""
    original_query: str
    intent: str
    sub_queries: List[str]
    rewritten_queries: List[str]
    complexity: float
    language: str = "en"
    requires_decomposition: bool = False


@dataclass
class RAGResponse:
    """Standardized RAG response with citations."""
    answer: str
    citations: List[Dict[str, Any]]
    confidence: float
    metadata: Dict[str, Any]
    retrieval_results: List[RetrievalResult]


# =============================================================================
# Configuration Interface
# =============================================================================

class IConfig(ABC):
    """Configuration interface for settings management."""
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        pass
    
    @abstractmethod
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        pass
    
    @abstractmethod
    def reload(self) -> None:
        """Reload configuration from source."""
        pass


# =============================================================================
# Document Loading Interfaces
# =============================================================================

class IDocumentLoader(ABC):
    """Interface for document loading."""
    
    @abstractmethod
    async def load(self, source: Union[str, bytes]) -> List[Dict[str, Any]]:
        """Load document from source."""
        pass
    
    @abstractmethod
    def supported_formats(self) -> List[str]:
        """Return list of supported file formats."""
        pass


class IDocumentValidator(ABC):
    """Interface for document validation."""
    
    @abstractmethod
    async def validate(self, document: Dict[str, Any]) -> bool:
        """Validate document meets requirements."""
        pass
    
    @abstractmethod
    def get_validation_errors(self) -> List[str]:
        """Get validation errors from last validation."""
        pass


# =============================================================================
# Chunking Interfaces
# =============================================================================

class IChunker(ABC):
    """Interface for document chunking."""
    
    @abstractmethod
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk text into segments with metadata."""
        pass
    
    @abstractmethod
    def get_chunk_size(self) -> int:
        """Get configured chunk size."""
        pass
    
    @abstractmethod
    def get_chunk_overlap(self) -> int:
        """Get configured chunk overlap."""
        pass


# =============================================================================
# Embedding Interfaces
# =============================================================================

class IEmbedder(ABC):
    """Interface for text embedding."""
    
    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        pass
    
    @abstractmethod
    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for single query."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get model name."""
        pass


# =============================================================================
# Vector Store Interfaces
# =============================================================================

class IVectorStore(ABC):
    """Interface for vector storage and retrieval."""
    
    @abstractmethod
    async def add(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        """Add chunks with embeddings."""
        pass
    
    @abstractmethod
    async def search(
        self, 
        query_embedding: List[float], 
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    async def delete(self, chunk_ids: List[str]) -> None:
        """Delete chunks by IDs."""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all vectors."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        pass


# =============================================================================
# Retrieval Interfaces
# =============================================================================

class IRetriever(ABC):
    """Interface for retrieval strategies."""
    
    @abstractmethod
    async def retrieve(
        self, 
        query: str, 
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Retrieve relevant documents."""
        pass
    
    @abstractmethod
    def get_strategy(self) -> RetrievalStrategy:
        """Get retrieval strategy."""
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
    ) -> List[RetrievalResult]:
        """Retrieve using multiple strategies with optional weights."""
        pass
    
    @abstractmethod
    def add_retriever(self, retriever: IRetriever) -> None:
        """Add a retriever to the hybrid."""
        pass


# =============================================================================
# Reranking Interfaces
# =============================================================================

class IReranker(ABC):
    """Interface for result reranking."""
    
    @abstractmethod
    async def rerank(
        self, 
        query: str, 
        results: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """Rerank retrieval results."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get reranker model name."""
        pass


# =============================================================================
# Query Processing Interfaces
# =============================================================================

class IQueryAnalyzer(ABC):
    """Interface for query analysis."""
    
    @abstractmethod
    async def analyze(self, query: str) -> QueryAnalysis:
        """Analyze query for intent, complexity, etc."""
        pass


class IQueryRewriter(ABC):
    """Interface for query rewriting (HyDE, etc.)."""
    
    @abstractmethod
    async def rewrite(self, query: str, analysis: QueryAnalysis) -> List[str]:
        """Rewrite query for better retrieval."""
        pass


class IQueryRouter(ABC):
    """Interface for query routing."""
    
    @abstractmethod
    async def route(self, analysis: QueryAnalysis) -> List[RetrievalStrategy]:
        """Determine retrieval strategies for query."""
        pass


# =============================================================================
# Generation Interfaces
# =============================================================================

class IGenerator(ABC):
    """Interface for answer generation."""
    
    @abstractmethod
    async def generate(
        self,
        query: str,
        context: List[RetrievalResult],
        analysis: QueryAnalysis
    ) -> RAGResponse:
        """Generate answer with citations."""
        pass
    
    @abstractmethod
    async def stream_generate(
        self,
        query: str,
        context: List[RetrievalResult],
        analysis: QueryAnalysis
    ) -> AsyncIterator[str]:
        """Stream generated answer."""
        pass


class ICitationGenerator(IGenerator):
    """Interface for citation-constrained generation."""
    
    @abstractmethod
    async def generate_with_citations(
        self,
        query: str,
        context: List[RetrievalResult],
        analysis: QueryAnalysis,
        citation_style: str = "inline"
    ) -> RAGResponse:
        """Generate answer with specific citation style."""
        pass


# =============================================================================
# Agentic Interfaces
# =============================================================================

class IAgent(ABC):
    """Base agent interface."""
    
    @abstractmethod
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent step."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get agent name."""
        pass


class IPlannerAgent(IAgent):
    """Interface for planning agent."""
    
    @abstractmethod
    async def plan(self, query: str, analysis: QueryAnalysis) -> List[Dict[str, Any]]:
        """Create execution plan."""
        pass


class IRetrieverAgent(IAgent):
    """Interface for retrieval agent."""
    
    @abstractmethod
    async def retrieve(
        self, 
        plan: List[Dict[str, Any]], 
        query: str
    ) -> List[RetrievalResult]:
        """Execute retrieval per plan."""
        pass


class ICriticAgent(IAgent):
    """Interface for critique agent."""
    
    @abstractmethod
    async def critique(
        self, 
        query: str, 
        context: List[RetrievalResult],
        draft_answer: str
    ) -> Dict[str, Any]:
        """Critique draft answer."""
        pass


class IRefinerAgent(IAgent):
    """Interface for refinement agent."""
    
    @abstractmethod
    async def refine(
        self,
        query: str,
        context: List[RetrievalResult],
        draft_answer: str,
        critique: Dict[str, Any]
    ) -> str:
        """Refine answer based on critique."""
        pass


# =============================================================================
# Knowledge Graph Interfaces
# =============================================================================

class IKnowledgeGraph(ABC):
    """Interface for knowledge graph operations."""
    
    @abstractmethod
    async def add_entity(self, entity: Dict[str, Any]) -> str:
        """Add entity to graph."""
        pass
    
    @abstractmethod
    async def add_relation(
        self, 
        source: str, 
        target: str, 
        relation: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add relation between entities."""
        pass
    
    @abstractmethod
    async def query(
        self, 
        cypher: str, 
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute Cypher query."""
        pass
    
    @abstractmethod
    async def get_neighbors(
        self, 
        entity_id: str, 
        relation_types: Optional[List[str]] = None,
        max_depth: int = 2
    ) -> List[Dict[str, Any]]:
        """Get neighboring entities."""
        pass
    
    @abstractmethod
    async def get_authority_path(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get authority propagation path."""
        pass


# =============================================================================
# Evaluation Interfaces
# =============================================================================

class IEvaluator(ABC):
    """Interface for RAG evaluation."""
    
    @abstractmethod
    async def evaluate(
        self,
        questions: List[str],
        ground_truths: List[str],
        rag_pipeline: Any
    ) -> Dict[str, float]:
        """Evaluate RAG pipeline."""
        pass
    
    @abstractmethod
    def get_metrics(self) -> List[str]:
        """Get supported metrics."""
        pass


# =============================================================================
# Multimodal Interfaces
# =============================================================================

class IMultimodalExtractor(ABC):
    """Interface for multimodal content extraction."""
    
    @abstractmethod
    async def extract(self, source: Union[str, bytes]) -> Dict[str, List[Any]]:
        """Extract text, tables, images, charts from document."""
        pass
    
    @abstractmethod
    def supported_types(self) -> List[str]:
        """Get supported content types."""
        pass


# =============================================================================
# Pipeline Interfaces
# =============================================================================

class IIngestionPipeline(ABC):
    """Interface for ingestion pipeline."""
    
    @abstractmethod
    async def ingest(
        self,
        sources: List[Union[str, bytes]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run ingestion pipeline."""
        pass


class IRAGPipeline(ABC):
    """Interface for RAG pipeline."""
    
    @abstractmethod
    async def query(self, query: str, **kwargs) -> RAGResponse:
        """Execute RAG query."""
        pass
    
    @abstractmethod
    async def stream_query(self, query: str, **kwargs) -> AsyncIterator[str]:
        """Stream RAG query response."""
        pass


# =============================================================================
# Authority System Interfaces
# =============================================================================

class IAuthoritySystem(ABC):
    """Interface for source authority system."""
    
    @abstractmethod
    def classify_source(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Classify source authority tier."""
        pass
    
    @abstractmethod
    def propagate_authority(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Propagate authority through retrieval results."""
        pass
    
    @abstractmethod
    def resolve_conflicts(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Resolve conflicting information."""
        pass


# =============================================================================
# Multilingual Interfaces
# =============================================================================

class ITranslator(ABC):
    """Interface for translation."""
    
    @abstractmethod
    async def translate(self, text: str, target_lang: str, source_lang: Optional[str] = None) -> str:
        """Translate text."""
        pass
    
    @abstractmethod
    def supported_languages(self) -> List[str]:
        """Get supported languages."""
        pass


class IMultilingualEmbedder(IEmbedder):
    """Interface for multilingual embedding."""
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """Get supported languages."""
        pass
    
    @abstractmethod
    async def embed_multilingual(
        self, 
        texts: List[str], 
        languages: List[str]
    ) -> List[List[float]]:
        """Embed texts with language awareness."""
        pass


# =============================================================================
# Production Interfaces
# =============================================================================

class ICache(ABC):
    """Interface for caching."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached value."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete cached value."""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear cache."""
        pass


class ICircuitBreaker(ABC):
    """Interface for circuit breaker pattern."""
    
    @abstractmethod
    async def call(self, func, *args, **kwargs) -> Any:
        """Execute function with circuit breaker."""
        pass
    
    @abstractmethod
    def get_state(self) -> str:
        """Get circuit breaker state."""
        pass


class IMetricsCollector(ABC):
    """Interface for metrics collection."""
    
    @abstractmethod
    def increment(self, metric: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment counter metric."""
        pass
    
    @abstractmethod
    def histogram(self, metric: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record histogram metric."""
        pass
    
    @abstractmethod
    def gauge(self, metric: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record gauge metric."""
        pass