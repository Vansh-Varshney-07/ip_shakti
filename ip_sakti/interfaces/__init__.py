"""
Interfaces package - Abstract base classes for all IP-SAKTI components.
Following the modular architecture pattern from NVIDIA/Anthropic.
"""
from ip_sakti.interfaces.base import (
    IComponent,
    IConfigurable,
    IHealthCheckable,
)
from ip_sakti.interfaces.retriever import (
    IRetriever,
    IHybridRetriever,
    RetrievalRequest,
    RetrievalResult,
    RetrievalStrategy,
    QueryIntent,
    QueryComplexity,
)
from ip_sakti.interfaces.embedder import (
    IEmbedder,
    IMultilingualEmbedder,
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingModel,
)
from ip_sakti.interfaces.reranker import (
    IReranker,
    RerankRequest,
    RerankResult,
    RerankStrategy,
    RerankerModel,
)
from ip_sakti.interfaces.generator import (
    IGenerator,
    ICitationGenerator,
    GenerationRequest,
    GenerationResult,
    GenerationStrategy,
)
from ip_sakti.interfaces.query_processor import (
    IQueryProcessor,
    IQueryAnalyzer,
    IQueryRewriter,
    IQueryRouter,
    QueryAnalysis,
    QueryRewrite,
    QueryRoute,
    QueryIntent,
    QueryComplexity,
)
from ip_sakti.interfaces.ingestion import (
    IDocumentLoader,
    IDocumentValidator,
    IChunker,
    IngestionRequest,
    IngestionResult,
    ChunkingStrategy,
    LoaderType,
    Document,
    Chunk,
    DocumentMetadata,
    DocumentType,
    ProcessingStatus,
)
from ip_sakti.interfaces.ingestion_pipeline import (
    IIngestionPipeline,
)
from ip_sakti.interfaces.rag_pipeline import (
    IRAGPipeline,
    RAGResponse,
)
from ip_sakti.interfaces.vector_store import (
    IVectorStore,
    VectorStoreConfig,
    IndexRequest,
    SearchRequest,
    SearchResult,
    VectorStoreBackend,
    DistanceMetric,
    VectorStoreHealth,
)
from ip_sakti.interfaces.authority import (
    IAuthoritySystem,
    AuthorityScore,
    AuthorityTier,
    AuthoritySourceType,
    AuthorityContext,
)
from ip_sakti.interfaces.kg import (
    IKnowledgeGraph,
    GraphNode,
    GraphEdge,
    GraphQuery,
    GraphResult,
    GraphNodeType,
    GraphEdgeType,
)
from ip_sakti.interfaces.orchestration import (
    IPipelineOrchestrator,
    PipelineRequest,
    PipelineResult,
    PipelineStage,
    PipelineMode,
)
from ip_sakti.interfaces.agent import (
    IAgent,
    IPlannerAgent,
    IRetrieverAgent,
    ICriticAgent,
    IRefinerAgent,
)
from ip_sakti.interfaces.multimodal import (
    IMultimodalExtractor,
)
from ip_sakti.interfaces.evaluation import (
    IEvaluator,
)
from ip_sakti.interfaces.cache import (
    ICache,
    ICircuitBreaker,
    CacheStats,
)
from ip_sakti.interfaces.metrics import (
    IMetricsCollector,
)
from ip_sakti.interfaces.config import (
    IConfig,
)
from ip_sakti.interfaces.multilingual import (
    ITranslator,
    QueryLanguage,
)

__all__ = [
    # Base
    "IComponent",
    "IConfigurable",
    "IHealthCheckable",
    # Config
    "IConfig",
    # Retriever
    "IRetriever",
    "IHybridRetriever",
    "RetrievalRequest",
    "RetrievalResult",
    "RetrievalStrategy",
    "QueryIntent",
    "QueryComplexity",
    # Embedder
    "IEmbedder",
    "IMultilingualEmbedder",
    "EmbeddingRequest",
    "EmbeddingResult",
    "EmbeddingModel",
    # Reranker
    "IReranker",
    "RerankRequest",
    "RerankResult",
    "RerankStrategy",
    "RerankerModel",
    # Generator
    "IGenerator",
    "ICitationGenerator",
    "GenerationRequest",
    "GenerationResult",
    "GenerationStrategy",
    # Query Processor
    "IQueryProcessor",
    "IQueryAnalyzer",
    "IQueryRewriter",
    "IQueryRouter",
    "QueryAnalysis",
    "QueryRewrite",
    "QueryRoute",
    # Ingestion
    "IDocumentLoader",
    "IDocumentValidator",
    "IChunker",
    "IngestionRequest",
    "IngestionResult",
    "ChunkingStrategy",
    "LoaderType",
    "Document",
    "Chunk",
    "DocumentMetadata",
    "DocumentType",
    "ProcessingStatus",
    # Ingestion Pipeline
    "IIngestionPipeline",
    # RAG Pipeline
    "IRAGPipeline",
    "RAGResponse",
    # Vector Store
    "IVectorStore",
    "VectorStoreConfig",
    "IndexRequest",
    "SearchRequest",
    "SearchResult",
    "VectorStoreBackend",
    "DistanceMetric",
    "VectorStoreHealth",
    # Authority
    "IAuthoritySystem",
    "AuthorityScore",
    "AuthorityTier",
    "AuthoritySourceType",
    "AuthorityContext",
    # Knowledge Graph
    "IKnowledgeGraph",
    "GraphNode",
    "GraphEdge",
    "GraphQuery",
    "GraphResult",
    "GraphNodeType",
    "GraphEdgeType",
    # Orchestration
    "IPipelineOrchestrator",
    "PipelineRequest",
    "PipelineResult",
    "PipelineStage",
    "PipelineMode",
    # Agent
    "IAgent",
    "IPlannerAgent",
    "IRetrieverAgent",
    "ICriticAgent",
    "IRefinerAgent",
    # Multimodal
    "IMultimodalExtractor",
    # Evaluation
    "IEvaluator",
    # Cache
    "ICache",
    "ICircuitBreaker",
    "CacheStats",
    # Metrics
    "IMetricsCollector",
    # Multilingual
    "ITranslator",
    "QueryLanguage",
]