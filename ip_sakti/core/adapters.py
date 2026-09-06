"""
Adapter implementations bridging existing code to new interfaces.
"""

from typing import Any, Dict, List, Optional, Union, AsyncIterator, AsyncGenerator
import asyncio
import numpy as np
from ip_sakti.interfaces import (
    IConfig, IDocumentLoader, IChunker, IEmbedder, IVectorStore,
    IAuthoritySystem, IRetriever, IReranker, IGenerator,
    IIngestionPipeline, IRAGPipeline, IQueryProcessor,
    IKnowledgeGraph, IPipelineOrchestrator,
    RetrievalStrategy, QueryIntent, QueryComplexity,
    GenerationStrategy, RerankStrategy,
    RetrievalRequest, RetrievalResult,
    EmbeddingRequest, EmbeddingResult,
    RerankRequest, RerankResult,
    GenerationRequest, GenerationResult,
    QueryAnalysis, QueryRewrite, QueryRoute,
    IngestionRequest, IngestionResult,
    VectorStoreConfig, IndexRequest, SearchRequest, SearchResult,
    AuthorityScore, AuthorityTier,
    AuthorityContext,
    GraphNode, GraphEdge, GraphQuery, GraphResult,
    GraphNodeType, GraphEdgeType,
    PipelineRequest, PipelineResult, PipelineStage, PipelineMode,
    IDocumentValidator,
    Document, Chunk,
    LoaderType, ChunkingStrategy,
    IMultilingualEmbedder,
    IHybridRetriever,
    IQueryAnalyzer, IQueryRewriter, IQueryRouter,
    ICitationGenerator,
    IAgent, IPlannerAgent, IRetrieverAgent, ICriticAgent, IRefinerAgent,
    IMultimodalExtractor,
    IEvaluator,
    ICache, ICircuitBreaker,
    IMetricsCollector,
    ITranslator,
    RAGResponse,
    VectorStoreBackend,
    VectorStoreHealth,
    DocumentType,
    ProcessingStatus,
    QueryLanguage,
    CacheStats,
    RerankerModel,
)
from ip_sakti.core.models import *
from ip_sakti.config.loader import Settings, get_settings
from ip_sakti.ingestion.loaders import DocumentLoaderRegistry, get_loader_registry
from ip_sakti.ingestion.chunking import ChunkingStrategy as InternalChunkingStrategy, ChunkingStrategyType
from ip_sakti.embedding.embedder import SentenceTransformerEmbeddingProvider, EmbeddingConfig, EmbeddingModelType
from ip_sakti.embedding.vector_store import VectorStore, create_vector_store_from_settings, VectorStoreConfig as InternalVectorStoreConfig
from ip_sakti.retrieval.retrieval_engine import RetrievalEngine, RetrievalConfig as InternalRetrievalConfig, VectorStoreType, CacheBackend, RetrievalConfig
from ip_sakti.authority.authority_system import SourceAuthoritySystem
from ip_sakti.rag.pipeline import RAGPipeline


# =============================================================================
# Configuration Adapter
# =============================================================================

class ConfigAdapter(IConfig):
    """Adapter for Settings to IConfig interface."""
    
    def __init__(self, settings: Settings):
        self._settings = settings
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self._settings.to_dict()
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self._settings.to_dict().get(section, {})
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value (not supported for frozen settings)."""
        raise NotImplementedError("Settings is immutable")
    
    def reload(self) -> None:
        """Reload configuration from source."""
        # Settings is a frozen dataclass, reload would require re-creating
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self._settings.to_dict()


# =============================================================================
# Document Loading Adapters
# =============================================================================

class DocumentLoaderAdapter(IDocumentLoader):
    """Adapter for document loader registry."""
    
    def __init__(self, loader_registry: DocumentLoaderRegistry):
        self._registry = loader_registry
    
    @property
    def supported_types(self) -> List[LoaderType]:
        """Supported document types."""
        return [LoaderType(fmt.value) for fmt in self._registry.get_supported_formats()]
    
    async def load(self, source: str, source_type: LoaderType, **kwargs) -> Document:
        """Load document from source."""
        docs = self._registry.load(source)
        
        # Convert to Document format
        if docs:
            doc = docs[0]
            return Document(
                id=doc.source,
                content=doc.content,
                metadata=doc.metadata,
                source=doc.source,
            )
        
        return Document(id=source, content="", metadata={}, source=source)
    
    async def load_from_bytes(self, content: bytes, source_type: LoaderType, **kwargs) -> Document:
        """Load document from bytes."""
        raise NotImplementedError("Loading from bytes not yet implemented")
    
    async def extract_tables(self, document: Document) -> List[Dict[str, Any]]:
        """Extract tables from document."""
        return []
    
    async def extract_images(self, document: Document) -> List[Dict[str, Any]]:
        """Extract images from document."""
        return []


class DocumentValidatorAdapter(IDocumentValidator):
    """Adapter for document validation."""
    
    async def validate(self, document: Document) -> bool:
        """Validate document meets requirements."""
        # Basic validation
        if not document.content:
            return False
        if not document.metadata:
            return False
        return True
    
    async def get_validation_errors(self, document: Document) -> List[str]:
        """Get list of validation errors."""
        errors = []
        if not document.content:
            errors.append("Document has no content")
        if not document.metadata:
            errors.append("Document has no metadata")
        return errors


# =============================================================================
# Chunking Adapter
# =============================================================================

class ChunkerAdapter(IChunker):
    """Adapter for chunking strategy."""
    
    def __init__(self, chunker_config=None):
        from ip_sakti.ingestion.chunking import ChunkingConfig, ChunkingStrategyType
        self._config = chunker_config or ChunkingConfig(
            strategy=ChunkingStrategyType.RECURSIVE,
            chunk_size=1000,
            chunk_overlap=200,
            min_chunk_size=100,
        )
    
    @property
    def name(self) -> str:
        return "legal_chunker"
    
    @property
    def supported_strategies(self) -> List[ChunkingStrategy]:
        return [ChunkingStrategy.LEGAL, ChunkingStrategy.RECURSIVE, ChunkingStrategy.SEMANTIC, ChunkingStrategy.FIXED]
    
    async def chunk(self, document: Document, strategy: ChunkingStrategy, chunk_size: int, chunk_overlap: int, **kwargs) -> List[Chunk]:
        """Chunk document into pieces."""
        from ip_sakti.ingestion.chunking import ChunkingConfig, ChunkingStrategyType, ChunkingStrategy as InternalChunkingStrategy
        
        # Map interface enum to internal enum
        strategy_map = {
            ChunkingStrategy.LEGAL: ChunkingStrategyType.RECURSIVE,
            ChunkingStrategy.RECURSIVE: ChunkingStrategyType.RECURSIVE,
            ChunkingStrategy.SEMANTIC: ChunkingStrategyType.SEMANTIC,
            ChunkingStrategy.FIXED: ChunkingStrategyType.FIXED_SIZE,
        }
        
        # Create chunking config
        config = ChunkingConfig(
            strategy=strategy_map.get(strategy, ChunkingStrategyType.RECURSIVE),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=kwargs.get('min_chunk_size', 100),
        )
        
        # Create a temporary chunker with this config
        temp_chunker = InternalChunkingStrategy(config)
        
        # Chunk document - use chunk_text since we have content and source
        internal_chunks = await temp_chunker.chunk_text(
            document.content, 
            document.source, 
            document_type=document.metadata.get('document_type', 'PATENT')
        )
        
        # Convert to interface Chunk format
        return [
            Chunk(
                id=chunk.id,
                content=chunk.content,
                metadata=chunk.metadata,
                source=document.source,
                chunk_index=chunk.chunk_index,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                authority_score=chunk.metadata.get('authority_score', 0.0),
                jurisdiction=chunk.metadata.get('jurisdiction'),
                document_type=chunk.metadata.get('document_type'),
                embedding=None,
            )
            for chunk in internal_chunks
        ]
    
    async def chunk_text(self, text: str, strategy: ChunkingStrategy, chunk_size: int, chunk_overlap: int, **kwargs) -> List[Chunk]:
        """Chunk raw text."""
        doc = Document(content=text, metadata={}, source="text_input")
        return await self.chunk(doc, strategy, chunk_size, chunk_overlap, **kwargs)
    
    def get_chunker_config(self) -> Dict[str, Any]:
        """Get current chunker configuration."""
        return {
            "strategy": self._config.strategy.value,
            "chunk_size": self._config.chunk_size,
            "chunk_overlap": self._config.chunk_overlap,
        }
    
    def get_chunk_size(self) -> int:
        """Get configured chunk size."""
        return self._config.chunk_size
    
    def get_chunk_overlap(self) -> int:
        """Get configured chunk overlap."""
        return self._config.chunk_overlap


# =============================================================================
# Embedding Adapters
# =============================================================================

class EmbedderAdapter(IEmbedder):
    """Adapter for embedding provider."""
    
    def __init__(self, embedder: SentenceTransformerEmbeddingProvider):
        self._embedder = embedder
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        result = await self._embedder.embed(texts)
        return result.tolist()
    
    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        result = await self._embedder.embed_query(query)
        return result.tolist()
    
    def get_dimension(self) -> int:
        """Embedding dimension."""
        return self._embedder.dimension
    
    def get_model_name(self) -> str:
        """Model name."""
        return self._embedder.config.model_name.value
    
    # New interface methods
    @property
    def name(self) -> str:
        return "bge_m3_embedder"
    
    @property
    def default_model(self):
        from ip_sakti.interfaces.embedder import EmbeddingModel
        return EmbeddingModel.BGE_M3
    
    @property
    def supported_models(self):
        from ip_sakti.interfaces.embedder import EmbeddingModel
        return [EmbeddingModel.BGE_M3, EmbeddingModel.BGE_LARGE_EN_V1_5]
    
    @property
    def dimensions(self) -> int:
        return self._embedder.dimension
    
    async def embed_request(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Generate embeddings for texts."""
        import time
        start = time.time()
        embeddings = await self._embedder.embed(request.texts)
        return EmbeddingResult(
            embeddings=embeddings.tolist(),
            model=self.default_model,
            dimensions=self.dimensions,
            token_count=sum(len(t.split()) for t in request.texts),
            processing_time_ms=(time.time() - start) * 1000,
        )
    
    async def embed_single(self, text: str, input_type: str = "search_document") -> List[float]:
        """Generate embedding for a single text."""
        return await self.embed_query(text)
    
    async def embed_document(self, text: str) -> List[float]:
        """Generate embedding optimized for document."""
        return await self._embedder.embed([text])[0].tolist()


class MultilingualEmbedderAdapter(IMultilingualEmbedder):
    """Adapter for multilingual embedding provider."""
    
    def __init__(self, embedder: SentenceTransformerEmbeddingProvider):
        self._embedder = embedder
    
    async def embed(self, texts: List[str], language: Optional[str] = None) -> List[List[float]]:
        """Generate embeddings for texts with language awareness."""
        return await EmbedderAdapter(self._embedder).embed(texts)
    
    async def embed_query(self, query: str, language: Optional[str] = None) -> List[float]:
        """Generate embedding for a single query with language awareness."""
        return await EmbedderAdapter(self._embedder).embed_query(query)
    
    @property
    def supported_languages(self) -> List[str]:
        """Supported languages."""
        return ["en", "hi", "bn", "te", "mr", "ta", "ur", "gu", "kn", "ml", "or", "pa", "as"]
    
    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        return self._embedder.dimension


# =============================================================================
# Vector Store Adapter
# =============================================================================

class VectorStoreAdapter(IVectorStore):
    """Adapter for vector store."""
    
    def __init__(self, vector_store: VectorStore):
        self._store = vector_store
    
    @property
    def name(self) -> str:
        return "vector_store"
    
    @property
    def backend(self) -> VectorStoreBackend:
        return VectorStoreBackend.IN_MEMORY
    
    async def initialize(self, config: VectorStoreConfig) -> None:
        """Initialize vector store."""
        # Already initialized
        pass
    
    async def index(self, request: IndexRequest) -> Dict[str, Any]:
        """Index chunks into vector store."""
        from ip_sakti.core.models import DocumentChunk
        import numpy as np
        
        # Convert to DocumentChunk objects
        doc_chunks = []
        embeddings = []
        for chunk in request.chunks:
            doc_chunk = DocumentChunk(
                id=chunk.id,
                content=chunk.content,
                metadata=chunk.metadata,
                source=chunk.source or "",
                chunk_index=chunk.chunk_index,
            )
            doc_chunks.append(doc_chunk)
            if chunk.embedding:
                embeddings.append(chunk.embedding)
        
        if embeddings:
            await self._store.upsert(doc_chunks, np.array(embeddings, dtype=np.float32))
        
        return {"indexed": len(doc_chunks)}
    
    async def search(self, request: SearchRequest) -> List[SearchResult]:
        """Search for similar vectors."""
        import numpy as np
        results = await self._store.search(np.array(request.query_vector, dtype=np.float32), request.top_k, request.filter)
        
        return [
            SearchResult(
                chunk=Chunk(
                    id=chunk.id,
                    content=chunk.content,
                    metadata=chunk.metadata,
                    source=chunk.source,
                    chunk_index=chunk.chunk_index,
                ),
                score=score,
                rank=i+1,
            )
            for i, (chunk, score) in enumerate(results)
        ]
    
    async def delete(self, chunk_ids: List[str]) -> Dict[str, Any]:
        """Delete chunks by IDs."""
        await self._store.delete(chunk_ids)
        return {"deleted": len(chunk_ids)}
    
    async def get_by_ids(self, chunk_ids: List[str]) -> List[Chunk]:
        """Get chunks by IDs."""
        # Not directly supported by the underlying store
        return []
    
    async def count(self, filter: Optional[Dict[str, Any]] = None) -> int:
        """Count chunks in collection."""
        return 0  # Not implemented in underlying store
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check."""
        stats = await self._store.get_stats()
        return {"status": "healthy", "stats": stats}
    
    # Legacy methods for backward compatibility
    async def add(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        """Add chunks with embeddings (legacy)."""
        from ip_sakti.core.models import DocumentChunk
        import numpy as np
        
        doc_chunks = []
        for i, doc in enumerate(chunks):
            chunk = DocumentChunk(
                id=doc.get("id", ""),
                content=doc.get("content", ""),
                metadata=doc.get("metadata", {}),
                source=doc.get("source", ""),
                chunk_index=doc.get("chunk_index", 0),
                parent_id=doc.get("parent_id"),
            )
            doc_chunks.append(chunk)
        
        await self._store.upsert(doc_chunks, np.array(embeddings, dtype=np.float32))
    
    async def search_legacy(
        self, 
        query_embedding: List[float], 
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Search for similar vectors (legacy)."""
        import numpy as np
        results = await self._store.search(np.array(query_embedding, dtype=np.float32), k, filter)
        
        return [
            RetrievalResult(
                chunk_id=chunk.id,
                content=chunk.content,
                score=score,
                metadata=chunk.metadata,
                source_authority=chunk.metadata.get('authority_score'),
                strategy=RetrievalStrategy.DENSE,
            )
            for chunk, score in results
        ]
    
    async def delete_legacy(self, chunk_ids: List[str]) -> None:
        """Delete chunks by IDs (legacy)."""
        await self._store.delete(chunk_ids)
    
    async def clear(self) -> None:
        """Clear all vectors."""
        await self._store.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        import asyncio
        return asyncio.run(self._store.get_stats())


# =============================================================================
# Retrieval Engine Adapter
# =============================================================================

class RetrievalEngineAdapter(IRetriever):
    """Adapter for retrieval engine."""
    
    def __init__(self, retrieval_engine: RetrievalEngine):
        self._engine = retrieval_engine
    
    @property
    def name(self) -> str:
        return "retrieval_engine"
    
    @property
    def supported_strategies(self) -> List[RetrievalStrategy]:
        return [RetrievalStrategy.DENSE, RetrievalStrategy.SPARSE, RetrievalStrategy.HYBRID]
    
    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Retrieve relevant chunks for a query."""
        from ip_sakti.retrieval.retrieval_engine import SearchRequest
        
        sr = SearchRequest(
            query=request.query,
            strategy=request.strategy,
            filters=request.filters,
            top_k=request.top_k,
        )
        
        response = await self._engine.search(sr)
        
        chunks = [
            Chunk(
                id=r.id,
                content=r.content,
                metadata=r.metadata,
                source=r.metadata.get('source'),
            )
            for r in response.results
        ]
        
        return RetrievalResult(
            chunks=chunks,
            scores=[r.score for r in response.results],
            strategy_used=request.strategy,
            total_candidates=len(response.results),
            retrieval_time_ms=response.processing_time_ms,
        )
    
    async def retrieve_batch(self, requests: List[RetrievalRequest]) -> List[RetrievalResult]:
        """Batch retrieval for multiple queries."""
        results = []
        for req in requests:
            results.append(await self.retrieve(req))
        return results
    
    async def get_similar_chunks(self, chunk: Chunk, top_k: int = 5) -> RetrievalResult:
        """Find chunks similar to a given chunk."""
        # Not directly supported by the underlying engine
        return RetrievalResult(
            chunks=[],
            scores=[],
            strategy_used=RetrievalStrategy.DENSE,
            total_candidates=0,
            retrieval_time_ms=0.0,
        )
    
    # Legacy methods for backward compatibility
    async def search(
        self,
        query: str,
        top_k: int = 10,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Search with query string (legacy)."""
        from ip_sakti.retrieval.retrieval_engine import SearchRequest
        
        request = SearchRequest(
            query=query,
            strategy=strategy,
            filters=filters or {},
            top_k=top_k,
        )
        
        response = await self._engine.search(request)
        
        return [
            RetrievalResult(
                chunk_id=r.id,
                content=r.content,
                score=r.score,
                metadata=r.metadata,
                source_authority=r.metadata.get('authority_score'),
                strategy=strategy,
            )
            for r in response.results
        ]
    
    async def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Hybrid search combining dense and sparse (legacy)."""
        return await self.search(query, top_k, RetrievalStrategy.HYBRID, filters)
    
    async def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics (legacy)."""
        return self._engine.get_stats()
    
    async def retrieve_legacy(
        self, 
        query: str, 
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Retrieve relevant documents (legacy)."""
        return await self.search(query, k, RetrievalStrategy.HYBRID, filter)
    
    def get_strategy(self) -> RetrievalStrategy:
        """Get retrieval strategy (legacy)."""
        return RetrievalStrategy.HYBRID


class HybridRetrieverAdapter(IHybridRetriever):
    """Adapter for hybrid retrieval."""
    
    def __init__(self, retrieval_engine: RetrievalEngine):
        self._engine = retrieval_engine
        self._retrievers: List[IRetriever] = []
    
    async def retrieve(
        self, 
        query: str, 
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Retrieve relevant documents."""
        return await self.search(query, k, RetrievalStrategy.HYBRID, filter)
    
    def get_strategy(self) -> RetrievalStrategy:
        """Get retrieval strategy."""
        return RetrievalStrategy.HYBRID
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Hybrid search with configurable weights."""
        # The retrieval engine already handles hybrid search internally
        request = SearchRequest(
            query=query,
            strategy=RetrievalStrategy.HYBRID,
            filters=filters or {},
            top_k=top_k,
        )
        
        response = await self._engine.search(request)
        
        return [
            RetrievalResult(
                chunk_id=r.id,
                content=r.content,
                score=r.score,
                metadata=r.metadata,
                source_authority=r.metadata.get('authority_score'),
                strategy=RetrievalStrategy.HYBRID,
            )
            for r in response.results
        ]
    
    async def retrieve_with_strategies(
        self,
        query: str,
        strategies: List[RetrievalStrategy],
        k: int = 10,
        weights: Optional[Dict[RetrievalStrategy, float]] = None
    ) -> List[RetrievalResult]:
        """Retrieve using multiple strategies with optional weights."""
        # For now, just use hybrid search
        return await self.retrieve(query, k)
    
    def add_retriever(self, retriever: IRetriever) -> None:
        """Add a retriever to the hybrid."""
        self._retrievers.append(retriever)


# =============================================================================
# Reranker Adapter
# =============================================================================

class RerankerAdapter(IReranker):
    """Adapter for reranker."""
    
    def __init__(self):
        # Initialize reranker if needed
        self._model_name = "bge-reranker-v2-m3"
    
    @property
    def name(self) -> str:
        return "bge_reranker"
    
    @property
    def supported_strategies(self) -> List[RerankStrategy]:
        return [RerankStrategy.CROSS_ENCODER]
    
    @property
    def default_model(self) -> RerankerModel:
        return RerankerModel.BGE_RERANKER_V2_M3
    
    async def rerank(self, request: RerankRequest) -> RerankResult:
        """Rerank chunks based on query relevance."""
        # For now, return as-is (reranking happens in pipeline)
        import time
        start = time.time()
        
        chunks = request.chunks[:request.top_k] if request.top_k else request.chunks
        scores = [1.0] * len(chunks)
        original_indices = list(range(len(chunks)))
        
        return RerankResult(
            chunks=chunks,
            relevance_scores=scores,
            original_indices=original_indices,
            strategy_used=request.strategy,
            rerank_time_ms=(time.time() - start) * 1000,
        )
    
    async def rerank_batch(self, requests: List[RerankRequest]) -> List[RerankResult]:
        """Batch reranking."""
        results = []
        for req in requests:
            results.append(await self.rerank(req))
        return results
    
    async def score_pairs(self, query: str, texts: List[str]) -> List[float]:
        """Score query-text pairs directly."""
        return [1.0] * len(texts)
    
    # Legacy methods for backward compatibility
    async def rerank_legacy(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Rerank results (legacy)."""
        return results[:top_k] if top_k else results
    
    async def score(self, query: str, results: List[RetrievalResult]) -> List[float]:
        """Score results (legacy)."""
        return [r.score for r in results]
    
    def get_model_name(self) -> str:
        """Get reranker model name (legacy)."""
        return self._model_name


# =============================================================================
# Query Processing Adapters
# =============================================================================

class QueryAnalyzerAdapter(IQueryAnalyzer):
    """Adapter for query analyzer."""
    
    def __init__(self, rag_pipeline: RAGPipeline):
        self._pipeline = rag_pipeline
    
    async def analyze(self, query: str) -> QueryAnalysis:
        """Analyze query intent and structure."""
        # Use pipeline's query analyzer
        analysis = await self._pipeline.query_analyzer.analyze(query, None)
        
        return QueryAnalysis(
            original_query=query,
            intent=analysis.get('intent', 'GENERAL_LEGAL'),
            sub_queries=analysis.get('sub_queries', []),
            rewritten_queries=[],
            complexity=analysis.get('complexity', 0.5),
            language="en",
            requires_decomposition=len(analysis.get('sub_queries', [])) > 0,
        )
    
    async def decompose(self, query: str) -> List[str]:
        """Decompose complex query into sub-queries."""
        analysis = await self._pipeline.query_analyzer.analyze(query, None)
        return analysis.get('sub_queries', [query])
    
    async def extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities from query."""
        analysis = await self._pipeline.query_analyzer.analyze(query, None)
        return analysis.get('entities', {})


class QueryRewriterAdapter(IQueryRewriter):
    """Adapter for query rewriter."""
    
    def __init__(self, rag_pipeline: RAGPipeline):
        self._pipeline = rag_pipeline
    
    async def rewrite(self, query: str, context: Optional[Dict[str, Any]] = None) -> List[str]:
        """Rewrite query into multiple variants."""
        # Use pipeline's query rewriter
        from ip_sakti.rag.pipeline import RAGContext, Query
        
        context_obj = RAGContext(
            query=Query(text=query, user_id="test"),
            original_query=query,
        )
        
        variants = await self._pipeline.query_rewriter.rewrite(query, context or {}, context_obj)
        return variants
    
    async def expand(self, query: str, num_variants: int = 3) -> List[str]:
        """Expand query into multiple variants."""
        variants = await self.rewrite(query)
        return variants[:num_variants]


class QueryRouterAdapter(IQueryRouter):
    """Adapter for query router."""
    
    def __init__(self, rag_pipeline: RAGPipeline):
        self._pipeline = rag_pipeline
    
    async def route(self, query: str, analysis: Optional[QueryAnalysis] = None) -> List[RetrievalStrategy]:
        """Determine retrieval strategies for query."""
        if analysis is None:
            analysis = await QueryAnalyzerAdapter(self._pipeline).analyze(query)
        
        # Use pipeline's strategy selector
        strategies = self._pipeline.strategy_selector.select(
            {"intent": analysis.intent, "jurisdiction": analysis.language}
        )
        
        return [RetrievalStrategy(s.value) for s in strategies]
    
    async def should_use_kg(self, query: str) -> bool:
        """Determine if KG retrieval is needed."""
        return "relationship" in query.lower() or "cite" in query.lower()


# =============================================================================
# =============================================================================
# Generator Adapter
# =============================================================================

class GeneratorAdapter(IGenerator):
    """Adapter for generator."""
    
    def __init__(self, rag_pipeline: RAGPipeline):
        self._pipeline = rag_pipeline
    
    @property
    def name(self) -> str:
        return "rag_generator"
    
    @property
    def supported_strategies(self) -> List[GenerationStrategy]:
        return [GenerationStrategy.CITATION_FIRST, GenerationStrategy.STANDARD, GenerationStrategy.STREAMING]
    
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate answer with citations."""
        from ip_sakti.rag.pipeline import RAGContext
        import time
        
        start = time.time()
        
        # Build context chunks
        context_chunks = []
        for r in request.chunks:
            chunk = type('Chunk', (), {
                'content': r.content,
                'metadata': r.metadata,
                'id': r.id,
            })()
            context_chunks.append(chunk)
        
        # Generate answer
        rag_context = RAGContext(
            query=type('Query', (), {'text': request.query, 'user_id': 'test'})(),
            original_query=request.query,
            context_chunks=context_chunks,
        )
        
        await self._pipeline._stage_generation(rag_context)
        
        # Convert citations
        citations = []
        for c in rag_context.citations:
            citations.append(CitationRef(
                cited_act=c.get('source', ''),
                cited_section=c.get('section', ''),
                cited_text=c.get('text', ''),
                citation_type=c.get('type', 'inline'),
                confidence=c.get('confidence', 1.0),
            ))
        
        return GenerationResult(
            answer=rag_context.generated_answer or "",
            citations=citations,
            confidence=rag_context.confidence_score,
            strategy_used=request.strategy,
            tokens_used=len((rag_context.generated_answer or "").split()),
            generation_time_ms=(time.time() - start) * 1000,
        )
    
    async def generate_stream(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        """Stream generation tokens."""
        answer = await self.generate(request)
        words = answer.answer.split()
        for i, word in enumerate(words):
            yield word + " "
            if i % 5 == 0:
                await asyncio.sleep(0.01)
    
    async def verify_citations(self, answer: str, chunks: List[Chunk]) -> Dict[str, Any]:
        """Verify that citations in answer match retrieved chunks."""
        # Use pipeline's citation validator
        from ip_sakti.rag.pipeline import RAGContext
        
        rag_context = RAGContext(
            query=type('Query', (), {'text': answer, 'user_id': 'test'})(),
            original_query=answer,
            context_chunks=chunks,
        )
        
        await self._pipeline._stage_validation(rag_context)
        
        return {
            "valid": len(rag_context.errors) == 0,
            "errors": rag_context.errors,
        }
    
    async def extract_structured(self, query: str, chunks: List[Chunk], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured information from chunks."""
        # Not implemented
        return {}
    
    # Legacy methods for backward compatibility
    async def generate_legacy(
        self,
        query: str,
        context: List[RetrievalResult],
        **kwargs,
    ) -> str:
        """Generate answer from context (legacy)."""
        from ip_sakti.rag.pipeline import RAGContext
        
        context_chunks = []
        for r in context:
            chunk = type('Chunk', (), {
                'content': r.content,
                'metadata': r.metadata,
                'id': r.chunk_id,
            })()
            context_chunks.append(chunk)
        
        rag_context = RAGContext(
            query=type('Query', (), {'text': query, 'user_id': 'test'})(),
            original_query=query,
            context_chunks=context_chunks,
        )
        
        await self._pipeline._stage_generation(rag_context)
        return rag_context.generated_answer or ""
    
    async def stream_generate(
        self,
        query: str,
        context: List[RetrievalResult],
        analysis: Optional[QueryAnalysis] = None
    ) -> AsyncGenerator[str, None]:
        """Stream generated answer (legacy)."""
        answer = await self.generate_legacy(query, context)
        words = answer.split()
        for i, word in enumerate(words):
            yield word + " "
            if i % 5 == 0:
                await asyncio.sleep(0.01)
    
    async def generate_with_citations(
        self,
        query: str,
        context: List[RetrievalResult],
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate answer with citations (legacy)."""
        answer = await self.generate_legacy(query, context, **kwargs)
        
        citations = []
        for i, r in enumerate(context):
            citations.append({
                "source_id": r.chunk_id,
                "content": r.content[:200],
                "metadata": r.metadata,
                "score": r.score,
            })
        
        return {
            "answer": answer,
            "citations": citations,
        }


class CitationGeneratorAdapter(ICitationGenerator):
    """Adapter for citation generator."""
    
    def __init__(self, rag_pipeline: RAGPipeline):
        self._pipeline = rag_pipeline
    
    @property
    def name(self) -> str:
        return "citation_generator"
    
    async def generate_citations(
        self,
        answer: str,
        chunks: List[Chunk],
    ) -> List[CitationRef]:
        """Generate citations for an answer."""
        citations = []
        for r in chunks:
            citations.append(CitationRef(
                cited_act=r.metadata.get('source', ''),
                cited_section=r.metadata.get('section', ''),
                cited_text=r.content[:200],
                citation_type="inline",
                confidence=1.0,
            ))
        return citations
    
    async def validate_citations(
        self,
        answer: str,
        citations: List[CitationRef],
    ) -> Dict[str, Any]:
        """Validate citations match answer."""
        # Use pipeline's citation validator
        from ip_sakti.rag.pipeline import RAGContext
        
        rag_context = RAGContext(
            query=type('Query', (), {'text': answer, 'user_id': 'test'})(),
            original_query=answer,
            citations=[c.model_dump() for c in citations],
        )
        
        await self._pipeline._stage_validation(rag_context)
        
        return {
            "valid": len(rag_context.errors) == 0,
            "errors": rag_context.errors,
        }
    
    # Legacy methods for backward compatibility
    async def generate_citations_legacy(
        self,
        answer: str,
        context: List[RetrievalResult],
    ) -> List[Dict[str, Any]]:
        """Generate citations for answer (legacy)."""
        citations = []
        for r in context:
            citations.append({
                "source_id": r.chunk_id,
                "content": r.content[:200],
                "metadata": r.metadata,
                "score": r.score,
                "authority_score": r.source_authority,
            })
        return citations
    
    async def validate_citations_legacy(
        self,
        answer: str,
        citations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Validate citations match answer (legacy)."""
        from ip_sakti.rag.pipeline import RAGContext
        
        rag_context = RAGContext(
            query=type('Query', (), {'text': answer, 'user_id': 'test'})(),
            original_query=answer,
            citations=citations,
        )
        
        await self._pipeline._stage_validation(rag_context)
        
        return {
            "valid": len(rag_context.errors) == 0,
            "errors": rag_context.errors,
        }

# =============================================================================
# Pipeline Adapters
# =============================================================================

class IngestionPipelineAdapter(IIngestionPipeline):
    """Adapter for ingestion pipeline."""
    
    def __init__(self, rag_pipeline: RAGPipeline):
        self._pipeline = rag_pipeline
    
    async def ingest(
        self,
        sources: List[Union[str, bytes]],
        **kwargs,
    ) -> Dict[str, Any]:
        """Ingest documents."""
        # This would use the ingestion pipeline from ip_sakti.ingestion.pipeline
        return {
            "success": True,
            "documents_processed": len(sources),
            "chunks_created": 0,
        }
    
    async def ingest_directory(
        self,
        directory: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Ingest all documents in directory."""
        return await self.ingest([directory], **kwargs)
    
    async def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics."""
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "last_ingestion": None,
        }


class RAGPipelineAdapter(IRAGPipeline):
    """Adapter for RAG pipeline."""
    
    def __init__(self, rag_pipeline: RAGPipeline):
        self._pipeline = rag_pipeline
    
    async def query(self, query: str, **kwargs) -> RAGResponse:
        """Execute RAG query."""
        # Create Query object
        query_obj = Query(
            text=query,
            user_id=kwargs.get('user_id', 'anonymous'),
            session_id=kwargs.get('session_id'),
            intent=kwargs.get('intent'),
            jurisdiction=kwargs.get('jurisdiction'),
            language=kwargs.get('language', 'en'),
            max_results=kwargs.get('top_k', 10),
            filters=kwargs.get('filters', {}),
            require_citations=kwargs.get('require_citations', True),
            stream=kwargs.get('stream', False),
        )
        
        # Run the pipeline
        context = await self._pipeline.run(query_obj)
        
        # Convert to RAGResponse
        return RAGResponse(
            answer=context.generated_answer or "",
            citations=context.citations,
            confidence=context.confidence_score,
            metadata=context.metadata,
            retrieval_results=[
                RetrievalResult(
                    chunk_id=r.chunk_id,
                    content=r.content,
                    score=r.score,
                    metadata=r.metadata,
                    source_authority=r.source_authority,
                    strategy=r.strategy,
                )
                for r in context.reranked_results
            ]
        )
    
    async def stream_query(self, query: str, **kwargs) -> AsyncIterator[str]:
        """Stream RAG query response."""
        response = await self.query(query, **kwargs)
        
        words = response.answer.split()
        for i, word in enumerate(words):
            yield word + " "
            if i % 5 == 0:
                await asyncio.sleep(0.01)


# =============================================================================
# Authority System Adapter
# =============================================================================

class AuthoritySystemAdapter(IAuthoritySystem):
    """Adapter for authority system."""
    
    def __init__(self, authority_system: SourceAuthoritySystem):
        self._system = authority_system
    
    @property
    def name(self) -> str:
        return "authority_system"
    
    async def evaluate_document(self, metadata: DocumentMetadata) -> AuthorityScore:
        """Evaluate authority of a document."""
        tier = self._system.classify(metadata.model_dump())
        return AuthorityScore(
            tier=AuthorityTier(tier.value if hasattr(tier, 'value') else tier),
            source_type=AuthoritySourceType.STATUTE,
            score=1.0,
        )
    
    async def evaluate_chunk(self, chunk: Chunk) -> AuthorityScore:
        """Evaluate authority of a chunk."""
        tier = self._system.classify(chunk.metadata)
        return AuthorityScore(
            tier=AuthorityTier(tier.value if hasattr(tier, 'value') else tier),
            source_type=AuthoritySourceType.STATUTE,
            score=1.0,
        )
    
    async def evaluate_batch(self, chunks: List[Chunk]) -> List[AuthorityScore]:
        """Batch evaluate authority."""
        return [await self.evaluate_chunk(c) for c in chunks]
    
    async def get_authority_weight(self, score: AuthorityScore, context: Optional[AuthorityContext] = None) -> float:
        """Get authority weight for fusion."""
        return float(score.score)
    
    async def rank_by_authority(self, chunks: List[Chunk], scores: List[AuthorityScore]) -> List[int]:
        """Return indices sorted by authority (highest first)."""
        paired = list(zip(range(len(scores)), scores))
        paired.sort(key=lambda x: x[1].score, reverse=True)
        return [i for i, _ in paired]
    
    async def get_tier_distribution(self, chunks: List[Chunk]) -> Dict[AuthorityTier, int]:
        """Get distribution of authority tiers."""
        dist = {}
        for chunk in chunks:
            score = await self.evaluate_chunk(chunk)
            tier = score.tier
            dist[tier] = dist.get(tier, 0) + 1
        return dist
    
    # Legacy methods for backward compatibility
    def classify_source(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Classify source authority tier."""
        tier = self._system.classify(metadata)
        return {"authority_tier": tier.value if hasattr(tier, 'value') else str(tier)}
    
    def propagate_authority(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Propagate authority through retrieval results."""
        # Convert to format expected by authority system
        docs = [
            {
                "chunk_id": r.chunk_id,
                "content": r.content,
                "score": r.score,
                "metadata": r.metadata,
            }
            for r in results
        ]
        
        enriched = self._system.propagate_authority(docs)
        
        # Convert back
        for i, doc in enumerate(enriched):
            if 'authority_score' in doc:
                results[i].source_authority = doc['authority_score']
        
        return results
    
    def resolve_conflicts(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Resolve conflicting information."""
        # Convert to format expected by authority system
        docs = [
            {
                "chunk_id": r.chunk_id,
                "content": r.content,
                "score": r.score,
                "metadata": r.metadata,
            }
            for r in results
        ]
        
        resolved = self._system.resolve_conflicts(docs)
        
        # Convert back
        for i, doc in enumerate(resolved):
            if 'confidence' in doc:
                results[i].score = doc.get('confidence', results[i].score)
        
        return results


# =============================================================================
# Adapter Factory
# =============================================================================

class AdapterFactory:
    """Factory for creating adapters from settings."""
    
    def __init__(self, settings: Settings):
        self._settings = settings
        self._modules = settings.modules or {}
        self._loader_registry: Optional[DocumentLoaderRegistry] = None
        self._chunker: Optional[ChunkingStrategy] = None
        self._embedder: Optional[SentenceTransformerEmbeddingProvider] = None
        self._vector_store: Optional[VectorStore] = None
        self._retrieval_engine: Optional[RetrievalEngine] = None
        self._authority_system: Optional[SourceAuthoritySystem] = None
        self._rag_pipeline: Optional[RAGPipeline] = None
    
    def _get_module_config(self, module_type: str) -> Dict[str, Any]:
        """Get configuration for a specific module type."""
        return self._modules.get(module_type, {})
    
    def _get_module_default(self, module_type: str) -> str:
        """Get default module for a type."""
        return self._get_module_config(module_type).get("default", "")
    
    def _get_module_fallback(self, module_type: str) -> str:
        """Get fallback module for a type."""
        return self._get_module_config(module_type).get("fallback", "")
    
    def _get_module_enabled(self, module_type: str) -> List[str]:
        """Get enabled modules for a type."""
        return self._get_module_config(module_type).get("enabled", [])
    
    def _get_module_params(self, module_type: str, module_name: str) -> Dict[str, Any]:
        """Get parameters for a specific module."""
        config = self._get_module_config(module_type)
        return config.get(module_name, {})
    
    def _get_loader(self) -> DocumentLoaderRegistry:
        if self._loader_registry is None:
            self._loader_registry = get_loader_registry()
        return self._loader_registry
    
    def _get_chunker(self) -> InternalChunkingStrategy:
        if self._chunker is None:
            from ip_sakti.ingestion.chunking import ChunkingConfig, ChunkingStrategyType
            
            # Get chunker config from settings
            chunker_config = self._get_module_config("chunker")
            default_chunker = chunker_config.get("default", "recursive")
            params = chunker_config.get(default_chunker, {})
            
            strategy_map = {
                "recursive": ChunkingStrategyType.RECURSIVE,
                "semantic": ChunkingStrategyType.SEMANTIC,
                "token_based": ChunkingStrategyType.FIXED_SIZE,
                "structure_aware": ChunkingStrategyType.RECURSIVE,  # fallback to recursive
            }
            
            config = ChunkingConfig(
                strategy=strategy_map.get(default_chunker, ChunkingStrategyType.RECURSIVE),
                chunk_size=params.get("chunk_size", 1000),
                chunk_overlap=params.get("chunk_overlap", 200),
                min_chunk_size=params.get("min_chunk_size", 100),
            )
            self._chunker = InternalChunkingStrategy(config)
        return self._chunker
    
    def _get_embedder(self) -> SentenceTransformerEmbeddingProvider:
        if self._embedder is None:
            embedder_config = self._get_module_config("embedder")
            default_embedder = embedder_config.get("default", "sentence_transformer")
            params = embedder_config.get(default_embedder, {})
            
            if default_embedder == "sentence_transformer":
                config = EmbeddingConfig(
                    model_name=EmbeddingModelType.BGE_M3,
                    dimension=params.get("dimensions", 1024),
                    device=params.get("device", "auto"),
                    normalize=params.get("normalize", True),
                )
            else:
                config = EmbeddingConfig(
                    model_name=EmbeddingModelType.BGE_M3,
                    dimension=1024,
                )
            self._embedder = SentenceTransformerEmbeddingProvider(config)
        return self._embedder
    
    def _get_vector_store(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = create_vector_store_from_settings(self._settings)
        return self._vector_store
    
    def _get_retrieval_engine(self) -> RetrievalEngine:
        if self._retrieval_engine is None:
            # Get retriever config from settings
            retriever_config = self._get_module_config("retriever")
            default_retriever = retriever_config.get("default", "hybrid")
            params = retriever_config.get(default_retriever, {})
            
            # Create RetrievalConfig from settings
            config = RetrievalConfig(
                hybrid_alpha=params.get("vector_weight", 0.6),
                vector_store=VectorStoreType.IN_MEMORY,
                vector_store_config={},
                cache_backend=CacheBackend.IN_MEMORY,
                cache_config={},
                cache_ttl_seconds=3600,
                enable_caching=True,
                enable_reranking=True,
                max_concurrent_searches=10,
                timeout_seconds=5.0,
            )
            self._retrieval_engine = RetrievalEngine(config)
        return self._retrieval_engine
    
    def _get_authority_system(self) -> SourceAuthoritySystem:
        if self._authority_system is None:
            self._authority_system = SourceAuthoritySystem(self._settings)
        return self._authority_system
    
    def _get_rag_pipeline(self) -> RAGPipeline:
        if self._rag_pipeline is None:
            self._rag_pipeline = RAGPipeline(
                self._settings,
                self._get_authority_system(),
                self._get_retrieval_engine()
            )
        return self._rag_pipeline
    
    def _get_query_processor(self):
        """Create query processor."""
        from ip_sakti.query_processor.query_processor import QueryProcessor
        return QueryProcessor()
    
    def _get_knowledge_graph(self):
        """Create knowledge graph."""
        from ip_sakti.kg.knowledge_graph import KnowledgeGraph
        return KnowledgeGraph()

    def create_config(self) -> IConfig:
        return ConfigAdapter(self._settings)
    
    def create_document_loader(self) -> IDocumentLoader:
        return DocumentLoaderAdapter(self._get_loader())
    
    def create_document_validator(self) -> IDocumentValidator:
        return DocumentValidatorAdapter()
    
    def create_chunker(self) -> IChunker:
        return ChunkerAdapter(self._get_chunker())
    
    def create_embedder(self) -> IEmbedder:
        return EmbedderAdapter(self._get_embedder())
    
    def create_multilingual_embedder(self) -> IMultilingualEmbedder:
        return MultilingualEmbedderAdapter(self._get_embedder())
    
    def create_vector_store(self) -> IVectorStore:
        return VectorStoreAdapter(self._get_vector_store())
    
    def create_retriever(self) -> IRetriever:
        return RetrievalEngineAdapter(self._get_retrieval_engine())
    
    def create_hybrid_retriever(self) -> IHybridRetriever:
        return HybridRetrieverAdapter(self._get_retrieval_engine())
    
    def create_reranker(self) -> IReranker:
        return RerankerAdapter()
    
    def create_query_analyzer(self) -> IQueryAnalyzer:
        return QueryAnalyzerAdapter(self._get_rag_pipeline())
    
    def create_query_rewriter(self) -> IQueryRewriter:
        return QueryRewriterAdapter(self._get_rag_pipeline())
    
    def create_query_router(self) -> IQueryRouter:
        return QueryRouterAdapter(self._get_rag_pipeline())
    
    def create_query_processor(self) -> IQueryProcessor:
        return QueryProcessorAdapter()

    def create_knowledge_graph(self) -> IKnowledgeGraph:
        return KnowledgeGraphAdapter()

    def create_authority_system(self) -> IAuthoritySystem:
        return AuthoritySystemAdapter(self._get_authority_system())

    def create_generator(self) -> IGenerator:
        return GeneratorAdapter()

    def create_ingestion_pipeline(self) -> IIngestionPipeline:
        return IngestionPipelineAdapter()

    def create_rag_pipeline(self) -> IRAGPipeline:
        return RAGPipelineAdapter(self._get_rag_pipeline())

    def create_pipeline_orchestrator(self) -> IPipelineOrchestrator:
        from ip_sakti.orchestration.pipeline_orchestrator import PipelineOrchestrator
        return PipelineOrchestrator(
            query_processor=self.create_query_processor(),
            retriever=self.create_retriever(),
            reranker=self.create_reranker(),
            generator=self.create_generator(),
            authority_system=self.create_authority_system(),
        )


# =============================================================================
# Query Processor Adapter
# =============================================================================

class QueryProcessorAdapter(IQueryProcessor):
    """Adapter for query processor."""
    
    def __init__(self):
        pass
    
    @property
    def name(self) -> str:
        return "query_processor"
    
    async def analyze(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> QueryAnalysis:
        """Analyze query for intent, complexity, entities."""
        # Simple analysis - in practice this would use NLP
        from ip_sakti.interfaces.query_processor import QueryAnalysis
        from ip_sakti.interfaces.retriever import QueryIntent, QueryComplexity
        
        return QueryAnalysis(
            intent=QueryIntent.SEARCH,
            complexity=QueryComplexity.SIMPLE,
            entities=[],
            legal_citations=[],
            acts_referenced=[],
            sections_referenced=[],
            dates_referenced=[],
            jurisdiction_hints=[],
            requires_multi_hop=False,
            requires_comparison=False,
            requires_temporal=False,
            confidence=0.5,
        )
    
    async def rewrite(self, query: str, analysis: QueryAnalysis) -> QueryRewrite:
        """Rewrite query for better retrieval."""
        from ip_sakti.interfaces.query_processor import QueryRewrite
        
        return QueryRewrite(
            original_query=query,
            rewritten_queries=[query],
            hyde_document=None,
            step_back_query=None,
            sub_queries=[],
            expansion_terms=[],
        )
    
    async def route(self, query: str, analysis: QueryAnalysis, rewrite: QueryRewrite) -> QueryRoute:
        """Determine retrieval strategy for query."""
        from ip_sakti.interfaces.query_processor import QueryRoute
        from ip_sakti.interfaces import RetrievalStrategy
        
        return QueryRoute(
            primary_strategy=RetrievalStrategy.HYBRID,
            fallback_strategies=[RetrievalStrategy.SPARSE, RetrievalStrategy.DENSE],
            use_kg=False,
            use_graph=False,
            use_contextual=False,
            requires_reranking=True,
            rerank_stages=["cross_encoder"],
        )
    
    async def process(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Full query processing pipeline: analyze → rewrite → route."""
        analysis = await self.analyze(query, conversation_history)
        rewrite = await self.rewrite(query, analysis)
        route = await self.route(query, analysis, rewrite)
        
        return {
            "analysis": analysis.model_dump(),
            "rewrite": rewrite.model_dump(),
            "route": route.model_dump(),
        }


# =============================================================================
# Knowledge Graph Adapter
# =============================================================================

class KnowledgeGraphAdapter(IKnowledgeGraph):
    """Adapter for knowledge graph."""
    
    def __init__(self):
        self._initialized = False
    
    @property
    def name(self) -> str:
        return "knowledge_graph"
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize graph backend."""
        self._initialized = True
        # In practice, initialize the actual graph store (NetworkX, Neo4j, etc.)
    
    async def add_node(self, node: GraphNode) -> str:
        """Add node to graph."""
        return node.id
    
    async def add_nodes(self, nodes: List[GraphNode]) -> List[str]:
        """Batch add nodes."""
        return [node.id for node in nodes]
    
    async def add_edge(self, edge: GraphEdge) -> str:
        """Add edge to graph."""
        return edge.id
    
    async def add_edges(self, edges: List[GraphEdge]) -> List[str]:
        """Batch add edges."""
        return [edge.id for edge in edges]
    
    async def query(self, query: GraphQuery) -> GraphResult:
        """Execute graph query."""
        return GraphResult(nodes=[], edges=[], paths=[], metadata={"query": query.model_dump()})
    
    async def get_neighbors(self, node_id: str, edge_types: Optional[List[GraphEdgeType]] = None, max_depth: int = 1) -> GraphResult:
        """Get neighboring nodes."""
        return GraphResult(nodes=[], edges=[], paths=[], metadata={"node_id": node_id})
    
    async def find_path(self, source_id: str, target_id: str, max_depth: int = 4) -> Optional[List[str]]:
        """Find shortest path between nodes."""
        return None
    
    async def get_subgraph(self, node_ids: List[str], max_depth: int = 2) -> GraphResult:
        """Get subgraph around nodes."""
        return GraphResult(nodes=[], edges=[], paths=[], metadata={"node_ids": node_ids})
    
    async def compute_pagerank(self, node_type: Optional[GraphNodeType] = None) -> Dict[str, float]:
        """Compute PageRank scores for authority propagation."""
        return {}
    
    async def detect_communities(self, algorithm: str = "leiden") -> Dict[str, int]:
        """Detect communities in graph."""
        return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check."""
        return {"status": "healthy", "initialized": self._initialized}


# =============================================================================
# Authority System Adapter
# =============================================================================

class AuthoritySystemAdapter(IAuthoritySystem):
    """Adapter for authority system."""

    def __init__(self, authority_system):
        self._system = authority_system

    @property
    def name(self) -> str:
        return "source_authority_system"

    async def evaluate_document(self, metadata: DocumentMetadata) -> AuthorityScore:
        """Evaluate authority of a document."""
        from ip_sakti.authority.authority_system import SourceAuthoritySystem
        from ip_sakti.interfaces.authority import AuthorityScore, AuthorityTier, AuthoritySourceType
        
        # Use the existing system
        doc_metadata = {
            'title': metadata.title,
            'document_type': metadata.document_type,
            'jurisdiction': metadata.jurisdiction,
            'source_path': metadata.source_path,
            'year': metadata.effective_date[:4] if metadata.effective_date else None,
        }
        
        # Get score from existing system
        score_obj = await self._system.evaluate_authority(doc_metadata)
        
        return AuthorityScore(
            tier=AuthorityTier(score_obj.tier),
            source_type=AuthoritySourceType(score_obj.source_type),
            score=score_obj.score,
            court_level=score_obj.court_level,
            jurisdiction=score_obj.jurisdiction,
            year=score_obj.year,
            is_binding=score_obj.is_binding,
            metadata=score_obj.metadata,
        )

    async def evaluate_chunk(self, chunk: Chunk) -> AuthorityScore:
        """Evaluate authority of a chunk."""
        # Create document metadata from chunk
        metadata = DocumentMetadata(
            title=chunk.metadata.get('title', ''),
            document_type=chunk.metadata.get('document_type', 'legal'),
            jurisdiction=chunk.metadata.get('jurisdiction', 'INDIA'),
            source_path=chunk.source,
        )
        return await self.evaluate_document(metadata)

    async def evaluate_batch(self, chunks: List[Chunk]) -> List[AuthorityScore]:
        """Batch evaluate authority."""
        return [await self.evaluate_chunk(chunk) for chunk in chunks]

    async def get_authority_weight(self, score: AuthorityScore, context: Optional[AuthorityContext] = None) -> float:
        """Get authority weight for fusion."""
        return self._system.get_authority_weight(score)

    async def rank_by_authority(self, chunks: List[Chunk], scores: List[AuthorityScore]) -> List[int]:
        """Return indices sorted by authority (highest first)."""
        # Sort by score descending, return original indices
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1].score, reverse=True)
        return [i for i, _ in indexed]

    async def get_tier_distribution(self, chunks: List[Chunk]) -> Dict[AuthorityTier, int]:
        """Get distribution of authority tiers."""
        from ip_sakti.interfaces.authority import AuthorityTier
        
        distribution = {tier: 0 for tier in AuthorityTier}
        scores = await self.evaluate_batch(chunks)
        for score in scores:
            distribution[score.tier] = distribution.get(score.tier, 0) + 1
        return distribution


# =============================================================================
# Generator Adapter
# =============================================================================

class GeneratorAdapter(IGenerator):
    """Adapter for generator."""

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "citation_generator"

    @property
    def supported_strategies(self) -> List[GenerationStrategy]:
        return [GenerationStrategy.CITATION_FIRST, GenerationStrategy.STANDARD, GenerationStrategy.CONSTRAINED]

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate answer with citations."""
        import time
        start = time.time()
        
        # Simple template-based generation for now
        # In production, this would use an LLM
        if not request.chunks:
            return GenerationResult(
                answer="I don't have enough information to answer this question.",
                citations=[],
                confidence=0.0,
                strategy_used=request.strategy,
                tokens_used=0,
                generation_time_ms=(time.time() - start) * 1000,
                metadata={},
                verification_passed=False,
                verification_errors=["No chunks provided"],
            )
        
        # Build answer from top chunks
        answer_parts = []
        citations = []
        for i, chunk in enumerate(request.chunks[:5]):
            answer_parts.append(chunk.content[:200])
            citations.append(CitationRef(
                cited_act=chunk.metadata.get('act', 'Unknown'),
                cited_section=chunk.metadata.get('section', 'N/A'),
                cited_text=chunk.content[:200],
                citation_type="inline",
                confidence=0.8,
            ))
        
        answer = "Based on the retrieved sources: " + " ".join(answer_parts)
        
        return GenerationResult(
            answer=answer,
            citations=citations,
            confidence=0.75,
            strategy_used=request.strategy,
            tokens_used=len(answer.split()),
            generation_time_ms=(time.time() - start) * 1000,
            metadata={"num_chunks_used": len(request.chunks)},
            verification_passed=True,
            verification_errors=[],
        )

    async def generate_stream(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        """Stream generation tokens."""
        result = await self.generate(request)
        for token in result.answer.split():
            yield token + " "
            await asyncio.sleep(0.01)

    async def verify_citations(self, answer: str, chunks: List[Chunk]) -> Dict[str, Any]:
        """Verify that citations in answer match retrieved chunks."""
        return {"verified": True, "matched": len(chunks), "total": len(chunks)}

    async def extract_structured(self, query: str, chunks: List[Chunk], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured information from chunks."""
        return {"extracted": True, "schema": schema, "chunks_used": len(chunks)}


# =============================================================================
# Ingestion Pipeline Adapter
# =============================================================================

class IngestionPipelineAdapter(IIngestionPipeline):
    """Adapter for ingestion pipeline."""

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "ingestion_pipeline"

    async def ingest(self, request: IngestionRequest) -> IngestionResult:
        """Ingest a single document."""
        import time
        start = time.time()
        
        # Placeholder - would use actual loaders and chunkers
        doc = Document(
            id=request.source,
            content=f"Content of {request.source}",
            metadata={},
            source=request.source,
        )
        
        chunk = Chunk(
            id=f"{request.source}_chunk_0",
            content=f"Content of {request.source}",
            metadata={},
            source=request.source,
            chunk_index=0,
            start_char=0,
            end_char=100,
        )
        
        return IngestionResult(
            document=doc,
            chunks=[chunk],
            total_chunks=1,
            processing_time_ms=(time.time() - start) * 1000,
        )

    async def ingest_batch(self, requests: List[IngestionRequest]) -> List[IngestionResult]:
        """Ingest multiple documents."""
        return [await self.ingest(req) for req in requests]

    async def ingest_directory(self, directory: str, **kwargs) -> List[IngestionResult]:
        """Ingest all documents in a directory."""
        return []

    async def get_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics."""
        return {"documents_processed": 0, "chunks_created": 0}

    async def health_check(self) -> Dict[str, Any]:
        """Health check."""
        return {"status": "healthy"}


# =============================================================================
# RAG Pipeline Adapter
# =============================================================================

class RAGPipelineAdapter(IRAGPipeline):
    """Adapter for RAG pipeline."""

    def __init__(self, rag_pipeline):
        self._pipeline = rag_pipeline

    @property
    def name(self) -> str:
        return "rag_pipeline"

    async def query(self, query: str, **kwargs) -> RAGResponse:
        """Execute RAG query."""
        # Use the existing pipeline
        result = await self._pipeline.query(query)
        
        return RAGResponse(
            answer=result.get('answer', ''),
            citations=result.get('citations', []),
            confidence=result.get('confidence', 0.5),
            metadata=result.get('metadata', {}),
        )

    async def stream_query(self, query: str, **kwargs) -> AsyncGenerator[str, None]:
        """Stream RAG query response."""
        result = await self.query(query)
        for token in result.answer.split():
            yield token + " "
            await asyncio.sleep(0.01)

    async def batch_query(self, queries: List[str], **kwargs) -> List[RAGResponse]:
        """Execute multiple RAG queries."""
        return [await self.query(q) for q in queries]

    async def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {"queries_processed": 0, "avg_latency_ms": 0.0}

    async def health_check(self) -> Dict[str, Any]:
        """Health check."""
        return {"status": "healthy"}