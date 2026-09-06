"""
IP-SAKTI Retrieval Engine
Phase 10: Infrastructure/service-level retrieval engine.
References Phase 9 (RAG pipeline) for pipeline logic.
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ip_sakti.config.loader import Settings, get_settings
from ip_sakti.core.models import (
    DocumentChunk,
    DocumentType,
    Jurisdiction,
    RetrievalResult,
    RetrievalStrategy,
    SourceAuthorityTier,
)
from ip_sakti.authority.authority_system import SourceAuthoritySystem
# Import new embedding module
from ip_sakti.embedding import (
    EmbeddingProvider,
    AsyncEmbeddingProvider,
    EmbeddingConfig,
    SentenceTransformerEmbeddingProvider,
    VectorStore,
    VectorStoreConfig,
    VectorStoreType,
    SearchResult,
    create_vector_store,
    create_vector_store_from_settings,
)

logger = logging.getLogger(__name__)


class CacheBackend(str, Enum):
    """Cache backend types."""
    REDIS = "redis"
    MEMCACHED = "memcached"
    IN_MEMORY = "in_memory"


# VectorStoreType is imported from embedding module


@dataclass
class RetrievalConfig:
    hybrid_alpha: float = 0.5  # Weight for semantic vs keyword (0=keyword, 1=semantic)
    vector_store: VectorStoreType = VectorStoreType.IN_MEMORY
    vector_store_config: Dict[str, Any] = field(default_factory=dict)
    cache_backend: CacheBackend = CacheBackend.IN_MEMORY
    cache_config: Dict[str, Any] = field(default_factory=dict)
    cache_ttl_seconds: int = 3600
    enable_caching: bool = True
    enable_reranking: bool = True
    max_concurrent_searches: int = 10
    timeout_seconds: float = 5.0


@dataclass
class SearchRequest:
    """Search request parameters."""
    query: str
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    filters: Dict[str, Any] = field(default_factory=dict)
    top_k: int = 10
    alpha: Optional[float] = None
    include_vectors: bool = False
    rerank: bool = True


@dataclass
class SearchResponse:
    """Search response."""
    results: List[RetrievalResult]
    query: str
    strategy: RetrievalStrategy
    took_ms: float
    total_hits: int
    cached: bool = False


class VectorStore(ABC):
    """Abstract base for vector stores."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store."""
        pass
    
    @abstractmethod
    async def upsert(self, chunks: List[DocumentChunk], vectors: np.ndarray) -> None:
        """Insert or update chunks with vectors."""
        pass
    
    @abstractmethod
    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Search by vector similarity. Returns (chunk, score)."""
        pass
    
    @abstractmethod
    async def delete(self, chunk_ids: List[str]) -> None:
        """Delete chunks by ID."""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close connections."""
        pass


class InMemoryVectorStore(VectorStore):
    """In-memory vector store for development/testing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.chunks: Dict[str, DocumentChunk] = {}
        self.vectors: Dict[str, np.ndarray] = {}
        self._index_built = False
    
    async def initialize(self) -> None:
        self._index_built = True
        logger.info("In-memory vector store initialized")
    
    async def upsert(self, chunks: List[DocumentChunk], vectors: np.ndarray) -> None:
        for chunk, vector in zip(chunks, vectors):
            self.chunks[chunk.id] = chunk
            self.vectors[chunk.id] = vector
        self._index_built = True
    
    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        if not self._index_built or not self.vectors:
            return []
        
        # Compute cosine similarity
        scores = {}
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return []
        
        for chunk_id, vector in self.vectors.items():
            chunk = self.chunks[chunk_id]
            
            # Apply filters
            if filters and not self._matches_filters(chunk, filters):
                continue
            
            # Cosine similarity
            vec_norm = np.linalg.norm(vector)
            if vec_norm == 0:
                continue
            score = float(np.dot(query_vector, vector) / (query_norm * vec_norm))
            scores[chunk_id] = score
        
        # Sort by score
        sorted_ids = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        
        return [(self.chunks[cid], scores[cid]) for cid in sorted_ids[:top_k]]
    
    def _matches_filters(self, chunk: DocumentChunk, filters: Dict[str, Any]) -> bool:
        """Check if chunk matches filters."""
        for key, value in filters.items():
            if value is None:
                continue
            if key == 'jurisdiction':
                if chunk.metadata.get('jurisdiction') != value:
                    return False
            elif key == 'document_type':
                if chunk.metadata.get('document_type') != value:
                    return False
            elif key == 'authority_tier':
                if chunk.metadata.get('source_authority_tier') != value:
                    return False
            elif key == 'claim_number':
                if chunk.metadata.get('claim_number') != value:
                    return False
        return True
    
    async def delete(self, chunk_ids: List[str]) -> None:
        for cid in chunk_ids:
            self.chunks.pop(cid, None)
            self.vectors.pop(cid, None)
    
    async def get_stats(self) -> Dict[str, Any]:
        return {
            'total_chunks': len(self.chunks),
            'total_vectors': len(self.vectors),
            'store_type': 'in_memory',
        }
    
    async def close(self) -> None:
        self.chunks.clear()
        self.vectors.clear()


class SQLiteVectorStore(InMemoryVectorStore):
    """Local persistent vector store for the default single-node deployment."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.path = config.get("persist_path", "data/runtime/ip_sakti_index.sqlite3")
        self._connection: Optional[sqlite3.Connection] = None

    async def initialize(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute("CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, payload TEXT NOT NULL, vector BLOB NOT NULL)")
        self._connection.commit()
        rows = self._connection.execute("SELECT payload, vector FROM chunks").fetchall()
        for payload, vector_blob in rows:
            data = json.loads(payload)
            chunk = DocumentChunk(
                id=data["id"], document_id=data["document_id"], content=data["content"],
                chunk_index=data.get("chunk_index", 0), start_char=data.get("start_char", 0),
                end_char=data.get("end_char", 0), metadata=data.get("metadata", {}),
                authority_score=data.get("authority_score", 0.0),
            )
            self.chunks[chunk.id] = chunk
            self.vectors[chunk.id] = np.frombuffer(vector_blob, dtype=np.float32)
        self._index_built = True

    async def upsert(self, chunks: List[DocumentChunk], vectors: np.ndarray) -> None:
        await super().upsert(chunks, vectors)
        if not self._connection:
            return
        for chunk, vector in zip(chunks, vectors):
            payload = json.dumps({
                "id": chunk.id, "document_id": chunk.document_id, "content": chunk.content,
                "chunk_index": chunk.chunk_index, "start_char": chunk.start_char, "end_char": chunk.end_char,
                "metadata": chunk.metadata, "authority_score": chunk.authority_score,
            }, default=str)
            self._connection.execute(
                "INSERT OR REPLACE INTO chunks(id, payload, vector) VALUES (?, ?, ?)",
                (chunk.id, payload, sqlite3.Binary(np.asarray(vector, dtype=np.float32).tobytes())),
            )
        self._connection.commit()

    async def delete(self, chunk_ids: List[str]) -> None:
        await super().delete(chunk_ids)
        if self._connection:
            self._connection.executemany("DELETE FROM chunks WHERE id = ?", [(chunk_id,) for chunk_id in chunk_ids])
            self._connection.commit()

    async def close(self) -> None:
        if self._connection:
            self._connection.close()
            self._connection = None
        self.chunks.clear()
        self.vectors.clear()


class KeywordIndex(ABC):
    """Abstract base for keyword/BM25 index."""
    
    @abstractmethod
    async def initialize(self) -> None:
        pass
    
    @abstractmethod
    async def upsert(self, chunks: List[DocumentChunk]) -> None:
        pass
    
    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        pass
    
    @abstractmethod
    async def delete(self, chunk_ids: List[str]) -> None:
        pass
    
    @abstractmethod
    async def close(self) -> None:
        pass


class InMemoryKeywordIndex(KeywordIndex):
    """In-memory BM25-style keyword index."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.chunks: Dict[str, DocumentChunk] = {}
        self.inverted_index: Dict[str, Set[str]] = {}  # term -> chunk_ids
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length = 0
        self.k1 = config.get('bm25_k1', 1.5)
        self.b = config.get('bm25_b', 0.75)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        import re
        return [t.lower() for t in re.findall(r'\b\w+\b', text) if len(t) > 2]
    
    async def initialize(self) -> None:
        logger.info("In-memory keyword index initialized")
    
    async def upsert(self, chunks: List[DocumentChunk]) -> None:
        for chunk in chunks:
            self.chunks[chunk.id] = chunk
            tokens = self._tokenize(chunk.content)
            self.doc_lengths[chunk.id] = len(tokens)
            
            # Update inverted index
            for token in set(tokens):  # Unique terms per doc
                if token not in self.inverted_index:
                    self.inverted_index[token] = set()
                self.inverted_index[token].add(chunk.id)
        
        # Update average doc length
        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)
    
    async def search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        
        # Find candidate chunks
        candidate_ids = set()
        for token in query_tokens:
            if token in self.inverted_index:
                candidate_ids.update(self.inverted_index[token])
        
        # Apply filters
        if filters:
            filtered_ids = set()
            for cid in candidate_ids:
                chunk = self.chunks.get(cid)
                if chunk and self._matches_filters(chunk, filters):
                    filtered_ids.add(cid)
            candidate_ids = filtered_ids
        
        # BM25 scoring
        scores = {}
        N = len(self.chunks)
        
        for chunk_id in candidate_ids:
            chunk = self.chunks[chunk_id]
            doc_len = self.doc_lengths.get(chunk_id, 0)
            
            score = 0.0
            for token in query_tokens:
                if token not in self.inverted_index:
                    continue
                
                df = len(self.inverted_index[token])  # Document frequency
                idf = np.log((N - df + 0.5) / (df + 0.5) + 1)
                
                # Term frequency in this document
                tf = self._tokenize(chunk.content).count(token)
                
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                score += idf * numerator / denominator
            
            scores[chunk_id] = score
        
        # Sort by score
        sorted_ids = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        
        return [(self.chunks[cid], scores[cid]) for cid in sorted_ids[:top_k]]
    
    def _matches_filters(self, chunk: DocumentChunk, filters: Dict[str, Any]) -> bool:
        for key, value in filters.items():
            if value is None:
                continue
            if key == 'jurisdiction':
                if chunk.metadata.get('jurisdiction') != value:
                    return False
            elif key == 'document_type':
                if chunk.metadata.get('document_type') != value:
                    return False
            elif key == 'authority_tier':
                if chunk.metadata.get('source_authority_tier') != value:
                    return False
        return True
    
    async def delete(self, chunk_ids: List[str]) -> None:
        for cid in chunk_ids:
            self.chunks.pop(cid, None)
            self.doc_lengths.pop(cid, None)
            # Note: inverted index cleanup would be needed in production
    
    async def close(self) -> None:
        self.chunks.clear()
        self.inverted_index.clear()
        self.doc_lengths.clear()


class Cache(ABC):
    """Abstract base for cache."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int) -> None:
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        pass


class InMemoryCache(Cache):
    """In-memory cache with TTL."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._cache: Dict[str, Tuple[Any, float]] = {}  # key -> (value, expiry_time)
    
    async def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            value, expiry = self._cache[key]
            if time.time() < expiry:
                return value
            else:
                del self._cache[key]
        return None
    
    async def set(self, key: str, value: Any, ttl: int) -> None:
        self._cache[key] = (value, time.time() + ttl)
    
    async def delete(self, key: str) -> None:
        self._cache.pop(key, None)
    
    async def clear(self) -> None:
        self._cache.clear()


class EmbeddingProvider(ABC):
    """Abstract base for embedding providers."""
    
    @abstractmethod
    async def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts. Returns (n, dim) array."""
        pass
    
    @abstractmethod
    async def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension."""
        pass


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for development."""
    
    def __init__(self, dimension: int = 384):
        self._dimension = dimension
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    async def embed(self, texts: List[str]) -> np.ndarray:
        # Deterministic mock embeddings based on text hash
        embeddings = []
        for text in texts:
            # Create deterministic "embedding" from hash
            hash_val = hash(text)
            np.random.seed(abs(hash_val) % (2**32))
            emb = np.random.randn(self._dimension).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
        return np.array(embeddings)
    
    async def embed_query(self, query: str) -> np.ndarray:
        result = await self.embed([query])
        return result[0]


class RetrievalEngine:
    """Main retrieval engine - infrastructure level.
    
    This handles the infra/service concerns: scaling, caching, deployment topology.
    Pipeline logic (query rewriting, reranking, generation) is in Phase 9 - RAG Pipeline.
    """
    
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.settings = get_settings()
        self.config.vector_store_config.setdefault("persist_path", self.settings.index_path)
        
        # Components
        self.vector_store = self._create_vector_store()
        self.keyword_index = self._create_keyword_index()
        self.cache = self._create_cache()
        self.embedding_provider = self._create_embedding_provider()
        self.knowledge_graph = None
        
        # Authority system for filtering
        self.authority_system = SourceAuthoritySystem(self.settings)
        
        # Semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(config.max_concurrent_searches)
        
        # Stats
        self.stats = {
            'total_searches': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
        }
    
    def _create_vector_store(self) -> VectorStore:
        """Use SQLite persistence in normal mode and memory only in explicit test mode."""
        if self.settings.test_mode:
            return InMemoryVectorStore(self.config.vector_store_config)
        return SQLiteVectorStore(self.config.vector_store_config)
    
    def _create_keyword_index(self) -> KeywordIndex:
        return InMemoryKeywordIndex(self.config.vector_store_config)
    
    def _create_cache(self) -> Cache:
        cache_type = self.config.cache_backend
        cache_config = self.config.cache_config
        
        if cache_type == CacheBackend.IN_MEMORY:
            return InMemoryCache(cache_config)
        # elif cache_type == CacheBackend.REDIS:
        #     return RedisCache(cache_config)
        
        return InMemoryCache(cache_config)
    
    def _create_embedding_provider(self) -> EmbeddingProvider:
        if self.settings.test_mode:
            logger.warning("Using deterministic mock embeddings because IP_SAKTI_TEST_MODE is enabled")
            return MockEmbeddingProvider(dimension=self.settings.embedding_dimensions)

        return AsyncEmbeddingProvider(SentenceTransformerEmbeddingProvider(EmbeddingConfig(
            model_name=self.settings.embedding_model,
            fallback_model=self.settings.embedding_fallback,
            dimension=self.settings.embedding_dimensions,
            max_tokens=self.settings.embedding_max_tokens,
            batch_size=self.settings.embedding_batch_size,
            device="cpu",
        )))
    
    async def initialize(self) -> None:
        """Initialize all components."""
        await asyncio.gather(
            self.vector_store.initialize(),
            self.keyword_index.initialize(),
            self.cache.initialize() if hasattr(self.cache, 'initialize') else asyncio.sleep(0),
        )
        # Rebuild the sparse index from persisted chunks after a restart.
        persisted_chunks = list(getattr(self.vector_store, "chunks", {}).values())
        if persisted_chunks:
            await self.keyword_index.upsert(persisted_chunks)
        from ip_sakti.kg.knowledge_graph import KnowledgeGraph
        self.knowledge_graph = KnowledgeGraph(settings=self.settings)
        await self.knowledge_graph.initialize()
        # Rehydrate the local graph from the persisted corpus so graph
        # retrieval remains available after an API restart.
        if persisted_chunks:
            await self.knowledge_graph.ingest_chunks(persisted_chunks)
        logger.info("Retrieval engine initialized")
    
    async def index_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Index chunks into both vector store and keyword index."""
        if not chunks:
            return
        
        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        vectors = await self.embedding_provider.embed(texts)
        
        # Upsert to both stores
        await asyncio.gather(
            self.vector_store.upsert(chunks, vectors),
            self.keyword_index.upsert(chunks),
        )
        if self.knowledge_graph:
            await self.knowledge_graph.ingest_chunks(chunks)
        
        logger.info(f"Indexed {len(chunks)} chunks")
    
    async def search(self, request: SearchRequest) -> SearchResponse:
        """Execute search with specified strategy."""
        start_time = time.time()
        
        async with self._semaphore:
            # Check cache
            cache_key = self._make_cache_key(request)
            cached_response = None
            
            if self.config.enable_caching:
                cached_response = await self.cache.get(cache_key)
                if cached_response:
                    self.stats['cache_hits'] += 1
                    cached_response.cached = True
                    return cached_response
            
            self.stats['cache_misses'] += 1
            self.stats['total_searches'] += 1
            
            # Execute search based on strategy
            try:
                results = await self._execute_search(request)
            except Exception as e:
                self.stats['errors'] += 1
                logger.exception(f"Search failed: {e}")
                raise
            
            took_ms = (time.time() - start_time) * 1000
            
            response = SearchResponse(
                results=results,
                query=request.query,
                strategy=request.strategy,
                took_ms=took_ms,
                total_hits=len(results),
            )
            
            # Cache response
            if self.config.enable_caching:
                await self.cache.set(cache_key, response, self.config.cache_ttl_seconds)
            
            return response
    
    async def _execute_search(self, request: SearchRequest) -> List[RetrievalResult]:
        """Execute search based on strategy."""
        strategy = request.strategy
        top_k = request.top_k
        alpha = request.alpha if request.alpha is not None else self.config.hybrid_alpha
        filters = request.filters
        
        if strategy == RetrievalStrategy.SEMANTIC:
            return await self._semantic_search(request.query, top_k, filters)
        elif strategy == RetrievalStrategy.KEYWORD:
            return await self._keyword_search(request.query, top_k, filters)
        elif strategy == RetrievalStrategy.HYBRID:
            return await self._hybrid_search(request.query, top_k, filters, alpha)
        elif strategy == RetrievalStrategy.GRAPH:
            return await self._graph_search(request.query, top_k, filters)
        else:
            return await self._hybrid_search(request.query, top_k, filters, alpha)
    
    async def _semantic_search(
        self,
        query: str,
        top_k: int,
        filters: Dict[str, Any],
    ) -> List[RetrievalResult]:
        """Pure vector similarity search."""
        query_vector = await self.embedding_provider.embed_query(query)
        results = await self.vector_store.search(query_vector, top_k, filters)
        
        return [
            RetrievalResult(
                chunk=chunk,
                score=score,
                strategy=RetrievalStrategy.SEMANTIC,
            )
            for chunk, score in results
        ]
    
    async def _keyword_search(
        self,
        query: str,
        top_k: int,
        filters: Dict[str, Any],
    ) -> List[RetrievalResult]:
        """Pure keyword/BM25 search."""
        results = await self.keyword_index.search(query, top_k, filters)
        
        return [
            RetrievalResult(
                chunk=chunk,
                score=score,
                strategy=RetrievalStrategy.KEYWORD,
            )
            for chunk, score in results
        ]
    
    async def _hybrid_search(
        self,
        query: str,
        top_k: int,
        filters: Dict[str, Any],
        alpha: float,
    ) -> List[RetrievalResult]:
        """Hybrid search combining semantic and keyword."""
        # Run both searches in parallel
        semantic_task = self._semantic_search(query, top_k * 2, filters)
        keyword_task = self._keyword_search(query, top_k * 2, filters)
        
        semantic_results, keyword_results = await asyncio.gather(semantic_task, keyword_task)
        
        # Combine scores using reciprocal rank fusion (RRF) or weighted sum
        return self._fuse_results(semantic_results, keyword_results, alpha, top_k)
    
    async def _graph_search(
        self,
        query: str,
        top_k: int,
        filters: Dict[str, Any],
    ) -> List[RetrievalResult]:
        """Retrieve indexed chunks referenced by entities in the knowledge graph."""
        if not self.knowledge_graph:
            return []
        terms = {term.lower() for term in query.split() if len(term) > 2}
        graph_store = self.knowledge_graph.graph_store
        matched_chunks: Dict[str, float] = {}
        for entity in graph_store.entities.values():
            name = entity.name.lower()
            if any(term in name for term in terms):
                for chunk_id in entity.source_chunks:
                    matched_chunks[chunk_id] = max(matched_chunks.get(chunk_id, 0.0), entity.confidence)
        results = []
        for chunk_id, score in matched_chunks.items():
            chunk = getattr(self.vector_store, "chunks", {}).get(chunk_id)
            if chunk and self.vector_store._matches_filters(chunk, filters):
                results.append(RetrievalResult(chunk=chunk, score=score, strategy=RetrievalStrategy.GRAPH))
        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k]
    
    def _fuse_results(
        self,
        semantic_results: List[RetrievalResult],
        keyword_results: List[RetrievalResult],
        alpha: float,
        top_k: int,
    ) -> List[RetrievalResult]:
        """Fuse semantic and keyword results using weighted score combination."""
        # Normalize scores to [0, 1] range
        def normalize(results: List[RetrievalResult]) -> Dict[str, float]:
            if not results:
                return {}
            scores = [r.score for r in results]
            min_s, max_s = min(scores), max(scores)
            if max_s == min_s:
                return {r.chunk.id: 1.0 for r in results}
            return {
                r.chunk.id: (r.score - min_s) / (max_s - min_s)
                for r in results
            }
        
        sem_scores = normalize(semantic_results)
        kw_scores = normalize(keyword_results)
        
        # Combine
        all_chunk_ids = set(sem_scores.keys()) | set(kw_scores.keys())
        fused = []
        
        for chunk_id in all_chunk_ids:
            sem_score = sem_scores.get(chunk_id, 0)
            kw_score = kw_scores.get(chunk_id, 0)
            
            # Weighted combination
            combined = alpha * sem_score + (1 - alpha) * kw_score
            
            # Find the chunk object
            chunk = None
            for r in semantic_results:
                if r.chunk.id == chunk_id:
                    chunk = r.chunk
                    break
            if chunk is None:
                for r in keyword_results:
                    if r.chunk.id == chunk_id:
                        chunk = r.chunk
                        break
            
            if chunk:
                fused.append(RetrievalResult(
                    chunk=chunk,
                    score=combined,
                    strategy=RetrievalStrategy.HYBRID,
                ))
        
        # Sort and return top_k
        fused.sort(key=lambda r: r.score, reverse=True)
        return fused[:top_k]
    
    def _make_cache_key(self, request: SearchRequest) -> str:
        """Create cache key from search request."""
        import hashlib
        import json
        
        key_data = {
            'query': request.query,
            'strategy': request.strategy.value,
            'filters': request.filters,
            'top_k': request.top_k,
            'alpha': request.alpha,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return f"search:{hashlib.sha256(key_str.encode()).hexdigest()[:32]}"
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        vector_stats = await self.vector_store.get_stats()
        
        return {
            **self.stats,
            'vector_store': vector_stats,
            'cache_size': len(self.cache._cache) if hasattr(self.cache, '_cache') else 0,
        }
    
    async def close(self) -> None:
        """Close all connections."""
        await asyncio.gather(
            self.vector_store.close(),
            self.keyword_index.close(),
            self.cache.clear() if hasattr(self.cache, 'clear') else asyncio.sleep(0),
        )
        if self.knowledge_graph:
            await self.knowledge_graph.close()
        logger.info("Retrieval engine closed")


# Factory function
def create_retrieval_engine(config: Optional[RetrievalConfig] = None) -> RetrievalEngine:
    """Factory to create retrieval engine."""
    if config is None:
        config = RetrievalConfig()
    return RetrievalEngine(config)
