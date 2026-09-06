"""
Phase 3: Vector/Semantic Retrieval Strategy
"""

import asyncio
import logging
import time
import uuid
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from ip_sakti.core.models import DocumentChunk, RetrievalResult, RetrievalStrategy
from ip_sakti.retrieval.strategies.base import RetrievalConfig, SearchRequest, SearchResult, RetrievalStrategy as BaseRetrievalStrategy
from ip_sakti.embedding.embedding_provider import EmbeddingProvider

logger = logging.getLogger(__name__)


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for development."""
    
    def __init__(self, dimension: int = 384):
        self._dimension = dimension
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    async def embed(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            hash_val = hash(text)
            np.random.seed(abs(hash_val) % (2**32))
            emb = np.random.randn(self._dimension).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
        return np.array(embeddings)
    
    async def embed_query(self, query: str) -> np.ndarray:
        result = await self.embed([query])
        return result[0]


class VectorRetriever(BaseRetrievalStrategy):
    """Vector-based semantic retrieval strategy."""
    
    def __init__(self, config: RetrievalConfig, embedding_provider: Optional[EmbeddingProvider] = None, **kwargs):
        self.config = config
        self.embedding_provider = embedding_provider or MockEmbeddingProvider(dimension=384)
        self.chunks: Dict[str, DocumentChunk] = {}
        self.vectors: Dict[str, np.ndarray] = {}
        self._initialized = False
    
    @property
    def name(self) -> str:
        return RetrievalStrategy.VECTOR.value
    
    async def index(self, chunks: List[DocumentChunk]) -> None:
        """Index chunks into vector store."""
        start = time.time()
        
        if not chunks:
            return
        
        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        vectors = await self.embedding_provider.embed(texts)
        
        for chunk, vector in zip(chunks, vectors):
            self.chunks[chunk.id] = chunk
            self.vectors[chunk.id] = vector
        
        self._initialized = True
        logger.info(f"Vector index built: {len(self.chunks)} chunks in {time.time() - start:.2f}s")
    
    async def search(self, request: SearchRequest) -> SearchResult:
        """Execute vector similarity search."""
        start_time = time.time()
        
        if not self._initialized or not self.vectors:
            return SearchResult(
                results=[],
                query=request.query,
                strategy=RetrievalStrategy.VECTOR,
                took_ms=0,
                total_hits=0,
            )
        
        # Generate query embedding
        query_vector = await self.embedding_provider.embed_query(request.query)
        
        # Compute cosine similarities
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return SearchResult(
                results=[],
                query=request.query,
                strategy=RetrievalStrategy.VECTOR,
                took_ms=(time.time() - start_time) * 1000,
                total_hits=0,
            )
        
        scores = {}
        for chunk_id, vector in self.vectors.items():
            chunk = self.chunks[chunk_id]
            
            # Apply filters
            if request.filters and not self._matches_filters(chunk, request.filters):
                continue
            
            vec_norm = np.linalg.norm(vector)
            if vec_norm == 0:
                continue
            
            score = float(np.dot(query_vector, vector) / (query_norm * vec_norm))
            scores[chunk_id] = score
        
        # Sort by score
        sorted_ids = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        
        results = [
            RetrievalResult(
                chunk=self.chunks[cid],
                score=scores[cid],
                strategy=RetrievalStrategy.VECTOR,
            )
            for cid in sorted_ids[:request.top_k]
        ]
        
        took_ms = (time.time() - start_time) * 1000
        
        return SearchResult(
            results=results,
            query=request.query,
            strategy=RetrievalStrategy.VECTOR,
            took_ms=took_ms,
            total_hits=len(results),
        )
    
    def _matches_filters(self, chunk: DocumentChunk, filters: Dict[str, Any]) -> bool:
        """Check if chunk matches filters."""
        for key, value in filters.items():
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
    
    async def close(self) -> None:
        """Cleanup resources."""
        self.chunks.clear()
        self.vectors.clear()
        self._initialized = False


# Register the strategy
from ip_sakti.retrieval.strategies.base import RetrievalStrategyFactory
RetrievalStrategyFactory.register(RetrievalStrategy.VECTOR.value, VectorRetriever)