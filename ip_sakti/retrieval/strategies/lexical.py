"""
Phase 3: Lexical/BM25 Retrieval Strategy
"""

import asyncio
import logging
import re
import time
import numpy as np
from typing import Any, Dict, List, Optional, Set, Tuple

from ip_sakti.core.models import DocumentChunk, RetrievalResult, RetrievalStrategy
from ip_sakti.retrieval.strategies.base import RetrievalConfig, SearchRequest, SearchResult, RetrievalStrategy as BaseRetrievalStrategy

logger = logging.getLogger(__name__)


class LexicalRetriever(BaseRetrievalStrategy):
    """BM25-based lexical retrieval strategy."""
    
    def __init__(self, config: RetrievalConfig, **kwargs):
        self.config = config
        self.chunks: Dict[str, DocumentChunk] = {}
        self.inverted_index: Dict[str, Set[str]] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length = 0
        self.k1 = config.filters.get('bm25_k1', 1.5)
        self.b = config.filters.get('bm25_b', 0.75)
        self._initialized = False
    
    @property
    def name(self) -> str:
        return RetrievalStrategy.LEXICAL.value
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return [t.lower() for t in re.findall(r'\b\w+\b', text) if len(t) > 2]
    
    async def index(self, chunks: List[DocumentChunk]) -> None:
        """Index chunks into BM25 index."""
        start = time.time()
        
        for chunk in chunks:
            self.chunks[chunk.id] = chunk
            tokens = self._tokenize(chunk.content)
            self.doc_lengths[chunk.id] = len(tokens)
            
            # Update inverted index
            for token in set(tokens):
                if token not in self.inverted_index:
                    self.inverted_index[token] = set()
                self.inverted_index[token].add(chunk.id)
        
        # Update average doc length
        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)
        
        self._initialized = True
        logger.info(f"Lexical index built: {len(self.chunks)} chunks, {len(self.inverted_index)} terms in {time.time() - start:.2f}s")
    
    async def search(self, request: SearchRequest) -> SearchResult:
        """Execute BM25 search."""
        start_time = time.time()
        
        if not self._initialized or not self.chunks:
            return SearchResult(
                results=[],
                query=request.query,
                strategy=RetrievalStrategy.LEXICAL,
                took_ms=0,
                total_hits=0,
            )
        
        query_tokens = self._tokenize(request.query)
        if not query_tokens:
            return SearchResult(
                results=[],
                query=request.query,
                strategy=RetrievalStrategy.LEXICAL,
                took_ms=(time.time() - start_time) * 1000,
                total_hits=0,
            )
        
        # Find candidate chunks
        candidate_ids = set()
        for token in query_tokens:
            if token in self.inverted_index:
                candidate_ids.update(self.inverted_index[token])
        
        # Apply filters
        if request.filters:
            filtered_ids = set()
            for cid in candidate_ids:
                chunk = self.chunks.get(cid)
                if chunk and self._matches_filters(chunk, request.filters):
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
                
                df = len(self.inverted_index[token])
                idf = np.log((N - df + 0.5) / (df + 0.5) + 1)
                
                tf = self._tokenize(chunk.content).count(token)
                
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                score += idf * numerator / denominator
            
            scores[chunk_id] = score
        
        # Sort by score
        sorted_ids = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        
        results = [
            RetrievalResult(
                chunk=self.chunks[cid],
                score=scores[cid],
                strategy=RetrievalStrategy.LEXICAL,
            )
            for cid in sorted_ids[:request.top_k]
        ]
        
        took_ms = (time.time() - start_time) * 1000
        
        return SearchResult(
            results=results,
            query=request.query,
            strategy=RetrievalStrategy.LEXICAL,
            took_ms=took_ms,
            total_hits=len(results),
        )
    
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
        return True
    
    async def close(self) -> None:
        """Cleanup resources."""
        self.chunks.clear()
        self.inverted_index.clear()
        self.doc_lengths.clear()
        self._initialized = False


# Register the strategy
from ip_sakti.retrieval.strategies.base import RetrievalStrategyFactory
RetrievalStrategyFactory.register(RetrievalStrategy.LEXICAL.value, LexicalRetriever)
