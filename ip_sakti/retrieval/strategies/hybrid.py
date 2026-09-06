"""
Phase 3: Hybrid Retrieval Strategy
"""

import asyncio
import logging
import time
import numpy as np
from typing import Any, Dict, List, Optional, Set, Tuple

from ip_sakti.core.models import DocumentChunk, RetrievalResult, RetrievalStrategy
from ip_sakti.retrieval.strategies.base import RetrievalConfig, SearchRequest, SearchResult, RetrievalStrategy as BaseRetrievalStrategy
from ip_sakti.retrieval.strategies.lexical import LexicalRetriever
from ip_sakti.retrieval.strategies.vector import VectorRetriever

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetrievalStrategy):
    """Hybrid retrieval combining lexical and vector search."""
    
    def __init__(self, config: RetrievalConfig, lexical: Optional[LexicalRetriever] = None, vector: Optional[VectorRetriever] = None, **kwargs):
        self.config = config
        self.lexical = lexical or LexicalRetriever(config)
        self.vector = vector or VectorRetriever(config)
        self.fusion_method = config.filters.get('fusion_method', 'rrf')  # 'rrf' or 'weighted'
        self._initialized = False
    
    @property
    def name(self) -> str:
        return RetrievalStrategy.HYBRID.value
    
    async def index(self, chunks: List[DocumentChunk]) -> None:
        """Index chunks into both lexical and vector stores."""
        await asyncio.gather(
            self.lexical.index(chunks),
            self.vector.index(chunks),
        )
        self._initialized = True
        logger.info(f"Hybrid index built for {len(chunks)} chunks")
    
    async def search(self, request: SearchRequest) -> SearchResult:
        """Execute hybrid search."""
        start_time = time.time()
        
        if not self._initialized:
            # Try to search anyway
            pass
        
        top_k = request.top_k
        alpha = request.alpha if request.alpha is not None else self.config.hybrid_alpha
        
        # Run both searches in parallel
        lexical_request = SearchRequest(
            query=request.query,
            strategy=RetrievalStrategy.LEXICAL,
            filters=request.filters,
            top_k=top_k * 2,
            rerank=False,
        )
        vector_request = SearchRequest(
            query=request.query,
            strategy=RetrievalStrategy.VECTOR,
            filters=request.filters,
            top_k=top_k * 2,
            rerank=False,
        )
        
        lexical_response, vector_response = await asyncio.gather(
            self.lexical.search(lexical_request),
            self.vector.search(vector_request),
        )
        
        # Fuse results
        if self.fusion_method == 'rrf':
            fused_results = self._reciprocal_rank_fusion(
                lexical_response.results,
                vector_response.results,
                top_k,
            )
        else:
            fused_results = self._weighted_fusion(
                lexical_response.results,
                vector_response.results,
                alpha,
                top_k,
            )
        
        took_ms = (time.time() - start_time) * 1000
        
        return SearchResult(
            results=fused_results,
            query=request.query,
            strategy=RetrievalStrategy.HYBRID,
            took_ms=took_ms,
            total_hits=len(fused_results),
            metadata={
                'lexical_time_ms': lexical_response.took_ms,
                'vector_time_ms': vector_response.took_ms,
                'fusion_method': self.fusion_method,
                'alpha': alpha,
            }
        )
    
    def _reciprocal_rank_fusion(
        self,
        lexical_results: List[RetrievalResult],
        vector_results: List[RetrievalResult],
        top_k: int,
        k: int = 60,
    ) -> List[RetrievalResult]:
        """Reciprocal Rank Fusion (RRF) for combining results."""
        # Create rank maps
        lexical_ranks = {r.chunk.id: i + 1 for i, r in enumerate(lexical_results)}
        vector_ranks = {r.chunk.id: i + 1 for i, r in enumerate(vector_results)}
        
        all_chunk_ids = set(lexical_ranks.keys()) | set(vector_ranks.keys())
        
        fused_scores = {}
        for chunk_id in all_chunk_ids:
            lex_rank = lexical_ranks.get(chunk_id, float('inf'))
            vec_rank = vector_ranks.get(chunk_id, float('inf'))
            
            rrf_score = 0.0
            if lex_rank != float('inf'):
                rrf_score += 1.0 / (k + lex_rank)
            if vec_rank != float('inf'):
                rrf_score += 1.0 / (k + vec_rank)
            
            fused_scores[chunk_id] = rrf_score
        
        # Get chunk objects
        chunk_map = {}
        for r in lexical_results + vector_results:
            if r.chunk.id not in chunk_map:
                chunk_map[r.chunk.id] = r.chunk
        
        # Sort and create results
        sorted_ids = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)
        
        return [
            RetrievalResult(
                chunk=chunk_map[cid],
                score=fused_scores[cid],
                strategy=RetrievalStrategy.HYBRID,
            )
            for cid in sorted_ids[:top_k]
        ]
    
    def _weighted_fusion(
        self,
        lexical_results: List[RetrievalResult],
        vector_results: List[RetrievalResult],
        alpha: float,
        top_k: int,
    ) -> List[RetrievalResult]:
        """Weighted score fusion."""
        # Normalize scores to [0, 1]
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
        
        lex_scores = normalize(lexical_results)
        vec_scores = normalize(vector_results)
        
        all_chunk_ids = set(lex_scores.keys()) | set(vec_scores.keys())
        
        chunk_map = {}
        for r in lexical_results + vector_results:
            if r.chunk.id not in chunk_map:
                chunk_map[r.chunk.id] = r.chunk
        
        fused = []
        for chunk_id in all_chunk_ids:
            lex_score = lex_scores.get(chunk_id, 0)
            vec_score = vec_scores.get(chunk_id, 0)
            
            combined = alpha * vec_score + (1 - alpha) * lex_score
            
            if chunk_id in chunk_map:
                fused.append(RetrievalResult(
                    chunk=chunk_map[chunk_id],
                    score=combined,
                    strategy=RetrievalStrategy.HYBRID,
                ))
        
        fused.sort(key=lambda r: r.score, reverse=True)
        return fused[:top_k]
    
    async def close(self) -> None:
        """Cleanup resources."""
        await asyncio.gather(
            self.lexical.close(),
            self.vector.close(),
        )
        self._initialized = False


# Register the strategy
from ip_sakti.retrieval.strategies.base import RetrievalStrategyFactory
RetrievalStrategyFactory.register(RetrievalStrategy.HYBRID.value, HybridRetriever)