"""
Phase 4: Advanced Reranking Module
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from ip_sakti.core.models import DocumentChunk, RetrievalResult

logger = logging.getLogger(__name__)


class RerankerType(str, Enum):
    """Types of rerankers."""
    CROSS_ENCODER = "cross_encoder"
    LLM_BASED = "llm_based"
    HYBRID = "hybrid"
    DIVERSITY = "diversity"
    MMR = "mmr"  # Maximal Marginal Relevance


@dataclass
class RerankerConfig:
    """Configuration for rerankers."""
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = 32
    max_length: int = 512
    device: str = "auto"
    threshold: float = 0.0
    # For MMR
    mmr_lambda: float = 0.5
    # For diversity
    diversity_threshold: float = 0.7


@dataclass
class RerankResult:
    """Result of reranking."""
    results: List[RetrievalResult]
    scores: List[float]
    took_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseReranker(ABC):
    """Abstract base class for rerankers."""
    
    def __init__(self, config: RerankerConfig):
        self.config = config
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the reranker."""
        pass
    
    @abstractmethod
    async def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> RerankResult:
        """Rerank results."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Cleanup resources."""
        pass


class CrossEncoderReranker(BaseReranker):
    """Cross-encoder based reranker using sentence-transformers."""
    
    def __init__(self, config: RerankerConfig):
        super().__init__(config)
        self._model = None
    
    async def initialize(self) -> None:
        try:
            from sentence_transformers import CrossEncoder
            import torch
            
            if self.config.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.config.device
            
            self._model = CrossEncoder(self.config.model_name, device=device)
            self._initialized = True
            logger.info(f"Cross-encoder reranker initialized: {self.config.model_name} on {device}")
        except ImportError:
            logger.warning("sentence-transformers not available, using fallback")
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize cross-encoder: {e}")
            self._initialized = True
    
    async def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> RerankResult:
        import time
        start = time.time()
        
        if not self._initialized:
            await self.initialize()
        
        if not results:
            return RerankResult(results=[], scores=[], took_ms=0)
        
        if self._model is None:
            # Fallback: return original order
            final = results[:top_k] if top_k else results
            return RerankResult(
                results=final,
                scores=[r.score for r in final],
                took_ms=(time.time() - start) * 1000,
                metadata={"fallback": True},
            )
        
        # Prepare pairs
        pairs = [(query, r.chunk.content) for r in results]
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            lambda: self._model.predict(pairs, batch_size=self.config.batch_size)
        )
        
        # Apply threshold and sort
        scored_results = list(zip(results, scores))
        scored_results = [(r, s) for r, s in scored_results if s >= self.config.threshold]
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        final_results = [r for r, _ in scored_results]
        final_scores = [s for _, s in scored_results]
        
        if top_k:
            final_results = final_results[:top_k]
            final_scores = final_scores[:top_k]
        
        took_ms = (time.time() - start) * 1000
        
        return RerankResult(
            results=final_results,
            scores=final_scores,
            took_ms=took_ms,
            metadata={
                "model": self.config.model_name,
                "original_count": len(results),
                "threshold": self.config.threshold,
            }
        )
    
    async def close(self) -> None:
        self._model = None
        self._initialized = False


class MMRReranker(BaseReranker):
    """Maximal Marginal Relevance reranker for diversity."""
    
    def __init__(self, config: RerankerConfig):
        super().__init__(config)
        self._embedding_model = None
    
    async def initialize(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            if self.config.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.config.device
            
            # Use a lightweight embedding model for similarity
            self._embedding_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device=device
            )
            self._initialized = True
            logger.info(f"MMR reranker initialized on {device}")
        except ImportError:
            logger.warning("sentence-transformers not available, using fallback")
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize MMR reranker: {e}")
            self._initialized = True
    
    async def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> RerankResult:
        import time
        start = time.time()
        
        if not self._initialized:
            await self.initialize()
        
        if not results:
            return RerankResult(results=[], scores=[], took_ms=0)
        
        if self._embedding_model is None:
            final = results[:top_k] if top_k else results
            return RerankResult(
                results=final,
                scores=[r.score for r in final],
                took_ms=(time.time() - start) * 1000,
                metadata={"fallback": True},
            )
        
        k = top_k or len(results)
        lambda_param = self.config.mmr_lambda
        
        # Get embeddings
        loop = asyncio.get_event_loop()
        query_emb = await loop.run_in_executor(
            None,
            lambda: self._embedding_model.encode([query], show_progress_bar=False)
        )
        query_emb = query_emb[0]
        
        doc_embs = await loop.run_in_executor(
            None,
            lambda: self._embedding_model.encode([r.chunk.content for r in results], show_progress_bar=False)
        )
        
        # MMR algorithm
        selected = []
        remaining = list(range(len(results)))
        
        while remaining and len(selected) < k:
            best_idx = None
            best_score = -1
            
            for idx in remaining:
                relevance = results[idx].score
                diversity = 0.0
                
                if selected:
                    # Max similarity to already selected
                    similarities = [
                        self._cosine_similarity(doc_embs[idx], doc_embs[s])
                        for s in selected
                    ]
                    diversity = max(similarities) if similarities else 0
                
                mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
        
        final_results = [results[i] for i in selected]
        final_scores = [best_score] * len(final_results)  # Placeholder
        
        took_ms = (time.time() - start) * 1000
        
        return RerankResult(
            results=final_results,
            scores=final_scores,
            took_ms=took_ms,
            metadata={
                "lambda": lambda_param,
                "original_count": len(results),
            }
        )
    
    def _cosine_similarity(self, a, b):
        import numpy as np
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    async def close(self) -> None:
        self._embedding_model = None
        self._initialized = False


class DiversityReranker(BaseReranker):
    """Diversity-based reranker using embedding similarity."""
    
    def __init__(self, config: RerankerConfig):
        super().__init__(config)
        self._embedding_model = None
    
    async def initialize(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            if self.config.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.config.device
            
            self._embedding_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device=device
            )
            self._initialized = True
        except ImportError:
            logger.warning("sentence-transformers not available")
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize diversity reranker: {e}")
            self._initialized = True
    
    async def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> RerankResult:
        import time
        start = time.time()
        
        if not self._initialized:
            await self.initialize()
        
        if not results:
            return RerankResult(results=[], scores=[], took_ms=0)
        
        if self._embedding_model is None or top_k is None or top_k >= len(results):
            final = results[:top_k] if top_k else results
            return RerankResult(
                results=final,
                scores=[r.score for r in final],
                took_ms=(time.time() - start) * 1000,
            )
        
        # Get embeddings
        loop = asyncio.get_event_loop()
        doc_embs = await loop.run_in_executor(
            None,
            lambda: self._embedding_model.encode([r.chunk.content for r in results], show_progress_bar=False)
        )
        
        # Greedy diversity selection
        threshold = self.config.diversity_threshold
        selected = [0]  # Always keep top result
        
        for i in range(1, min(len(results), top_k * 3)):  # Consider more candidates
            if len(selected) >= top_k:
                break
            
            # Check diversity against selected
            emb_i = doc_embs[i]
            max_sim = max(
                self._cosine_similarity(emb_i, doc_embs[j])
                for j in selected
            )
            
            if max_sim < threshold:
                selected.append(i)
        
        final_results = [results[i] for i in selected[:top_k]]
        
        took_ms = (time.time() - start) * 1000
        
        return RerankResult(
            results=final_results,
            scores=[r.score for r in final_results],
            took_ms=took_ms,
            metadata={
                "threshold": threshold,
                "original_count": len(results),
            }
        )
    
    def _cosine_similarity(self, a, b):
        import numpy as np
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    async def close(self) -> None:
        self._embedding_model = None
        self._initialized = False


class RerankerFactory:
    """Factory for creating rerankers."""
    
    _rerankers: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, reranker_class: type) -> None:
        cls._rerankers[name] = reranker_class
    
    @classmethod
    def create(cls, config: RerankerConfig, reranker_type: str = "cross_encoder") -> BaseReranker:
        if reranker_type not in cls._rerankers:
            raise ValueError(f"Unknown reranker type: {reranker_type}")
        return cls._rerankers[reranker_type](config)
    
    @classmethod
    def get_available(cls) -> List[str]:
        return list(cls._rerankers.keys())


# Register built-in rerankers
RerankerFactory.register(RerankerType.CROSS_ENCODER.value, CrossEncoderReranker)
RerankerFactory.register(RerankerType.MMR.value, MMRReranker)
RerankerFactory.register(RerankerType.DIVERSITY.value, DiversityReranker)