"""
Phase 3: Multi-Strategy Retrieval Enhancement - Base Classes
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ip_sakti.core.models import DocumentChunk, RetrievalResult, RetrievalStrategy

logger = logging.getLogger(__name__)


class RetrievalConfig:
    """Configuration for retrieval strategies."""
    
    def __init__(
        self,
        top_k: int = 10,
        hybrid_alpha: float = 0.5,
        vector_weight: float = 0.6,
        lexical_weight: float = 0.4,
        rrf_k: int = 60,
        enable_reranking: bool = True,
        timeout_seconds: float = 5.0,
        max_concurrent: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ):
        self.top_k = top_k
        self.hybrid_alpha = hybrid_alpha
        self.vector_weight = vector_weight
        self.lexical_weight = lexical_weight
        self.rrf_k = rrf_k
        self.enable_reranking = enable_reranking
        self.timeout_seconds = timeout_seconds
        self.max_concurrent = max_concurrent
        self.filters = filters or {}


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
    variants: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Search response."""
    results: List[RetrievalResult]
    query: str
    strategy: RetrievalStrategy
    took_ms: float
    total_hits: int
    cached: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class RetrievalStrategy(ABC):
    """Abstract base for retrieval strategies."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass
    
    @abstractmethod
    async def search(self, request: SearchRequest) -> SearchResult:
        """Execute search."""
        pass
    
    @abstractmethod
    async def index(self, chunks: List[DocumentChunk]) -> None:
        """Index documents."""
        pass
    
    async def close(self) -> None:
        """Cleanup resources."""
        pass


class RetrievalStrategyFactory:
    """Factory for creating retrieval strategies."""
    
    _strategies: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, strategy_class: type) -> None:
        """Register a strategy class."""
        cls._strategies[name] = strategy_class
    
    @classmethod
    def create(cls, name: str, config: RetrievalConfig, **kwargs) -> RetrievalStrategy:
        """Create a strategy instance."""
        if name not in cls._strategies:
            raise ValueError(f"Unknown strategy: {name}")
        return cls._strategies[name](config, **kwargs)
    
    @classmethod
    def get_available(cls) -> List[str]:
        """Get available strategy names."""
        return list(cls._strategies.keys())