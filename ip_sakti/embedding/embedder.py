"""
IP-SAKTI Embedding Module
Phase 3: Embedding generation using sentence-transformers.
Supports bge-m3 (primary), e5-large-v2 (fallback), and other models.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingModelType(str, Enum):
    """Supported embedding model types."""
    BGE_M3 = "BAAI/bge-m3"
    E5_LARGE_V2 = "intfloat/e5-large-v2"
    BGE_LARGE_EN = "BAAI/bge-large-en-v1.5"
    BGE_BASE_EN = "BAAI/bge-base-en-v1.5"
    JINA_V3 = "jinaai/jina-embeddings-v3"
    JINA_V2_BASE_EN = "jinaai/jina-embeddings-v2-base-en"
    GTE_LARGE = "thenlper/gte-large"
    GTE_BASE = "thenlper/gte-base"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    model_name: str = EmbeddingModelType.BGE_M3
    fallback_model: str = EmbeddingModelType.E5_LARGE_V2
    dimension: int = 1024
    max_tokens: int = 8192
    batch_size: int = 32
    device: str = "auto"  # auto, cpu, cuda, mps
    normalize: bool = True
    use_fp16: bool = True
    trust_remote_code: bool = False


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    embeddings: np.ndarray
    model_name: str
    dimension: int
    token_counts: List[int]
    processing_time_ms: float


class EmbeddingProvider(ABC):
    """Abstract base for embedding providers."""
    
    @abstractmethod
    async def embed(self, texts: List[str]) -> EmbeddingResult:
        """Generate embeddings for a list of texts."""
        pass
    
    @abstractmethod
    async def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query."""
        pass
    
    @abstractmethod
    async def embed_documents(self, documents: List[str]) -> np.ndarray:
        """Generate embeddings for documents (may use different prefix/prompt)."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model name."""
        pass


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using sentence-transformers library."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._model = None
        self._fallback_model = None
        self._model_loaded = False
        self._fallback_loaded = False
    
    def _load_model(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        
        logger.info(f"Loading embedding model: {model_name}")
        model = SentenceTransformer(
            model_name,
            device=self.config.device,
            trust_remote_code=self.config.trust_remote_code
        )
        
        if self.config.use_fp16 and model.device.type != 'cpu':
            model.half()
        
        return model
    
    def _ensure_model_loaded(self):
        """Ensure primary model is loaded."""
        if not self._model_loaded:
            self._model = self._load_model(self.config.model_name)
            self._model_loaded = True
    
    def _ensure_fallback_loaded(self):
        """Ensure fallback model is loaded."""
        if not self._fallback_loaded:
            self._fallback_model = self._load_model(self.config.fallback_model)
            self._fallback_loaded = True
    
    @property
    def dimension(self) -> int:
        return self.config.dimension
    
    @property
    def model_name(self) -> str:
        return self.config.model_name
    
    def _embed_batch(self, texts: List[str], model, is_query: bool = False) -> np.ndarray:
        """Embed a batch of texts using the given model."""
        import time
        start_time = time.perf_counter()
        
        # For bge-m3 and e5, we might want different prompts for queries vs documents
        # bge-m3 handles this internally, e5 uses "query: " and "passage: " prefixes
        if is_query and ("e5" in self.config.model_name.lower() or "gte" in self.config.model_name.lower()):
            texts = [f"query: {t}" for t in texts]
        elif not is_query and ("e5" in self.config.model_name.lower() or "gte" in self.config.model_name.lower()):
            texts = [f"passage: {t}" for t in texts]
        
        # Tokenize to get token counts
        token_counts = []
        for text in texts:
            tokens = model.tokenize([text])
            token_counts.append(len(tokens['input_ids'][0]))
        
        # Generate embeddings
        embeddings = model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
        )
        
        elapsed = (time.perf_counter() - start_time) * 1000
        
        return embeddings, token_counts, elapsed
    
    async def embed(self, texts: List[str]) -> EmbeddingResult:
        """Generate embeddings for a list of texts using primary model."""
        self._ensure_model_loaded()
        
        # Run in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        embeddings, token_counts, elapsed = await loop.run_in_executor(
            None, self._embed_batch, texts, self._model, False
        )
        
        return EmbeddingResult(
            embeddings=embeddings,
            model_name=self.config.model_name,
            dimension=self.config.dimension,
            token_counts=token_counts,
            processing_time_ms=elapsed,
        )
    
    async def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query."""
        result = await self.embed([query])
        return result.embeddings[0]
    
    async def embed_documents(self, documents: List[str]) -> np.ndarray:
        """Generate embeddings for documents."""
        result = await self.embed(documents)
        return result.embeddings
    
    async def embed_with_fallback(self, texts: List[str]) -> EmbeddingResult:
        """Try primary model, fallback on failure."""
        try:
            return await self.embed(texts)
        except Exception as e:
            logger.warning(f"Primary model {self.config.model_name} failed: {e}, trying fallback")
            self._ensure_fallback_loaded()
            
            loop = asyncio.get_event_loop()
            embeddings, token_counts, elapsed = await loop.run_in_executor(
                None, self._embed_batch, texts, self._fallback_model, False
            )
            
            return EmbeddingResult(
                embeddings=embeddings,
                model_name=self.config.fallback_model,
                dimension=self.config.dimension,
                token_counts=token_counts,
                processing_time_ms=elapsed,
            )


class HybridEmbeddingProvider(EmbeddingProvider):
    """Hybrid provider that can use multiple models for different purposes."""
    
    def __init__(self, configs: Dict[str, EmbeddingConfig]):
        self.configs = configs
        self._providers: Dict[str, SentenceTransformerEmbeddingProvider] = {}
    
    def _get_provider(self, model_key: str) -> SentenceTransformerEmbeddingProvider:
        if model_key not in self._providers:
            self._providers[model_key] = SentenceTransformerEmbeddingProvider(self.configs[model_key])
        return self._providers[model_key]
    
    @property
    def dimension(self) -> int:
        # Return primary model dimension
        primary = self.configs.get('primary', list(self.configs.values())[0])
        return primary.dimension
    
    @property
    def model_name(self) -> str:
        return "hybrid"
    
    async def embed(self, texts: List[str]) -> EmbeddingResult:
        """Embed using primary model."""
        provider = self._get_provider('primary')
        return await provider.embed(texts)
    
    async def embed_query(self, query: str) -> np.ndarray:
        provider = self._get_provider('primary')
        return await provider.embed_query(query)
    
    async def embed_documents(self, documents: List[str]) -> np.ndarray:
        provider = self._get_provider('primary')
        return await provider.embed_documents(documents)
    
    async def embed_with_model(self, texts: List[str], model_key: str) -> EmbeddingResult:
        """Embed using a specific model."""
        provider = self._get_provider(model_key)
        return await provider.embed(texts)


def create_embedding_provider(config: Optional[EmbeddingConfig] = None) -> EmbeddingProvider:
    """Factory to create embedding provider from config."""
    if config is None:
        config = EmbeddingConfig()
    return SentenceTransformerEmbeddingProvider(config)


def create_hybrid_embedding_provider(settings: Any) -> HybridEmbeddingProvider:
    """Create hybrid embedding provider from settings."""
    configs = {
        'primary': EmbeddingConfig(
            model_name=getattr(settings, 'embedding_model', 'BAAI/bge-m3'),
            dimension=getattr(settings, 'embedding_dimensions', 1024),
            max_tokens=getattr(settings, 'embedding_max_tokens', 8192),
            batch_size=getattr(settings, 'embedding_batch_size', 32),
        ),
        'fallback': EmbeddingConfig(
            model_name=getattr(settings, 'embedding_fallback', 'intfloat/e5-large-v2'),
            dimension=getattr(settings, 'embedding_dimensions', 1024),
            max_tokens=getattr(settings, 'embedding_max_tokens', 8192),
            batch_size=getattr(settings, 'embedding_batch_size', 32),
        ),
    }
    return HybridEmbeddingProvider(configs)


# Async wrapper for compatibility with existing retrieval engine interface
class AsyncEmbeddingProvider:
    """Async wrapper for EmbeddingProvider to match retrieval_engine interface."""
    
    def __init__(self, provider: EmbeddingProvider):
        self.provider = provider
    
    async def embed(self, texts: List[str]) -> np.ndarray:
        result = await self.provider.embed(texts)
        return result.embeddings
    
    async def embed_query(self, query: str) -> np.ndarray:
        return await self.provider.embed_query(query)
    
    @property
    def dimension(self) -> int:
        return self.provider.dimension


# Alias for backward compatibility
SentenceTransformerProvider = SentenceTransformerEmbeddingProvider