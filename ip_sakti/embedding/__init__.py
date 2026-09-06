"""
IP-SAKTI Embedding Module
Phase 3: Embedding generation and vector store abstraction.
Supports sentence-transformers (bge-m3, e5-large-v2) with Qdrant, ChromaDB, FAISS, and in-memory stores.
"""

from ip_sakti.embedding.embedder import (
    EmbeddingProvider,
    SentenceTransformerProvider,
    SentenceTransformerEmbeddingProvider,
    create_embedding_provider,
    create_hybrid_embedding_provider,
    EmbeddingConfig,
    EmbeddingModelType,
    EmbeddingResult,
    AsyncEmbeddingProvider,
)
from ip_sakti.embedding.vector_store import (
    VectorStore,
    VectorStoreConfig,
    VectorStoreType,
    SearchResult,
    VectorStoreStats,
    InMemoryVectorStore,
    ChromaVectorStore,
    FAISSVectorStore,
    QdrantVectorStore,
    AsyncVectorStore,
    create_vector_store,
    create_vector_store_from_settings,
)

__all__ = [
    # Embedder
    "EmbeddingProvider",
    "SentenceTransformerProvider",
    "SentenceTransformerEmbeddingProvider",
    "create_embedding_provider",
    "create_hybrid_embedding_provider",
    "EmbeddingConfig",
    "EmbeddingModelType",
    "EmbeddingResult",
    "AsyncEmbeddingProvider",
    # Vector Store
    "VectorStore",
    "VectorStoreConfig",
    "VectorStoreType",
    "SearchResult",
    "VectorStoreStats",
    "InMemoryVectorStore",
    "ChromaVectorStore",
    "FAISSVectorStore",
    "QdrantVectorStore",
    "AsyncVectorStore",
    "create_vector_store",
    "create_vector_store_from_settings",
]
