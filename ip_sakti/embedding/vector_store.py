"""
IP-SAKTI Vector Store Module
Phase 3: Vector store abstraction with ChromaDB (persistent) and FAISS (in-memory) support.
"""

import asyncio
import logging
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class VectorStoreType(str, Enum):
    """Supported vector store types."""
    QDRANT = "qdrant"
    CHROMA = "chroma"
    FAISS = "faiss"
    IN_MEMORY = "in_memory"


@dataclass
class VectorStoreConfig:
    """Configuration for vector stores."""
    store_type: VectorStoreType = VectorStoreType.QDRANT
    collection_name: str = "ip_sakti_chunks"
    persist_directory: str = "./chroma_db"
    
    # Qdrant specific
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_api_key: Optional[str] = None
    qdrant_https: bool = False
    qdrant_prefer_grpc: bool = True
    qdrant_hnsw_m: int = 16
    qdrant_hnsw_ef_construct: int = 100
    qdrant_quantization: str = "scalar"  # scalar, binary, none
    
    # ChromaDB specific
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_ssl: bool = False
    
    # FAISS specific
    faiss_index_path: str = "./faiss_index"
    faiss_index_type: str = "flat"  # flat, ivf, hnsw
    faiss_nlist: int = 100
    faiss_nprobe: int = 10
    
    # Common
    dimension: int = 1024
    metric: str = "cosine"  # cosine, l2, ip
    batch_size: int = 100


@dataclass
class SearchResult:
    """Vector search result."""
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    vector: Optional[np.ndarray] = None


@dataclass
class VectorStoreStats:
    """Vector store statistics."""
    total_vectors: int
    dimension: int
    store_type: str
    collection_name: str
    disk_usage_mb: float = 0.0
    index_size: int = 0


class VectorStore(ABC):
    """Abstract base for vector stores."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store."""
        pass
    
    @abstractmethod
    async def upsert(
        self,
        ids: List[str],
        vectors: np.ndarray,
        contents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """Insert or update vectors with content and metadata."""
        pass
    
    @abstractmethod
    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        include_vectors: bool = False,
    ) -> List[SearchResult]:
        """Search by vector similarity."""
        pass
    
    @abstractmethod
    async def delete(self, ids: List[str]) -> None:
        """Delete vectors by IDs."""
        pass
    
    @abstractmethod
    async def get_stats(self) -> VectorStoreStats:
        """Get store statistics."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close connections."""
        pass


class InMemoryVectorStore(VectorStore):
    """In-memory vector store for development/testing."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self._vectors: Dict[str, np.ndarray] = {}
        self._contents: Dict[str, str] = {}
        self._metadatas: Dict[str, Dict[str, Any]] = {}
        self._initialized = False
    
    async def initialize(self) -> None:
        self._initialized = True
        logger.info("In-memory vector store initialized")
    
    async def upsert(
        self,
        ids: List[str],
        vectors: np.ndarray,
        contents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        for i, id in enumerate(ids):
            self._vectors[id] = vectors[i]
            self._contents[id] = contents[i]
            self._metadatas[id] = metadatas[i]
        logger.debug(f"Upserted {len(ids)} vectors to in-memory store")
    
    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        include_vectors: bool = False,
    ) -> List[SearchResult]:
        if not self._vectors:
            return []
        
        # Compute cosine similarity
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return []
        
        scores = {}
        for id, vector in self._vectors.items():
            # Apply filters
            if filters and not self._matches_filters(self._metadatas[id], filters):
                continue
            
            vec_norm = np.linalg.norm(vector)
            if vec_norm == 0:
                continue
            
            # Cosine similarity
            score = float(np.dot(query_vector, vector) / (query_norm * vec_norm))
            scores[id] = score
        
        # Sort by score
        sorted_ids = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        
        results = []
        for id in sorted_ids[:top_k]:
            result = SearchResult(
                id=id,
                content=self._contents[id],
                metadata=self._metadatas[id],
                score=scores[id],
                vector=self._vectors[id] if include_vectors else None,
            )
            results.append(result)
        
        return results
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        for key, value in filters.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        return True
    
    async def delete(self, ids: List[str]) -> None:
        for id in ids:
            self._vectors.pop(id, None)
            self._contents.pop(id, None)
            self._metadatas.pop(id, None)
    
    async def get_stats(self) -> VectorStoreStats:
        return VectorStoreStats(
            total_vectors=len(self._vectors),
            dimension=self.config.dimension,
            store_type="in_memory",
            collection_name=self.config.collection_name,
        )
    
    async def close(self) -> None:
        self._vectors.clear()
        self._contents.clear()
        self._metadatas.clear()


class ChromaVectorStore(VectorStore):
    """ChromaDB vector store for persistent storage."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self._client = None
        self._collection = None
        self._initialized = False
    
    async def initialize(self) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        
        # Create persist directory
        persist_dir = Path(self.config.persist_directory)
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize client
        if self.config.chroma_host and self.config.chroma_port:
            # HTTP client
            self._client = chromadb.HttpClient(
                host=self.config.chroma_host,
                port=self.config.chroma_port,
                ssl=self.config.chroma_ssl,
            )
        else:
            # Persistent client
            self._client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        
        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": self.config.metric},
        )
        
        self._initialized = True
        logger.info(f"ChromaDB initialized: {self.config.collection_name} at {persist_dir}")
    
    async def upsert(
        self,
        ids: List[str],
        vectors: np.ndarray,
        contents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        if not self._initialized:
            await self.initialize()
        
        # ChromaDB expects lists
        self._collection.upsert(
            ids=ids,
            embeddings=vectors.tolist(),
            documents=contents,
            metadatas=metadatas,
        )
        logger.debug(f"Upserted {len(ids)} vectors to ChromaDB")
    
    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        include_vectors: bool = False,
    ) -> List[SearchResult]:
        if not self._initialized:
            await self.initialize()
        
        # Prepare where clause for filters
        where = None
        if filters:
            where = self._build_where_clause(filters)
        
        # Search
        results = self._collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances", "embeddings"] if include_vectors else ["documents", "metadatas", "distances"],
        )
        
        # Convert to SearchResult
        search_results = []
        if results['ids'] and results['ids'][0]:
            for i, id in enumerate(results['ids'][0]):
                result = SearchResult(
                    id=id,
                    content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i],
                    score=1.0 - results['distances'][0][i],  # Convert distance to similarity
                    vector=np.array(results['embeddings'][0][i]) if include_vectors and 'embeddings' in results and results['embeddings'][0] else None,
                )
                search_results.append(result)
        
        return search_results
    
    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build ChromaDB where clause from filters."""
        where = {}
        for key, value in filters.items():
            if isinstance(value, list):
                where[key] = {"$in": value}
            else:
                where[key] = {"$eq": value}
        return where
    
    async def delete(self, ids: List[str]) -> None:
        if not self._initialized:
            await self.initialize()
        
        self._collection.delete(ids=ids)
    
    async def get_stats(self) -> VectorStoreStats:
        if not self._initialized:
            await self.initialize()
        
        count = self._collection.count()
        
        # Estimate disk usage
        disk_usage = 0.0
        persist_dir = Path(self.config.persist_directory)
        if persist_dir.exists():
            for f in persist_dir.rglob("*"):
                if f.is_file():
                    disk_usage += f.stat().st_size
            disk_usage /= 1024 * 1024  # MB
        
        return VectorStoreStats(
            total_vectors=count,
            dimension=self.config.dimension,
            store_type="chroma",
            collection_name=self.config.collection_name,
            disk_usage_mb=disk_usage,
        )
    
    async def close(self) -> None:
        # ChromaDB persistent client doesn't need explicit close
        pass


class QdrantVectorStore(VectorStore):
    """Qdrant vector store for high-performance persistent storage."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self._client = None
        self._initialized = False
    
    async def initialize(self) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models
        from qdrant_client.http.models import Distance, VectorParams, HnswConfigDiff, QuantizationConfig, ScalarQuantization
        
        # Initialize client
        if self.config.qdrant_prefer_grpc:
            self._client = QdrantClient(
                host=self.config.qdrant_host,
                grpc_port=self.config.qdrant_grpc_port,
                prefer_grpc=True,
                api_key=self.config.qdrant_api_key,
                https=self.config.qdrant_https,
            )
        else:
            self._client = QdrantClient(
                host=self.config.qdrant_host,
                port=self.config.qdrant_port,
                api_key=self.config.qdrant_api_key,
                https=self.config.qdrant_https,
            )
        
        # Check if collection exists, create if not
        collections = self._client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.config.collection_name not in collection_names:
            # Configure quantization
            quantization_config = None
            if self.config.qdrant_quantization == "scalar":
                quantization_config = QuantizationConfig(
                    scalar=ScalarQuantization(
                        type="int8",
                        quantile=0.99,
                        always_ram=True,
                    )
                )
            elif self.config.qdrant_quantization == "binary":
                quantization_config = QuantizationConfig(
                    binary=models.BinaryQuantization(
                        always_ram=True,
                    )
                )
            
            # Configure HNSW
            hnsw_config = HnswConfigDiff(
                m=self.config.qdrant_hnsw_m,
                ef_construct=self.config.qdrant_hnsw_ef_construct,
            )
            
            # Distance metric
            distance = Distance.COSINE
            if self.config.metric == "l2":
                distance = Distance.EUCLID
            elif self.config.metric == "ip":
                distance = Distance.DOT
            
            self._client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.config.dimension,
                    distance=distance,
                    hnsw_config=hnsw_config,
                    quantization_config=quantization_config,
                ),
            )
            logger.info(f"Created Qdrant collection: {self.config.collection_name}")
        else:
            logger.info(f"Using existing Qdrant collection: {self.config.collection_name}")
        
        self._initialized = True
    
    async def upsert(
        self,
        ids: List[str],
        vectors: np.ndarray,
        contents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        if not self._initialized:
            await self.initialize()
        
        from qdrant_client.http.models import PointStruct
        
        points = []
        for i, id in enumerate(ids):
            payload = {
                "content": contents[i],
                **metadatas[i],
            }
            points.append(PointStruct(
                id=id,
                vector=vectors[i].tolist(),
                payload=payload,
            ))
        
        # Batch upsert
        batch_size = self.config.batch_size
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self._client.upsert(
                collection_name=self.config.collection_name,
                points=batch,
                wait=True,
            )
        
        logger.debug(f"Upserted {len(ids)} vectors to Qdrant")
    
    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        include_vectors: bool = False,
    ) -> List[SearchResult]:
        if not self._initialized:
            await self.initialize()
        
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny
        
        # Build filter
        query_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if isinstance(value, list):
                    conditions.append(FieldCondition(
                        key=key,
                        match=MatchAny(any=value),
                    ))
                else:
                    conditions.append(FieldCondition(
                        key=key,
                        match=MatchValue(value=value),
                    ))
            query_filter = Filter(must=conditions)
        
        # Search
        search_results = self._client.search(
            collection_name=self.config.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=include_vectors,
        )
        
        # Convert to SearchResult
        results = []
        for hit in search_results:
            payload = hit.payload or {}
            content = payload.pop("content", "")
            
            result = SearchResult(
                id=str(hit.id),
                content=content,
                metadata=payload,
                score=hit.score,  # Qdrant returns similarity score (higher = better for cosine)
                vector=np.array(hit.vector) if include_vectors and hit.vector else None,
            )
            results.append(result)
        
        return results
    
    async def delete(self, ids: List[str]) -> None:
        if not self._initialized:
            await self.initialize()
        
        from qdrant_client.http.models import PointIdsList
        
        self._client.delete(
            collection_name=self.config.collection_name,
            points_selector=PointIdsList(points=ids),
            wait=True,
        )
    
    async def get_stats(self) -> VectorStoreStats:
        if not self._initialized:
            await self.initialize()
        
        info = self._client.get_collection(self.config.collection_name)
        
        return VectorStoreStats(
            total_vectors=info.vectors_count,
            dimension=self.config.dimension,
            store_type="qdrant",
            collection_name=self.config.collection_name,
            disk_usage_mb=0.0,  # Would need disk usage check
            index_size=info.vectors_count,
        )
    
    async def close(self) -> None:
        if self._client:
            self._client.close()


class FAISSVectorStore(VectorStore):
    """FAISS vector store for high-performance in-memory search."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self._index = None
        self._id_map: Dict[int, str] = {}  # FAISS index -> our ID
        self._reverse_id_map: Dict[str, int] = {}  # our ID -> FAISS index
        self._contents: Dict[str, str] = {}
        self._metadatas: Dict[str, Dict[str, Any]] = {}
        self._initialized = False
        self._next_index = 0
    
    def _create_index(self):
        """Create FAISS index based on config."""
        import faiss
        
        dim = self.config.dimension
        metric = self.config.metric
        
        if self.config.faiss_index_type == "flat":
            if metric == "cosine":
                # For cosine, use inner product with normalized vectors
                self._index = faiss.IndexFlatIP(dim)
            elif metric == "l2":
                self._index = faiss.IndexFlatL2(dim)
            else:  # ip
                self._index = faiss.IndexFlatIP(dim)
        elif self.config.faiss_index_type == "ivf":
            nlist = self.config.faiss_nlist
            quantizer = faiss.IndexFlatIP(dim) if metric != "l2" else faiss.IndexFlatL2(dim)
            self._index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT if metric != "l2" else faiss.METRIC_L2)
        elif self.config.faiss_index_type == "hnsw":
            self._index = faiss.IndexHNSWFlat(dim, 32)
            if metric != "l2":
                self._index.hnsw.efSearch = 128
        else:
            # Default to flat
            self._index = faiss.IndexFlatIP(dim)
    
    async def initialize(self) -> None:
        self._create_index()
        
        # Load existing index if exists
        index_path = Path(self.config.faiss_index_path)
        if index_path.exists():
            await self._load_index(index_path)
        
        self._initialized = True
        logger.info(f"FAISS index initialized: {self.config.faiss_index_type}")
    
    async def _load_index(self, index_path: Path):
        """Load FAISS index and metadata from disk."""
        import faiss
        import json
        
        # Load index
        self._index = faiss.read_index(str(index_path / "index.faiss"))
        
        # Load metadata
        meta_path = index_path / "metadata.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                data = json.load(f)
                self._id_map = {int(k): v for k, v in data.get('id_map', {}).items()}
                self._reverse_id_map = data.get('reverse_id_map', {})
                self._contents = data.get('contents', {})
                self._metadatas = data.get('metadatas', {})
                self._next_index = data.get('next_index', 0)
        
        logger.info(f"Loaded FAISS index with {self._index.ntotal} vectors")
    
    async def _save_index(self):
        """Save FAISS index and metadata to disk."""
        import faiss
        import json
        
        index_path = Path(self.config.faiss_index_path)
        index_path.mkdir(parents=True, exist_ok=True)
        
        # Save index
        faiss.write_index(self._index, str(index_path / "index.faiss"))
        
        # Save metadata
        meta_path = index_path / "metadata.json"
        with open(meta_path, 'w') as f:
            json.dump({
                'id_map': {str(k): v for k, v in self._id_map.items()},
                'reverse_id_map': self._reverse_id_map,
                'contents': self._contents,
                'metadatas': self._metadatas,
                'next_index': self._next_index,
            }, f)
    
    async def upsert(
        self,
        ids: List[str],
        vectors: np.ndarray,
        contents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        if not self._initialized:
            await self.initialize()
        
        # Normalize vectors for cosine similarity
        if self.config.metric == "cosine":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / (norms + 1e-8)
        
        # Train IVF index if needed
        if hasattr(self._index, 'is_trained') and not self._index.is_trained:
            # Need training data - use current vectors
            self._index.train(vectors)
        
        # Add to index
        start_idx = self._next_index
        self._index.add(vectors)
        
        # Update mappings
        for i, id in enumerate(ids):
            idx = start_idx + i
            self._id_map[idx] = id
            self._reverse_id_map[id] = idx
            self._contents[id] = contents[i]
            self._metadatas[id] = metadatas[i]
        
        self._next_index += len(ids)
        
        # Save to disk
        await self._save_index()
        
        logger.debug(f"Upserted {len(ids)} vectors to FAISS (total: {self._index.ntotal})")
    
    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        include_vectors: bool = False,
    ) -> List[SearchResult]:
        if not self._initialized:
            await self.initialize()
        
        if self._index.ntotal == 0:
            return []
        
        # Normalize query for cosine
        if self.config.metric == "cosine":
            query_norm = np.linalg.norm(query_vector)
            if query_norm > 0:
                query_vector = query_vector / query_norm
        
        # Set nprobe for IVF
        if hasattr(self._index, 'nprobe'):
            self._index.nprobe = self.config.faiss_nprobe
        
        # Search
        k = min(top_k * 3, self._index.ntotal)  # Get more for filtering
        distances, indices = self._index.search(query_vector.reshape(1, -1), k)
        
        # Convert to results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid
                continue
            
            id = self._id_map.get(idx)
            if not id:
                continue
            
            # Apply filters
            if filters and not self._matches_filters(self._metadatas.get(id, {}), filters):
                continue
            
            # Convert distance to similarity score
            if self.config.metric == "cosine" or self.config.metric == "ip":
                score = float(dist)  # Inner product (higher = better)
            else:
                score = 1.0 / (1.0 + float(dist))  # L2 distance to similarity
            
            result = SearchResult(
                id=id,
                content=self._contents.get(id, ""),
                metadata=self._metadatas.get(id, {}),
                score=score,
                vector=None,  # FAISS doesn't easily return vectors
            )
            results.append(result)
            
            if len(results) >= top_k:
                break
        
        return results
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        for key, value in filters.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        return True
    
    async def delete(self, ids: List[str]) -> None:
        # FAISS doesn't support easy deletion - mark as deleted in metadata
        # For full rebuild, we'd need to reconstruct the index
        for id in ids:
            if id in self._metadatas:
                self._metadatas[id]['_deleted'] = True
                self._contents.pop(id, None)
                # Remove from reverse map
                if id in self._reverse_id_map:
                    idx = self._reverse_id_map.pop(id)
                    self._id_map.pop(idx, None)
        
        await self._save_index()
    
    async def get_stats(self) -> VectorStoreStats:
        if not self._initialized:
            await self.initialize()
        
        # Count non-deleted
        active = sum(1 for m in self._metadatas.values() if not m.get('_deleted', False))
        
        # Disk usage
        disk_usage = 0.0
        index_path = Path(self.config.faiss_index_path)
        if index_path.exists():
            for f in index_path.rglob("*"):
                if f.is_file():
                    disk_usage += f.stat().st_size
            disk_usage /= 1024 * 1024  # MB
        
        return VectorStoreStats(
            total_vectors=active,
            dimension=self.config.dimension,
            store_type="faiss",
            collection_name=self.config.collection_name,
            disk_usage_mb=disk_usage,
            index_size=self._index.ntotal if self._index else 0,
        )
    
    async def close(self) -> None:
        await self._save_index()


def create_vector_store(config: Optional[VectorStoreConfig] = None) -> VectorStore:
    """Factory to create vector store from config."""
    if config is None:
        config = VectorStoreConfig()
    
    if config.store_type == VectorStoreType.CHROMA:
        return ChromaVectorStore(config)
    elif config.store_type == VectorStoreType.QDRANT:
        return QdrantVectorStore(config)
    elif config.store_type == VectorStoreType.FAISS:
        return FAISSVectorStore(config)
    else:
        return InMemoryVectorStore(config)


def create_vector_store_from_settings(settings: Any) -> VectorStore:
    """Create vector store from settings object."""
    store_type_str = getattr(settings, 'vector_db_type', 'qdrant')
    if store_type_str.lower() == 'qdrant':
        store_type = VectorStoreType.QDRANT
    elif store_type_str.lower() == 'faiss':
        store_type = VectorStoreType.FAISS
    elif store_type_str.lower() == 'chroma':
        store_type = VectorStoreType.CHROMA
    else:
        store_type = VectorStoreType.IN_MEMORY
    
    config = VectorStoreConfig(
        store_type=store_type,
        collection_name=getattr(settings, 'vector_db_collection', 'ip_sakti_chunks'),
        persist_directory=f"./chroma_db_{getattr(settings, 'deployment_environment', 'dev')}",
        chroma_host=getattr(settings, 'vector_db_host', 'localhost'),
        chroma_port=getattr(settings, 'vector_db_port', 8000),
        dimension=getattr(settings, 'embedding_dimensions', 1024),
        metric="cosine",
        # Qdrant-specific
        qdrant_host=getattr(settings, 'qdrant_host', 'localhost'),
        qdrant_port=getattr(settings, 'qdrant_port', 6333),
        qdrant_grpc_port=getattr(settings, 'qdrant_grpc_port', 6334),
        qdrant_prefer_grpc=getattr(settings, 'qdrant_prefer_grpc', True),
        qdrant_api_key=getattr(settings, 'qdrant_api_key', None),
        qdrant_https=getattr(settings, 'qdrant_https', False),
        qdrant_quantization=getattr(settings, 'qdrant_quantization', 'scalar'),
        qdrant_hnsw_m=getattr(settings, 'qdrant_hnsw_m', 16),
        qdrant_hnsw_ef_construct=getattr(settings, 'qdrant_hnsw_ef_construct', 128),
        # FAISS-specific
        faiss_index_path=f"./faiss_index_{getattr(settings, 'deployment_environment', 'dev')}",
        faiss_index_type=getattr(settings, 'faiss_index_type', 'IVF'),
        faiss_nlist=getattr(settings, 'faiss_nlist', 100),
        faiss_nprobe=getattr(settings, 'faiss_nprobe', 10),
    )
    return create_vector_store(config)


# Async wrapper for compatibility with existing retrieval engine interface
class AsyncVectorStore:
    """Async wrapper for VectorStore to match retrieval_engine interface."""
    
    def __init__(self, store: VectorStore):
        self.store = store
    
    async def initialize(self) -> None:
        await self.store.initialize()
    
    async def upsert(self, chunks: List[Any], vectors: np.ndarray) -> None:
        """Upsert chunks with vectors. Chunks are expected to have id, content, metadata."""
        ids = [chunk.id for chunk in chunks]
        contents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        await self.store.upsert(ids, vectors, contents, metadatas)
    
    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Any, float]]:
        """Search and return (chunk, score) tuples for compatibility."""
        results = await self.store.search(query_vector, top_k, filters, include_vectors=False)
        
        # Create mock chunk objects
        chunk_results = []
        for r in results:
            chunk = type('Chunk', (), {
                'id': r.id,
                'content': r.content,
                'metadata': r.metadata,
            })()
            chunk_results.append((chunk, r.score))
        
        return chunk_results
    
    async def delete(self, chunk_ids: List[str]) -> None:
        await self.store.delete(chunk_ids)
    
    async def get_stats(self) -> Dict[str, Any]:
        stats = await self.store.get_stats()
        return {
            'total_chunks': stats.total_vectors,
            'total_vectors': stats.total_vectors,
            'store_type': stats.store_type,
            'collection_name': stats.collection_name,
        }
    
    async def close(self) -> None:
        await self.store.close()