"""
Ingestion interfaces for IP-SAKTI.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, BinaryIO
from pydantic import BaseModel, Field
from enum import Enum


class Document(BaseModel):
    """Document model for ingestion."""
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source: Optional[str] = None
    chunks: List[str] = Field(default_factory=list)


class DocumentMetadata(BaseModel):
    """Document metadata."""
    title: str = ""
    document_type: str = "legal"
    jurisdiction: str = "INDIA"
    language: str = "en"
    mime_type: Optional[str] = None
    source_path: Optional[str] = None
    version: str = "1.0"
    effective_date: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    custom_fields: Dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """Chunk model for ingestion."""
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source: Optional[str] = None
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0
    authority_score: float = 0.0
    jurisdiction: Optional[str] = None
    document_type: Optional[str] = None
    embedding: Optional[List[float]] = None


class LoaderType(str, Enum):
    """Document loader types."""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    HTML = "html"
    MARKDOWN = "markdown"
    CSV = "csv"
    JSON = "json"
    XML = "xml"
    PPTX = "pptx"
    XLSX = "xlsx"


class ChunkingStrategy(str, Enum):
    """Chunking strategies."""
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    LEGAL = "legal"
    MARKDOWN = "markdown"
    FIXED = "fixed"
    HIERARCHICAL = "hierarchical"
    MULTIMODAL = "multimodal"


class DocumentType(str, Enum):
    """Document types for ingestion."""
    LEGAL = "legal"
    REGULATORY = "regulatory"
    CASE_LAW = "case_law"
    STATUTE = "statute"
    REGULATION = "regulation"
    TREATY = "treaty"
    GUIDELINE = "guideline"
    FORM = "form"
    OTHER = "other"


class ProcessingStatus(str, Enum):
    """Processing status for ingestion."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class IngestionRequest(BaseModel):
    """Request for document ingestion."""
    source: str  # file path, URL, or content
    source_type: LoaderType
    metadata: Optional[DocumentMetadata] = None
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.LEGAL
    chunk_size: int = Field(default=800, ge=100, le=4000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    extract_tables: bool = True
    extract_images: bool = False
    generate_contextual_embeddings: bool = True
    assign_authority_tier: bool = True
    jurisdiction: Optional[str] = None


class IngestionResult(BaseModel):
    """Result from document ingestion."""
    document: Document
    chunks: List[Chunk]
    total_chunks: int
    processing_time_ms: float
    tables_extracted: int = 0
    images_extracted: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IDocumentLoader(ABC):
    """Interface for document loaders."""
    
    @property
    @abstractmethod
    def supported_types(self) -> List[LoaderType]:
        """Supported document types."""
        pass
    
    @abstractmethod
    async def load(self, source: str, source_type: LoaderType, **kwargs) -> Document:
        """Load document from source."""
        pass
    
    @abstractmethod
    async def load_from_bytes(self, content: bytes, source_type: LoaderType, **kwargs) -> Document:
        """Load document from bytes."""
        pass
    
    @abstractmethod
    async def extract_tables(self, document: Document) -> List[Dict[str, Any]]:
        """Extract tables from document."""
        pass
    
    @abstractmethod
    async def extract_images(self, document: Document) -> List[Dict[str, Any]]:
        """Extract images from document."""
        pass


class IChunker(ABC):
    """Interface for chunkers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Chunker name."""
        pass
    
    @property
    @abstractmethod
    def supported_strategies(self) -> List[ChunkingStrategy]:
        """Supported chunking strategies."""
        pass
    
    @abstractmethod
    async def chunk(self, document: Document, strategy: ChunkingStrategy, chunk_size: int, chunk_overlap: int, **kwargs) -> List[Chunk]:
        """Chunk document into pieces."""
        pass
    
    @abstractmethod
    async def chunk_text(self, text: str, strategy: ChunkingStrategy, chunk_size: int, chunk_overlap: int, **kwargs) -> List[Chunk]:
        """Chunk raw text."""
        pass


class IDocumentValidator(ABC):
    """Interface for document validation."""
    
    @abstractmethod
    async def validate(self, document: Document) -> bool:
        """Validate document meets requirements."""
        pass
    
    @abstractmethod
    async def get_validation_errors(self, document: Document) -> List[str]:
        """Get list of validation errors."""
        pass


class IngestionMetrics(BaseModel):
    """Metrics for ingestion evaluation."""
    documents_per_sec: float = 0.0
    chunks_per_document: float = 0.0
    avg_chunk_size: float = 0.0
    table_extraction_rate: float = 0.0
    processing_time_ms: float = 0.0