"""
Multimodal interfaces for IP-SAKTI.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class ModalityType(str, Enum):
    """Supported modalities."""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    AUDIO = "audio"
    VIDEO = "video"


class MultimodalContent(BaseModel):
    """Multimodal content item."""
    type: ModalityType
    content: Any  # base64 for images, text for tables, etc.
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_ref: Optional[str] = None


class MultimodalDocument(BaseModel):
    """Document with multimodal content."""
    id: str
    text: str
    multimodal_content: List[MultimodalContent] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExtractionResult(BaseModel):
    """Result from multimodal extraction."""
    document: MultimodalDocument
    extracted_tables: List[Dict[str, Any]] = Field(default_factory=list)
    extracted_images: List[Dict[str, Any]] = Field(default_factory=list)
    extracted_formulas: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time_ms: float


class IMultimodalExtractor(ABC):
    """Interface for multimodal extraction."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Extractor name."""
        pass
    
    @property
    @abstractmethod
    def supported_modalities(self) -> List[ModalityType]:
        """Supported modalities."""
        pass
    
    @abstractmethod
    async def extract(self, source: str, source_type: str) -> ExtractionResult:
        """Extract multimodal content from source."""
        pass
    
    @abstractmethod
    async def extract_from_bytes(self, content: bytes, source_type: str) -> ExtractionResult:
        """Extract from bytes."""
        pass
    
    @abstractmethod
    async def extract_tables(self, document: MultimodalDocument) -> List[Dict[str, Any]]:
        """Extract tables from document."""
        pass
    
    @abstractmethod
    async def extract_images(self, document: MultimodalDocument) -> List[Dict[str, Any]]:
        """Extract images from document."""
        pass
    
    @abstractmethod
    async def extract_formulas(self, document: MultimodalDocument) -> List[Dict[str, Any]]:
        """Extract mathematical formulas from document."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Health check."""
        pass


class MultimodalMetrics(BaseModel):
    """Metrics for multimodal evaluation."""
    table_extraction_accuracy: float = 0.0
    image_caption_quality: float = 0.0
    formula_accuracy: float = 0.0
    processing_latency_ms: float = 0.0