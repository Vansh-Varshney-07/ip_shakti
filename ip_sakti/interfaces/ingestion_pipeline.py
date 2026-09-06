"""
Ingestion pipeline interface for IP-SAKTI.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum

from ip_sakti.interfaces.ingestion import IngestionRequest, IngestionResult


class IIngestionPipeline(ABC):
    """Interface for ingestion pipeline."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Pipeline name."""
        pass
    
    @abstractmethod
    async def ingest(self, request: IngestionRequest) -> IngestionResult:
        """Ingest a single document."""
        pass
    
    @abstractmethod
    async def ingest_batch(self, requests: List[IngestionRequest]) -> List[IngestionResult]:
        """Ingest multiple documents."""
        pass
    
    @abstractmethod
    async def ingest_directory(self, directory: str, **kwargs) -> List[IngestionResult]:
        """Ingest all documents in a directory."""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Health check."""
        pass


class IngestionPipelineMetrics(BaseModel):
    """Metrics for ingestion pipeline evaluation."""
    documents_processed: int = 0
    chunks_created: int = 0
    avg_processing_time_ms: float = 0.0
    error_rate: float = 0.0