"""
Metrics interfaces for IP-SAKTI.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class MetricType(str, Enum):
    """Metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class IMetricsCollector(ABC):
    """Interface for metrics collection."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Collector name."""
        pass
    
    @abstractmethod
    async def increment(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment counter."""
        pass
    
    @abstractmethod
    async def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Set gauge value."""
        pass
    
    @abstractmethod
    async def histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record histogram value."""
        pass
    
    @abstractmethod
    async def timing(self, name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record timing."""
        pass
    
    @abstractmethod
    async def flush(self) -> None:
        """Flush metrics to backend."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Health check."""
        pass


class MetricsMetrics(BaseModel):
    """Metrics for metrics collector."""
    metrics_collected: int = 0
    flush_latency_ms: float = 0.0