"""
Cache interfaces for IP-SAKTI.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class CacheBackend(str, Enum):
    """Cache backends."""
    IN_MEMORY = "in_memory"
    REDIS = "redis"
    MEMCACHED = "memcached"
    DISK = "disk"


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class ICache(ABC):
    """Interface for cache components."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Cache name."""
        pass
    
    @property
    @abstractmethod
    def backend(self) -> CacheBackend:
        """Cache backend."""
        pass
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache."""
        pass
    
    @abstractmethod
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values."""
        pass
    
    @abstractmethod
    async def set_many(self, items: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple values."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Health check."""
        pass


class ICircuitBreaker(ABC):
    """Interface for circuit breaker."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Circuit breaker name."""
        pass
    
    @abstractmethod
    async def call(self, func, *args, **kwargs) -> Any:
        """Execute function with circuit breaker."""
        pass
    
    @abstractmethod
    async def get_state(self) -> CircuitState:
        """Get current state."""
        pass
    
    @abstractmethod
    async def reset(self) -> None:
        """Reset circuit breaker."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Health check."""
        pass


class CacheMetrics(BaseModel):
    """Metrics for cache evaluation."""
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    avg_latency_ms: float = 0.0
    memory_usage_mb: float = 0.0
    eviction_count: int = 0


class CacheStats(BaseModel):
    """Cache statistics."""
    backend: str
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    avg_latency_ms: float = 0.0
    memory_usage_mb: float = 0.0
    eviction_count: int = 0
    total_keys: int = 0
    expired_keys: int = 0