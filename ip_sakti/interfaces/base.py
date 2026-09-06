"""
Base interfaces for all IP-SAKTI components.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from enum import Enum


class ComponentStatus(str, Enum):
    """Health status of a component."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"


class IComponent(ABC):
    """Base interface for all components."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique component name."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Component version."""
        pass
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the component with configuration."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Graceful shutdown."""
        pass
    
    @abstractmethod
    async def health_check(self) -> ComponentStatus:
        """Health check endpoint."""
        pass


class IConfigurable(ABC):
    """Interface for components that can be configured at runtime."""
    
    @abstractmethod
    async def update_config(self, config: Dict[str, Any]) -> None:
        """Update configuration without restart."""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        pass


class IHealthCheckable(ABC):
    """Interface for detailed health checks."""
    
    @abstractmethod
    async def detailed_health_check(self) -> Dict[str, Any]:
        """Return detailed health information."""
        pass


class ComponentMetadata(BaseModel):
    """Metadata about a component."""
    name: str
    version: str
    description: str = ""
    dependencies: list[str] = Field(default_factory=list)
    config_schema: Dict[str, Any] = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=list)