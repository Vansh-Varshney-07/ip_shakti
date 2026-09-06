"""
Agent interfaces for IP-SAKTI.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class AgentRole(str, Enum):
    """Agent roles in the system."""
    PLANNER = "planner"
    RETRIEVER = "retriever"
    CRITIC = "critic"
    REFINER = "refiner"
    COORDINATOR = "coordinator"


class AgentMessage(BaseModel):
    """Message between agents."""
    role: AgentRole
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    requires_response: bool = False


class AgentResponse(BaseModel):
    """Response from an agent."""
    role: AgentRole
    content: str
    success: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)
    citations: List[str] = Field(default_factory=list)


class IAgent(ABC):
    """Base interface for all agents."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name."""
        pass
    
    @property
    @abstractmethod
    def role(self) -> AgentRole:
        """Agent role."""
        pass
    
    @abstractmethod
    async def process(self, message: AgentMessage) -> AgentResponse:
        """Process a message."""
        pass
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize agent with configuration."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Health check."""
        pass


class IPlannerAgent(IAgent):
    """Interface for planner agent."""
    
    @abstractmethod
    async def create_plan(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create execution plan for query."""
        pass
    
    @abstractmethod
    async def decompose_query(self, query: str) -> List[str]:
        """Decompose complex query into sub-queries."""
        pass


class IRetrieverAgent(IAgent):
    """Interface for retriever agent."""
    
    @abstractmethod
    async def retrieve(self, query: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute retrieval based on plan."""
        pass
    
    @abstractmethod
    async def select_strategy(self, query: str) -> List[str]:
        """Select retrieval strategies."""
        pass


class ICriticAgent(IAgent):
    """Interface for critic agent."""
    
    @abstractmethod
    async def evaluate(self, answer: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate answer quality and correctness."""
        pass
    
    @abstractmethod
    async def find_issues(self, answer: str, citations: List[str]) -> List[str]:
        """Find issues in answer and citations."""
        pass


class IRefinerAgent(IAgent):
    """Interface for refiner agent."""
    
    @abstractmethod
    async def refine(self, answer: str, issues: List[str], context: List[Dict[str, Any]]) -> str:
        """Refine answer based on issues."""
        pass
    
    @abstractmethod
    async def verify_citations(self, answer: str, citations: List[str]) -> Dict[str, Any]:
        """Verify citations match answer."""
        pass


class AgentMetrics(BaseModel):
    """Metrics for agent evaluation."""
    success_rate: float = 0.0
    avg_latency_ms: float = 0.0
    tokens_per_sec: float = 0.0
    error_rate: float = 0.0