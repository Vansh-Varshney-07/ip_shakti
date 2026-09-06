"""
Generator interfaces for IP-SAKTI.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncGenerator
from pydantic import BaseModel, Field
from enum import Enum


class Chunk(BaseModel):
    """Chunk model for generation."""
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source: Optional[str] = None
    chunk_index: int = 0
    parent_id: Optional[str] = None


class CitationRef(BaseModel):
    """Citation reference."""
    cited_act: str
    cited_section: str
    cited_text: str
    citation_type: str
    hierarchy_path: Optional[Dict[str, Any]] = None
    confidence: float = 1.0


class GenerationStrategy(str, Enum):
    """Generation strategies."""
    STANDARD = "standard"
    CITATION_FIRST = "citation_first"
    CONSTRAINED = "constrained"
    MULTI_HOP = "multi_hop"
    STREAMING = "streaming"
    STRUCTURED = "structured"


class GenerationRequest(BaseModel):
    """Request for generation."""
    query: str
    chunks: List[Chunk]
    strategy: GenerationStrategy = GenerationStrategy.CITATION_FIRST
    max_tokens: int = Field(default=2048, ge=1, le=8192)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    system_prompt: Optional[str] = None
    response_format: Optional[Dict[str, Any]] = None  # For structured output
    citation_format: str = "inline"  # inline, footnote, bracket
    include_confidence: bool = True
    verify_citations: bool = True
    stream: bool = False


class GenerationResult(BaseModel):
    """Result from generation."""
    answer: str
    citations: List[CitationRef]
    confidence: float
    strategy_used: GenerationStrategy
    tokens_used: int
    generation_time_ms: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    verification_passed: bool = True
    verification_errors: List[str] = Field(default_factory=list)


class IGenerator(ABC):
    """Interface for generation components."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Generator name."""
        pass
    
    @property
    @abstractmethod
    def supported_strategies(self) -> List[GenerationStrategy]:
        """List of supported generation strategies."""
        pass
    
    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate answer with citations."""
        pass
    
    @abstractmethod
    async def generate_stream(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        """Stream generation tokens."""
        pass
    
    @abstractmethod
    async def verify_citations(self, answer: str, chunks: List[Chunk]) -> Dict[str, Any]:
        """Verify that citations in answer match retrieved chunks."""
        pass
    
    @abstractmethod
    async def extract_structured(self, query: str, chunks: List[Chunk], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured information from chunks."""
        pass


class GeneratorMetrics(BaseModel):
    """Metrics for generation evaluation."""
    latency_ms: float = 0.0
    tokens_per_sec: float = 0.0
    faithfulness_score: float = 0.0
    citation_accuracy: float = 0.0
    hallucination_rate: float = 0.0


class ICitationGenerator(ABC):
    """Interface for citation generation."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Citation generator name."""
        pass
    
    @abstractmethod
    async def generate_citations(
        self,
        answer: str,
        chunks: List[Chunk],
    ) -> List[CitationRef]:
        """Generate citations for an answer."""
        pass
    
    @abstractmethod
    async def validate_citations(
        self,
        answer: str,
        citations: List[CitationRef],
    ) -> Dict[str, Any]:
        """Validate citations match answer."""
        pass