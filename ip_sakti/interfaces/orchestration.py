"""
Orchestration interfaces for IP-SAKTI.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncGenerator
from pydantic import BaseModel, Field
from enum import Enum
from ip_sakti.interfaces.retriever import RetrievalRequest, RetrievalResult, RetrievalStrategy
from ip_sakti.interfaces.generator import GenerationRequest, GenerationResult, GenerationStrategy
from ip_sakti.interfaces.query_processor import QueryAnalysis, QueryRewrite, QueryRoute


class Chunk(BaseModel):
    """Chunk model for orchestration."""
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


class PipelineStage(str, Enum):
    """Pipeline stages."""
    QUERY_ANALYSIS = "query_analysis"
    QUERY_REWRITE = "query_rewrite"
    QUERY_ROUTE = "query_route"
    RETRIEVAL = "retrieval"
    RERANKING = "reranking"
    CONTEXT_BUILDING = "context_building"
    GENERATION = "generation"
    CITATION_VERIFICATION = "citation_verification"
    POST_PROCESSING = "post_processing"


class PipelineMode(str, Enum):
    """Pipeline execution modes."""
    SYNC = "sync"
    ASYNC = "async"
    STREAMING = "streaming"
    BATCH = "batch"
    AGENTIC = "agentic"


class PipelineRequest(BaseModel):
    """Request for pipeline execution."""
    query: str
    conversation_history: Optional[List[Dict[str, str]]] = None
    mode: PipelineMode = PipelineMode.SYNC
    retrieval_request: Optional[RetrievalRequest] = None
    generation_request: Optional[GenerationRequest] = None
    enable_self_correction: bool = False
    max_iterations: int = 3
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PipelineResult(BaseModel):
    """Result from pipeline execution."""
    answer: str
    citations: List[CitationRef]
    confidence: float
    chunks_used: List[Chunk]
    stages_completed: List[PipelineStage]
    stage_timings: Dict[str, float]
    query_analysis: Optional[QueryAnalysis] = None
    query_rewrite: Optional[QueryRewrite] = None
    query_route: Optional[QueryRoute] = None
    retrieval_result: Optional[RetrievalResult] = None
    generation_result: Optional[GenerationResult] = None
    iterations: int = 1
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IPipelineOrchestrator(ABC):
    """Interface for pipeline orchestration."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Orchestrator name."""
        pass
    
    @abstractmethod
    async def execute(self, request: PipelineRequest) -> PipelineResult:
        """Execute pipeline synchronously."""
        pass
    
    @abstractmethod
    async def execute_stream(self, request: PipelineRequest) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute pipeline with streaming updates."""
        pass
    
    @abstractmethod
    async def execute_batch(self, requests: List[PipelineRequest]) -> List[PipelineResult]:
        """Execute pipeline for batch of requests."""
        pass
    
    @abstractmethod
    async def execute_agentic(self, request: PipelineRequest) -> PipelineResult:
        """Execute with agentic self-correction loop."""
        pass
    
    @abstractmethod
    def get_stage_order(self, mode: PipelineMode) -> List[PipelineStage]:
        """Get stage order for a given mode."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Health check."""
        pass


class PipelineMetrics(BaseModel):
    """Metrics for pipeline evaluation."""
    total_latency_ms: float = 0.0
    stage_latencies: Dict[str, float] = Field(default_factory=dict)
    throughput_queries_per_sec: float = 0.0
    success_rate: float = 1.0
    error_rate_by_stage: Dict[str, float] = Field(default_factory=dict)