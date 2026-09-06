"""
Evaluation interfaces for IP-SAKTI.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class EvaluationMetric(str, Enum):
    """Evaluation metrics."""
    RECALL_AT_K = "recall_at_k"
    PRECISION_AT_K = "precision_at_k"
    MRR = "mrr"
    NDCG_AT_K = "ndcg_at_k"
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCE = "answer_relevance"
    CITATION_ACCURACY = "citation_accuracy"
    HALLUCINATION_RATE = "hallucination_rate"
    LATENCY_MS = "latency_ms"
    THROUGHPUT = "throughput"


class EvaluationDataset(BaseModel):
    """Evaluation dataset."""
    name: str
    queries: List[Dict[str, Any]]
    expected_answers: Optional[List[str]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationRequest(BaseModel):
    """Request for evaluation."""
    dataset: EvaluationDataset
    metrics: List[EvaluationMetric]
    config: Dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    """Result from evaluation."""
    dataset_name: str
    metrics: Dict[str, float]
    per_query_results: List[Dict[str, Any]] = Field(default_factory=list)
    aggregate_scores: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IEvaluator(ABC):
    """Interface for evaluation components."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Evaluator name."""
        pass
    
    @property
    @abstractmethod
    def supported_metrics(self) -> List[EvaluationMetric]:
        """Supported evaluation metrics."""
        pass
    
    @abstractmethod
    async def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        """Run evaluation."""
        pass
    
    @abstractmethod
    async def evaluate_retrieval(
        self, 
        queries: List[str], 
        retrieved_chunks: List[List[str]], 
        ground_truth: List[List[str]]
    ) -> Dict[str, float]:
        """Evaluate retrieval quality."""
        pass
    
    @abstractmethod
    async def evaluate_generation(
        self, 
        queries: List[str], 
        generated_answers: List[str], 
        reference_answers: List[str],
        citations: List[List[str]] = None
    ) -> Dict[str, float]:
        """Evaluate generation quality."""
        pass
    
    @abstractmethod
    async def evaluate_end_to_end(self, queries: List[str], responses: List[Dict[str, Any]]) -> EvaluationResult:
        """Evaluate end-to-end pipeline."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Health check."""
        pass


class EvaluatorMetrics(BaseModel):
    """Metrics for evaluator."""
    evaluation_latency_ms: float = 0.0
    queries_evaluated: int = 0