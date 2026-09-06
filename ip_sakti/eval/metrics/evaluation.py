"""
Phase 9: Comprehensive Evaluation Framework
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import defaultdict
import statistics

from ip_sakti.core.models import DocumentChunk, RetrievalResult

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of evaluation metrics."""
    # Retrieval metrics
    PRECISION_AT_K = "precision_at_k"
    RECALL_AT_K = "recall_at_k"
    MRR = "mrr"  # Mean Reciprocal Rank
    NDCG = "ndcg"  # Normalized Discounted Cumulative Gain
    MAP = "map"  # Mean Average Precision
    HIT_RATE = "hit_rate"
    
    # Generation metrics
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCY = "answer_relevancy"
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"
    CONTEXT_RELEVANCY = "context_relevancy"
    
    # Citation metrics
    CITATION_PRECISION = "citation_precision"
    CITATION_RECALL = "citation_recall"
    CITATION_F1 = "citation_f1"
    CITATION_COVERAGE = "citation_coverage"
    
    # System metrics
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    COST = "cost"


@dataclass
class MetricResult:
    """Result of a metric computation."""
    name: str
    value: float
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TestCase:
    """A single test case for evaluation."""
    id: str
    query: str
    expected_answer: Optional[str] = None
    relevant_docs: List[str] = field(default_factory=list)  # chunk IDs
    expected_citations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    category: str = "general"
    difficulty: str = "medium"


@dataclass
class EvaluationResult:
    """Result of evaluating a single test case."""
    test_case_id: str
    query: str
    generated_answer: str
    retrieved_docs: List[str]
    citations: List[str]
    metrics: Dict[str, MetricResult] = field(default_factory=dict)
    latency_ms: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class EvaluationSummary:
    """Summary of evaluation results."""
    total_cases: int
    metrics: Dict[str, float]  # metric_name -> mean_value
    metrics_std: Dict[str, float]  # metric_name -> std_dev
    by_category: Dict[str, Dict[str, float]]
    by_difficulty: Dict[str, Dict[str, float]]
    failed_cases: int
    total_latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseMetric(ABC):
    """Abstract base class for evaluation metrics."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    async def compute(
        self,
        test_case: TestCase,
        result: EvaluationResult,
    ) -> MetricResult:
        """Compute the metric."""
        pass


# Retrieval Metrics
class PrecisionAtK(BaseMetric):
    """Precision at K."""
    
    def __init__(self, k: int = 5):
        super().__init__(f"precision_at_{k}")
        self.k = k
    
    async def compute(self, test_case: TestCase, result: EvaluationResult) -> MetricResult:
        if not test_case.relevant_docs:
            return MetricResult(name=self.name, value=0.0, details={"reason": "no_relevant_docs"})
        
        retrieved_top_k = result.retrieved_docs[:self.k]
        relevant_set = set(test_case.relevant_docs)
        
        hits = sum(1 for doc in retrieved_top_k if doc in relevant_set)
        precision = hits / min(self.k, len(retrieved_top_k)) if retrieved_top_k else 0.0
        
        return MetricResult(
            name=self.name,
            value=precision,
            details={"hits": hits, "k": self.k, "relevant_count": len(relevant_set)}
        )


class RecallAtK(BaseMetric):
    """Recall at K."""
    
    def __init__(self, k: int = 5):
        super().__init__(f"recall_at_{k}")
        self.k = k
    
    async def compute(self, test_case: TestCase, result: EvaluationResult) -> MetricResult:
        if not test_case.relevant_docs:
            return MetricResult(name=self.name, value=0.0, details={"reason": "no_relevant_docs"})
        
        retrieved_top_k = set(result.retrieved_docs[:self.k])
        relevant_set = set(test_case.relevant_docs)
        
        hits = len(retrieved_top_k & relevant_set)
        recall = hits / len(relevant_set) if relevant_set else 0.0
        
        return MetricResult(
            name=self.name,
            value=recall,
            details={"hits": hits, "k": self.k, "relevant_count": len(relevant_set)}
        )


class MRR(BaseMetric):
    """Mean Reciprocal Rank."""
    
    def __init__(self):
        super().__init__("mrr")
    
    async def compute(self, test_case: TestCase, result: EvaluationResult) -> MetricResult:
        if not test_case.relevant_docs:
            return MetricResult(name=self.name, value=0.0, details={"reason": "no_relevant_docs"})
        
        relevant_set = set(test_case.relevant_docs)
        
        for rank, doc_id in enumerate(result.retrieved_docs, 1):
            if doc_id in relevant_set:
                return MetricResult(
                    name=self.name,
                    value=1.0 / rank,
                    details={"rank": rank}
                )
        
        return MetricResult(name=self.name, value=0.0, details={"rank": "not_found"})


class NDCG(BaseMetric):
    """Normalized Discounted Cumulative Gain."""
    
    def __init__(self, k: int = 10):
        super().__init__(f"ndcg_at_{k}")
        self.k = k
    
    def _dcg(self, relevance: List[float]) -> float:
        return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance))
    
    async def compute(self, test_case: TestCase, result: EvaluationResult) -> MetricResult:
        import math
        
        if not test_case.relevant_docs:
            return MetricResult(name=self.name, value=0.0, details={"reason": "no_relevant_docs"})
        
        relevant_set = set(test_case.relevant_docs)
        
        # Actual relevance (1 if relevant, 0 otherwise)
        actual_relevance = [
            1.0 if doc_id in relevant_set else 0.0
            for doc_id in result.retrieved_docs[:self.k]
        ]
        
        # Ideal relevance
        ideal_relevance = [1.0] * min(len(relevant_set), self.k)
        
        dcg = self._dcg(actual_relevance)
        idcg = self._dcg(ideal_relevance)
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        return MetricResult(
            name=self.name,
            value=ndcg,
            details={"dcg": dcg, "idcg": idcg, "k": self.k}
        )


class HitRate(BaseMetric):
    """Hit Rate - whether any relevant doc is in top K."""
    
    def __init__(self, k: int = 5):
        super().__init__(f"hit_rate_at_{k}")
        self.k = k
    
    async def compute(self, test_case: TestCase, result: EvaluationResult) -> MetricResult:
        if not test_case.relevant_docs:
            return MetricResult(name=self.name, value=0.0, details={"reason": "no_relevant_docs"})
        
        retrieved_top_k = set(result.retrieved_docs[:self.k])
        relevant_set = set(test_case.relevant_docs)
        
        hit = 1.0 if (retrieved_top_k & relevant_set) else 0.0
        
        return MetricResult(
            name=self.name,
            value=hit,
            details={"hit": bool(hit), "k": self.k}
        )


# Generation Metrics (using LLM-as-judge)
class LLMJudgeMetric(BaseMetric):
    """Base class for LLM-based evaluation metrics."""
    
    def __init__(self, name: str, prompt_template: str, llm_client=None):
        super().__init__(name)
        self.prompt_template = prompt_template
        self.llm_client = llm_client
    
    async def compute(self, test_case: TestCase, result: EvaluationResult) -> MetricResult:
        if not self.llm_client:
            return MetricResult(name=self.name, value=0.0, details={"reason": "no_llm_client"})
        
        prompt = self.prompt_template.format(
            query=test_case.query,
            expected_answer=test_case.expected_answer or "N/A",
            generated_answer=result.generated_answer,
            context="\n".join(result.metadata.get("context_texts", [])),
        )
        
        try:
            response = await self._call_llm(prompt)
            score = self._parse_score(response)
            
            return MetricResult(
                name=self.name,
                value=score,
                details={"raw_response": response}
            )
        except Exception as e:
            logger.error(f"LLM judge failed for {self.name}: {e}")
            return MetricResult(name=self.name, value=0.0, details={"error": str(e)})
    
    async def _call_llm(self, prompt: str) -> str:
        # Implementation depends on LLM client
        # This is a placeholder
        return "Score: 0.5"
    
    def _parse_score(self, response: str) -> float:
        import re
        match = re.search(r'Score:\s*([0-9.]+)', response)
        if match:
            return float(match.group(1))
        return 0.0


class FaithfulnessMetric(LLMJudgeMetric):
    """Faithfulness - does the answer stay faithful to the context?"""
    
    def __init__(self, llm_client=None):
        prompt = """You are evaluating the faithfulness of an answer to its context.
Query: {query}
Context: {context}
Generated Answer: {generated_answer}

Does the answer contain any claims not supported by the context? 
Score from 0 (completely hallucinated) to 1 (fully faithful).
Format: Score: X.X"""
        super().__init__("faithfulness", prompt, llm_client)


class AnswerRelevancyMetric(LLMJudgeMetric):
    """Answer Relevancy - does the answer address the query?"""
    
    def __init__(self, llm_client=None):
        prompt = """You are evaluating the relevancy of an answer to the query.
Query: {query}
Generated Answer: {generated_answer}

Does the answer directly address the query?
Score from 0 (irrelevant) to 1 (highly relevant).
Format: Score: X.X"""
        super().__init__("answer_relevancy", prompt, llm_client)


class ContextPrecisionMetric(LLMJudgeMetric):
    """Context Precision - are retrieved chunks relevant?"""
    
    def __init__(self, llm_client=None):
        prompt = """You are evaluating the precision of retrieved context.
Query: {query}
Context: {context}

Are the retrieved chunks relevant to answering the query?
Score from 0 (irrelevant) to 1 (highly precise).
Format: Score: X.X"""
        super().__init__("context_precision", prompt, llm_client)


class ContextRecallMetric(LLMJudgeMetric):
    """Context Recall - does context contain all needed info?"""
    
    def __init__(self, llm_client=None):
        prompt = """You are evaluating the recall of retrieved context.
Query: {query}
Expected Answer: {expected_answer}
Context: {context}

Does the context contain all information needed to answer the query?
Score from 0 (missing key info) to 1 (complete).
Format: Score: X.X"""
        super().__init__("context_recall", prompt, llm_client)


# Citation Metrics
class CitationPrecision(BaseMetric):
    """Citation Precision - are citations correct?"""
    
    def __init__(self):
        super().__init__("citation_precision")
    
    async def compute(self, test_case: TestCase, result: EvaluationResult) -> MetricResult:
        if not result.citations:
            return MetricResult(name=self.name, value=0.0, details={"reason": "no_citations"})
        
        if not test_case.expected_citations:
            return MetricResult(name=self.name, value=1.0, details={"reason": "no_expected_citations"})
        
        expected_set = set(test_case.expected_citations)
        correct = sum(1 for c in result.citations if c in expected_set)
        precision = correct / len(result.citations) if result.citations else 0.0
        
        return MetricResult(
            name=self.name,
            value=precision,
            details={"correct": correct, "total": len(result.citations)}
        )


class CitationRecall(BaseMetric):
    """Citation Recall - are all expected citations present?"""
    
    def __init__(self):
        super().__init__("citation_recall")
    
    async def compute(self, test_case: TestCase, result: EvaluationResult) -> MetricResult:
        if not test_case.expected_citations:
            return MetricResult(name=self.name, value=1.0, details={"reason": "no_expected_citations"})
        
        if not result.citations:
            return MetricResult(name=self.name, value=0.0, details={"reason": "no_citations"})
        
        expected_set = set(test_case.expected_citations)
        found = sum(1 for c in test_case.expected_citations if c in result.citations)
        recall = found / len(expected_set) if expected_set else 0.0
        
        return MetricResult(
            name=self.name,
            value=recall,
            details={"found": found, "expected": len(expected_set)}
        )


class CitationF1(BaseMetric):
    """Citation F1 Score."""
    
    def __init__(self):
        super().__init__("citation_f1")
    
    async def compute(self, test_case: TestCase, result: EvaluationResult) -> MetricResult:
        if not test_case.expected_citations and not result.citations:
            return MetricResult(name=self.name, value=1.0)
        
        expected_set = set(test_case.expected_citations)
        actual_set = set(result.citations)
        
        tp = len(expected_set & actual_set)
        fp = len(actual_set - expected_set)
        fn = len(expected_set - actual_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return MetricResult(
            name=self.name,
            value=f1,
            details={"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}
        )


class CitationCoverage(BaseMetric):
    """Citation Coverage - does every claim have a citation?"""
    
    def __init__(self):
        super().__init__("citation_coverage")
    
    async def compute(self, test_case: TestCase, result: EvaluationResult) -> MetricResult:
        # Simple heuristic: count sentences vs citations
        import re
        sentences = re.split(r'[.!?]+', result.generated_answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return MetricResult(name=self.name, value=1.0)
        
        # Count sentences with citations
        cited_sentences = 0
        citation_pattern = r'\[.*?\]'
        for sent in sentences:
            if re.search(citation_pattern, sent):
                cited_sentences += 1
        
        coverage = cited_sentences / len(sentences)
        
        return MetricResult(
            name=self.name,
            value=coverage,
            details={"total_sentences": len(sentences), "cited_sentences": cited_sentences}
        )


# System Metrics
class LatencyMetric(BaseMetric):
    """Latency metric."""
    
    def __init__(self):
        super().__init__("latency_ms")
    
    async def compute(self, test_case: TestCase, result: EvaluationResult) -> MetricResult:
        return MetricResult(
            name=self.name,
            value=result.latency_ms,
            details={"latency_ms": result.latency_ms}
        )


class EvaluationHarness:
    """Main evaluation harness."""
    
    def __init__(
        self,
        metrics: Optional[List[BaseMetric]] = None,
        llm_client=None,
    ):
        self.metrics = metrics or self._default_metrics(llm_client)
        self.llm_client = llm_client
    
    def _default_metrics(self, llm_client) -> List[BaseMetric]:
        return [
            PrecisionAtK(1),
            PrecisionAtK(3),
            PrecisionAtK(5),
            RecallAtK(1),
            RecallAtK(3),
            RecallAtK(5),
            MRR(),
            NDCG(5),
            NDCG(10),
            HitRate(5),
            HitRate(10),
            FaithfulnessMetric(llm_client),
            AnswerRelevancyMetric(llm_client),
            ContextPrecisionMetric(llm_client),
            ContextRecallMetric(llm_client),
            CitationPrecision(),
            CitationRecall(),
            CitationF1(),
            CitationCoverage(),
            LatencyMetric(),
        ]
    
    async def evaluate(
        self,
        test_cases: List[TestCase],
        system_func: Callable[[str], Tuple[str, List[str], List[str], float]],  # query -> (answer, retrieved, citations, latency)
        **kwargs,
    ) -> EvaluationSummary:
        """Run evaluation on test cases."""
        results = []
        
        for test_case in test_cases:
            try:
                start = time.time()
                answer, retrieved, citations, latency = await system_func(test_case.query)
                
                eval_result = EvaluationResult(
                    test_case_id=test_case.id,
                    query=test_case.query,
                    generated_answer=answer,
                    retrieved_docs=retrieved,
                    citations=citations,
                    latency_ms=latency,
                )
                
                # Compute metrics
                for metric in self.metrics:
                    metric_result = await metric.compute(test_case, eval_result)
                    eval_result.metrics[metric.name] = metric_result
                
                results.append(eval_result)
                
            except Exception as e:
                logger.error(f"Evaluation failed for {test_case.id}: {e}")
                results.append(EvaluationResult(
                    test_case_id=test_case.id,
                    query=test_case.query,
                    generated_answer="ERROR",
                    retrieved_docs=[],
                    citations=[],
                    latency_ms=0,
                    metadata={"error": str(e)},
                ))
        
        return self._summarize(results)
    
    def _summarize(self, results: List[EvaluationResult]) -> EvaluationSummary:
        """Summarize evaluation results."""
        if not results:
            return EvaluationSummary(
                total_cases=0,
                metrics={},
                metrics_std={},
                by_category={},
                by_difficulty={},
                failed_cases=0,
                total_latency_ms=0,
            )
        
        # Collect metric values
        metric_values = defaultdict(list)
        for r in results:
            for name, m in r.metrics.items():
                metric_values[name].append(m.value)
        
        # Compute means and stds
        metrics_mean = {}
        metrics_std = {}
        for name, values in metric_values.items():
            metrics_mean[name] = statistics.mean(values) if values else 0.0
            metrics_std[name] = statistics.stdev(values) if len(values) > 1 else 0.0
        
        # By category
        by_category = defaultdict(lambda: defaultdict(list))
        by_difficulty = defaultdict(lambda: defaultdict(list))
        
        for r in results:
            test_case = next((tc for tc in [] if tc.id == r.test_case_id), None)
            # We'd need access to test_cases here - simplified
            pass
        
        return EvaluationSummary(
            total_cases=len(results),
            metrics=metrics_mean,
            metrics_std=metrics_std,
            by_category={},
            by_difficulty={},
            failed_cases=sum(1 for r in results if "error" in r.metadata),
            total_latency_ms=sum(r.latency_ms for r in results),
        )
    
    def get_metric_names(self) -> List[str]:
        return [m.name for m in self.metrics]


class EvaluationReport:
    """Generate evaluation reports."""
    
    @staticmethod
    def to_json(summary: EvaluationSummary, results: List[EvaluationResult]) -> str:
        """Export as JSON."""
        return json.dumps({
            "summary": {
                "total_cases": summary.total_cases,
                "metrics": summary.metrics,
                "metrics_std": summary.metrics_std,
                "failed_cases": summary.failed_cases,
                "total_latency_ms": summary.total_latency_ms,
            },
            "results": [
                {
                    "test_case_id": r.test_case_id,
                    "query": r.query,
                    "generated_answer": r.generated_answer,
                    "retrieved_docs": r.retrieved_docs,
                    "citations": r.citations,
                    "metrics": {k: v.value for k, v in r.metrics.items()},
                    "latency_ms": r.latency_ms,
                }
                for r in results
            ],
        }, indent=2)
    
    @staticmethod
    def to_markdown(summary: EvaluationSummary) -> str:
        """Export as Markdown table."""
        lines = [
            "# Evaluation Report",
            "",
            f"Total Cases: {summary.total_cases}",
            f"Failed Cases: {summary.failed_cases}",
            f"Avg Latency: {summary.total_latency_ms / max(summary.total_cases, 1):.2f}ms",
            "",
            "## Metrics",
            "",
            "| Metric | Mean | Std Dev |",
            "|--------|------|---------|",
        ]
        
        for name in sorted(summary.metrics.keys()):
            mean = summary.metrics[name]
            std = summary.metrics_std.get(name, 0)
            lines.append(f"| {name} | {mean:.4f} | {std:.4f} |")
        
        return "\n".join(lines)


class RegressionDetector:
    """Detect performance regressions."""
    
    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold
        self.baseline: Optional[Dict[str, float]] = None
    
    def set_baseline(self, metrics: Dict[str, float]) -> None:
        self.baseline = metrics
    
    def check_regression(self, current: Dict[str, float]) -> Dict[str, Any]:
        if not self.baseline:
            return {"has_baseline": False}
        
        regressions = {}
        improvements = {}
        
        for metric, baseline_value in self.baseline.items():
            if metric not in current:
                continue
            
            current_value = current[metric]
            diff = current_value - baseline_value
            pct_change = diff / baseline_value if baseline_value != 0 else 0
            
            if diff < -self.threshold * abs(baseline_value):
                regressions[metric] = {
                    "baseline": baseline_value,
                    "current": current_value,
                    "change": diff,
                    "pct_change": pct_change,
                }
            elif diff > self.threshold * abs(baseline_value):
                improvements[metric] = {
                    "baseline": baseline_value,
                    "current": current_value,
                    "change": diff,
                    "pct_change": pct_change,
                }
        
        return {
            "has_regressions": len(regressions) > 0,
            "regressions": regressions,
            "improvements": improvements,
        }


async def create_evaluation_harness(
    metrics: Optional[List[BaseMetric]] = None,
    llm_client=None,
) -> EvaluationHarness:
    """Factory to create evaluation harness."""
    return EvaluationHarness(metrics, llm_client)