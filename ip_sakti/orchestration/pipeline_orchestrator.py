"""
Pipeline Orchestrator for IP-SAKTI.
Coordinates all stages of the RAG pipeline with proper error handling, retries, and observability.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncGenerator, Callable
import asyncio
import time
import logging
from dataclasses import dataclass, field
from enum import Enum

from ip_sakti.interfaces import (
    IPipelineOrchestrator,
    PipelineRequest,
    PipelineResult,
    PipelineStage,
    PipelineMode,
    QueryAnalysis,
    QueryRewrite,
    QueryRoute,
    RetrievalRequest,
    RetrievalResult,
    GenerationRequest,
    GenerationResult,
    Chunk,
    CitationRef,
    PipelineMetrics,
    RetrievalStrategy,
    GenerationStrategy,
)

logger = logging.getLogger(__name__)


class PipelineOrchestrator(IPipelineOrchestrator):
    """Main pipeline orchestrator coordinating all RAG stages."""

    def __init__(
        self,
        query_processor=None,
        retriever=None,
        reranker=None,
        generator=None,
        authority_system=None,
        knowledge_graph=None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self._config = config or {}
        self._query_processor = query_processor
        self._retriever = retriever
        self._reranker = reranker
        self._generator = generator
        self._authority_system = authority_system
        self._knowledge_graph = knowledge_graph
        self._stage_handlers: Dict[PipelineStage, Callable] = {}
        self._initialize_stage_handlers()
        self._metrics = PipelineMetrics()

    def _initialize_stage_handlers(self):
        """Initialize stage execution handlers."""
        self._stage_handlers = {
            PipelineStage.QUERY_ANALYSIS: self._execute_query_analysis,
            PipelineStage.QUERY_REWRITE: self._execute_query_rewrite,
            PipelineStage.QUERY_ROUTE: self._execute_query_route,
            PipelineStage.RETRIEVAL: self._execute_retrieval,
            PipelineStage.RERANKING: self._execute_reranking,
            PipelineStage.CONTEXT_BUILDING: self._execute_context_building,
            PipelineStage.GENERATION: self._execute_generation,
            PipelineStage.CITATION_VERIFICATION: self._execute_citation_verification,
            PipelineStage.POST_PROCESSING: self._execute_post_processing,
        }

    @property
    def name(self) -> str:
        return "ip_sakti_pipeline_orchestrator"

    def get_stage_order(self, mode: PipelineMode) -> List[PipelineStage]:
        """Get stage order for a given mode."""
        base_order = [
            PipelineStage.QUERY_ANALYSIS,
            PipelineStage.QUERY_REWRITE,
            PipelineStage.QUERY_ROUTE,
            PipelineStage.RETRIEVAL,
            PipelineStage.RERANKING,
            PipelineStage.CONTEXT_BUILDING,
            PipelineStage.GENERATION,
            PipelineStage.CITATION_VERIFICATION,
            PipelineStage.POST_PROCESSING,
        ]

        if mode == PipelineMode.AGENTIC:
            # Add self-correction loop stages
            return base_order

        return base_order

    async def execute(self, request: PipelineRequest) -> PipelineResult:
        """Execute pipeline synchronously."""
        start_time = time.time()
        stages_completed = []
        stage_timings = {}
        
        # Initialize state
        state = {
            "query": request.query,
            "conversation_history": request.conversation_history,
            "mode": request.mode,
            "retrieval_request": request.retrieval_request,
            "generation_request": request.generation_request,
            "query_analysis": None,
            "query_rewrite": None,
            "query_route": None,
            "retrieval_result": None,
            "reranked_chunks": None,
            "context": None,
            "generation_result": None,
            "answer": "",
            "citations": [],
            "confidence": 0.0,
            "iterations": 1,
        }

        stage_order = self.get_stage_order(request.mode)

        for stage in stage_order:
            stage_start = time.time()
            try:
                handler = self._stage_handlers.get(stage)
                if handler:
                    await handler(state, request)
                    stages_completed.append(stage)
                    stage_timings[stage.value] = (time.time() - stage_start) * 1000
                else:
                    logger.warning(f"No handler for stage: {stage}")
            except Exception as e:
                logger.error(f"Stage {stage} failed: {e}")
                stage_timings[stage.value] = (time.time() - stage_start) * 1000
                # Continue with fallback
                await self._handle_stage_failure(stage, state, e)

        # Build result
        total_time = (time.time() - start_time) * 1000
        
        # Update metrics
        self._metrics.total_latency_ms = total_time
        self._metrics.stage_latencies = stage_timings
        self._metrics.success_rate = 1.0 if stages_completed == stage_order else 0.8

        return PipelineResult(
            answer=state.get("answer", ""),
            citations=state.get("citations", []),
            confidence=state.get("confidence", 0.0),
            chunks_used=state.get("chunks_used", []),
            stages_completed=stages_completed,
            stage_timings=stage_timings,
            query_analysis=state.get("query_analysis"),
            query_rewrite=state.get("query_rewrite"),
            query_route=state.get("query_route"),
            retrieval_result=state.get("retrieval_result"),
            generation_result=state.get("generation_result"),
            iterations=state.get("iterations", 1),
            metadata={
                "total_time_ms": total_time,
                "mode": request.mode.value,
            },
        )

    async def execute_stream(self, request: PipelineRequest) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute pipeline with streaming updates."""
        yield {"type": "start", "query": request.query}
        
        # Execute with streaming updates
        state = await self._execute_streaming(request)
        
        yield {
            "type": "complete",
            "answer": state.get("answer", ""),
            "citations": state.get("citations", []),
            "confidence": state.get("confidence", 0.0),
        }

    async def execute_batch(self, requests: List[PipelineRequest]) -> List[PipelineResult]:
        """Execute pipeline for batch of requests."""
        results = []
        for request in requests:
            result = await self.execute(request)
            results.append(result)
        return results

    async def execute_agentic(self, request: PipelineRequest) -> PipelineResult:
        """Execute with agentic self-correction loop."""
        max_iterations = request.max_iterations
        current_iteration = 1
        
        while current_iteration <= max_iterations:
            request.metadata["iteration"] = current_iteration
            request.metadata["max_iterations"] = max_iterations
            
            result = await self.execute(request)
            
            # Check if self-correction is needed
            if not request.enable_self_correction or current_iteration >= max_iterations:
                return result
            
            # Evaluate result quality
            if result.confidence >= 0.8:
                return result
            
            # Self-correction: modify query and retry
            if result.query_rewrite:
                request.query = f"Refine: {result.query_rewrite.step_back_query or request.query}"
            
            current_iteration += 1
            result.iterations = current_iteration
        
        return result

    async def health_check(self) -> Dict[str, Any]:
        """Health check."""
        components = {}
        
        if self._query_processor:
            try:
                await self._query_processor.health_check()
                components["query_processor"] = "healthy"
            except Exception:
                components["query_processor"] = "unhealthy"
        
        if self._retriever:
            try:
                await self._retriever.health_check()
                components["retriever"] = "healthy"
            except Exception:
                components["retriever"] = "unhealthy"
        
        if self._generator:
            try:
                await self._generator.health_check()
                components["generator"] = "healthy"
            except Exception:
                components["generator"] = "unhealthy"

        return {
            "status": "healthy" if all(v == "healthy" for v in components.values()) else "degraded",
            "components": components,
            "metrics": self._metrics.model_dump(),
        }

    # ============================================================
    # Stage Execution Handlers
    # ============================================================

    async def _execute_query_analysis(self, state: Dict[str, Any], request: PipelineRequest):
        """Execute query analysis stage."""
        if self._query_processor:
            state["query_analysis"] = await self._query_processor.analyze(
                state["query"], 
                state.get("conversation_history")
            )

    async def _execute_query_rewrite(self, state: Dict[str, Any], request: PipelineRequest):
        """Execute query rewrite stage."""
        if self._query_processor and state.get("query_analysis"):
            state["query_rewrite"] = await self._query_processor.rewrite(
                state["query"],
                state["query_analysis"]
            )

    async def _execute_query_route(self, state: Dict[str, Any], request: PipelineRequest):
        """Execute query routing stage."""
        if self._query_processor and state.get("query_analysis") and state.get("query_rewrite"):
            state["query_route"] = await self._query_processor.route(
                state["query"],
                state["query_analysis"],
                state["query_rewrite"]
            )

    async def _execute_retrieval(self, state: Dict[str, Any], request: PipelineRequest):
        """Execute retrieval stage."""
        if self._retriever:
            # Build retrieval request from routing
            retrieval_req = request.retrieval_request or RetrievalRequest(
                query=state["query"],
                top_k=20,
                strategy=RetrievalStrategy.HYBRID,
            )
            
            if state.get("query_route"):
                retrieval_req.strategy = state["query_route"].primary_strategy
            
            state["retrieval_result"] = await self._retriever.retrieve(retrieval_req)
            state["chunks_used"] = state["retrieval_result"].chunks

    async def _execute_reranking(self, state: Dict[str, Any], request: PipelineRequest):
        """Execute reranking stage."""
        if self._reranker and state.get("retrieval_result"):
            from ip_sakti.interfaces import RerankRequest, RerankStrategy
            
            rerank_req = RerankRequest(
                query=state["query"],
                chunks=state["retrieval_result"].chunks,
                top_k=10,
                strategy=RerankStrategy.CROSS_ENCODER,
            )
            
            if state.get("query_route") and state["query_route"].rerank_stages:
                rerank_req.strategy = RerankStrategy(state["query_route"].rerank_stages[0])
            
            rerank_result = await self._reranker.rerank(rerank_req)
            state["reranked_chunks"] = rerank_result.chunks
            state["chunks_used"] = rerank_result.chunks

    async def _execute_context_building(self, state: Dict[str, Any], request: PipelineRequest):
        """Execute context building stage."""
        # Use reranked chunks if available, otherwise retrieval result
        chunks = state.get("reranked_chunks") or state.get("chunks_used", [])
        
        # Add authority scores if available
        if self._authority_system and chunks:
            authority_scores = await self._authority_system.evaluate_batch(chunks)
            for chunk, score in zip(chunks, authority_scores):
                chunk.metadata["authority_score"] = score.score
                chunk.metadata["authority_tier"] = score.tier.value if hasattr(score.tier, 'value') else score.tier

        # Build context string
        context_parts = []
        for i, chunk in enumerate(chunks[:10]):
            citation_marker = f"[{i+1}]"
            context_parts.append(f"{citation_marker} {chunk.content}")
        
        state["context"] = "\n\n".join(context_parts)

    async def _execute_generation(self, state: Dict[str, Any], request: PipelineRequest):
        """Execute generation stage."""
        if self._generator:
            chunks = state.get("reranked_chunks") or state.get("chunks_used", [])
            
            gen_req = request.generation_request or GenerationRequest(
                query=state["query"],
                chunks=chunks,
                strategy=GenerationStrategy.CITATION_FIRST,
                max_tokens=2048,
                temperature=0.1,
            )
            
            state["generation_result"] = await self._generator.generate(gen_req)
            state["answer"] = state["generation_result"].answer
            state["citations"] = state["generation_result"].citations
            state["confidence"] = state["generation_result"].confidence

    async def _execute_citation_verification(self, state: Dict[str, Any], request: PipelineRequest):
        """Execute citation verification stage."""
        if self._generator and state.get("generation_result") and state.get("chunks_used"):
            verification = await self._generator.verify_citations(
                state["generation_result"].answer,
                state["chunks_used"]
            )
            state["verification"] = verification
            
            # Update confidence based on verification
            if verification.get("verified"):
                state["confidence"] = min(state["confidence"] * 1.1, 1.0)

    async def _execute_post_processing(self, state: Dict[str, Any], request: PipelineRequest):
        """Execute post-processing stage."""
        # Final answer formatting, citation formatting, etc.
        answer = state.get("answer", "")
        
        # Ensure answer is not empty
        if not answer and state.get("chunks_used"):
            answer = "Based on the retrieved documents, I found relevant information but could not generate a complete answer."
            state["confidence"] = 0.3
        
        state["answer"] = answer

    async def _handle_stage_failure(self, stage: PipelineStage, state: Dict[str, Any], error: Exception):
        """Handle stage failure with fallback."""
        logger.warning(f"Stage {stage} failed with error: {error}, using fallback")
        
        if stage == PipelineStage.QUERY_ANALYSIS:
            state["query_analysis"] = QueryAnalysis(
                intent="SEARCH",
                complexity="SIMPLE",
                entities=[],
                legal_citations=[],
                acts_referenced=[],
                sections_referenced=[],
                dates_referenced=[],
                jurisdiction_hints=[],
                requires_multi_hop=False,
                requires_comparison=False,
                requires_temporal=False,
                confidence=0.5,
            )
        
        elif stage == PipelineStage.QUERY_REWRITE:
            state["query_rewrite"] = QueryRewrite(
                original_query=state["query"],
                rewritten_queries=[state["query"]],
                hyde_document=None,
                step_back_query=None,
                sub_queries=[],
                expansion_terms=[],
            )
        
        elif stage == PipelineStage.QUERY_ROUTE:
            state["query_route"] = QueryRoute(
                primary_strategy=RetrievalStrategy.HYBRID,
                fallback_strategies=[RetrievalStrategy.SPARSE, RetrievalStrategy.DENSE],
                use_kg=False,
                use_graph=False,
                use_contextual=False,
                requires_reranking=True,
                rerank_stages=["cross_encoder"],
            )

    async def _execute_streaming(self, request: PipelineRequest) -> Dict[str, Any]:
        """Execute pipeline with streaming intermediate results."""
        state = {
            "query": request.query,
            "conversation_history": request.conversation_history,
            "mode": request.mode,
            "query_analysis": None,
            "query_rewrite": None,
            "query_route": None,
            "retrieval_result": None,
            "reranked_chunks": None,
            "context": None,
            "generation_result": None,
            "answer": "",
            "citations": [],
            "confidence": 0.0,
        }

        stage_order = self.get_stage_order(request.mode)

        for stage in stage_order:
            handler = self._stage_handlers.get(stage)
            if handler:
                try:
                    await handler(state, request)
                    yield {"stage": stage.value, "status": "completed", "data": self._get_stage_output(state, stage)}
                except Exception as e:
                    logger.error(f"Streaming stage {stage} failed: {e}")
                    yield {"stage": stage.value, "status": "failed", "error": str(e)}
                    await self._handle_stage_failure(stage, state, e)

        return state

    def _get_stage_output(self, state: Dict[str, Any], stage: PipelineStage) -> Dict[str, Any]:
        """Get output data for a completed stage."""
        outputs = {
            PipelineStage.QUERY_ANALYSIS: state.get("query_analysis"),
            PipelineStage.QUERY_REWRITE: state.get("query_rewrite"),
            PipelineStage.QUERY_ROUTE: state.get("query_route"),
            PipelineStage.RETRIEVAL: state.get("retrieval_result"),
            PipelineStage.RERANKING: state.get("reranked_chunks"),
            PipelineStage.GENERATION: state.get("generation_result"),
        }
        return outputs.get(stage, {}).model_dump() if outputs.get(stage) else {}


def create_pipeline_orchestrator(
    query_processor=None,
    retriever=None,
    reranker=None,
    generator=None,
    authority_system=None,
    knowledge_graph=None,
    config: Optional[Dict[str, Any]] = None,
) -> PipelineOrchestrator:
    """Factory function to create a pipeline orchestrator."""
    return PipelineOrchestrator(
        query_processor=query_processor,
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        authority_system=authority_system,
        knowledge_graph=knowledge_graph,
        config=config,
    )