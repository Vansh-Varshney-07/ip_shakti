"""
Phase 6: Agentic RAG with Self-Correction Module
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable

from ip_sakti.core.models import DocumentChunk, RetrievalResult
from ip_sakti.generation.strategies.citation_generator import (
    GenerationConfig,
    GeneratedAnswer,
    CitationFirstGenerator,
)

logger = logging.getLogger(__name__)


class AgentAction(str, Enum):
    """Possible agent actions."""
    RETRIEVE = "retrieve"
    RERANK = "rerank"
    GENERATE = "generate"
    VERIFY = "verify"
    REFINED_RETRIEVE = "refined_retrieve"
    CLARIFY = "clarify"
    ABSTAIN = "abstain"


@dataclass
class AgentStep:
    """A single step in the agent's reasoning trace."""
    action: AgentAction
    reasoning: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    took_ms: float
    success: bool
    error: Optional[str] = None


@dataclass
class AgentState:
    """Current state of the agent."""
    query: str
    original_query: str
    context: List[RetrievalResult] = field(default_factory=list)
    generated_answer: Optional[GeneratedAnswer] = None
    steps: List[AgentStep] = field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 3
    confidence: float = 0.0
    should_continue: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base class for agents."""
    
    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations
    
    @abstractmethod
    async def run(self, query: str, **kwargs) -> Tuple[GeneratedAnswer, List[AgentStep]]:
        """Run the agent on a query."""
        pass


class SelfCorrectingAgent(BaseAgent):
    """Agent that iteratively refines its answer through self-correction."""
    
    def __init__(
        self,
        generator: CitationFirstGenerator,
        retriever_func: Callable,
        reranker_func: Optional[Callable] = None,
        verifier_func: Optional[Callable] = None,
        max_iterations: int = 3,
    ):
        super().__init__(max_iterations)
        self.generator = generator
        self.retriever_func = retriever_func
        self.reranker_func = reranker_func
        self.verifier_func = verifier_func
    
    async def run(self, query: str, **kwargs) -> Tuple[GeneratedAnswer, List[AgentStep]]:
        state = AgentState(
            query=query,
            original_query=query,
            max_iterations=self.max_iterations,
        )
        
        while state.should_continue and state.iteration < state.max_iterations:
            state.iteration += 1
            logger.info(f"Agent iteration {state.iteration}/{state.max_iterations}")
            
            # Decide action based on current state
            action = await self._decide_action(state)
            
            # Execute action
            await self._execute_action(state, action)
            
            # Check if we should continue
            state.should_continue = self._should_continue(state)
        
        # Final answer
        final_answer = state.generated_answer or GeneratedAnswer(
            answer="Unable to generate a reliable answer.",
            citations=[],
            raw_output="",
            verified=False,
            metadata={"failed": True},
        )
        
        return final_answer, state.steps
    
    async def _decide_action(self, state: AgentState) -> AgentAction:
        """Decide the next action based on current state."""
        if state.iteration == 1:
            return AgentAction.RETRIEVE
        
        if state.generated_answer is None:
            if state.context:
                return AgentAction.GENERATE
            return AgentAction.RETRIEVE
        
        if not state.generated_answer.verified:
            return AgentAction.VERIFY
        
        if state.confidence < 0.7 and state.iteration < state.max_iterations:
            return AgentAction.REFINED_RETRIEVE
        
        if state.confidence < 0.5:
            return AgentAction.CLARIFY
        
        return AgentAction.ABSTAIN
    
    async def _execute_action(self, state: AgentState, action: AgentAction) -> None:
        """Execute the decided action."""
        start = time.time()
        
        try:
            if action == AgentAction.RETRIEVE:
                await self._action_retrieve(state)
            elif action == AgentAction.REFINED_RETRIEVE:
                await self._action_refined_retrieve(state)
            elif action == AgentAction.RERANK:
                await self._action_rerank(state)
            elif action == AgentAction.GENERATE:
                await self._action_generate(state)
            elif action == AgentAction.VERIFY:
                await self._action_verify(state)
            elif action == AgentAction.CLARIFY:
                await self._action_clarify(state)
            
            took_ms = (time.time() - start) * 1000
            state.steps.append(AgentStep(
                action=action,
                reasoning=f"Executed {action.value}",
                input_data={"query": state.query, "iteration": state.iteration},
                output_data={"context_count": len(state.context)},
                took_ms=took_ms,
                success=True,
            ))
            
        except Exception as e:
            took_ms = (time.time() - start) * 1000
            state.steps.append(AgentStep(
                action=action,
                reasoning=f"Failed {action.value}: {e}",
                input_data={"query": state.query},
                output_data={},
                took_ms=took_ms,
                success=False,
                error=str(e),
            ))
            logger.error(f"Agent action {action} failed: {e}")
    
    async def _action_retrieve(self, state: AgentState) -> None:
        """Initial retrieval."""
        results = await self.retriever_func(state.query)
        state.context = results
    
    async def _action_refined_retrieve(self, state: AgentState) -> None:
        """Refined retrieval based on gaps in previous answer."""
        # Analyze what's missing
        if state.generated_answer and state.generated_answer.verification_notes:
            # Create a refined query based on verification notes
            refined_query = f"{state.original_query} " + " ".join(state.generated_answer.verification_notes)
            state.query = refined_query
        else:
            # Broaden the query
            state.query = f"{state.original_query} related concepts definitions"
        
        results = await self.retriever_func(state.query)
        state.context.extend(results)
        # Deduplicate
        seen = set()
        unique = []
        for r in state.context:
            if r.chunk.id not in seen:
                seen.add(r.chunk.id)
                unique.append(r)
        state.context = unique
    
    async def _action_rerank(self, state: AgentState) -> None:
        """Rerank current context."""
        if self.reranker_func and state.context:
            reranked = await self.reranker_func(state.query, state.context)
            state.context = reranked
    
    async def _action_generate(self, state: AgentState) -> None:
        """Generate answer from context."""
        answer = await self.generator.generate(state.query, state.context)
        state.generated_answer = answer
        state.confidence = self._calculate_confidence(answer)
    
    async def _action_verify(self, state: AgentState) -> None:
        """Verify the generated answer."""
        if self.verifier_func and state.generated_answer:
            verified, notes = await self.verifier_func(
                state.generated_answer.answer,
                state.generated_answer.citations,
                state.context,
            )
            state.generated_answer.verified = verified
            state.generated_answer.verification_notes = notes
    
    async def _action_clarify(self, state: AgentState) -> None:
        """Signal that clarification is needed."""
        state.metadata["needs_clarification"] = True
        state.metadata["clarification_reason"] = "Low confidence after maximum iterations"
    
    def _calculate_confidence(self, answer: GeneratedAnswer) -> float:
        """Calculate confidence score for the answer."""
        if not answer.citations:
            return 0.0
        
        # Base confidence on citation coverage
        citation_confidence = sum(c.confidence for c in answer.citations) / len(answer.citations)
        
        # Boost if verified
        verification_boost = 0.2 if answer.verified else 0.0
        
        # Penalize if too few citations
        coverage = min(len(answer.citations) / 3.0, 1.0)
        
        return min(citation_confidence * coverage + verification_boost, 1.0)
    
    def _should_continue(self, state: AgentState) -> bool:
        """Determine if agent should continue iterating."""
        if state.iteration >= state.max_iterations:
            return False
        
        if state.generated_answer is None:
            return True
        
        if not state.generated_answer.verified:
            return True
        
        if state.confidence < 0.7:
            return True
        
        return False


class MultiAgentOrchestrator:
    """Orchestrates multiple specialized agents."""
    
    def __init__(self, agents: Dict[str, BaseAgent]):
        self.agents = agents
    
    async def run(self, query: str, agent_name: str = "default", **kwargs) -> Tuple[GeneratedAnswer, List[AgentStep]]:
        """Run a specific agent."""
        if agent_name not in self.agents:
            raise ValueError(f"Unknown agent: {agent_name}")
        
        agent = self.agents[agent_name]
        return await agent.run(query, **kwargs)
    
    async def run_ensemble(self, query: str, **kwargs) -> Tuple[GeneratedAnswer, Dict[str, List[AgentStep]]]:
        """Run multiple agents and combine results."""
        results = {}
        for name, agent in self.agents.items():
            try:
                answer, steps = await agent.run(query, **kwargs)
                results[name] = (answer, steps)
            except Exception as e:
                logger.error(f"Agent {name} failed: {e}")
                results[name] = (GeneratedAnswer(
                    answer=f"Agent {name} failed: {e}",
                    citations=[],
                    raw_output="",
                    verified=False,
                ), [])
        
        # Select best answer (highest confidence)
        best_name = max(results.keys(), key=lambda n: results[n][0].metadata.get("confidence", 0))
        best_answer, best_steps = results[best_name]
        best_answer.metadata["ensemble"] = True
        best_answer.metadata["agents_run"] = list(results.keys())
        best_answer.metadata["selected_agent"] = best_name
        
        all_steps = {name: steps for name, (_, steps) in results.items()}
        
        return best_answer, all_steps


@dataclass
class AgentConfig:
    """Configuration for the agentic system."""
    max_iterations: int = 3
    confidence_threshold: float = 0.7
    enable_verification: bool = True
    enable_reranking: bool = True
    retrieval_top_k: int = 10
    reranker_type: str = "cross_encoder"
    generator_config: Optional[GenerationConfig] = None


class AgenticRAGSystem:
    """Main entry point for agentic RAG with self-correction."""
    
    def __init__(
        self,
        config: AgentConfig,
        retriever_func: Callable,
        reranker_func: Optional[Callable] = None,
        verifier_func: Optional[Callable] = None,
    ):
        self.config = config
        
        # Initialize generator
        gen_config = config.generator_config or GenerationConfig()
        self.generator = CitationFirstGenerator(gen_config)
        
        # Initialize agents
        self.main_agent = SelfCorrectingAgent(
            generator=self.generator,
            retriever_func=retriever_func,
            reranker_func=reranker_func if config.enable_reranking else None,
            verifier_func=verifier_func if config.enable_verification else None,
            max_iterations=config.max_iterations,
        )
        
        self.orchestrator = MultiAgentOrchestrator({
            "default": self.main_agent,
        })
    
    async def initialize(self) -> None:
        """Initialize all components."""
        await self.generator.initialize()
        logger.info("Agentic RAG system initialized")
    
    async def query(
        self,
        query: str,
        agent: str = "default",
        **kwargs,
    ) -> Tuple[GeneratedAnswer, List[AgentStep]]:
        """Process a query through the agentic system."""
        return await self.orchestrator.run(query, agent, **kwargs)
    
    async def query_ensemble(self, query: str, **kwargs) -> Tuple[GeneratedAnswer, Dict[str, List[AgentStep]]]:
        """Process a query through all agents and select best."""
        return await self.orchestrator.run_ensemble(query, **kwargs)
    
    async def close(self) -> None:
        """Cleanup resources."""
        await self.generator.close()