"""
Phase 5: Citation-First Constrained Generation Module
"""

import asyncio
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ip_sakti.core.models import DocumentChunk, RetrievalResult

logger = logging.getLogger(__name__)


class GenerationStrategy(str, Enum):
    """Generation strategies."""
    CITATION_FIRST = "citation_first"
    CONSTRAINED = "constrained"
    STRUCTURED = "structured"
    CHAIN_OF_THOUGHT = "chain_of_thought"


@dataclass
class GenerationConfig:
    """Configuration for generation."""
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.1
    max_tokens: int = 2048
    top_p: float = 0.95
    # Citation settings
    citation_format: str = "bracket"  # bracket, inline, footnote
    max_citations_per_claim: int = 3
    require_citations: bool = True
    # Verification
    verify_citations: bool = True
    max_verification_retries: int = 2
    # Structured output
    response_schema: Optional[Dict[str, Any]] = None


@dataclass
class Citation:
    """A citation linking a claim to a source."""
    claim: str
    source_ids: List[str]
    source_texts: List[str]
    confidence: float
    citation_id: str = ""
    
    def format(self, style: str = "bracket") -> str:
        """Format citation for output."""
        if style == "bracket":
            return f"[{', '.join(self.source_ids)}]"
        elif style == "inline":
            return f" (Sources: {', '.join(self.source_ids)})"
        elif style == "footnote":
            return f"^{self.citation_id}^"
        return str(self.source_ids)


@dataclass
class GeneratedAnswer:
    """Complete generated answer with citations."""
    answer: str
    citations: List[Citation]
    raw_output: str
    verified: bool = False
    verification_notes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseGenerator(ABC):
    """Abstract base class for generators."""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the generator."""
        pass
    
    @abstractmethod
    async def generate(
        self,
        query: str,
        context: List[RetrievalResult],
        **kwargs,
    ) -> GeneratedAnswer:
        """Generate answer with citations."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Cleanup resources."""
        pass


class CitationFirstGenerator(BaseGenerator):
    """Citation-first constrained generator."""
    
    def __init__(self, config: GenerationConfig):
        super().__init__(config)
        self._llm = None
    
    async def initialize(self) -> None:
        try:
            # Try to import available LLM clients
            import os
            if os.getenv("OPENAI_API_KEY"):
                from openai import AsyncOpenAI
                self._llm = AsyncOpenAI()
                self._llm_type = "openai"
            elif os.getenv("ANTHROPIC_API_KEY"):
                from anthropic import AsyncAnthropic
                self._llm = AsyncAnthropic()
                self._llm_type = "anthropic"
            else:
                logger.warning("No LLM API key found, using fallback")
                self._llm_type = "fallback"
            
            self._initialized = True
            logger.info(f"Generator initialized: {self._llm_type}")
        except ImportError:
            logger.warning("LLM client not available, using fallback")
            self._llm_type = "fallback"
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize generator: {e}")
            self._llm_type = "fallback"
            self._initialized = True
    
    async def generate(
        self,
        query: str,
        context: List[RetrievalResult],
        **kwargs,
    ) -> GeneratedAnswer:
        import time
        start = time.time()
        
        if not self._initialized:
            await self.initialize()
        
        if self._llm_type == "fallback":
            return await self._fallback_generate(query, context)
        
        # Prepare context with citation IDs
        context_with_ids = self._prepare_context(context)
        
        # Build prompt
        prompt = self._build_prompt(query, context_with_ids)
        
        # Generate
        raw_output = await self._call_llm(prompt)
        
        # Parse and validate citations
        answer, citations = self._parse_output(raw_output, context)
        
        # Verify citations if enabled
        verified = False
        verification_notes = []
        if self.config.verify_citations:
            verified, verification_notes = await self._verify_citations(
                answer, citations, context
            )
            
            # Retry if verification failed
            for attempt in range(self.config.max_verification_retries):
                if verified:
                    break
                logger.info(f"Citation verification failed, retrying ({attempt + 1}/{self.config.max_verification_retries})")
                raw_output = await self._call_llm(
                    prompt + f"\n\nPrevious answer had issues: {'; '.join(verification_notes)}\nPlease fix and regenerate."
                )
                answer, citations = self._parse_output(raw_output, context)
                verified, verification_notes = await self._verify_citations(
                    answer, citations, context
                )
        
        took_ms = (time.time() - start) * 1000
        
        return GeneratedAnswer(
            answer=answer,
            citations=citations,
            raw_output=raw_output,
            verified=verified,
            verification_notes=verification_notes,
            metadata={
                "model": self.config.model_name,
                "context_count": len(context),
                "took_ms": took_ms,
                "strategy": GenerationStrategy.CITATION_FIRST.value,
            }
        )
    
    def _prepare_context(self, context: List[RetrievalResult]) -> List[Dict[str, Any]]:
        """Prepare context with citation IDs."""
        result = []
        for i, r in enumerate(context):
            citation_id = f"DOC-{i+1}"
            result.append({
                "id": citation_id,
                "content": r.chunk.content[:2000],  # Limit length
                "metadata": r.chunk.metadata,
                "score": r.score,
            })
        return result
    
    def _build_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Build citation-first prompt."""
        context_str = "\n\n".join([
            f"[{c['id']}] {c['content']}"
            for c in context
        ])
        
        citation_format_instruction = {
            "bracket": "Use square bracket citations like [DOC-1, DOC-2] after each claim.",
            "inline": "Use inline citations like (Sources: DOC-1, DOC-2) after each claim.",
            "footnote": "Use footnote citations like ^1^ after each claim.",
        }[self.config.citation_format]
        
        return f"""You are a precise legal research assistant for Indian IP law, Ayurveda, Traditional Knowledge, and Biodiversity law.

QUERY: {query}

CONTEXT DOCUMENTS:
{context_str}

INSTRUCTIONS:
1. Answer the query based ONLY on the provided context documents.
2. {citation_format_instruction}
3. Every factual claim MUST have at least one citation.
4. If the context doesn't contain enough information, state that clearly.
5. Be precise and cite specific sections, articles, or clauses when mentioned.
6. Maximum {self.config.max_citations_per_claim} citations per claim.
7. Do not hallucinate or add information not in the context.

ANSWER:"""
    
    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM API."""
        if self._llm_type == "openai":
            response = await self._llm.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
            )
            return response.choices[0].message.content
        elif self._llm_type == "anthropic":
            response = await self._llm.messages.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        return ""
    
    def _parse_output(self, raw: str, context: List[RetrievalResult]) -> Tuple[str, List[Citation]]:
        """Parse output and extract citations."""
        # Extract citations from text
        citation_pattern = r'\[([A-Z0-9\-\,\s]+)\]'
        citations_map = {}
        
        # Find all citations
        matches = re.findall(citation_pattern, raw)
        for match in matches:
            ids = [id.strip() for id in match.split(',')]
            for cid in ids:
                if cid not in citations_map:
                    # Find the matching context
                    ctx = next((c for c in context if c.chunk.id == cid or f"DOC-{context.index(c)+1}" == cid), None)
                    if ctx:
                        citations_map[cid] = {
                            "chunk": ctx.chunk,
                            "content": ctx.chunk.content[:500],
                        }
        
        # Build citation objects
        citation_objects = []
        for cid, info in citations_map.items():
            citation_objects.append(Citation(
                claim="",  # Would need more sophisticated parsing
                source_ids=[cid],
                source_texts=[info["content"]],
                confidence=info["chunk"].score if hasattr(info["chunk"], 'score') else 1.0,
                citation_id=cid,
            ))
        
        # Clean answer text (remove citation markers for clean version)
        clean_answer = re.sub(citation_pattern, '', raw).strip()
        
        return clean_answer, citation_objects
    
    async def _verify_citations(
        self,
        answer: str,
        citations: List[Citation],
        context: List[RetrievalResult],
    ) -> Tuple[bool, List[str]]:
        """Verify that citations support the claims."""
        notes = []
        
        # Simple verification: check that all cited sources exist in context
        context_ids = {f"DOC-{i+1}" for i in range(len(context))}
        
        for citation in citations:
            for sid in citation.source_ids:
                if sid not in context_ids:
                    notes.append(f"Citation {sid} not found in context")
        
        # Check if answer has content
        if not answer or len(answer) < 10:
            notes.append("Answer too short or empty")
        
        return len(notes) == 0, notes
    
    async def _fallback_generate(
        self,
        query: str,
        context: List[RetrievalResult],
    ) -> GeneratedAnswer:
        """Fallback generation without LLM."""
        if not context:
            return GeneratedAnswer(
                answer="No relevant documents found for your query.",
                citations=[],
                raw_output="",
                verified=True,
                metadata={"fallback": True},
            )
        
        # Simple extractive answer
        top_chunks = context[:3]
        answer_parts = []
        citations = []
        
        for i, r in enumerate(top_chunks):
            cid = f"DOC-{i+1}"
            content = r.chunk.content[:500]
            answer_parts.append(f"{content}")
            citations.append(Citation(
                claim=content[:100],
                source_ids=[cid],
                source_texts=[content],
                confidence=r.score,
                citation_id=cid,
            ))
        
        answer = "\n\n".join(answer_parts)
        formatted_citations = " ".join([c.format(self.config.citation_format) for c in citations])
        answer = f"{answer} {formatted_citations}"
        
        return GeneratedAnswer(
            answer=answer,
            citations=citations,
            raw_output=answer,
            verified=True,
            metadata={"fallback": True},
        )
    
    async def close(self) -> None:
        self._llm = None
        self._initialized = False


class StructuredGenerator(CitationFirstGenerator):
    """Generator that outputs structured JSON responses."""
    
    def __init__(self, config: GenerationConfig):
        super().__init__(config)
    
    def _build_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        base_prompt = super()._build_prompt(query, context)
        
        schema = self.config.response_schema or {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "citations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim": {"type": "string"},
                            "source_ids": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["claim", "source_ids"],
                    },
                },
                "confidence": {"type": "number"},
            },
            "required": ["answer", "citations", "confidence"],
        }
        
        return f"""{base_prompt}

IMPORTANT: Output ONLY valid JSON matching this schema:
{json.dumps(schema, indent=2)}"""


class GeneratorFactory:
    """Factory for creating generators."""
    
    _generators: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, generator_class: type) -> None:
        cls._generators[name] = generator_class
    
    @classmethod
    def create(cls, config: GenerationConfig, strategy: str = "citation_first") -> BaseGenerator:
        if strategy not in cls._generators:
            raise ValueError(f"Unknown generator strategy: {strategy}")
        return cls._generators[strategy](config)
    
    @classmethod
    def get_available(cls) -> List[str]:
        return list(cls._generators.keys())


# Register built-in generators
GeneratorFactory.register(GenerationStrategy.CITATION_FIRST.value, CitationFirstGenerator)
GeneratorFactory.register(GenerationStrategy.STRUCTURED.value, StructuredGenerator)