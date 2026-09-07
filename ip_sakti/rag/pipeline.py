"""
IP-SAKTI RAG Architecture
Phase 9: Implements the core RAG (Retrieval-Augmented Generation) pipeline.
This is the pipeline logic - retrieval engine infra is in Phase 10.
"""

import asyncio
import os
import re
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple, Union

from ip_sakti.config.loader import Settings, get_settings
from ip_sakti.core.models import (
    Document,
    DocumentChunk,
    DocumentType,
    JurisdictionCode,
    Query,
    QueryIntent,
    RetrievalResult,
    RetrievalStrategy,
    SourceAuthorityTier,
)
from ip_sakti.authority.authority_system import SourceAuthoritySystem
from ip_sakti.ingestion.chunking import ChunkingStrategy, ChunkingConfig, ChunkingStrategyType
from ip_sakti.retrieval.retrieval_engine import RetrievalEngine, RetrievalConfig, SearchRequest

logger = logging.getLogger(__name__)


class RAGStage(str, Enum):
    """Stages in the RAG pipeline."""
    QUERY_ANALYSIS = "query_analysis"
    QUERY_REWRITING = "query_rewriting"
    RETRIEVAL = "retrieval"
    RERANKING = "reranking"
    CONTEXT_CONSTRUCTION = "context_construction"
    GENERATION = "generation"
    CITATION = "citation"
    VALIDATION = "validation"
    COMPLETE = "complete"


@dataclass
class RAGContext:
    """Context passed through RAG pipeline stages."""
    query: Query
    original_query: str
    rewritten_queries: List[str] = field(default_factory=list)
    negative_queries: List[str] = field(default_factory=list)
    retrieval_results: List[RetrievalResult] = field(default_factory=list)
    negative_retrieval_results: List[RetrievalResult] = field(default_factory=list)
    reranked_results: List[RetrievalResult] = field(default_factory=list)
    context_chunks: List[DocumentChunk] = field(default_factory=list)
    generated_answer: Optional[str] = None
    citations: List[Dict[str, Any]] = field(default_factory=list)
    confidence_score: float = 0.0
    errors: List[str] = field(default_factory=list)
    current_stage: RAGStage = RAGStage.QUERY_ANALYSIS
    stage_start_time: datetime = field(default_factory=datetime.utcnow)
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class QueryAnalyzer(ABC):
    """Abstract base for query analysis."""
    
    @abstractmethod
    async def analyze(self, query: str, context: RAGContext) -> Dict[str, Any]:
        """Analyze query and return analysis results."""
        pass


class IntentClassifier(QueryAnalyzer):
    """Classify query intent for IP domain."""
    
    INTENT_PATTERNS = {
        QueryIntent.FORMULATION_CLASSIFY: [
            r'\bdrug\b', r'\bmedicine\b', r'\bformulation\b', r'\bayurved', r'\bherbal\b',
            r'\bplant(?:s)?\b', r'\bchemical(?:s)?\b', r'\bphytopharmaceutical\b',
            r'\bnutraceutical\b', r'\bcosmetic\b', r'\bmanufactur', r'\bregister(?:ed|ation)?\b',
            r'आयुर्वेदिक', r'औषधि', r'दवा',
        ],
        QueryIntent.PROCEDURE_QUERY: [
            r'\bhow\s+(?:do|can|should)\b', r'\bprocedure\b', r'\bprocess\b', r'\bsteps?\b',
            r'\bwhere\s+should\b', r'\bwhich\s+office\b', r'\bhow\s+to\s+register\b',
            r'\bregistration pathway\b', r'\bnext steps?\b', r'कदम', r'शुरुआती', r'कैसे',
        ],
        QueryIntent.COMPLIANCE_CHECK: [
            r'\bcompli(?:ant|ance)\b', r'\ballowed\b', r'\blegal(?:ly)?\b', r'\bunder\s+which\s+act\b',
            r'\bwhat\s+rules?\b', r'\bapproval\b', r'\blicen[cs]e\b', r'\bclearance\b',
            r'\badvertis(?:e|ing)\b', r'\blabell?ing\b', r'\brestriction\b', r'\bregulat(?:e|ory|ion)\b',
        ],
        QueryIntent.PATENT_SEARCH: [
            r'\bpatent\b', r'\binvention\b', r'\bclaim\b', r'\bprior art\b',
            r'\bnovelty\b', r'\binventive step\b', r'\bpatentability\b'
        ],
        QueryIntent.TRADEMARK_SEARCH: [
            r'\btrademark\b', r'\bmark\b', r'\bbrand\b', r'\blikelihood of confusion\b',
            r'\bdistinctive\b', r'\btrademark search\b'
        ],
        QueryIntent.COPYRIGHT_SEARCH: [
            r'\bcopyright\b', r'\bwork\b', r'\bauthor\b', r'\binfringement\b',
            r'\bfair use\b', r'\bderivative work\b'
        ],
        QueryIntent.DESIGN_SEARCH: [
            r'\bdesign\b', r'\bappearance\b', r'\bvisual\b', r'\bornamental\b'
        ],
        QueryIntent.LEGAL_RESEARCH: [
            r'\bcase law\b', r'\bprecedent\b', r'\bjudgment\b', r'\bruling\b',
            r'\bstatute\b', r'\bregulation\b', r'\blegal opinion\b', r'\bcourt decision\b',
            r'\babs\b', r'\bbenefit[- ]sharing\b', r'\bbiodiversity\b', r'\bnagoya\b',
            r'\btraditional knowledge\b', r'\btkdl\b', r'\bgeographical identity\b', r'\bgi\b',
        ],
        QueryIntent.FREEDOM_TO_OPERATE: [
            r'\bfreedom to operate\b', r'\bfto\b', r'\brisk\b', r'\bclearance\b'
        ],
        QueryIntent.VALIDITY_CHALLENGE: [
            r'\bvalidity\b', r'\binvalid\b', r'\brevocation\b', r'\bopposition\b',
            r'\bchallenge\b'
        ],
        QueryIntent.LICENSING: [
            r'\blicens\b', r'\broyalt\b', r'\bassign\b', r'\btransfer\b'
        ],
    }
    
    async def analyze(self, query: str, context: RAGContext) -> Dict[str, Any]:
        query_lower = query.lower()
        intent_scores = {}
        
        for intent, patterns in self.INTENT_PATTERNS.items():
            score = sum(1 for p in patterns if re.search(p, query_lower))
            if score > 0:
                intent_scores[intent] = score
        
        # Prefer explicit legal action over broad product vocabulary. A query
        # mentioning a plant and a patent is a patent search, while a query
        # asking how to register a herbal product is a procedure query.
        intent_priority = {
            QueryIntent.PATENT_SEARCH: 5,
            QueryIntent.TRADEMARK_SEARCH: 5,
            QueryIntent.COPYRIGHT_SEARCH: 5,
            QueryIntent.DESIGN_SEARCH: 5,
            QueryIntent.FREEDOM_TO_OPERATE: 5,
            QueryIntent.VALIDITY_CHALLENGE: 5,
            QueryIntent.LICENSING: 5,
            QueryIntent.PROCEDURE_QUERY: 4,
            QueryIntent.COMPLIANCE_CHECK: 4,
            QueryIntent.LEGAL_RESEARCH: 5,
            QueryIntent.FORMULATION_CLASSIFY: 1,
        }
        primary_intent = (
            max(intent_scores, key=lambda intent: (
                intent_scores[intent] + (intent_priority.get(intent, 0) if intent_priority.get(intent, 0) >= 4 else 0.5),
                intent_scores[intent],
            ))
            if intent_scores else QueryIntent.GENERAL_LEGAL
        )
        
        # Extract entities (simplified - would use NER in production)
        entities = self._extract_entities(query)
        
        # Determine jurisdiction from query
        jurisdiction = self._detect_jurisdiction(query)
        
        return {
            'intent': primary_intent,
            'intent_scores': intent_scores,
            'entities': entities,
            'jurisdiction': jurisdiction,
            'complexity': self._assess_complexity(query),
        }
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities from query (patent numbers, case citations, etc.)."""
        entities = {
            'patent_numbers': [],
            'case_citations': [],
            'trademark_numbers': [],
            'dates': [],
            'organizations': [],
        }
        
        # Patent numbers (various formats)
        patent_patterns = [
            r'\b(?:US|EP|WO|IN|CN|JP)\d{7,10}[A-Z]?\b',
            r'\b\d{4,}\/\d{4,}\b',
        ]
        for pattern in patent_patterns:
            entities['patent_numbers'].extend(re.findall(pattern, query, re.IGNORECASE))
        
        # Case citations
        case_pattern = r'\b\d{4}\s+[A-Z]+\s+\d+\b'
        entities['case_citations'].extend(re.findall(case_pattern, query))
        
        # Years
        entities['dates'].extend(re.findall(r'\b(?:19|20)\d{2}\b', query))
        
        return entities
    
    def _detect_jurisdiction(self, query: str) -> Optional[JurisdictionCode]:
        """Detect jurisdiction from query."""
        jurisdiction_keywords = {
            JurisdictionCode.INDIA: ['india', 'indian', 'ipab', 'controller of patents', 'delhi high court', 'supreme court of india'],
            JurisdictionCode.US: ['us', 'united states', 'uspto', 'federal circuit', 'supreme court', 'america'],
            JurisdictionCode.EP: ['european', 'epo', 'europe', 'ep patent'],
            JurisdictionCode.WO: ['pct', 'wipo', 'international', 'wo '],
            JurisdictionCode.UK: ['uk', 'united kingdom', 'ukipo', 'ewhc', 'ewca'],
            JurisdictionCode.CN: ['china', 'cnipa', 'chinese'],
            JurisdictionCode.JP: ['japan', 'jpo', 'japanese'],
        }
        
        query_lower = query.lower()
        for jurisdiction, keywords in jurisdiction_keywords.items():
            if any(kw in query_lower for kw in keywords):
                return jurisdiction
        
        return None
    
    def _assess_complexity(self, query: str) -> str:
        """Assess query complexity."""
        word_count = len(query.split())
        has_boolean = any(op in query.upper() for op in ['AND', 'OR', 'NOT'])
        has_quotes = '"' in query
        has_entities = bool(self._extract_entities(query)['patent_numbers'] or self._extract_entities(query)['case_citations'])
        
        if word_count > 50 or has_boolean or has_entities:
            return "complex"
        elif word_count > 20 or has_quotes:
            return "moderate"
        return "simple"


class QueryRewriter(ABC):
    """Abstract base for query rewriting."""
    
    @abstractmethod
    async def rewrite(self, query: str, analysis: Dict[str, Any], context: RAGContext) -> List[str]:
        """Rewrite query into multiple variants."""
        pass


class MultiQueryRewriter(QueryRewriter):
    """Rewrite query into multiple variants for better recall."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
    
    async def rewrite(self, query: str, analysis: Dict[str, Any], context: RAGContext) -> List[str]:
        variants = [query]  # Original query
        
        intent = analysis.get('intent', QueryIntent.GENERAL_LEGAL)
        entities = analysis.get('entities', {})
        jurisdiction = analysis.get('jurisdiction')
        
        # Add jurisdiction-specific variants
        if jurisdiction:
            variants.append(f"{query} {jurisdiction.value}")
        
        # Add intent-specific variants
        if intent == QueryIntent.PATENT_SEARCH:
            variants.extend(self._patent_variants(query, entities))
        elif intent == QueryIntent.TRADEMARK_SEARCH:
            variants.extend(self._trademark_variants(query, entities))
        elif intent == QueryIntent.LEGAL_RESEARCH:
            variants.extend(self._legal_variants(query, entities))
        
        # Add synonym expansion (simplified)
        variants.extend(self._synonym_expansion(query))
        
        # Deduplicate and limit
        unique_variants = list(dict.fromkeys(variants))
        return unique_variants[:self.settings.max_query_variants]

    def expand_for_semantic_retrieval(
        self, query: str, analysis: Dict[str, Any], limit: int
    ) -> List[str]:
        """Create diverse positive semantic probes for multi-angle legal recall."""
        angles = [
            "applicable Indian statute and regulatory category",
            "drug or formulation classification and definition",
            "registration authority and filing procedure",
            "safety efficacy manufacturing and licensing requirements",
            "patentability novelty inventive step and excluded subject matter",
            "traditional knowledge and prior-art protection",
            "biological resource access and benefit sharing obligations",
            "Ayurveda classical proprietary or new-drug treatment",
            "food nutraceutical or cosmetic alternative classification",
            "labelling packaging advertising and claims restrictions",
            "intellectual property ownership and protection options",
            "trade secret confidentiality and research disclosure risk",
            "plant variety biodiversity and source disclosure requirements",
            "geographical indication or trademark pathway",
            "international export market and treaty obligations",
            "competent authority forms fees and timelines",
            "clinical evidence pharmacopoeial standard and quality control",
            "environmental and public-health compliance concerns",
            "enforcement penalties and post-registration duties",
            "practical next steps for an AYUSH startup or researcher",
            "relevant official registry database or government portal",
            "exceptions exclusions and provisions that do not apply",
            "case law and authoritative interpretation",
            "cross-reference between patents biodiversity and drug law",
        ]
        variants = [query]
        for angle in angles:
            variants.append(f"{query}\nFocus on {angle}.")
        return variants[:max(1, limit)]

    def build_negative_probes(self, query: str, analysis: Dict[str, Any], limit: int) -> List[str]:
        """Build contrastive probes used to down-weight semantically adjacent noise."""
        return [
            f"{query}\nFind unrelated or non-applicable legal regimes to exclude.",
            f"{query}\nFind counterexamples, exclusions, and provisions that do not govern this fact pattern.",
            f"{query}\nFind foreign or purely commercial sources that should not be treated as Indian regulatory authority.",
        ][:max(0, limit)]
    
    def _patent_variants(self, query: str, entities: Dict) -> List[str]:
        variants = []
        # Add patent-specific synonyms
        synonyms = {
            'invention': ['innovation', 'device', 'method', 'system'],
            'claim': ['claims', 'claim language', 'claim scope'],
            'prior art': ['prior art references', 'background art', 'state of the art'],
            'novelty': ['new', 'novel', 'not anticipated'],
            'inventive step': ['non-obvious', 'inventive', 'non-obviousness'],
        }
        
        for term, syns in synonyms.items():
            if term in query.lower():
                for syn in syns:
                    variants.append(query.replace(term, syn))
        
        return variants
    
    def _trademark_variants(self, query: str, entities: Dict) -> List[str]:
        variants = []
        synonyms = {
            'trademark': ['mark', 'brand', 'trade mark'],
            'confusion': ['likelihood of confusion', 'confusingly similar'],
            'distinctive': ['distinctiveness', 'inherently distinctive'],
        }
        
        for term, syns in synonyms.items():
            if term in query.lower():
                for syn in syns:
                    variants.append(query.replace(term, syn))
        
        return variants
    
    def _legal_variants(self, query: str, entities: Dict) -> List[str]:
        variants = []
        synonyms = {
            'case law': ['precedent', 'case law', 'judicial precedent'],
            'statute': ['legislation', 'act', 'law'],
            'regulation': ['rule', 'regulatory provision'],
        }
        
        for term, syns in synonyms.items():
            if term in query.lower():
                for syn in syns:
                    variants.append(query.replace(term, syn))
        
        return variants
    
    def _synonym_expansion(self, query: str) -> List[str]:
        """General synonym expansion."""
        # Simplified - in production would use word embeddings or thesaurus
        return []


import re


class RetrievalStrategySelector:
    """Select optimal retrieval strategy based on query analysis."""
    
    STRATEGY_MAP = {
        QueryIntent.FORMULATION_CLASSIFY: [RetrievalStrategy.HYBRID, RetrievalStrategy.SEMANTIC, RetrievalStrategy.KEYWORD],
        QueryIntent.PROCEDURE_QUERY: [RetrievalStrategy.HYBRID, RetrievalStrategy.SEMANTIC, RetrievalStrategy.KEYWORD],
        QueryIntent.COMPLIANCE_CHECK: [RetrievalStrategy.HYBRID, RetrievalStrategy.SEMANTIC, RetrievalStrategy.KEYWORD],
        QueryIntent.PATENT_SEARCH: [RetrievalStrategy.HYBRID, RetrievalStrategy.SEMANTIC, RetrievalStrategy.KEYWORD],
        QueryIntent.TRADEMARK_SEARCH: [RetrievalStrategy.HYBRID, RetrievalStrategy.KEYWORD, RetrievalStrategy.SEMANTIC],
        QueryIntent.COPYRIGHT_SEARCH: [RetrievalStrategy.SEMANTIC, RetrievalStrategy.HYBRID],
        QueryIntent.DESIGN_SEARCH: [RetrievalStrategy.SEMANTIC, RetrievalStrategy.HYBRID],
        QueryIntent.LEGAL_RESEARCH: [RetrievalStrategy.HYBRID, RetrievalStrategy.SEMANTIC, RetrievalStrategy.GRAPH],
        QueryIntent.FREEDOM_TO_OPERATE: [RetrievalStrategy.HYBRID, RetrievalStrategy.GRAPH, RetrievalStrategy.SEMANTIC],
        QueryIntent.VALIDITY_CHALLENGE: [RetrievalStrategy.HYBRID, RetrievalStrategy.GRAPH, RetrievalStrategy.KEYWORD],
        QueryIntent.LICENSING: [RetrievalStrategy.SEMANTIC, RetrievalStrategy.HYBRID],
    }
    
    def select(self, analysis: Dict[str, Any]) -> List[RetrievalStrategy]:
        """Select retrieval strategies based on query analysis."""
        intent = analysis.get('intent', QueryIntent.GENERAL_LEGAL)
        complexity = analysis.get('complexity', 'moderate')
        
        strategies = self.STRATEGY_MAP.get(intent, [RetrievalStrategy.HYBRID])
        
        # Adjust for complexity
        if complexity == 'simple':
            # For simple queries, prefer faster strategies
            return strategies[:2]
        elif complexity == 'complex':
            # For complex queries, use more strategies
            return strategies
        
        return strategies


class Reranker(ABC):
    """Abstract base for reranking."""
    
    @abstractmethod
    async def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        context: RAGContext,
    ) -> List[RetrievalResult]:
        """Rerank retrieval results."""
        pass


class AuthorityWeightedReranker(Reranker):
    """Rerank using authority scores from Phase 5."""
    
    def __init__(self, settings: Settings, authority_system: SourceAuthoritySystem):
        self.settings = settings
        self.authority_system = authority_system
    
    async def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        context: RAGContext,
    ) -> List[RetrievalResult]:
        for result in results:
            # Get authority score from chunk metadata
            authority_score = result.chunk.metadata.get('authority_score', 0.5)
            
            # Get source tier
            tier_str = result.chunk.metadata.get('source_authority_tier', 'SECONDARY_SOURCE')
            try:
                tier = SourceAuthorityTier(tier_str)
                tier_score = self.authority_system.get_tier_score(tier)
            except ValueError:
                tier_score = 0.5
            
            # Combine scores
            original_score = result.score
            # The YAML loader exposes the authority section as a raw config
            # mapping, so read the configured retrieval weight defensively.
            authority_weight = self.settings._config.get("authority", {}).get(
                "retrieval", {}
            ).get("authority_weight", 0.3)
            
            # Reranked score = (1 - authority_weight) * original + authority_weight * authority
            result.reranked_score = (
                (1 - authority_weight) * original_score + 
                authority_weight * ((authority_score + tier_score) / 2)
            )
        
        # Sort by reranked score
        results.sort(key=lambda r: r.reranked_score or 0, reverse=True)
        return results


class CrossEncoderReranker(Reranker):
    """Cross-encoder reranker with an explicit deterministic test path."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None

    def _load_model(self):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(self.settings.reranker_model)
    
    async def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        context: RAGContext,
    ) -> List[RetrievalResult]:
        if not results:
            return results
        if self.settings.test_mode:
            for i, result in enumerate(results):
                result.reranked_score = result.score * (1.0 - i * 0.01)
            return results
        # The configured multilingual cross-encoder is a multi-gigabyte model.
        # Keep the production path responsive by making that optional; the
        # authority-weighted retrieval score remains the default reranker.
        if os.getenv("IP_SAKTI_ENABLE_CROSS_ENCODER", "false").lower() != "true":
            for result in results:
                result.reranked_score = result.score
            return results
        if self.model is None:
            self._load_model()
        pairs = [(query, result.chunk.content) for result in results]
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(None, self.model.predict, pairs)
        for result, score in zip(results, scores):
            result.reranked_score = float(score)
        results.sort(key=lambda r: r.reranked_score or 0, reverse=True)

        return results


class ContextBuilder:
    """Build context for generation from retrieved chunks."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.max_context_tokens = settings.rag_generation.get("max_context_tokens", 8000)
        self.citation_format = settings.rag_generation.get("citation_format", "bracket")
    
    def build_context(
        self,
        results: List[RetrievalResult],
        context: RAGContext,
    ) -> Tuple[str, List[DocumentChunk], List[Dict[str, Any]]]:
        """Build context string and citation list from results."""
        context_chunks = []
        citations = []
        context_parts = []
        current_tokens = 0
        seen_sources = set()
        
        for i, result in enumerate(results):
            chunk = result.chunk

            # Keep the evidence list source-diverse. Multiple probes often
            # return adjacent chunks from the same document; repeating the
            # same source makes the answer look grounded without adding new
            # authority.
            source_key = (
                chunk.metadata.get("source_path")
                or chunk.metadata.get("source_name")
                or chunk.metadata.get("title")
                or chunk.document_id
            )
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)
            
            # Estimate tokens (rough: 1 token ≈ 4 chars)
            chunk_tokens = len(chunk.content) // 4
            
            if current_tokens + chunk_tokens > self.max_context_tokens:
                break
            
            context_chunks.append(chunk)
            current_tokens += chunk_tokens
            
            # Format chunk with citation marker
            citation_id = i + 1
            citation_marker = f"[{citation_id}]"
            
            chunk_text = f"{citation_marker} {chunk.content}"
            context_parts.append(chunk_text)
            
            # Build citation
            citation = self._build_citation(chunk, citation_id, result)
            citations.append(citation)
        
        context_string = "\n\n".join(context_parts)
        return context_string, context_chunks, citations
    
    def _build_citation(
        self,
        chunk: DocumentChunk,
        citation_id: int,
        result: RetrievalResult,
    ) -> Dict[str, Any]:
        """Build citation object from chunk."""
        return {
            'id': citation_id,
            'chunk_id': chunk.id,
            'document_id': chunk.document_id,
            'content_preview': chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
            'score': result.score,
            'reranked_score': getattr(result, 'reranked_score', None),
            'authority_score': chunk.metadata.get('authority_score'),
            'source_tier': chunk.metadata.get('source_authority_tier'),
            'jurisdiction': chunk.metadata.get('jurisdiction'),
            'document_type': chunk.metadata.get('document_type'),
            'section': chunk.metadata.get('section_title') or chunk.metadata.get('patent_section') or chunk.metadata.get('statute_section') or chunk.metadata.get('case_section'),
            'claim_number': chunk.metadata.get('claim_number'),
            'source_url': chunk.metadata.get('source_url'),
            'source_path': chunk.metadata.get('source_path'),
            'source_name': chunk.metadata.get('source_name') or chunk.metadata.get('title'),
            'content_hash': chunk.metadata.get('content_hash'),
        }


class Generator(ABC):
    """Abstract base for answer generation."""
    
    @abstractmethod
    async def generate(
        self,
        query: str,
        context: str,
        citations: List[Dict[str, Any]],
        rag_context: RAGContext,
    ) -> Tuple[str, float]:
        """Generate answer with confidence score."""
        pass


class LLMGenerator(Generator):
    """LLM-based answer generator."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.provider = settings.llm_provider.lower()
        self.api_key = os.getenv("NVIDIA_API_KEY" if self.provider == "nvidia" else "OPENAI_API_KEY")
        self.model = settings.llm_primary
    
    async def generate(
        self,
        query: str,
        context: str,
        citations: List[Dict[str, Any]],
        rag_context: RAGContext,
    ) -> Tuple[str, float]:
        prompt = self._build_prompt(query, context, citations, rag_context)
        if self.settings.test_mode:
            return self._extractive_test_answer(query, citations, rag_context)
        if not self.api_key:
            variable = "NVIDIA_API_KEY" if self.provider == "nvidia" else "OPENAI_API_KEY"
            raise RuntimeError(f"{variable} is required when IP_SAKTI_TEST_MODE is disabled")

        import httpx
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                f"{self.settings.llm_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "temperature": self.settings.llm_temperature,
                    "max_tokens": self.settings.llm_max_tokens,
                    "messages": [
                        {"role": "system", "content": "You are Sahayak, a careful IP, Ayurveda, traditional-knowledge, biodiversity, and regulatory research assistant. Answer only from the supplied evidence. Use the user's facts to classify the formulation before recommending a pathway. Distinguish India from international regimes. Cite every material legal proposition with [n]. Format the answer as a legal research note: begin with a direct, qualified answer; then use the heading 'Here is the detailed legal analysis based on the provided sources:' followed by numbered reasoning points; finish with 'Conclusion', an information-not-legal-advice disclaimer, and 'Grounded Statutory Evidence (N Verified Sources)' listing cited sources with tier, jurisdiction, score, and a short supporting excerpt. Short questions may be concise, but multi-factor product or registration questions should include classification, applicable regimes, practical next steps, documents/evidence, uncertainties, and exclusions. Never invent an authority, section, form, fee, deadline, or approval."},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            response.raise_for_status()
            payload = response.json()
        answer = payload["choices"][0]["message"]["content"].strip()
        # NVIDIA-hosted models sometimes emit CJK or full-width citation
        # brackets even when the prompt requests [n]. Normalize at the
        # boundary so validation and the UI use one stable citation syntax.
        answer = re.sub(r"[【［]\s*(\d+)\s*[】］]", r"[\1]", answer)
        cited = set(int(value) for value in re.findall(r"\[(\d+)\]", answer))
        confidence = min(0.95, 0.55 + 0.1 * len(cited))
        return answer, confidence

    def _extractive_test_answer(
        self,
        query: str,
        citations: List[Dict[str, Any]],
        rag_context: RAGContext,
    ) -> Tuple[str, float]:
        """Return a relevant, clearly qualified local answer for test mode."""
        if not citations:
            return (
                "I could not find enough directly relevant evidence in the indexed "
                "sources to answer this question safely. Please add or verify the "
                "authoritative source before relying on a legal conclusion.",
                0.0,
            )

        stop_words = {
            "about", "after", "also", "which", "where", "would", "could",
            "should", "have", "from", "with", "this", "that", "under",
            "their", "there", "what", "when", "your", "into", "only",
        }
        query_terms = {
            term for term in re.findall(r"[a-z]{4,}", query.lower())
            if term not in stop_words
        }
        ranked = []
        for citation in citations:
            preview = citation.get("content_preview", "").strip()
            source = str(citation.get("source_name") or citation.get("source_path") or "")
            evidence_terms = set(re.findall(r"[a-z]{4,}", f"{source} {preview}".lower()))
            overlap = len(query_terms & evidence_terms)
            ranked.append((overlap, float(citation.get("score") or 0), citation))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        relevant = [item[2] for item in ranked if item[0] > 0][:3]

        if not relevant:
            return (
                "I could not find a directly relevant indexed passage for this "
                "question. I am not quoting nearby but potentially unrelated law. "
                "Please add or verify the authoritative source before relying on "
                "a legal conclusion.",
                0.0,
            )

        lines = [
            "Based on the provided authoritative context, the indexed material does "
            "not support an unqualified legal conclusion. The answer below is a "
            "preliminary, source-grounded research analysis of the facts supplied.",
            "",
            "Here is the detailed legal analysis based on the provided sources:",
        ]
        query_lower = query.lower()
        if re.search(r"classical|traditional|recipe|chamanprash|chyawanprash", query_lower):
            lines.extend([
                "",
                "1. Classification and patentability",
                "The product may involve a classical or traditional formulation. "
                "A classical status cannot be confirmed from the retrieved passages "
                "alone, and the indexed corpus does not contain a direct monograph "
                "for this named recipe. Verify the exact ingredients, proportions, "
                "dosage form, indications, and textual reference against an "
                "authoritative Ayurvedic source. Any genuinely new process, dosage, "
                "composition, or delivery system must be assessed separately from "
                "the traditional recipe.",
            ])
        else:
            lines.extend([
                "",
                "1. Scope of the retrieved material",
                "The retrieved passages identify potentially relevant legal material, "
                "but they do not by themselves establish every fact needed for a "
                "final filing, licence, or product-classification decision.",
            ])
        lines.extend(["", "2. Relevant source passages"])
        for citation in relevant:
            source = Path(str(citation.get("source_name") or citation.get("source_path") or "Indexed source")).name
            preview = citation.get("content_preview", "").strip()
            lines.append(f"[{citation['id']}] {source}: {preview}")
        lines.extend([
            "",
            "3. Practical next steps",
            "1. Confirm the exact formulation and its authoritative textual source.",
            "2. Separate traditional ingredients or steps from any claimed technical "
            "improvement and document the development history.",
            "3. Check the current primary statute, rules, and competent registry before "
            "filing or commercialising.",
            "4. Obtain qualified patent and regulatory advice for a filing decision.",
            "",
            "Conclusion",
            "The available evidence supports a cautious preliminary answer only. "
            "Confirm the exact product facts and consult the current primary source "
            "before taking legal, regulatory, or commercial action.",
            "",
            "Disclaimer: This is legal information, not legal advice. For specific "
            "regulatory filings or patent actions, consult a qualified patent agent "
            "or legal professional.",
            "",
            f"Grounded Statutory Evidence ({len(citations)} Verified Sources)",
        ])
        for citation in citations:
            source = Path(str(citation.get("source_name") or citation.get("source_path") or "Indexed source")).name
            tier = citation.get("source_tier") or "Unclassified"
            jurisdiction = citation.get("jurisdiction") or "Unknown jurisdiction"
            score = citation.get("score")
            score_text = f"{float(score) * 100:.1f}%" if score is not None else "n/a"
            preview = str(citation.get("content_preview") or "").strip().replace("\n", " ")
            lines.append(f"[{citation['id']}] {source}")
            lines.append(f"- {jurisdiction} - Score: {score_text}")
            lines.append(f"- {tier}")
            if preview:
                lines.append(f'"{preview}"')
        confidence = min(0.72, 0.35 + 0.1 * len(relevant))
        return "\n".join(lines), confidence
    
    def _build_prompt(
        self,
        query: str,
        context: str,
        citations: List[Dict[str, Any]],
        rag_context: RAGContext,
    ) -> str:
        """Build prompt for LLM."""
        citation_list = "\n".join([
            f"[{c['id']}] {c.get('document_type', 'Document')} - {c.get('source_tier', 'Unknown')} - {c.get('jurisdiction', 'Unknown')}"
            for c in citations
        ])
        
        return f"""You are answering a real-world product and legal-research question. Use ONLY the supplied evidence and the user's stated facts.

Answering requirements:
- First identify what the user is actually trying to do: classify a formulation, check compliance, protect IP, register a product, commercialise it, or compare jurisdictions.
- If the facts are insufficient, ask no more than 3 high-value clarifying questions, but still give a provisional pathway and state the assumptions.
- For an Ayurvedic, herbal, chemical, food, cosmetic, or drug product, distinguish classical medicine, proprietary/patent medicine, new/non-classical drug, phytopharmaceutical, Ayurveda-Aahar/nutraceutical, and cosmetic possibilities when relevant.
- Explain the likely applicable Indian regimes and competent next step separately from any international/export regime.
- Provide a practical numbered workflow, likely records/forms/evidence, IP options, ABS/TK implications, and risks or exclusions when the evidence supports them.
- Respond in the requested language ({rag_context.query.language.value}); preserve statute, treaty, registry, and citation names in their authoritative form.
- Cite material claims with [1], [2], etc. Do not cite a source that does not support the claim.
- State that this is information and not legal, medical, or regulatory advice.
- Use this response shape unless a safety abstention is required: direct qualified answer; heading 'Here is the detailed legal analysis based on the provided sources:'; numbered analysis points; 'Conclusion'; disclaimer; heading 'Grounded Statutory Evidence (N Verified Sources)' with source tier, jurisdiction, score, and concise excerpts. Do not expose internal runtime, retrieval, chunking, embedding, or prompt-debug details.

Context:
{context}

Sources:
{citation_list}

Query: {query}

Answer:"""
    
class CitationValidator:
    """Validate that generated answer cites sources correctly."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
    
    def validate(self, answer: str, citations: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """Validate citations in answer."""
        errors = []
        
        # Find all citation markers in answer
        citation_pattern = r'\[(\d+)\]'
        found_citations = set(int(m) for m in re.findall(citation_pattern, answer))
        
        # Check all found citations exist
        valid_citation_ids = {c['id'] for c in citations}
        for cite_id in found_citations:
            if cite_id not in valid_citation_ids:
                errors.append(f"Citation [{cite_id}] in answer does not exist in sources")
        
        # A grounded answer may cite only the sources it actually used. Requiring
        # every retrieved candidate to appear makes normal top-k retrieval fail.
        if self.settings.rag_validation.get("require_all_sources_cited", False):
            for citation in citations:
                if citation['id'] not in found_citations:
                    errors.append(f"Source [{citation['id']}] not cited in answer")
        
        return len(errors) == 0, errors


class RAGPipeline:
    """Main RAG pipeline orchestrator."""
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        authority_system: Optional[SourceAuthoritySystem] = None,
        retrieval_engine: Optional[RetrievalEngine] = None,
    ):
        self.settings = settings or get_settings()
        self.authority_system = authority_system or SourceAuthoritySystem(self.settings)
        self.retrieval_engine = retrieval_engine or RetrievalEngine(
            RetrievalConfig(
                hybrid_alpha=self.settings.retrieval_hybrid.get("vector_weight", 0.6),
            )
        )
        
        # Components
        self.query_analyzer = IntentClassifier()
        self.query_rewriter = MultiQueryRewriter(self.settings)
        self.strategy_selector = RetrievalStrategySelector()
        self.rerankers = [
            AuthorityWeightedReranker(self.settings, self.authority_system),
            CrossEncoderReranker(self.settings),
        ]
        self.context_builder = ContextBuilder(self.settings)
        self.generator = LLMGenerator(self.settings)
        self.citation_validator = CitationValidator(self.settings)
        
        # Stage handlers
        self.stage_handlers = {
            RAGStage.QUERY_ANALYSIS: self._stage_query_analysis,
            RAGStage.QUERY_REWRITING: self._stage_query_rewriting,
            RAGStage.RETRIEVAL: self._stage_retrieval,
            RAGStage.RERANKING: self._stage_reranking,
            RAGStage.CONTEXT_CONSTRUCTION: self._stage_context_construction,
            RAGStage.GENERATION: self._stage_generation,
            RAGStage.CITATION: self._stage_citation,
            RAGStage.VALIDATION: self._stage_validation,
        }
    
    async def run(
        self,
        query: Query,
        attachment_text: Optional[str] = None,
        attachment_name: Optional[str] = None,
    ) -> RAGContext:
        """Run the full RAG pipeline."""
        context = RAGContext(
            query=query,
            original_query=query.text,
        )
        if attachment_text:
            context.metadata["attachment_text"] = attachment_text[:60000]
            context.metadata["attachment_name"] = attachment_name or "uploaded document"
        
        try:
            for stage in RAGStage:
                if stage == RAGStage.COMPLETE:
                    break
                
                context.current_stage = stage
                context.stage_start_time = datetime.utcnow()
                
                handler = self.stage_handlers.get(stage)
                if handler:
                    await handler(context)
                
                if context.errors:
                    logger.error(f"RAG pipeline error at stage {stage}: {context.errors}")
                    break
            
            context.metrics['total_time_ms'] = sum(
                v for k, v in context.metrics.items() if k.endswith('_time_ms')
            )

            # Keep a bounded, process-local telemetry snapshot for the live
            # dashboard. These values describe completed API queries only and
            # are intentionally not treated as a durable analytics store.
            telemetry = self.retrieval_engine.stats.setdefault("query_telemetry", {
                "completed": 0,
                "cited": 0,
                "abstained": 0,
                "categories": {},
                "jurisdictions": {},
                "latency_ms": {},
            })
            telemetry["completed"] += 1
            if context.citations:
                telemetry["cited"] += 1
            if not context.generated_answer or "don't have" in context.generated_answer.lower():
                telemetry["abstained"] += 1
            category = getattr(context.query.intent, "value", None) or "general_legal"
            jurisdiction = getattr(context.query.jurisdiction, "value", None) or "INDIA"
            telemetry["categories"][category] = telemetry["categories"].get(category, 0) + 1
            telemetry["jurisdictions"][jurisdiction] = telemetry["jurisdictions"].get(jurisdiction, 0) + 1
            for key, value in context.metrics.items():
                if key.endswith("_time_ms"):
                    telemetry["latency_ms"][key] = round(float(value), 2)

            # Persist only aggregate counters; raw prompts and generated text
            # remain outside the dashboard telemetry store.
            self.retrieval_engine.persist_query_telemetry()
            
        except Exception as e:
            logger.exception("RAG pipeline failed")
            context.errors.append(str(e))
        
        return context
    
    async def _stage_query_analysis(self, context: RAGContext) -> None:
        """Analyze query intent and extract entities."""
        analysis = await self.query_analyzer.analyze(context.original_query, context)
        context.metadata['analysis'] = analysis
        context.query.intent = analysis.get('intent', QueryIntent.GENERAL_LEGAL)
        context.query.entities = analysis.get('entities', {})
        
        context.metrics['query_analysis_time_ms'] = (
            datetime.utcnow() - context.stage_start_time
        ).total_seconds() * 1000
    
    async def _stage_query_rewriting(self, context: RAGContext) -> None:
        """Rewrite query into multiple variants."""
        analysis = context.metadata.get('analysis', {})
        context.rewritten_queries = await self.query_rewriter.rewrite(
            context.original_query, analysis, context
        )
        
        context.metrics['query_rewriting_time_ms'] = (
            datetime.utcnow() - context.stage_start_time
        ).total_seconds() * 1000
        context.metrics['num_query_variants'] = len(context.rewritten_queries)
    
    async def _stage_retrieval(self, context: RAGContext) -> None:
        """Retrieve documents for all query variants."""
        analysis = context.metadata.get('analysis', {})
        strategies = self.strategy_selector.select(analysis)

        configured_positive = int(self.settings.rag_retrieval.get("positive_semantic_queries", 24))
        configured_negative = int(self.settings.rag_retrieval.get("negative_semantic_queries", 3))
        positive_limit = min(configured_positive, 4) if self.settings.test_mode else configured_positive
        positive_variants = self.query_rewriter.expand_for_semantic_retrieval(
            context.original_query, analysis, positive_limit
        )
        context.rewritten_queries = positive_variants
        context.negative_queries = self.query_rewriter.build_negative_probes(
            context.original_query, analysis, configured_negative
        )
        jurisdiction_filter = {"jurisdiction": context.query.jurisdiction.value}

        all_results = []
        # The positive pass deliberately uses semantic probes to cover the
        # formulation, IP, ABS, and procedure angles of a natural-language ask.
        positive_requests = [
            SearchRequest(
                query=variant,
                strategy=RetrievalStrategy.SEMANTIC,
                filters=jurisdiction_filter,
                top_k=self.settings.rag_retrieval.get("positive_top_k", 4),
            )
            for variant in positive_variants
        ]
        positive_responses = await asyncio.gather(*(
            self.retrieval_engine.search(request) for request in positive_requests
        ))
        for response in positive_responses:
            all_results.extend(response.results)

        # Preserve exact legal-term recall with a small hybrid/keyword pass.
        exact_requests = [
            SearchRequest(
                query=variant,
                strategy=strategy,
                # Query analysis is advisory; do not discard authoritative
                # sources when the classifier returns an unnormalised value.
                filters=jurisdiction_filter,
                top_k=self.settings.rag_retrieval.get("top_k_per_variant", 20),
            )
            for variant in [context.original_query, *context.rewritten_queries[:2]]
            for strategy in strategies[:2]
        ]
        exact_responses = await asyncio.gather(*(
            self.retrieval_engine.search(request) for request in exact_requests
        ))
        for response in exact_responses:
            all_results.extend(response.results)

        # Graph evidence adds local multi-hop/entity support without replacing
        # the citation-grounded semantic and lexical retrieval passes.
        graph_response = await self.retrieval_engine.search(SearchRequest(
            query=context.original_query,
            strategy=RetrievalStrategy.GRAPH,
            filters=jurisdiction_filter,
            top_k=self.settings.rag_retrieval.get("graph_top_k", 5),
        ))
        all_results.extend(graph_response.results)
        context.metrics['graph_searches'] = 1

        negative_requests = [
            SearchRequest(
                query=variant,
                strategy=RetrievalStrategy.SEMANTIC,
                filters=jurisdiction_filter,
                top_k=self.settings.rag_retrieval.get("negative_top_k", 3),
            )
            for variant in context.negative_queries
        ]
        negative_responses = await asyncio.gather(*(
            self.retrieval_engine.search(request) for request in negative_requests
        ))
        negative_results = [
            result
            for response in negative_responses
            for result in response.results
        ]
        context.negative_retrieval_results = negative_results

        negative_scores = {}
        for result in negative_results:
            negative_scores[result.chunk.id] = max(
                negative_scores.get(result.chunk.id, 0.0), float(result.score)
            )
        negative_weight = float(self.settings.rag_retrieval.get("negative_penalty", 0.20))
        for result in all_results:
            result.score = float(result.score) - negative_weight * negative_scores.get(result.chunk.id, 0.0)

        # Keep the canonical path resilient when an intent/strategy adapter
        # returns no candidates for an otherwise searchable corpus.
        if not all_results:
            response = await self.retrieval_engine.search(SearchRequest(
                query=context.original_query,
                strategy=RetrievalStrategy.HYBRID,
                filters=jurisdiction_filter,
                top_k=self.settings.rag_retrieval.get("top_k_final", 10),
            ))
            all_results.extend(response.results)
        
        # Deduplicate by chunk_id, then reject passages that only match broad
        # legal filler words. This prevents a patent/GI passage from being
        # presented as evidence for an unrelated Ayurvedic product question.
        seen = set()
        unique_results = []
        for result in all_results:
            if result.chunk.id not in seen:
                seen.add(result.chunk.id)
                unique_results.append(result)

        query_terms = set(re.findall(r"[a-z]{4,}", context.original_query.lower()))
        generic_terms = {
            "about", "answer", "apply", "authority", "based", "claims",
            "classified", "combination", "consider", "context", "design",
            "designed", "evidence", "exact", "explain", "filing", "from",
            "information", "ingredients", "legal", "licence", "license",
            "literature", "made", "mentioned", "method", "newly", "patent",
            "process", "product", "question", "required", "restrictions",
            "sell", "section", "source", "statute", "testing", "treated",
            "under", "what", "which", "with",
        }
        signal_terms = query_terms - generic_terms
        if signal_terms:
            minimum_overlap = 3 if len(signal_terms) >= 6 else 1
            domain_terms = {
                "ayurveda", "ayurvedic", "ashwagandha", "cosmetic", "dosage",
                "extract", "extraction", "formulation", "medicine", "neem",
                "nutraceutical", "proprietary", "therapeutic", "turmeric",
                "wellness",
            }
            requested_domain_terms = signal_terms & domain_terms
            unique_results = [
                result for result in unique_results
                if (
                    len(signal_terms & set(re.findall(r"[a-z]{4,}", result.chunk.content.lower())))
                    >= minimum_overlap
                    and (
                        not requested_domain_terms
                        or len(requested_domain_terms & set(re.findall(r"[a-z]{4,}", result.chunk.content.lower())))
                        >= min(2, len(requested_domain_terms))
                    )
                )
            ]
        
        # Sort by score
        unique_results.sort(key=lambda r: r.score, reverse=True)
        context.retrieval_results = unique_results[:self.settings.rag_retrieval.get("top_k_final", 10)]
        
        context.metrics['retrieval_time_ms'] = (
            datetime.utcnow() - context.stage_start_time
        ).total_seconds() * 1000
        context.metrics['num_results'] = len(context.retrieval_results)
        context.metrics['positive_semantic_searches'] = len(positive_variants)
        context.metrics['negative_semantic_searches'] = len(context.negative_queries)
    
    def _infer_document_type(self, intent: Optional[QueryIntent]) -> Optional[DocumentType]:
        """Infer document type from intent."""
        mapping = {
            QueryIntent.PATENT_SEARCH: DocumentType.PATENT,
            QueryIntent.TRADEMARK_SEARCH: DocumentType.TRADEMARK,
            QueryIntent.COPYRIGHT_SEARCH: DocumentType.COPYRIGHT,
            QueryIntent.DESIGN_SEARCH: DocumentType.DESIGN,
            QueryIntent.LEGAL_RESEARCH: DocumentType.CASE_LAW,
            QueryIntent.FREEDOM_TO_OPERATE: DocumentType.PATENT,
            QueryIntent.VALIDITY_CHALLENGE: DocumentType.PATENT,
            QueryIntent.LICENSING: DocumentType.PATENT,
        }
        return mapping.get(intent)
    
    async def _stage_reranking(self, context: RAGContext) -> None:
        """Rerank retrieval results."""
        results = context.retrieval_results
        
        for reranker in self.rerankers:
            results = await reranker.rerank(
                context.original_query,
                results,
                context,
            )
        
        context.reranked_results = results
        
        context.metrics['reranking_time_ms'] = (
            datetime.utcnow() - context.stage_start_time
        ).total_seconds() * 1000
    
    async def _stage_context_construction(self, context: RAGContext) -> None:
        """Build context from reranked results."""
        attachment_text = context.metadata.get("attachment_text")
        if attachment_text:
            attachment_name = context.metadata.get("attachment_name", "uploaded document")
            attachment_chunk = DocumentChunk(
                id=f"attachment:{context.query.query_id}",
                document_id=f"attachment:{context.query.query_id}",
                content=attachment_text,
                metadata={
                    "source_name": attachment_name,
                    "source_path": attachment_name,
                    "source_authority_tier": "TIER_5",
                    "document_type": "uploaded_attachment",
                    "jurisdiction": context.query.jurisdiction.value,
                },
            )
            context.reranked_results = [RetrievalResult(
                chunk=attachment_chunk,
                score=1.0,
                strategy=RetrievalStrategy.KEYWORD,
            ), *context.reranked_results]
        context_string, context_chunks, citations = self.context_builder.build_context(
            context.reranked_results,
            context,
        )
        
        context.context_chunks = context_chunks
        context.citations = citations
        context.metadata['context_string'] = context_string
        
        context.metrics['context_construction_time_ms'] = (
            datetime.utcnow() - context.stage_start_time
        ).total_seconds() * 1000
        context.metrics['context_tokens'] = len(context_string) // 4
        context.metrics['num_context_chunks'] = len(context_chunks)
    
    async def _stage_generation(self, context: RAGContext) -> None:
        """Generate answer using LLM."""
        context_string = context.metadata.get('context_string', '')
        
        answer, confidence = await self.generator.generate(
            context.original_query,
            context_string,
            context.citations,
            context,
        )
        
        context.generated_answer = answer
        context.confidence_score = confidence
        
        context.metrics['generation_time_ms'] = (
            datetime.utcnow() - context.stage_start_time
        ).total_seconds() * 1000
    
    async def _stage_citation(self, context: RAGContext) -> None:
        """Post-process citations in answer."""
        if not context.generated_answer:
            return
        # An abstention must not be followed by a bibliography of passages
        # that the answer explicitly rejected as irrelevant.
        if context.generated_answer.startswith((
            "I could not find enough directly relevant evidence",
            "I could not find a directly relevant indexed passage",
        )):
            context.citations = []
            context.context_chunks = []
            return
        valid_ids = {str(c["id"]) for c in context.citations}
        found_ids = set(re.findall(r"\[(\d+)\]", context.generated_answer))
        if not found_ids and valid_ids:
            context.generated_answer += "\n\nSources: " + ", ".join(f"[{i}]" for i in sorted(valid_ids, key=int))
    
    async def _stage_validation(self, context: RAGContext) -> None:
        """Validate generated answer."""
        if not context.generated_answer:
            context.errors.append("No answer generated")
            return
        
        is_valid, errors = self.citation_validator.validate(
            context.generated_answer,
            context.citations,
        )
        
        if not is_valid:
            context.errors.extend(errors)
        
        context.metrics['validation_time_ms'] = (
            datetime.utcnow() - context.stage_start_time
        ).total_seconds() * 1000


class StreamingRAGPipeline(RAGPipeline):
    """RAG pipeline with streaming generation support."""
    
    async def run_streaming(self, query: Query) -> AsyncGenerator[Dict[str, Any], None]:
        """Run pipeline with streaming updates."""
        context = RAGContext(query=query, original_query=query.text)
        
        # Run non-streaming stages
        for stage in [
            RAGStage.QUERY_ANALYSIS,
            RAGStage.QUERY_REWRITING,
            RAGStage.RETRIEVAL,
            RAGStage.RERANKING,
            RAGStage.CONTEXT_CONSTRUCTION,
        ]:
            context.current_stage = stage
            context.stage_start_time = datetime.utcnow()
            handler = self.stage_handlers.get(stage)
            if handler:
                await handler(context)
            
            yield {
                'stage': stage.value,
                'status': 'complete' if not context.errors else 'error',
                'metrics': context.metrics.copy(),
            }
            
            if context.errors:
                yield {'stage': 'error', 'errors': context.errors}
                return
        
        # Streaming generation
        context_string = context.metadata.get('context_string', '')
        
        # For streaming, we'd yield tokens as they're generated
        # This is a simplified version
        answer, confidence = await self.generator.generate(
            context.original_query,
            context_string,
            context.citations,
            context,
        )
        
        context.generated_answer = answer
        context.confidence_score = confidence
        
        # Yield answer in chunks (simulated streaming)
        words = answer.split()
        for i in range(0, len(words), 10):
            chunk = " ".join(words[i:i+10])
            yield {
                'stage': 'generation',
                'token': chunk + " ",
                'is_complete': i + 10 >= len(words),
            }
        
        # Validation
        await self._stage_validation(context)
        
        yield {
            'stage': 'complete',
            'answer': context.generated_answer,
            'citations': context.citations,
            'confidence': context.confidence_score,
            'metrics': context.metrics,
        }


# Factory function
def create_rag_pipeline(
    settings: Optional[Settings] = None,
    authority_system: Optional[SourceAuthoritySystem] = None,
    retrieval_engine: Optional[RetrievalEngine] = None,
    streaming: bool = False,
) -> Union[RAGPipeline, StreamingRAGPipeline]:
    """Factory to create RAG pipeline."""
    if streaming:
        return StreamingRAGPipeline(settings, authority_system, retrieval_engine)
    return RAGPipeline(settings, authority_system, retrieval_engine)
