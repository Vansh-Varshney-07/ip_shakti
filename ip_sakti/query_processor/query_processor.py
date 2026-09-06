"""
Advanced Query Processing for IP-SAKTI.
Implements query analysis, rewriting, and routing.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from ip_sakti.interfaces.query_processor import (
    IQueryProcessor, IQueryAnalyzer, IQueryRewriter, IQueryRouter,
    QueryAnalysis, QueryRewrite, QueryRoute,
)
from ip_sakti.interfaces.retriever import QueryIntent, QueryComplexity, RetrievalStrategy

logger = logging.getLogger(__name__)


# =============================================================================
# Legal Entity Patterns
# =============================================================================

# Indian legal citation patterns
ACT_PATTERN = re.compile(
    r'\b(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+Act,?\s+(\d{4})\b'
)
SECTION_PATTERN = re.compile(
    r'\b(?:section|sec\.?|§)\s*(\d+[A-Z]?(?:\(\d+\))?(?:\(\w+\))?)\b', re.IGNORECASE
)
DATE_PATTERN = re.compile(
    r'\b(?:on|dated|from|since|after|before)\s+(\d{1,2}\s+\w+\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4})\b', re.IGNORECASE
)
JURISDICTION_PATTERN = re.compile(
    r'\b(?:in|under|pursuant to)\s+(?:the\s+)?(India|Indian|Central|State|Supreme Court|High Court|District Court|Tribunal|Authority)\b', re.IGNORECASE
)

# Legal intent keywords
INTENT_KEYWORDS = {
    QueryIntent.SEARCH: [
        "find", "search", "look for", "locate", "what is", "where is", 
        "show me", "list", "get", "retrieve"
    ],
    QueryIntent.LEGAL_RESEARCH: [
        "research", "analyze", "examine", "study", "investigate", 
        "compare", "review", "interpret", "understand"
    ],
    QueryIntent.CITATION_LOOKUP: [
        "cite", "citation", "reference", "quote", "source", "authority"
    ],
    QueryIntent.COMPLIANCE_CHECK: [
        "comply", "compliance", "requirement", "obligation", "must", 
        "shall", "required", "mandatory", "checklist"
    ],
    QueryIntent.PRECEDENT_SEARCH: [
        "precedent", "case law", "judgment", "ruling", "decision", 
        "order", "held that", "established"
    ],
    QueryIntent.DRAFTING: [
        "draft", "prepare", "create", "write", "formulate", "template",
        "clause", "provision", "agreement", "contract"
    ],
    QueryIntent.COMPARISON: [
        "compare", "contrast", "difference", "similar", "versus", "vs",
        "better", "advantage", "disadvantage"
    ],
    QueryIntent.TEMPORAL: [
        "when", "date", "timeline", "history", "evolution", "amendment",
        "repealed", "effective", "commencement"
    ],
}

COMPLEXITY_INDICATORS = {
    QueryComplexity.SIMPLE: {
        "max_entities": 2,
        "max_sub_queries": 1,
        "keywords": ["what", "where", "who", "define", "meaning"]
    },
    QueryComplexity.MODERATE: {
        "max_entities": 5,
        "max_sub_queries": 3,
        "keywords": ["how", "why", "explain", "process", "procedure", "requirements"]
    },
    QueryComplexity.COMPLEX: {
        "max_entities": 10,
        "max_sub_queries": 5,
        "keywords": ["analyze", "compare", "evaluate", "assess", "implications"]
    },
    QueryComplexity.MULTI_HOP: {
        "max_entities": 20,
        "max_sub_queries": 10,
        "keywords": ["multi-step", "chain", "dependency", "cascade", "implications of"]
    },
}


# =============================================================================
# Query Analyzer
# =============================================================================

class LegalQueryAnalyzer(IQueryAnalyzer):
    """Legal domain query analyzer with entity extraction and decomposition."""
    
    def __init__(self):
        self._compiled_patterns = {
            "acts": ACT_PATTERN,
            "sections": SECTION_PATTERN,
            "dates": DATE_PATTERN,
            "jurisdictions": JURISDICTION_PATTERN,
        }
    
    @property
    def name(self) -> str:
        return "legal_query_analyzer"
    
    async def analyze(
        self, 
        query: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> QueryAnalysis:
        """Analyze query for intent, complexity, entities."""
        
        # Extract entities
        entities = await self.extract_entities(query)
        
        # Determine intent
        intent = self._classify_intent(query)
        
        # Determine complexity
        complexity = self._assess_complexity(query, entities)
        
        # Extract legal citations
        legal_citations = self._extract_legal_citations(query)
        acts_referenced = entities.get("acts", [])
        sections_referenced = entities.get("sections", [])
        dates_referenced = entities.get("dates", [])
        jurisdiction_hints = entities.get("jurisdictions", [])
        
        # Check for special requirements
        requires_multi_hop = complexity == QueryComplexity.MULTI_HOP or len(acts_referenced) > 1
        requires_comparison = intent == QueryIntent.COMPARISON or "compare" in query.lower()
        requires_temporal = intent == QueryIntent.TEMPORAL or len(dates_referenced) > 0
        
        return QueryAnalysis(
            intent=intent,
            complexity=complexity,
            entities=[{"type": k, "values": v} for k, v in entities.items() if v],
            legal_citations=legal_citations,
            acts_referenced=acts_referenced,
            sections_referenced=sections_referenced,
            dates_referenced=dates_referenced,
            jurisdiction_hints=jurisdiction_hints,
            requires_multi_hop=requires_multi_hop,
            requires_comparison=requires_comparison,
            requires_temporal=requires_temporal,
            confidence=0.8,
            metadata={
                "query_length": len(query),
                "entity_count": sum(len(v) for v in entities.values()),
                "has_conversation_history": bool(conversation_history),
            }
        )
    
    async def extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities from query."""
        entities = {
            "acts": [],
            "sections": [],
            "dates": [],
            "jurisdictions": [],
            "organizations": [],
            "persons": [],
        }
        
        # Extract acts
        for match in ACT_PATTERN.finditer(query):
            act_name = match.group(1).strip()
            year = match.group(2)
            entities["acts"].append(f"{act_name} Act, {year}")
        
        # Extract sections
        for match in SECTION_PATTERN.finditer(query):
            entities["sections"].append(match.group(1))
        
        # Extract dates
        for match in DATE_PATTERN.finditer(query):
            entities["dates"].append(match.group(1))
        
        # Extract jurisdictions
        for match in JURISDICTION_PATTERN.finditer(query):
            entities["jurisdictions"].append(match.group(1))
        
        return entities
    
    async def decompose(self, query: str) -> List[str]:
        """Decompose complex query into sub-queries."""
        sub_queries = []
        
        # Split by conjunctions that indicate separate questions
        conjunctions = [" and ", " or ", " also ", " additionally ", " furthermore ", " moreover "]
        parts = [query]
        for conj in conjunctions:
            new_parts = []
            for part in parts:
                new_parts.extend(part.split(conj))
            parts = [p.strip() for p in new_parts if p.strip()]
        
        # If we have multiple parts, use them as sub-queries
        if len(parts) > 1:
            sub_queries = parts
        else:
            # Try to decompose based on question marks
            question_parts = [p.strip() for p in query.split("?") if p.strip()]
            if len(question_parts) > 1:
                sub_queries = question_parts
        
        return sub_queries if sub_queries else [query]
    
    def _classify_intent(self, query: str) -> QueryIntent:
        """Classify query intent based on keywords."""
        query_lower = query.lower()
        scores = {intent: 0 for intent in QueryIntent}
        
        for intent, keywords in INTENT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    scores[intent] += 1
        
        # Return intent with highest score, default to SEARCH
        max_intent = max(scores, key=scores.get)
        return max_intent if scores[max_intent] > 0 else QueryIntent.SEARCH
    
    def _assess_complexity(self, query: str, entities: Dict[str, List[str]]) -> QueryComplexity:
        """Assess query complexity based on various factors."""
        entity_count = sum(len(v) for v in entities.values())
        sub_queries = len([p for p in query.split("?") if p.strip()]) or 1
        
        # Check for multi-hop indicators
        multi_hop_keywords = ["chain", "dependency", "cascade", "implications", "consequence", "lead to"]
        if any(kw in query.lower() for kw in multi_hop_keywords):
            return QueryComplexity.MULTI_HOP
        
        # Assess based on entity count and sub-queries
        for complexity, indicators in COMPLEXITY_INDICATORS.items():
            if entity_count <= indicators["max_entities"] and sub_queries <= indicators["max_sub_queries"]:
                return complexity
        
        return QueryComplexity.COMPLEX
    
    def _extract_legal_citations(self, query: str) -> List[str]:
        """Extract legal citations from query."""
        citations = []
        
        # Pattern: "Section X of the Y Act, Z"
        section_act_pattern = re.compile(
            r'(?:section|sec\.?|§)\s*(\d+[A-Z]?(?:\(\d+\))?(?:\(\w+\))?)\s+(?:of\s+the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+Act,?\s+(\d{4})',
            re.IGNORECASE
        )
        
        for match in section_act_pattern.finditer(query):
            citations.append(f"Section {match.group(1)} of the {match.group(2)} Act, {match.group(3)}")
        
        # Pattern: "Y Act, Z Section X"
        act_section_pattern = re.compile(
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+Act,?\s+(\d{4}),?\s*(?:section|sec\.?|§)\s*(\d+[A-Z]?(?:\(\d+\))?(?:\(\w+\))?)',
            re.IGNORECASE
        )
        
        for match in act_section_pattern.finditer(query):
            citations.append(f"{match.group(1)} Act, {match.group(2)}, Section {match.group(3)}")
        
        return citations


# =============================================================================
# Query Rewriter
# =============================================================================

class LegalQueryRewriter(IQueryRewriter):
    """Legal domain query rewriter with HyDE, step-back, and expansion."""
    
    def __init__(self, llm_client=None):
        self._llm_client = llm_client
    
    @property
    def name(self) -> str:
        return "legal_query_rewriter"
    
    async def rewrite(self, query: str, analysis: QueryAnalysis) -> QueryRewrite:
        """Rewrite query for better retrieval."""
        
        rewritten_queries = []
        
        # 1. Original query
        rewritten_queries.append(query)
        
        # 2. Expand with synonyms and related terms
        expanded = await self.expand(query, num_variants=3)
        rewritten_queries.extend(expanded)
        
        # 3. Step-back query (generalize)
        step_back = await self._generate_step_back(query, analysis)
        if step_back and step_back != query:
            rewritten_queries.append(step_back)
        
        # 4. HyDE - hypothetical document
        hyde_doc = await self._generate_hyde(query, analysis)
        
        # 5. Sub-queries from decomposition
        sub_queries = await self._generate_sub_queries(query, analysis)
        
        # Deduplicate while preserving order
        seen = set()
        unique_queries = []
        for q in rewritten_queries:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)
        
        return QueryRewrite(
            original_query=query,
            rewritten_queries=unique_queries[:5],  # Limit to 5 variants
            hyde_document=hyde_doc,
            step_back_query=step_back,
            sub_queries=sub_queries,
            expansion_terms=self._get_expansion_terms(query, analysis),
            metadata={
                "num_variants": len(unique_queries),
                "has_hyde": bool(hyde_doc),
                "has_step_back": bool(step_back),
            }
        )
    
    async def expand(self, query: str, num_variants: int = 3) -> List[str]:
        """Expand query into multiple variants."""
        variants = []
        query_lower = query.lower()
        
        # Legal synonyms
        synonyms = {
            "act": ["legislation", "statute", "law", "enactment"],
            "section": ["provision", "clause", "article"],
            "rule": ["regulation", "bye-law", "notification"],
            "judgment": ["decision", "ruling", "order", "precedent"],
            "court": ["tribunal", "authority", "bench"],
            "compliance": ["adherence", "conformance", "observance"],
            "requirement": ["obligation", "mandate", "necessity"],
            "penalty": ["punishment", "fine", "sanction", "consequence"],
        }
        
        # Generate variants by replacing terms
        for original, replacements in synonyms.items():
            if original in query_lower:
                for replacement in replacements[:num_variants]:
                    variant = query_lower.replace(original, replacement)
                    if variant != query_lower:
                        variants.append(variant)
        
        # Add jurisdiction-specific variants
        if "india" not in query_lower and "indian" not in query_lower:
            variants.append(f"{query} in India")
            variants.append(f"Indian {query}")
        
        # Add "legal" prefix for general queries
        if not any(term in query_lower for term in ["legal", "law", "act", "section", "court"]):
            variants.append(f"legal {query}")
        
        return variants[:num_variants]
    
    def _get_expansion_terms(self, query: str, analysis: QueryAnalysis) -> List[str]:
        """Get expansion terms based on query and analysis."""
        terms = []
        
        # Add act names as expansion terms
        terms.extend(analysis.acts_referenced)
        
        # Add section references
        terms.extend(analysis.sections_referenced)
        
        # Add jurisdiction terms
        terms.extend(analysis.jurisdiction_hints)
        
        # Add intent-specific terms
        if analysis.intent == QueryIntent.COMPLIANCE_CHECK:
            terms.extend(["requirements", "obligations", "checklist", "procedure"])
        elif analysis.intent == QueryIntent.PRECEDENT_SEARCH:
            terms.extend(["case law", "precedent", "judgment", "ruling"])
        
        return list(set(terms))
    
    async def _generate_step_back(self, query: str, analysis: QueryAnalysis) -> Optional[str]:
        """Generate a step-back (more general) query."""
        # If query is very specific, generalize it
        if analysis.sections_referenced and analysis.acts_referenced:
            # E.g., "Section 3(d) of Patents Act 1970" -> "Patents Act 1970 patentability criteria"
            act = analysis.acts_referenced[0] if analysis.acts_referenced else ""
            return f"{act} overview and key provisions"
        
        if analysis.intent == QueryIntent.CITATION_LOOKUP:
            return f"overview of {query}"
        
        return None
    
    async def _generate_hyde(self, query: str, analysis: QueryAnalysis) -> Optional[str]:
        """Generate hypothetical document (HyDE) for the query."""
        # In a real implementation, this would use an LLM
        # For now, construct a template-based hypothetical answer
        if analysis.intent == QueryIntent.LEGAL_RESEARCH:
            return f"This document discusses {query}. It covers the relevant legal provisions, "
            "case law interpretations, and practical implications."
        
        return None
    
    async def _generate_sub_queries(self, query: str, analysis: QueryAnalysis) -> List[str]:
        """Generate sub-queries for complex queries."""
        sub_queries = []
        
        # Use decomposition from analyzer
        decomposed = await self._decompose_for_sub_queries(query)
        sub_queries.extend(decomposed)
        
        # Add act-specific sub-queries
        for act in analysis.acts_referenced:
            sub_queries.append(f"key provisions of {act}")
            sub_queries.append(f"recent amendments to {act}")
        
        # Add section-specific sub-queries
        for section in analysis.sections_referenced:
            sub_queries.append(f"interpretation of section {section}")
        
        return sub_queries[:5]
    
    async def _decompose_for_sub_queries(self, query: str) -> List[str]:
        """Decompose query into sub-questions."""
        # Simple rule-based decomposition
        parts = []
        
        # Split by question marks
        question_parts = [p.strip() for p in query.split("?") if p.strip()]
        if len(question_parts) > 1:
            return question_parts
        
        # Split by "and" that connects separate questions
        if " and " in query.lower():
            # Check if it's connecting two questions
            and_parts = [p.strip() for p in query.split(" and ")]
            if len(and_parts) == 2 and any(q in p.lower() for p in and_parts for q in ["what", "how", "why", "when", "where", "who"]):
                return and_parts
        
        return []


# =============================================================================
# Query Router
# =============================================================================

class LegalQueryRouter(IQueryRouter):
    """Legal domain query router for retrieval strategy selection."""
    
    def __init__(self):
        pass
    
    @property
    def name(self) -> str:
        return "legal_query_router"
    
    async def route(
        self, 
        query: str, 
        analysis: Optional[QueryAnalysis] = None
    ) -> QueryRoute:
        """Determine retrieval strategy for query."""
        
        if analysis is None:
            # Quick analysis if not provided
            from ip_sakti.query_processor.query_processor import LegalQueryAnalyzer
            analyzer = LegalQueryAnalyzer()
            analysis = await analyzer.analyze(query)
        
        # Determine primary strategy
        primary_strategy = self._select_primary_strategy(query, analysis)
        
        # Determine fallback strategies
        fallback_strategies = self._select_fallback_strategies(primary_strategy, analysis)
        
        # Determine if KG is needed
        use_kg = await self.should_use_kg(query)
        
        # Determine if graph traversal is needed
        use_graph = await self.should_use_graph(query)
        
        # Determine reranking needs
        requires_reranking = analysis.complexity in [QueryComplexity.COMPLEX, QueryComplexity.MULTI_HOP]
        rerank_stages = ["cross_encoder"] if requires_reranking else []
        
        return QueryRoute(
            primary_strategy=primary_strategy,
            fallback_strategies=fallback_strategies,
            use_kg=use_kg,
            use_graph=use_graph,
            use_contextual=analysis.requires_multi_hop,
            requires_reranking=requires_reranking,
            rerank_stages=rerank_stages,
            metadata={
                "intent": analysis.intent.value,
                "complexity": analysis.complexity.value,
                "entity_count": len(analysis.entities),
            }
        )
    
    async def should_use_kg(self, query: str) -> bool:
        """Determine if KG retrieval is needed."""
        kg_indicators = [
            "relationship", "connection", "linked", "related to", "associated with",
            "entity", "person", "organization", "company", "inventor", "assignee",
            "patent family", "priority", "citation", "cited by", "cites"
        ]
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in kg_indicators)
    
    async def should_use_graph(self, query: str) -> bool:
        """Determine if graph traversal is needed."""
        graph_indicators = [
            "path", "traverse", "chain", "dependency", "hierarchy", "tree",
            "network", "graph", "connected", "link", "relationship between"
        ]
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in graph_indicators)
    
    def _select_primary_strategy(self, query: str, analysis: QueryAnalysis) -> str:
        """Select primary retrieval strategy based on query analysis."""
        
        query_lower = query.lower()
        
        # Exact citation lookup -> lexical search
        if analysis.intent == QueryIntent.CITATION_LOOKUP:
            return RetrievalStrategy.LEXICAL.value
        
        # Precedent search -> hybrid with heavy vector weight
        if analysis.intent == QueryIntent.PRECEDENT_SEARCH:
            return RetrievalStrategy.HYBRID.value
        
        # Multi-hop -> hybrid with KG
        if analysis.requires_multi_hop or analysis.complexity == QueryComplexity.MULTI_HOP:
            return RetrievalStrategy.HYBRID.value
        
        # Comparison -> hybrid
        if analysis.intent == QueryIntent.COMPARISON:
            return RetrievalStrategy.HYBRID.value
        
        # Temporal -> hybrid with date filtering
        if analysis.requires_temporal:
            return RetrievalStrategy.HYBRID.value
        
        # Compliance check -> hybrid with authority weighting
        if analysis.intent == QueryIntent.COMPLIANCE_CHECK:
            return RetrievalStrategy.HYBRID.value
        
        # Drafting -> vector for semantic similarity
        if analysis.intent == QueryIntent.DRAFTING:
            return RetrievalStrategy.VECTOR.value
        
        # Default to hybrid for balanced retrieval
        return RetrievalStrategy.HYBRID.value
    
    def _select_fallback_strategies(
        self, 
        primary: str, 
        analysis: QueryAnalysis
    ) -> List[str]:
        """Select fallback strategies."""
        all_strategies = [
            RetrievalStrategy.HYBRID.value,
            RetrievalStrategy.VECTOR.value,
            RetrievalStrategy.LEXICAL.value,
            RetrievalStrategy.HYBRID_RRF.value,
        ]
        
        # Remove primary from fallbacks
        fallbacks = [s for s in all_strategies if s != primary]
        
        # For complex queries, add more fallbacks
        if analysis.complexity in [QueryComplexity.COMPLEX, QueryComplexity.MULTI_HOP]:
            return fallbacks[:3]
        
        return fallbacks[:2]


# =============================================================================
# Composite Query Processor
# =============================================================================

class CompositeQueryProcessor(IQueryProcessor):
    """Composite query processor combining analyzer, rewriter, and router."""
    
    def __init__(
        self,
        analyzer: Optional[IQueryAnalyzer] = None,
        rewriter: Optional[IQueryRewriter] = None,
        router: Optional[IQueryRouter] = None,
    ):
        self._analyzer = analyzer or LegalQueryAnalyzer()
        self._rewriter = rewriter or LegalQueryRewriter()
        self._router = router or LegalQueryRouter()
    
    @property
    def name(self) -> str:
        return "composite_query_processor"
    
    async def analyze(
        self, 
        query: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> QueryAnalysis:
        """Analyze query for intent, complexity, entities."""
        return await self._analyzer.analyze(query, conversation_history)
    
    async def rewrite(self, query: str, analysis: QueryAnalysis) -> QueryRewrite:
        """Rewrite query for better retrieval."""
        return await self._rewriter.rewrite(query, analysis)
    
    async def route(
        self, 
        query: str, 
        analysis: QueryAnalysis, 
        rewrite: QueryRewrite
    ) -> QueryRoute:
        """Determine retrieval strategy for query."""
        return await self._router.route(query, analysis)
    
    async def process(
        self, 
        query: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Full query processing pipeline: analyze → rewrite → route."""
        import time
        
        start = time.time()
        
        # Step 1: Analyze
        analysis_start = time.time()
        analysis = await self.analyze(query, conversation_history)
        analysis_latency = (time.time() - analysis_start) * 1000
        
        # Step 2: Rewrite
        rewrite_start = time.time()
        rewrite = await self.rewrite(query, analysis)
        rewrite_latency = (time.time() - rewrite_start) * 1000
        
        # Step 3: Route
        route_start = time.time()
        route = await self.route(query, analysis, rewrite)
        route_latency = (time.time() - route_start) * 1000
        
        total_latency = (time.time() - start) * 1000
        
        return {
            "query": query,
            "analysis": analysis,
            "rewrite": rewrite,
            "route": route,
            "metrics": {
                "total_latency_ms": total_latency,
                "analysis_latency_ms": analysis_latency,
                "rewrite_latency_ms": rewrite_latency,
                "route_latency_ms": route_latency,
            },
            "conversation_history": conversation_history,
        }


# =============================================================================
# Simple Query Processor (fallback)
# =============================================================================

class SimpleQueryProcessor(IQueryProcessor):
    """Simple query processor for fast processing."""
    
    def __init__(self):
        self._analyzer = LegalQueryAnalyzer()
    
    @property
    def name(self) -> str:
        return "simple_query_processor"
    
    async def analyze(
        self, 
        query: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> QueryAnalysis:
        return await self._analyzer.analyze(query, conversation_history)
    
    async def rewrite(self, query: str, analysis: QueryAnalysis) -> QueryRewrite:
        # Minimal rewrite - just return original
        return QueryRewrite(
            original_query=query,
            rewritten_queries=[query],
            hyde_document=None,
            step_back_query=None,
            sub_queries=[query],
            expansion_terms=[],
            metadata={"processor": "simple"}
        )
    
    async def route(
        self, 
        query: str, 
        analysis: QueryAnalysis, 
        rewrite: QueryRewrite
    ) -> QueryRoute:
        return QueryRoute(
            primary_strategy=RetrievalStrategy.HYBRID.value,
            fallback_strategies=[RetrievalStrategy.VECTOR.value],
            use_kg=False,
            use_graph=False,
            use_contextual=False,
            requires_reranking=False,
            rerank_stages=[],
            metadata={"processor": "simple"}
        )
    
    async def process(
        self, 
        query: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        analysis = await self.analyze(query, conversation_history)
        rewrite = await self.rewrite(query, analysis)
        route = await self.route(query, analysis, rewrite)
        
        return {
            "query": query,
            "analysis": analysis,
            "rewrite": rewrite,
            "route": route,
            "metrics": {
                "total_latency_ms": 0,
                "analysis_latency_ms": 0,
                "rewrite_latency_ms": 0,
                "route_latency_ms": 0,
            },
        }


# =============================================================================
# Factory Function
# =============================================================================

def create_query_processor(config: Optional[Dict[str, Any]] = None) -> IQueryProcessor:
    """Create query processor based on configuration."""
    config = config or {}
    processor_type = config.get("type", "composite")
    
    if processor_type == "simple":
        return SimpleQueryProcessor()
    elif processor_type == "composite":
        return CompositeQueryProcessor()
    else:
        return CompositeQueryProcessor()