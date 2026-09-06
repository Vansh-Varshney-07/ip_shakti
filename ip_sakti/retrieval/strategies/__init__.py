"""
Phase 3: Multi-Strategy Retrieval Enhancement
Advanced retrieval strategies for IP-SAKTI.
"""

from ip_sakti.retrieval.strategies.base import RetrievalStrategy, RetrievalConfig, SearchRequest, SearchResult
from ip_sakti.retrieval.strategies.lexical import LexicalRetriever
from ip_sakti.retrieval.strategies.vector import VectorRetriever
from ip_sakti.retrieval.strategies.hybrid import HybridRetriever
from ip_sakti.retrieval.strategies.graph import GraphRetriever
from ip_sakti.retrieval.strategies.citation_based import CitationBasedRetriever
from ip_sakti.retrieval.strategies.multi_hop import MultiHopRetriever
from ip_sakti.retrieval.strategies.exact_match import ExactMatchRetriever
from ip_sakti.retrieval.strategies.fusion import FusionRetriever

__all__ = [
    "RetrievalStrategy",
    "RetrievalConfig",
    "SearchRequest",
    "SearchResult",
    "LexicalRetriever",
    "VectorRetriever",
    "HybridRetriever",
    "GraphRetriever",
    "CitationBasedRetriever",
    "MultiHopRetriever",
    "ExactMatchRetriever",
    "FusionRetriever",
]