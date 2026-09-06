"""
Core package for IP Sakti RAG system.
"""

from ip_sakti.core.models import *
from ip_sakti.core.interfaces import *
from ip_sakti.core.di import *
from ip_sakti.core.services import *

__all__ = [
    # Models
    "Document",
    "LegalChunk", 
    "QueryAnalysis",
    "RAGResponse",
    "RetrievalResult",
    "RetrievalStrategy",
    
    # Interfaces
    "IConfig",
    "IDocumentLoader",
    "IDocumentValidator",
    "IChunker",
    "IEmbedder",
    "IVectorStore",
    "IRetriever",
    "IHybridRetriever",
    "IReranker",
    "IQueryAnalyzer",
    "IQueryRewriter",
    "IQueryRouter",
    "IGenerator",
    "ICitationGenerator",
    "IAgent",
    "IPlannerAgent",
    "IRetrieverAgent",
    "ICriticAgent",
    "IRefinerAgent",
    "IKnowledgeGraph",
    "IEvaluator",
    "IMultimodalExtractor",
    "IIngestionPipeline",
    "IRAGPipeline",
    "IAuthoritySystem",
    "ITranslator",
    "IMultilingualEmbedder",
    "ICache",
    "ICircuitBreaker",
    "IMetricsCollector",
    
    # DI
    "ServiceLifetime",
    "ServiceDescriptor",
    "IServiceProvider",
    "IServiceScope",
    "ServiceCollection",
    "ServiceProvider",
    "ServiceScope",
    "ApplicationBuilder",
    "Application",
    "ServiceLocator",
    "create_application",
    "register_core_services",
    "singleton",
    "transient",
    "scoped",
    "inject",
    
    # Services
    "configure_services",
    "create_application",
    "get_service_provider",
    "get_rag_pipeline",
    "get_authority_system",
    "get_vector_store",
    "get_embedder",
]