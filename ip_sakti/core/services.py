"""
Service configuration and dependency injection setup for IP Sakti.
Uses the new modular interfaces from ip_sakti.interfaces.
"""
from ip_sakti.core.container import (
    ServiceCollection,
    ServiceProvider,
    ApplicationBuilder,
    ServiceLocator,
    register_core_services
)
from ip_sakti.config.loader import Settings, get_settings
from ip_sakti.core.adapters import AdapterFactory

# Import new interfaces
from ip_sakti.interfaces import (
    IConfig, IChunker, IDocumentLoader, IEmbedder, IVectorStore,
    IAuthoritySystem, IRetriever, IReranker, IGenerator,
    IIngestionPipeline, IRAGPipeline, IQueryProcessor,
    IKnowledgeGraph, IPipelineOrchestrator,
)


def _create_config_factory(adapter_factory: AdapterFactory):
    return adapter_factory.create_config()

def _create_document_loader_factory(adapter_factory: AdapterFactory):
    return adapter_factory.create_document_loader()

def _create_chunker_factory(adapter_factory: AdapterFactory):
    return adapter_factory.create_chunker()

def _create_embedder_factory(adapter_factory: AdapterFactory):
    return adapter_factory.create_embedder()

def _create_vector_store_factory(adapter_factory: AdapterFactory):
    return adapter_factory.create_vector_store()

def _create_authority_system_factory(adapter_factory: AdapterFactory):
    return adapter_factory.create_authority_system()

def _create_retriever_factory(adapter_factory: AdapterFactory):
    return adapter_factory.create_retriever()

def _create_reranker_factory(adapter_factory: AdapterFactory):
    return adapter_factory.create_reranker()

def _create_query_processor_factory(adapter_factory: AdapterFactory):
    return adapter_factory.create_query_processor()

def _create_generator_factory(adapter_factory: AdapterFactory):
    return adapter_factory.create_generator()

def _create_knowledge_graph_factory(adapter_factory: AdapterFactory):
    return adapter_factory.create_knowledge_graph()

def _create_ingestion_pipeline_factory(adapter_factory: AdapterFactory):
    return adapter_factory.create_ingestion_pipeline()

def _create_rag_pipeline_factory(adapter_factory: AdapterFactory):
    return adapter_factory.create_rag_pipeline()

def _create_pipeline_orchestrator_factory(adapter_factory: AdapterFactory):
    return adapter_factory.create_pipeline_orchestrator()


def configure_services(builder: ApplicationBuilder) -> ApplicationBuilder:
    """Configure all services for the application."""
    
    # Import adapters (will be created in Phase 1.3)
    from ip_sakti.core.adapters import AdapterFactory
    
    # Register factory as singleton
    builder.services.add_singleton(AdapterFactory)
    
    # Register adapters using factory pattern with constructor injection
    # We use a factory that receives the AdapterFactory as a dependency
    
    # Config
    builder.services.add_singleton(IConfig, factory=_create_config_factory)
    
    # Document processing
    builder.services.add_singleton(IDocumentLoader, factory=_create_document_loader_factory)
    builder.services.add_singleton(IChunker, factory=_create_chunker_factory)
    
    # Embedding
    builder.services.add_singleton(IEmbedder, factory=_create_embedder_factory)
    
    # Vector store
    builder.services.add_singleton(IVectorStore, factory=_create_vector_store_factory)
    
    # Authority
    builder.services.add_singleton(IAuthoritySystem, factory=_create_authority_system_factory)
    
    # Retrieval
    builder.services.add_singleton(IRetriever, factory=_create_retriever_factory)
    builder.services.add_singleton(IReranker, factory=_create_reranker_factory)
    
    # Query Processing (NEW)
    builder.services.add_singleton(IQueryProcessor, factory=_create_query_processor_factory)
    
    # Generation
    builder.services.add_singleton(IGenerator, factory=_create_generator_factory)
    
    # Knowledge Graph (NEW)
    builder.services.add_singleton(IKnowledgeGraph, factory=_create_knowledge_graph_factory)
    
    # Pipelines
    builder.services.add_singleton(IIngestionPipeline, factory=_create_ingestion_pipeline_factory)
    builder.services.add_singleton(IRAGPipeline, factory=_create_rag_pipeline_factory)
    
    # Pipeline Orchestrator (NEW)
    builder.services.add_singleton(IPipelineOrchestrator, factory=_create_pipeline_orchestrator_factory)

    return builder


def build_application() -> ApplicationBuilder:
    """Build and configure the application."""
    builder = ApplicationBuilder()
    from ip_sakti.core.container import register_core_services_builder
    builder = register_core_services_builder(builder)
    builder.configure(configure_services)
    return builder


# Global application builder
_app_builder: ApplicationBuilder = None


def get_app_builder() -> ApplicationBuilder:
    """Get the global application builder."""
    global _app_builder
    if _app_builder is None:
        _app_builder = build_application()
    return _app_builder


def get_service_provider() -> ServiceProvider:
    """Get the global service provider (builds if needed)."""
    return get_app_builder().build().services


def initialize_services() -> ServiceProvider:
    """Initialize all services and return the provider."""
    provider = get_service_provider()
    ServiceLocator.set_provider(provider)
    return provider


# Convenience function to get services
def get_service(service_type):
    """Get a service from the global service provider."""
    provider = get_service_provider()
    return provider.get_service(service_type)


def get_services(service_type):
    """Get all services of a type from the global service provider."""
    provider = get_service_provider()
    return provider.get_services(service_type)