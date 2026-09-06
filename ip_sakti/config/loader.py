"""
Configuration loader for IP-SAKTI Sahayak.
Loads settings from YAML with environment variable overrides.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import yaml
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)


@dataclass
class Settings:
    """Application settings loaded from YAML."""
    
    # Raw config dict
    _config: Dict[str, Any] = field(default_factory=dict, repr=False)

    # Test substitutes are enabled only by an explicit runtime switch.
    environment: str = "development"
    test_mode: bool = False
    corpus_root: str = ""
    index_path: str = "data/runtime/ip_sakti_index.sqlite3"
    api_cors_origins: List[str] = field(default_factory=list)
    
    # Technology stack
    embedding_model: str = "BAAI/bge-m3"
    embedding_fallback: str = "jinaai/jina-embeddings-v3"
    embedding_dimensions: int = 1024
    embedding_max_tokens: int = 8192
    embedding_batch_size: int = 32
    
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_fallback: str = "jinaai/jina-reranker-v2-base-multilingual"
    reranker_top_k: int = 100
    reranker_batch_size: int = 16
    
    llm_primary: str = "gpt-4o"
    llm_fallback: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096
    llm_top_p: float = 0.9
    llm_provider: str = "nvidia"
    llm_base_url: str = "https://integrate.api.nvidia.com/v1"
    
    vector_db_type: str = "qdrant"
    vector_db_host: str = "localhost"
    vector_db_port: int = 6333
    vector_db_collection: str = "ip_sakti_chunks"
    
    lexical_db_type: str = "elasticsearch"
    lexical_db_host: str = "localhost"
    lexical_db_port: int = 9200
    lexical_db_index: str = "ip_sakti_lexical"
    
    graph_db_type: str = "neo4j"
    graph_db_host: str = "localhost"
    graph_db_port: int = 7687
    graph_db_database: str = "neo4j"
    graph_db_username: str = "neo4j"
    
    relational_db_type: str = "postgresql"
    relational_db_host: str = "localhost"
    relational_db_port: int = 5432
    relational_db_database: str = "ip_sakti"
    relational_db_pool_size: int = 20
    
    cache_type: str = "redis"
    cache_host: str = "localhost"
    cache_port: int = 6379
    cache_db: int = 0
    cache_ttl_default: int = 3600
    
    # Authority tiers
    authority_tiers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Security
    jwt_algorithm: str = "RS256"
    jwt_issuer: str = "ip-sakti"
    jwt_audience: str = "ip-sakti-api"
    jwt_access_token_ttl: int = 900
    jwt_refresh_token_ttl: int = 604800
    jwt_key_rotation_days: int = 90
    
    api_key_hash_algorithm: str = "sha256"
    api_key_default_ttl_days: int = 365
    api_key_max_keys_per_user: int = 10
    
    mfa_methods: list = field(default_factory=lambda: ["totp", "webauthn"])
    mfa_totp_issuer: str = "IP-SAKTI Sahayak"
    mfa_webauthn_rp_id: str = "ip-sakti.local"
    
    encryption_algorithm: str = "AES-256-GCM"
    encryption_kms_backend: str = "local"
    encryption_key_rotation_days: int = 90
    
    rate_limit_tiers: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    mtls_enabled: bool = True
    tls_min_version: str = "TLSv1.3"
    
    # Ingestion allowed paths
    allowed_ingestion_paths: List[str] = field(default_factory=lambda: [
        str(Path(__file__).resolve().parents[2] / "data"),
        str(Path.cwd() / "data"),
    ])
    
    # Ingestion
    ingestion_sources: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    ingestion_parsing: Dict[str, Any] = field(default_factory=dict)
    ingestion_chunking: Dict[str, Any] = field(default_factory=dict)
    ingestion_deduplication: Dict[str, Any] = field(default_factory=dict)
    ingestion_validation: Dict[str, Any] = field(default_factory=dict)
    
    # Ingestion settings as properties for backward compatibility
    @property
    def ingestion(self) -> Dict[str, Any]:
        """Get ingestion configuration."""
        return self._config.get("ingestion", {})
    
    @property
    def max_document_size_mb(self) -> int:
        """Maximum document size in MB."""
        return self.ingestion.get("validation", {}).get("max_document_size_mb", 50)
    
    @property
    def fetch_timeout_seconds(self) -> int:
        """Fetch timeout in seconds."""
        return self.ingestion.get("sources", {}).get("fetch_timeout_seconds", 30)
    
    @property
    def chunk_size(self) -> int:
        """Chunk size for ingestion."""
        return self.ingestion.get("chunking", {}).get("chunk_size", 512)
    
    @property
    def chunk_overlap(self) -> int:
        """Chunk overlap for ingestion."""
        return self.ingestion.get("chunking", {}).get("chunk_overlap", 50)
    
    @property
    def chunking_strategy(self) -> str:
        """Chunking strategy for ingestion."""
        return self.ingestion.get("chunking", {}).get("strategy", "semantic")
    
    # Retrieval
    retrieval_hybrid: Dict[str, Any] = field(default_factory=dict)
    retrieval_filters: Dict[str, Any] = field(default_factory=dict)
    retrieval_caching: Dict[str, Any] = field(default_factory=dict)
    retrieval_scaling: Dict[str, Any] = field(default_factory=dict)
    
    # RAG Pipeline
    rag_pipeline_stages: list = field(default_factory=list)
    rag_retrieval: Dict[str, Any] = field(default_factory=dict)
    rag_generation: Dict[str, Any] = field(default_factory=dict)
    rag_validation: Dict[str, Any] = field(default_factory=dict)
    
    # Evaluation
    evaluation_benchmarks: list = field(default_factory=list)
    evaluation_metrics: Dict[str, list] = field(default_factory=dict)
    evaluation_experiment_tracking: Dict[str, Any] = field(default_factory=dict)
    
    # Deployment
    deployment_environment: str = "development"
    deployment_replicas: Dict[str, int] = field(default_factory=dict)
    deployment_resources: Dict[str, Dict[str, str]] = field(default_factory=dict)
    deployment_autoscaling: Dict[str, Any] = field(default_factory=dict)
    
    # Observability
    observability_logging: Dict[str, Any] = field(default_factory=dict)
    observability_metrics: Dict[str, Any] = field(default_factory=dict)
    observability_tracing: Dict[str, Any] = field(default_factory=dict)
    observability_alerting: Dict[str, Any] = field(default_factory=dict)
    
    # Module selection configuration (Phase 1.5)
    modules: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self._apply_env_overrides()
    
    @property
    def rag(self) -> Dict[str, Any]:
        """Get RAG pipeline configuration."""
        return self._config.get("rag", {})
    
    @property
    def max_query_variants(self) -> int:
        """Maximum number of query variants for rewriting."""
        return self.rag.get("retrieval", {}).get("top_k_initial", 5)
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        self.environment = os.getenv("IP_SAKTI_ENV", os.getenv("ENVIRONMENT", self.environment))
        self.test_mode = os.getenv("IP_SAKTI_TEST_MODE", "true" if self.test_mode else "false").lower() in {
            "1", "true", "yes", "on"
        }
        self.corpus_root = os.getenv("IP_SAKTI_CORPUS_ROOT", self.corpus_root)
        self.index_path = os.getenv("IP_SAKTI_INDEX_PATH", self.index_path)
        cors = os.getenv("IP_SAKTI_CORS_ORIGINS")
        if cors:
            self.api_cors_origins = [origin.strip() for origin in cors.split(",") if origin.strip()]
        # Database URLs
        if db_url := os.getenv("DATABASE_URL"):
            self._config.setdefault("relational_db", {})["url"] = db_url
        
        if redis_url := os.getenv("REDIS_URL"):
            self._config.setdefault("cache", {})["url"] = redis_url
        
        if vector_url := os.getenv("VECTOR_DB_URL"):
            self._config.setdefault("vector_db", {})["url"] = vector_url
        
        if graph_url := os.getenv("GRAPH_DB_URL"):
            self._config.setdefault("graph_db", {})["url"] = graph_url
        
        # API keys
        if openai_key := os.getenv("OPENAI_API_KEY"):
            self._config.setdefault("llm", {})["api_key"] = openai_key
        if provider := os.getenv("IP_SAKTI_LLM_PROVIDER"):
            self.llm_provider = provider.lower()
        if base_url := os.getenv("IP_SAKTI_LLM_BASE_URL"):
            self.llm_base_url = base_url.rstrip("/")
        if model := os.getenv("IP_SAKTI_LLM_MODEL"):
            self.llm_primary = model
        
        # Environment
        if self.environment:
            self.deployment_environment = self.environment
        
        # Log level
        if log_level := os.getenv("LOG_LEVEL"):
            self._config.setdefault("observability", {}).setdefault("logging", {})["level"] = log_level


def load_settings(config_path: Optional[str] = None) -> Settings:
    """Load settings from YAML file."""
    if config_path is None:
        # Default to config/settings.yaml relative to this file
        base_dir = Path(__file__).parent
        config_path = base_dir / "settings.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    settings = Settings(_config=config)
    
    # Technology
    tech = config.get("technology", {})
    emb = tech.get("embedding", {})
    settings.embedding_model = emb.get("primary", settings.embedding_model)
    settings.embedding_fallback = emb.get("fallback", settings.embedding_fallback)
    settings.embedding_dimensions = emb.get("dimensions", settings.embedding_dimensions)
    settings.embedding_max_tokens = emb.get("max_tokens", settings.embedding_max_tokens)
    settings.embedding_batch_size = emb.get("batch_size", settings.embedding_batch_size)
    
    rerank = tech.get("reranker", {})
    settings.reranker_model = rerank.get("primary", settings.reranker_model)
    settings.reranker_fallback = rerank.get("fallback", settings.reranker_fallback)
    settings.reranker_top_k = rerank.get("top_k", settings.reranker_top_k)
    settings.reranker_batch_size = rerank.get("batch_size", settings.reranker_batch_size)
    
    llm = tech.get("llm", {})
    settings.llm_primary = llm.get("primary", settings.llm_primary)
    settings.llm_fallback = llm.get("fallback", settings.llm_fallback)
    settings.llm_temperature = llm.get("temperature", settings.llm_temperature)
    settings.llm_max_tokens = llm.get("max_tokens", settings.llm_max_tokens)
    settings.llm_top_p = llm.get("top_p", settings.llm_top_p)
    settings.llm_provider = llm.get("provider", settings.llm_provider)
    settings.llm_base_url = llm.get("base_url", settings.llm_base_url).rstrip("/")
    
    vdb = tech.get("vector_db", {})
    settings.vector_db_type = vdb.get("type", settings.vector_db_type)
    settings.vector_db_host = vdb.get("host", settings.vector_db_host)
    settings.vector_db_port = vdb.get("port", settings.vector_db_port)
    settings.vector_db_collection = vdb.get("collection", settings.vector_db_collection)
    
    ldb = tech.get("lexical_search", {})
    settings.lexical_db_type = ldb.get("type", settings.lexical_db_type)
    settings.lexical_db_host = ldb.get("host", settings.lexical_db_host)
    settings.lexical_db_port = ldb.get("port", settings.lexical_db_port)
    settings.lexical_db_index = ldb.get("index", settings.lexical_db_index)
    
    gdb = tech.get("graph_db", {})
    settings.graph_db_type = gdb.get("type", settings.graph_db_type)
    settings.graph_db_host = gdb.get("host", settings.graph_db_host)
    settings.graph_db_port = gdb.get("port", settings.graph_db_port)
    settings.graph_db_database = gdb.get("database", settings.graph_db_database)
    settings.graph_db_username = gdb.get("username", settings.graph_db_username)
    
    rdb = tech.get("relational_db", {})
    settings.relational_db_type = rdb.get("type", settings.relational_db_type)
    settings.relational_db_host = rdb.get("host", settings.relational_db_host)
    settings.relational_db_port = rdb.get("port", settings.relational_db_port)
    settings.relational_db_database = rdb.get("database", settings.relational_db_database)
    settings.relational_db_pool_size = rdb.get("pool_size", settings.relational_db_pool_size)
    
    cache = tech.get("cache", {})
    settings.cache_type = cache.get("type", settings.cache_type)
    settings.cache_host = cache.get("host", settings.cache_host)
    settings.cache_port = cache.get("port", settings.cache_port)
    settings.cache_db = cache.get("db", settings.cache_db)
    settings.cache_ttl_default = cache.get("ttl_default", settings.cache_ttl_default)
    
    # Authority
    settings.authority_tiers = config.get("authority", {}).get("tiers", {})
    
    # Security
    sec = config.get("security", {})
    jwt = sec.get("jwt", {})
    settings.jwt_algorithm = jwt.get("algorithm", settings.jwt_algorithm)
    settings.jwt_issuer = jwt.get("issuer", settings.jwt_issuer)
    settings.jwt_audience = jwt.get("audience", settings.jwt_audience)
    settings.jwt_access_token_ttl = jwt.get("access_token_ttl", settings.jwt_access_token_ttl)
    settings.jwt_refresh_token_ttl = jwt.get("refresh_token_ttl", settings.jwt_refresh_token_ttl)
    settings.jwt_key_rotation_days = jwt.get("key_rotation_days", settings.jwt_key_rotation_days)
    
    api_keys = sec.get("api_keys", {})
    settings.api_key_hash_algorithm = api_keys.get("hash_algorithm", settings.api_key_hash_algorithm)
    settings.api_key_default_ttl_days = api_keys.get("default_ttl_days", settings.api_key_default_ttl_days)
    settings.api_key_max_keys_per_user = api_keys.get("max_keys_per_user", settings.api_key_max_keys_per_user)
    
    mfa = sec.get("mfa", {})
    settings.mfa_methods = mfa.get("methods", settings.mfa_methods)
    settings.mfa_totp_issuer = mfa.get("totp_issuer", settings.mfa_totp_issuer)
    settings.mfa_webauthn_rp_id = mfa.get("webauthn_rp_id", settings.mfa_webauthn_rp_id)
    
    enc = sec.get("encryption", {})
    settings.encryption_algorithm = enc.get("algorithm", settings.encryption_algorithm)
    settings.encryption_kms_backend = enc.get("kms_backend", settings.encryption_kms_backend)
    settings.encryption_key_rotation_days = enc.get("key_rotation_days", settings.encryption_key_rotation_days)
    
    settings.rate_limit_tiers = sec.get("rate_limiting", {}).get("tiers", {})
    
    net = sec.get("network", {})
    settings.mtls_enabled = net.get("mtls_enabled", settings.mtls_enabled)
    settings.tls_min_version = net.get("tls_min_version", settings.tls_min_version)
    
    # Allowed ingestion paths
    settings.allowed_ingestion_paths = sec.get("allowed_ingestion_paths", settings.allowed_ingestion_paths)
    repo_data = str(Path(__file__).resolve().parents[2] / "data")
    if repo_data not in settings.allowed_ingestion_paths:
        settings.allowed_ingestion_paths.append(repo_data)
    
    # Ingestion
    ing = config.get("ingestion", {})
    settings.ingestion_sources = ing.get("sources", {})
    settings.ingestion_parsing = ing.get("parsing", {})
    settings.ingestion_chunking = ing.get("chunking", {})
    settings.ingestion_deduplication = ing.get("deduplication", {})
    settings.ingestion_validation = ing.get("validation", {})
    
    # Retrieval
    ret = config.get("retrieval", {})
    settings.retrieval_hybrid = ret.get("hybrid", {})
    settings.retrieval_filters = ret.get("filters", {})
    settings.retrieval_caching = ret.get("caching", {})
    settings.retrieval_scaling = ret.get("scaling", {})
    
    # RAG
    rag = config.get("rag", {})
    settings.rag_pipeline_stages = rag.get("pipeline", {}).get("stages", [])
    settings.rag_retrieval = rag.get("retrieval", {})
    settings.rag_generation = rag.get("generation", {})
    settings.rag_validation = rag.get("validation", {})
    
    # Evaluation
    eval_cfg = config.get("evaluation", {})
    settings.evaluation_benchmarks = eval_cfg.get("benchmarks", [])
    settings.evaluation_metrics = eval_cfg.get("metrics", {})
    settings.evaluation_experiment_tracking = eval_cfg.get("experiment_tracking", {})
    
    # Deployment
    dep = config.get("deployment", {})
    settings.deployment_environment = dep.get("environment", settings.deployment_environment)
    settings.deployment_replicas = dep.get("replicas", {})
    settings.deployment_resources = dep.get("resources", {})
    settings.deployment_autoscaling = dep.get("autoscaling", {})
    
    # Observability
    obs = config.get("observability", {})
    settings.observability_logging = obs.get("logging", {})
    settings.observability_metrics = obs.get("metrics", {})
    settings.observability_tracing = obs.get("tracing", {})
    settings.observability_alerting = obs.get("alerting", {})
    
    # Module selection (Phase 1.5)
    settings.modules = config.get("modules", {})

    # Re-apply environment variables after YAML so deployment settings win over
    # repository defaults without requiring edits to the checked-in config.
    settings._apply_env_overrides()
    
    return settings


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get global settings instance (singleton)."""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def reload_settings(config_path: Optional[str] = None) -> Settings:
    """Reload settings from config file."""
    global _settings
    _settings = load_settings(config_path)
    return _settings
