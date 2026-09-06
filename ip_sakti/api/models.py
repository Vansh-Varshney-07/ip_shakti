"""
API Request/Response Models
Phase 18: Pydantic models for all API endpoints.
"""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, field_validator


# ============================================================
# ENUMS
# ============================================================

class JurisdictionCode(str, Enum):
    """Jurisdiction identifiers."""
    INDIA = "INDIA"
    INTERNATIONAL = "INTERNATIONAL"
    US = "US"
    EP = "EP"
    WO = "WO"
    UK = "UK"
    CN = "CN"
    JP = "JP"
    # Indian states can be added as needed


class LanguageCode(str, Enum):
    """Supported languages (ISO 639-1)."""
    EN = "en"
    HI = "hi"
    TA = "ta"
    BN = "bn"
    TE = "te"
    MR = "mr"
    GU = "gu"
    KN = "kn"
    ML = "ml"
    PA = "pa"
    OR = "or"
    AS = "as"
    UR = "ur"


class QueryIntent(str, Enum):
    """User query intent classification."""
    PROVISION_LOOKUP = "provision_lookup"
    DEFINITION_QUERY = "definition_query"
    PROCEDURE_QUERY = "procedure_query"
    COMPLIANCE_CHECK = "compliance_check"
    CASE_LAW_SEARCH = "case_law_search"
    CROSS_REFERENCE = "cross_reference"
    AMENDMENT_HISTORY = "amendment_history"
    JURISDICTION_COMPARE = "jurisdiction_compare"
    FORMULATION_CLASSIFY = "formulation_classify"
    GENERAL_LEGAL = "general_legal"
    # Additional intents
    PATENT_SEARCH = "patent_search"
    TRADEMARK_SEARCH = "trademark_search"
    COPYRIGHT_SEARCH = "copyright_search"
    DESIGN_SEARCH = "design_search"
    LEGAL_RESEARCH = "legal_research"
    FREEDOM_TO_OPERATE = "freedom_to_operate"
    VALIDITY_CHALLENGE = "validity_challenge"
    LICENSING = "licensing"


class AuthorityTier(str, Enum):
    """Authority tier labels."""
    TIER_1 = "TIER_1"
    TIER_2 = "TIER_2"
    TIER_3 = "TIER_3"
    TIER_4 = "TIER_4"
    TIER_5 = "TIER_5"
    TIER_6 = "TIER_6"


class DocumentType(str, Enum):
    """Document types."""
    ACT = "act"
    RULE = "rule"
    REGULATION = "regulation"
    NOTIFICATION = "notification"
    ORDER = "order"
    CIRCULAR = "circular"
    GUIDELINE = "guideline"
    TREATY = "treaty"
    PROTOCOL = "protocol"
    CASE_LAW = "case_law"
    REGISTRY_RECORD = "registry_record"
    PHARMACOPOEIA = "pharmacopoeia"
    FORMULARY = "formulary"


class IngestionStatus(str, Enum):
    """Ingestion job status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class RetrievalStrategy(str, Enum):
    """Retrieval strategy."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    GRAPH = "graph"


# ============================================================
# CORE MODELS
# ============================================================

class SourceReference(BaseModel):
    """Source reference for citations."""
    source_id: str
    source_name: str
    canonical_url: str
    authority_tier: AuthorityTier
    retrieved_at: datetime
    content_hash: str


class TextSpan(BaseModel):
    """Text span within a chunk."""
    start: int
    end: int
    text: str


class Citation(BaseModel):
    """Complete citation for generated answers."""
    legal_citation: str
    source_reference: SourceReference
    chunk_id: str
    text_span: Optional[TextSpan] = None
    amendment_status: str = "original"
    effective_date: Optional[str] = None
    amendment_act: Optional[str] = None

    def format_legal(self) -> str:
        """Format as legal citation."""
        parts = [self.legal_citation]
        if self.amendment_status != "original":
            parts.append(f"({self.amendment_status.value if hasattr(self.amendment_status, 'value') else self.amendment_status}")
            if self.effective_date:
                parts.append(f"effective {self.effective_date}")
            if self.amendment_act:
                parts.append(f"by {self.amendment_act}")
            parts.append(")")
        parts.append(f"Source: {self.source_reference.source_name} ({self.source_reference.canonical_url})")
        return " ".join(parts)

    def format_inline(self) -> str:
        """Format as inline citation [1]."""
        return f"[{self.legal_citation}]"


class RetrievedChunk(BaseModel):
    """Retrieved chunk with scores."""
    chunk_id: str
    document_id: str
    text: str
    hierarchy: Dict[str, Any]
    retrieval_score: float
    authority_score: float
    combined_score: float
    rank: int
    matched_terms: List[str] = []
    source_provenance: Dict[str, Any]


class EvidenceChunk(BaseModel):
    """Evidence chunk for generation."""
    chunk: RetrievedChunk
    relevance_score: float
    authority_score: float
    citation: Citation
    claim_support: str = ""


class GeneratedSegment(BaseModel):
    """A segment of generated answer."""
    text: str
    claims: List[str] = []
    citations: List[Citation] = []


class Claim(BaseModel):
    """A verifiable claim in generated answer."""
    claim_id: str
    text: str
    claim_type: str
    confidence: float
    supporting_chunks: List[EvidenceChunk] = []
    authority: Optional[Dict[str, Any]] = None


class GeneratedAnswer(BaseModel):
    """Complete generated answer."""
    answer_id: str
    query: str
    segments: List[GeneratedSegment]
    citations: List[Citation]
    overall_confidence: float
    jurisdiction: JurisdictionCode
    language: LanguageCode
    processing_time_ms: int
    retrieval_stats: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}


class QueryResponseModel(BaseModel):
    """Complete query response."""
    query_id: str
    answer: GeneratedAnswer
    retrieved_chunks: List[RetrievedChunk]
    intent: QueryIntent
    jurisdiction: JurisdictionCode
    warnings: List[str] = []
    metadata: Dict[str, Any] = {}


# ============================================================
# QUERY ENDPOINT MODELS
# ============================================================

class QueryRequest(BaseModel):
    """Incoming query request."""
    query: str = Field(..., min_length=1, max_length=10000, description="User query text")
    user_id: str = Field(..., description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier for conversation continuity")
    jurisdiction: Optional[JurisdictionCode] = Field(None, description="Override jurisdiction detection")
    language: LanguageCode = Field(default=LanguageCode.EN, description="Response language")
    max_results: int = Field(default=10, ge=1, le=50, description="Maximum results to return")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata filters")
    intent: Optional[QueryIntent] = Field(None, description="Override intent classification")
    require_citations: bool = Field(default=True, description="Require citations in response")
    stream: bool = Field(default=False, description="Stream response tokens")


class StreamQueryRequest(QueryRequest):
    """Streaming query request."""
    stream: bool = Field(default=True, description="Must be true for streaming")


class QueryResponse(BaseModel):
    """Query response wrapper."""
    success: bool = True
    data: QueryResponseModel
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str = Field(default_factory=lambda: str(uuid4()))


class EscalationRequest(BaseModel):
    """Request a local human-facilitator review without external services."""
    query: str = Field(..., min_length=1, max_length=10000)
    reason: str = Field(..., min_length=5, max_length=2000)
    user_id: str = Field(..., min_length=1, max_length=200)
    query_id: Optional[str] = Field(None, max_length=100)
    jurisdiction: JurisdictionCode = JurisdictionCode.INDIA


class EscalationResponse(BaseModel):
    """Acknowledgement for a locally queued facilitator review."""
    success: bool = True
    escalation_id: str
    status: str = "queued"
    message: str


class BatchQueryRequest(BaseModel):
    """Batch query request."""
    queries: List[QueryRequest] = Field(..., min_length=1, max_length=100, description="List of queries")
    batch_id: Optional[str] = Field(None, description="Batch identifier")


class BatchQueryItem(BaseModel):
    """Single item in batch query response."""
    index: int
    request: QueryRequest
    response: Optional[QueryResponseModel] = None
    error: Optional[str] = None


class BatchQueryResponse(BaseModel):
    """Batch query response."""
    success: bool = True
    batch_id: str
    items: List[BatchQueryItem]
    completed: int = 0
    failed: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StreamEvent(BaseModel):
    """Server-sent event for streaming."""
    event: str  # "stage", "token", "complete", "error"
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================
# DOCUMENT ENDPOINT MODELS
# ============================================================

class DocumentSearchRequest(BaseModel):
    """Document search request."""
    query: Optional[str] = Field(None, description="Search query")
    jurisdiction: Optional[JurisdictionCode] = Field(None)
    document_type: Optional[DocumentType] = Field(None)
    authority_tier: Optional[AuthorityTier] = Field(None)
    date_from: Optional[datetime] = Field(None)
    date_to: Optional[datetime] = Field(None)
    language: Optional[LanguageCode] = Field(None)
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    sort_by: str = Field(default="relevance", pattern="^(relevance|date|title|authority)$")
    sort_order: str = Field(default="desc", pattern="^(asc|desc)$")


class DocumentSummary(BaseModel):
    """Document summary for search results."""
    document_id: str
    title: str
    document_type: DocumentType
    jurisdiction: JurisdictionCode
    authority_tier: AuthorityTier
    language: LanguageCode
    version: str
    effective_date: Optional[datetime] = None
    source_url: str
    snippet: Optional[str] = None
    chunk_count: int = 0


class DocumentSearchResponse(BaseModel):
    """Document search response."""
    success: bool = True
    results: List[DocumentSummary]
    total: int
    limit: int
    offset: int
    query: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DocumentChunkDetail(BaseModel):
    """Detailed chunk information."""
    chunk_id: str
    hierarchy: Dict[str, Any]
    text: str
    chunk_type: str
    token_count: int
    citations: List[Dict[str, Any]] = []


class DocumentDetailResponse(BaseModel):
    """Full document detail response."""
    success: bool = True
    document_id: str
    title: str
    document_type: DocumentType
    jurisdiction: JurisdictionCode
    authority_tier: AuthorityTier
    language: LanguageCode
    version: str
    effective_date: Optional[datetime] = None
    amendment_history: List[Dict[str, Any]] = []
    source_provenance: Dict[str, Any]
    chunks: List[DocumentChunkDetail]
    metadata: Dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================
# INGESTION ENDPOINT MODELS
# ============================================================

class IngestionSourceConfig(BaseModel):
    """Configuration for a single ingestion source."""
    source_id: str
    source_type: DocumentType
    url: Optional[str] = None
    file_path: Optional[str] = None
    jurisdiction: JurisdictionCode = JurisdictionCode.INDIA
    authority_tier: AuthorityTier = AuthorityTier.TIER_1
    language: LanguageCode = LanguageCode.EN
    metadata: Dict[str, Any] = {}


class IngestionRequest(BaseModel):
    """Ingestion request."""
    sources: List[IngestionSourceConfig] = Field(..., min_length=1)
    options: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=0, ge=-10, le=10)


class IngestionResponse(BaseModel):
    """Ingestion response."""
    success: bool = True
    job_id: str
    status: IngestionStatus = IngestionStatus.PENDING
    sources_queued: int
    estimated_chunks: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class IngestionStatusResponse(BaseModel):
    """Ingestion job status response."""
    success: bool = True
    job_id: str
    status: IngestionStatus
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    sources_processed: int = 0
    sources_total: int = 0
    chunks_created: int = 0
    errors: List[str] = []
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================
# ADMIN ENDPOINT MODELS
# ============================================================

class HealthComponent(BaseModel):
    """Health check for a single component."""
    name: str
    status: str  # "healthy", "degraded", "unhealthy"
    latency_ms: Optional[float] = None
    details: Dict[str, Any] = {}


class HealthResponse(BaseModel):
    """Health check response."""
    success: bool = True
    status: str  # "healthy", "degraded", "unhealthy"
    version: str = "1.0.0"
    components: List[HealthComponent]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MetricsResponse(BaseModel):
    """System metrics response."""
    success: bool = True
    metrics: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConfigResponse(BaseModel):
    """Configuration response (sanitized)."""
    success: bool = True
    config: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ExperimentSummary(BaseModel):
    """Experiment summary."""
    experiment_id: str
    name: str
    description: str
    hypothesis: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    metrics: Dict[str, float] = {}


class ExperimentListResponse(BaseModel):
    """Experiment list response."""
    success: bool = True
    experiments: List[ExperimentSummary]
    total: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================
# ERROR MODELS
# ============================================================

class ErrorDetail(BaseModel):
    """Error detail."""
    code: str
    message: str
    field: Optional[str] = None
    context: Dict[str, Any] = {}


class ErrorResponse(BaseModel):
    """Standard error response."""
    success: bool = False
    error: ErrorDetail
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================
# AUTH MODELS
# ============================================================

class TokenRequest(BaseModel):
    """Token request for OAuth2 password grant."""
    grant_type: str = "password"
    username: str
    password: str
    scope: str = "api"
    client_id: Optional[str] = None
    client_secret: Optional[str] = None


class TokenResponse(BaseModel):
    """Token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None
    scope: str = "api"


class APIKeyCreate(BaseModel):
    """API key creation request."""
    name: str = Field(..., min_length=1, max_length=100)
    scopes: List[str] = Field(default=["read"])
    expires_in_days: Optional[int] = Field(None, ge=1, le=3650)
    rate_limit_tier: str = Field(default="standard")


class APIKeyResponse(BaseModel):
    """API key response."""
    key_id: str
    name: str
    key: str  # Only returned once on creation
    scopes: List[str]
    rate_limit_tier: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None


class APIKeyListResponse(BaseModel):
    """API key list response."""
    keys: List[APIKeyResponse]
    total: int
