"""
Main FastAPI Application
Phase 18: API Architecture - Complete FastAPI app with all endpoints.
"""

from __future__ import annotations
import logging
import os
import time
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ip_sakti.config.loader import Settings, get_settings
from ip_sakti.api.models import (
    BatchQueryItem,
    BatchQueryRequest,
    BatchQueryResponse,
    ConfigResponse,
    DocumentDetailResponse,
    DocumentSearchRequest,
    DocumentSearchResponse,
    DocumentSummary,
    ErrorDetail,
    ErrorResponse,
    ExperimentListResponse,
    ExperimentSummary,
    HealthComponent,
    HealthResponse,
    IngestionRequest,
    IngestionSourceConfig,
    IngestionResponse,
    IngestionStatusResponse,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
    QueryResponseModel,
    StreamEvent,
    StreamQueryRequest,
    TokenRequest,
    TokenResponse,
    APIKeyCreate,
    APIKeyResponse,
    APIKeyListResponse,
    GeneratedAnswer,
    GeneratedSegment,
    Citation,
    SourceReference,
    AuthorityTier,
    JurisdictionCode,
    LanguageCode,
    DocumentType,
    TextSpan,
    RetrievedChunk,
    DocumentChunkDetail,
    Claim,
)
from ip_sakti.api.auth import (
    CurrentUser,
    APIKeyManager,
    JWTManager,
    RateLimiter,
    get_current_user,
    get_current_user_jwt,
    get_current_user_api_key,
    require_scopes,
    require_roles,
    rate_limit_dependency,
    rate_limit_middleware,
    set_global_jwt_manager,
    set_global_api_key_manager,
    set_global_rate_limiter,
)
from ip_sakti.rag.pipeline import RAGPipeline, StreamingRAGPipeline, create_rag_pipeline
from ip_sakti.retrieval.retrieval_engine import RetrievalEngine, create_retrieval_engine, SearchRequest, RetrievalStrategy
from ip_sakti.ingestion.pipeline import IngestionPipeline, IngestionJob, IngestionStatus, create_ingestion_pipeline
from ip_sakti.ingestion.corpus_validation import validate_corpus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# GLOBAL STATE
# ============================================================

_rag_pipeline: Optional[RAGPipeline] = None
_streaming_rag_pipeline: Optional[StreamingRAGPipeline] = None
_retrieval_engine: Optional[RetrievalEngine] = None
_ingestion_pipeline: Optional[IngestionPipeline] = None
_jwt_manager: Optional[JWTManager] = None
_api_key_manager: Optional[APIKeyManager] = None
_rate_limiter: Optional[RateLimiter] = None
_ingestion_jobs: Dict[str, Dict[str, Any]] = {}


def _api_authority_tier(value: Any) -> AuthorityTier:
    """Convert internal tier values without dropping provenance."""
    raw = getattr(value, "value", value) or "TIER_6"
    raw = str(raw).upper()
    if raw.isdigit():
        raw = f"TIER_{raw}"
    try:
        return AuthorityTier(raw)
    except ValueError:
        return AuthorityTier.TIER_6


def _citation_model(citation: Dict[str, Any]) -> Citation:
    source_url = citation.get("source_url") or citation.get("canonical_url") or ""
    source_name = citation.get("source_name") or citation.get("document_type") or "Indexed legal source"
    return Citation(
        legal_citation=citation.get("section") or f"Document {citation.get('document_id', 'unknown')}",
        source_reference=SourceReference(
            source_id=str(citation.get("document_id") or citation.get("chunk_id")),
            source_name=str(source_name),
            canonical_url=str(source_url),
            authority_tier=_api_authority_tier(citation.get("source_tier")),
            retrieved_at=datetime.utcnow(),
            content_hash=str(citation.get("content_hash") or ""),
        ),
        chunk_id=str(citation.get("chunk_id")),
    )


def _retrieved_chunk_model(result: Any, rank: int) -> RetrievedChunk:
    chunk = result.chunk
    metadata = chunk.metadata or {}
    return RetrievedChunk(
        chunk_id=str(chunk.id),
        document_id=str(chunk.document_id),
        text=chunk.content,
        hierarchy={"section": metadata.get("section_title") or metadata.get("statute_section")},
        retrieval_score=float(result.score),
        authority_score=float(chunk.authority_score or metadata.get("authority_score") or 0.0),
        combined_score=float(getattr(result, "reranked_score", None) or result.score),
        rank=rank,
        matched_terms=[],
        source_provenance={
            "source_url": metadata.get("source_url"),
            "source_path": metadata.get("source_path"),
            "authority_tier": metadata.get("source_authority_tier"),
            "jurisdiction": getattr(chunk.jurisdiction, "value", chunk.jurisdiction),
            "document_type": getattr(chunk.document_type, "value", chunk.document_type),
            "content_hash": metadata.get("content_hash"),
        },
    )


# ============================================================
# LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    global _rag_pipeline, _streaming_rag_pipeline, _retrieval_engine
    global _ingestion_pipeline, _jwt_manager, _api_key_manager, _rate_limiter

    settings = get_settings()

    # Initialize components
    logger.info("Initializing API components...")

    _jwt_manager = JWTManager(settings)
    _api_key_manager = APIKeyManager(settings)
    _rate_limiter = RateLimiter(settings)

    # Set global instances for dependencies
    set_global_jwt_manager(_jwt_manager)
    set_global_api_key_manager(_api_key_manager)
    set_global_rate_limiter(_rate_limiter)

    _retrieval_engine = create_retrieval_engine()
    await _retrieval_engine.initialize()

    # Reuse the initialized persistent engine so the RAG pipeline searches the
    # production SQLite index instead of creating an empty second engine.
    _rag_pipeline = create_rag_pipeline(
        streaming=False,
        retrieval_engine=_retrieval_engine,
    )
    _streaming_rag_pipeline = create_rag_pipeline(
        streaming=True,
        retrieval_engine=_retrieval_engine,
    )

    _ingestion_pipeline = create_ingestion_pipeline(retrieval_engine=_retrieval_engine)

    logger.info("API components initialized successfully")

    yield

    # Cleanup
    logger.info("Shutting down API components...")
    if _retrieval_engine:
        await _retrieval_engine.close()
    logger.info("API components shut down")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="IP-SAKTI Sahayak API",
    description="""
    **IP-SAKTI Sahayak** - Indian Intellectual Property Legal Assistant API
    
    A RAG-based legal research system for Indian IP law with:
    - Multi-jurisdiction retrieval (India, International treaties)
    - Authority-weighted ranking (Tier 1-6)
    - Citation-first generation with verification
    - Formulation classification (8 classes)
    - Multilingual support (12+ Indian languages)
    - Agentic orchestration for complex queries
    """,
    version="1.0.0",
    contact={
        "name": "IP-SAKTI Sahayak",
        "email": "support@ip-sakti.gov.in",
    },
    license_info={
        "name": "Government of India",
        "url": "https://www.ipindia.gov.in",
    },
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().api_cors_origins or ["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.middleware("http")(rate_limit_middleware)

# Serve the checked-in web client from the same origin as the API.
_web_dir = Path(__file__).resolve().parents[1] / "web"
if _web_dir.exists():
    app.mount("/web", StaticFiles(directory=str(_web_dir), html=True), name="web")


# ============================================================
# CUSTOM OPENAPI
# ============================================================

def custom_openapi():
    """Custom OpenAPI schema with security schemes."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        },
        "APIKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
        },
    }

    # Apply security globally
    openapi_schema["security"] = [
        {"BearerAuth": []},
        {"APIKeyAuth": []},
    ]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# ============================================================
# DEPENDENCY OVERRIDES
# ============================================================

def get_rag_pipeline() -> RAGPipeline:
    if _rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    return _rag_pipeline


def get_streaming_rag_pipeline() -> StreamingRAGPipeline:
    if _streaming_rag_pipeline is None:
        raise HTTPException(status_code=503, detail="Streaming RAG pipeline not initialized")
    return _streaming_rag_pipeline


def get_retrieval_engine() -> RetrievalEngine:
    if _retrieval_engine is None:
        raise HTTPException(status_code=503, detail="Retrieval engine not initialized")
    return _retrieval_engine


def get_ingestion_pipeline() -> IngestionPipeline:
    if _ingestion_pipeline is None:
        raise HTTPException(status_code=503, detail="Ingestion pipeline not initialized")
    return _ingestion_pipeline


# ============================================================
# EXCEPTION HANDLERS
# ============================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Standard error response format."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=ErrorDetail(
                code=f"HTTP_{exc.status_code}",
                message=exc.detail,
            ),
            request_id=request.headers.get("X-Request-ID", "unknown"),
        ).model_dump(mode="json"),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler."""
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=ErrorDetail(
                code="INTERNAL_ERROR",
                message="An internal server error occurred",
            ),
            request_id=request.headers.get("X-Request-ID", "unknown"),
        ).model_dump(mode="json"),
    )


# ============================================================
# AUTH ENDPOINTS
# ============================================================

@app.post("/auth/token", response_model=TokenResponse, tags=["Authentication"])
async def create_token(
    request: TokenRequest,
    jwt_manager: JWTManager = Depends(lambda: _jwt_manager),
):
    """
    OAuth2 password grant token endpoint.
    In production, validate credentials against user database.
    """
    # Demo: accept any credentials (replace with real auth)
    if request.username and request.password:
        access_token = jwt_manager.create_access_token(
            subject=request.username,
            scopes=request.scope.split(),
            user_id=request.username,
            roles=["user"],
        )
        refresh_token = jwt_manager.create_refresh_token(
            subject=request.username,
            user_id=request.username,
        )
        return TokenResponse(
            access_token=access_token,
            expires_in=jwt_manager.access_token_ttl,
            refresh_token=refresh_token,
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


@app.post("/auth/api-keys", response_model=APIKeyResponse, tags=["Authentication"])
async def create_api_key(
    request: APIKeyCreate,
    user: CurrentUser = Depends(get_current_user),
    api_key_manager: APIKeyManager = Depends(lambda: _api_key_manager),
):
    """Create a new API key for the authenticated user."""
    return api_key_manager.create_key(
        user_id=user.user_id,
        name=request.name,
        scopes=request.scopes,
        expires_in_days=request.expires_in_days,
        rate_limit_tier=request.rate_limit_tier,
    )


@app.get("/auth/api-keys", response_model=APIKeyListResponse, tags=["Authentication"])
async def list_api_keys(
    user: CurrentUser = Depends(get_current_user),
    api_key_manager: APIKeyManager = Depends(lambda: _api_key_manager),
):
    """List all API keys for the authenticated user."""
    return api_key_manager.list_keys(user.user_id)


@app.delete("/auth/api-keys/{key_id}", tags=["Authentication"])
async def revoke_api_key(
    key_id: str,
    user: CurrentUser = Depends(get_current_user),
    api_key_manager: APIKeyManager = Depends(lambda: _api_key_manager),
):
    """Revoke an API key."""
    success = api_key_manager.revoke_key(user.user_id, key_id)
    if not success:
        raise HTTPException(status_code=404, detail="API key not found")
    return {"success": True, "message": "API key revoked"}


# ============================================================
# QUERY ENDPOINTS
# ============================================================

@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(
    request: QueryRequest,
    user: CurrentUser = Depends(get_current_user),
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline),
    _rate_limit: None = Depends(rate_limit_dependency),
):
    """
    Execute a legal query and get a grounded answer with citations.
    
    - **query**: Natural language legal question
    - **jurisdiction**: Optional jurisdiction override
    - **language**: Response language (default: English)
    - **max_results**: Maximum evidence chunks to retrieve
    - **require_citations**: Whether citations are mandatory
    """
    from ip_sakti.core.models import Query, QueryIntent, JurisdictionCode, LanguageCode

    # Convert to internal Query model
    query = Query(
        text=request.query,
        user_id=request.user_id,
        session_id=request.session_id,
        intent=request.intent,
        jurisdiction=request.jurisdiction or JurisdictionCode.INDIA,
        language=request.language,
        max_results=request.max_results,
        filters=request.filters,
        require_citations=request.require_citations,
    )

    # Run RAG pipeline
    context = await rag_pipeline.run(query)

    if context.errors:
        return QueryResponse(
            success=False,
            data=QueryResponseModel(
                query_id=context.query.query_id,
                answer=GeneratedAnswer(
                    answer_id="error",
                    query=request.query,
                    segments=[],
                    citations=[],
                    overall_confidence=0.0,
                    jurisdiction=request.jurisdiction or JurisdictionCode.INDIA,
                    language=request.language,
                    processing_time_ms=int(context.metrics.get("total_time_ms", 0)),
                ),
                retrieved_chunks=[],
                intent=context.query.intent or QueryIntent.GENERAL_LEGAL,
                jurisdiction=request.jurisdiction or JurisdictionCode.INDIA,
                warnings=context.errors,
            ),
        )

    citation_models = [_citation_model(citation) for citation in context.citations]
    retrieved_results = context.reranked_results or context.retrieval_results
    retrieved_models = [_retrieved_chunk_model(result, index) for index, result in enumerate(retrieved_results, 1)]
    return QueryResponse(
        success=True,
        data=QueryResponseModel(
            query_id=context.query.query_id,
            answer=GeneratedAnswer(
                answer_id=str(context.query.query_id),
                query=request.query,
                segments=[
                    GeneratedSegment(
                        text=context.generated_answer or "No answer generated",
                        claims=[],
                        citations=citation_models,
                    )
                ],
                citations=citation_models,
                overall_confidence=context.confidence_score,
                jurisdiction=request.jurisdiction or JurisdictionCode.INDIA,
                language=request.language,
                processing_time_ms=int(context.metrics.get("total_time_ms", 0)),
                retrieval_stats=context.metrics,
            ),
            retrieved_chunks=retrieved_models,
            intent=context.query.intent or QueryIntent.GENERAL_LEGAL,
            jurisdiction=request.jurisdiction or JurisdictionCode.INDIA,
            warnings=context.errors,
        ),
    )


@app.post("/query/stream", tags=["Query"])
async def query_stream(
    request: StreamQueryRequest,
    user: CurrentUser = Depends(get_current_user),
    streaming_pipeline: StreamingRAGPipeline = Depends(get_streaming_rag_pipeline),
    _rate_limit: None = Depends(rate_limit_dependency),
):
    """
    Stream a legal query response token by token.
    Returns Server-Sent Events (SSE).
    """
    from ip_sakti.core.models import Query, JurisdictionCode, LanguageCode

    query = Query(
        text=request.query,
        user_id=request.user_id,
        session_id=request.session_id,
        intent=request.intent,
        jurisdiction=request.jurisdiction or JurisdictionCode.INDIA,
        language=request.language,
        max_results=request.max_results,
        filters=request.filters,
        require_citations=request.require_citations,
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        async for event_data in streaming_pipeline.run_streaming(query):
            event = StreamEvent(event="stage", data=event_data)
            yield f"data: {event.model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/query/batch", response_model=BatchQueryResponse, tags=["Query"])
async def query_batch(
    request: BatchQueryRequest,
    user: CurrentUser = Depends(require_scopes("query:read")),
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline),
    _rate_limit: None = Depends(rate_limit_dependency),
):
    """
    Execute multiple queries in a single batch request.
    Requires 'batch' scope.
    """
    batch_id = request.batch_id or str(datetime.utcnow().timestamp())
    items = []

    for idx, query_req in enumerate(request.queries):
        try:
            from ip_sakti.core.models import Query, JurisdictionCode, LanguageCode

            query = Query(
                text=query_req.query,
                user_id=query_req.user_id,
                session_id=query_req.session_id,
                intent=query_req.intent,
                jurisdiction=query_req.jurisdiction or JurisdictionCode.INDIA,
                language=query_req.language,
                max_results=query_req.max_results,
                filters=query_req.filters,
                require_citations=query_req.require_citations,
            )

            context = await rag_pipeline.run(query)

            # Build response (simplified)
            response_data = QueryResponseModel(
                query_id=context.query.query_id,
                answer=GeneratedAnswer(
                    answer_id=str(context.query.query_id),
                    query=query_req.query,
                    segments=[],
                    citations=[],
                    overall_confidence=context.confidence_score,
                    jurisdiction=query_req.jurisdiction or JurisdictionCode.INDIA,
                    language=query_req.language,
                    processing_time_ms=int(context.metrics.get("total_time_ms", 0)),
                ),
                retrieved_chunks=[],
                intent=context.query.intent or QueryIntent.GENERAL_LEGAL,
                jurisdiction=query_req.jurisdiction or JurisdictionCode.INDIA,
            )

            items.append(BatchQueryItem(index=idx, request=query_req, response=response_data))

        except Exception as e:
            logger.exception(f"Batch query {idx} failed")
            items.append(BatchQueryItem(index=idx, request=query_req, error=str(e)))

    completed = sum(1 for item in items if item.response is not None)
    failed = len(items) - completed

    return BatchQueryResponse(
        success=failed == 0,
        batch_id=batch_id,
        items=items,
        completed=completed,
        failed=failed,
    )


# ============================================================
# DOCUMENT ENDPOINTS
# ============================================================

@app.get("/documents", response_model=DocumentSearchResponse, tags=["Documents"])
async def search_documents(
    request: DocumentSearchRequest = Depends(),
    user: CurrentUser = Depends(get_current_user),
    retrieval_engine: RetrievalEngine = Depends(get_retrieval_engine),
    _rate_limit: None = Depends(rate_limit_dependency),
):
    """
    Search documents by metadata and content.
    """
    # Build filters
    filters = {}
    if request.jurisdiction:
        filters["jurisdiction"] = request.jurisdiction.value
    if request.document_type:
        filters["document_type"] = request.document_type.value
    if request.authority_tier:
        filters["authority_tier"] = request.authority_tier.value

    if request.query:
        search_req = SearchRequest(query=request.query, strategy=RetrievalStrategy.HYBRID, filters=filters, top_k=request.limit)
        candidate_chunks = [result.chunk for result in (await retrieval_engine.search(search_req)).results]
    else:
        candidate_chunks = list(getattr(retrieval_engine.vector_store, "chunks", {}).values())

    # Convert to document summaries (would need document-level aggregation)
    results = []
    seen_docs = set()

    for chunk in candidate_chunks:
        doc_id = chunk.document_id
        if doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)

        results.append(DocumentSummary(
            document_id=doc_id,
            title=chunk.metadata.get("title") or chunk.metadata.get("source_name", "Unknown"),
            document_type=DocumentType(chunk.metadata.get("document_type", "act")),
            jurisdiction=JurisdictionCode(chunk.metadata.get("jurisdiction", "INDIA")),
            authority_tier=_api_authority_tier(chunk.metadata.get("source_authority_tier")),
            language=LanguageCode(chunk.metadata.get("language", "en")),
            version=chunk.metadata.get("version", "1.0"),
            effective_date=None,
            source_url=chunk.metadata.get("canonical_url") or chunk.metadata.get("source_url") or "",
            snippet=chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
            chunk_count=sum(1 for candidate in candidate_chunks if candidate.document_id == doc_id),
        ))

    return DocumentSearchResponse(
        results=results,
        total=len(results),
        limit=request.limit,
        offset=request.offset,
        query=request.query,
    )


@app.get("/documents/{document_id}", response_model=DocumentDetailResponse, tags=["Documents"])
async def get_document(
    document_id: str,
    user: CurrentUser = Depends(get_current_user),
    retrieval_engine: RetrievalEngine = Depends(get_retrieval_engine),
    _rate_limit: None = Depends(rate_limit_dependency),
):
    """
    Get full document details including all chunks.
    """
    chunks = [chunk for chunk in getattr(retrieval_engine.vector_store, "chunks", {}).values() if chunk.document_id == document_id]
    if not chunks:
        raise HTTPException(status_code=404, detail="Document not found")
    first = chunks[0]
    metadata = first.metadata or {}
    details = [DocumentChunkDetail(
        chunk_id=chunk.id,
        hierarchy={"section": (chunk.metadata or {}).get("section_title")},
        text=chunk.content,
        chunk_type=(chunk.metadata or {}).get("chunk_type", "legal_text"),
        token_count=len(chunk.content.split()),
        citations=[],
    ) for chunk in chunks]
    return DocumentDetailResponse(
        document_id=document_id,
        title=metadata.get("title") or metadata.get("source_name", "Unknown"),
        document_type=DocumentType(metadata.get("document_type", "act")),
        jurisdiction=JurisdictionCode(metadata.get("jurisdiction", "INDIA")),
        authority_tier=_api_authority_tier(metadata.get("source_authority_tier")),
        language=LanguageCode(metadata.get("language", "en")),
        version=metadata.get("version", "1.0"),
        source_provenance={key: metadata.get(key) for key in ("source_url", "source_path", "content_hash", "source_authority_tier")},
        chunks=details,
        metadata=metadata,
    )


@app.post("/documents/search", response_model=DocumentSearchResponse, tags=["Documents"])
async def search_documents_post(
    request: DocumentSearchRequest,
    user: CurrentUser = Depends(get_current_user),
    retrieval_engine: RetrievalEngine = Depends(get_retrieval_engine),
    _rate_limit: None = Depends(rate_limit_dependency),
):
    """
    Search documents via POST (for complex queries).
    """
    return await search_documents(request, user, retrieval_engine, _rate_limit)


# ============================================================
# INGESTION ENDPOINTS
# ============================================================

@app.post("/ingest", response_model=IngestionResponse, tags=["Ingestion"])
async def start_ingestion(
    request: IngestionRequest,
    user: CurrentUser = Depends(require_scopes("ingest:write")),
    ingestion_pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
    _rate_limit: None = Depends(rate_limit_dependency),
):
    """
    Start document ingestion job.
    Requires 'ingest' or 'admin' scope.
    """
    job_id = f"ingest_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{str(hash(str(request.sources)))[:8]}"

    # Queue ingestion (async)
    # In production, would use message queue
    estimated_chunks = len(request.sources) * 100  # Rough estimate
    _ingestion_jobs[job_id] = {
        "status": IngestionStatus.PENDING,
        "sources_total": len(request.sources),
        "sources_processed": 0,
        "chunks_created": 0,
        "errors": [],
        "started_at": None,
        "completed_at": None,
    }

    # Start background task
    import asyncio
    asyncio.create_task(_run_ingestion(job_id, request, ingestion_pipeline))

    return IngestionResponse(
        job_id=job_id,
        status=IngestionStatus.PENDING,
        sources_queued=len(request.sources),
        estimated_chunks=estimated_chunks,
    )


@app.post("/ingest/upload", response_model=IngestionResponse, tags=["Ingestion"])
async def upload_ingestion(
    file: UploadFile = File(...),
    user: CurrentUser = Depends(require_scopes("ingest:write")),
    ingestion_pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
    _rate_limit: None = Depends(rate_limit_dependency),
):
    """Accept a browser upload and send it through the same ingestion pipeline."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename")
    content = await file.read()
    settings = get_settings()
    if len(content) > settings.max_document_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Uploaded document is too large")
    upload_dir = Path(__file__).resolve().parents[2] / "data" / "runtime" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename).name
    upload_path = upload_dir / safe_name
    upload_path.write_bytes(content)
    request = IngestionRequest(sources=[IngestionSourceConfig(
        source_id=safe_name,
        source_type=DocumentType.ACT,
        file_path=str(upload_path),
        jurisdiction=JurisdictionCode.INDIA,
        authority_tier=AuthorityTier.TIER_1,
    )])
    return await start_ingestion(request, user, ingestion_pipeline, _rate_limit)


@app.post("/query/upload", response_model=QueryResponse, tags=["Query"])
async def query_with_upload(
    file: UploadFile = File(...),
    query_text: str = Form(..., min_length=1, max_length=10000),
    jurisdiction: str = Form("INDIA"),
    user: CurrentUser = Depends(get_current_user),
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline),
    _rate_limit: None = Depends(rate_limit_dependency),
):
    """Answer using a PDF as ephemeral prompt context without indexing it."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded document must have a filename")
    content = await file.read()
    settings = get_settings()
    if len(content) > settings.max_document_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Uploaded document is too large")

    upload_dir = Path(__file__).resolve().parents[2] / "data" / "runtime" / "prompt_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = upload_dir / Path(file.filename).name
    upload_path.write_bytes(content)
    try:
        from ip_sakti.ingestion.loaders import load_document
        loaded = await load_document(str(upload_path))
        attachment_text = "\n\n".join(doc.content for doc in loaded if doc.content).strip()
    finally:
        upload_path.unlink(missing_ok=True)
    if not attachment_text:
        raise HTTPException(status_code=422, detail="No readable text was found in the uploaded document")

    from ip_sakti.core.models import Query, QueryIntent
    try:
        selected_jurisdiction = JurisdictionCode(jurisdiction.upper())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="jurisdiction must be INDIA or INTERNATIONAL") from exc
    internal_query = Query(
        text=query_text,
        user_id=user.user_id,
        jurisdiction=selected_jurisdiction,
        language=LanguageCode.EN,
        max_results=10,
        require_citations=True,
    )
    context = await rag_pipeline.run(
        internal_query,
        attachment_text=attachment_text,
        attachment_name=file.filename,
    )
    citation_models = [_citation_model(citation) for citation in context.citations]
    retrieved_results = context.reranked_results or context.retrieval_results
    retrieved_models = [_retrieved_chunk_model(result, index) for index, result in enumerate(retrieved_results, 1)]
    return QueryResponse(
        success=not bool(context.errors),
        data=QueryResponseModel(
            query_id=context.query.query_id,
            answer=GeneratedAnswer(
                answer_id=str(context.query.query_id),
                query=query_text,
                segments=[GeneratedSegment(text=context.generated_answer or "No answer generated", claims=[], citations=citation_models)],
                citations=citation_models,
                overall_confidence=context.confidence_score,
                jurisdiction=selected_jurisdiction,
                language=LanguageCode.EN,
                processing_time_ms=int(context.metrics.get("total_time_ms", 0)),
                retrieval_stats=context.metrics,
            ),
            retrieved_chunks=retrieved_models,
            intent=context.query.intent or QueryIntent.GENERAL_LEGAL,
            jurisdiction=selected_jurisdiction,
            warnings=context.errors,
        ),
    )


async def _run_ingestion(job_id: str, request: IngestionRequest, pipeline: IngestionPipeline):
    """Background ingestion task."""
    record = _ingestion_jobs[job_id]
    record["status"] = IngestionStatus.PROCESSING
    record["started_at"] = datetime.utcnow()
    try:
        # Convert request to pipeline format - create IngestionJob objects
        jobs = []
        for s in request.sources:
            job = IngestionJob(
                id=job_id,
                source_url=s.url,
                source_path=s.file_path,
                title=s.source_id,
                document_type=s.source_type,
                jurisdiction=s.jurisdiction,
                authority_tier=s.authority_tier,
                metadata=s.metadata,
                status=IngestionStatus.PENDING,
            )
            jobs.append(job)
        
        # Run ingestion for each source
        for job in jobs:
            result = await pipeline.ingest(job)
            record["sources_processed"] += 1
            record["chunks_created"] += len(result.chunks)
            if result.error_message:
                record["errors"].append(result.error_message)
        record["status"] = IngestionStatus.FAILED if record["errors"] else IngestionStatus.COMPLETED
    except Exception as e:
        logger.exception(f"Ingestion job {job_id} failed: {e}")
        record["status"] = IngestionStatus.FAILED
        record["errors"].append(str(e))
    finally:
        record["completed_at"] = datetime.utcnow()


@app.get("/ingest/{job_id}/status", response_model=IngestionStatusResponse, tags=["Ingestion"])
async def get_ingestion_status(
    job_id: str,
    user: CurrentUser = Depends(get_current_user),
    _rate_limit: None = Depends(rate_limit_dependency),
):
    """
    Get ingestion job status.
    """
    record = _ingestion_jobs.get(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Ingestion job not found")
    total = record["sources_total"] or 1
    return IngestionStatusResponse(
        job_id=job_id,
        status=record["status"],
        progress=record["sources_processed"] / total,
        sources_processed=record["sources_processed"],
        sources_total=record["sources_total"],
        chunks_created=record["chunks_created"],
        errors=record["errors"],
        started_at=record["started_at"],
        completed_at=record["completed_at"],
    )


@app.post("/ingest/batch", response_model=IngestionResponse, tags=["Ingestion"])
async def batch_ingest(
    request: IngestionRequest,
    user: CurrentUser = Depends(require_scopes("ingest", "admin")),
    ingestion_pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
    _rate_limit: None = Depends(rate_limit_dependency),
):
    """
    Batch ingestion - same as /ingest but for explicit batch processing.
    """
    return await start_ingestion(request, user, ingestion_pipeline, _rate_limit)


# ============================================================
# ADMIN ENDPOINTS
# ============================================================

@app.get("/health", response_model=HealthResponse, tags=["Admin"])
async def health_check(
    retrieval_engine: RetrievalEngine = Depends(get_retrieval_engine),
):
    """
    System health check.
    """
    components = []

    # Check retrieval engine
    try:
        stats = await retrieval_engine.get_stats()
        components.append(HealthComponent(
            name="retrieval_engine",
            status="healthy",
            latency_ms=stats.get("avg_latency_ms"),
            details=stats,
        ))
    except Exception as e:
        components.append(HealthComponent(
            name="retrieval_engine",
            status="unhealthy",
            details={"error": str(e)},
        ))

    # Check vector store
    try:
        if retrieval_engine.vector_store:
            vs_stats = await retrieval_engine.vector_store.get_stats()
            components.append(HealthComponent(
                name="vector_store",
                status="healthy",
                details=vs_stats,
            ))
    except Exception as e:
        components.append(HealthComponent(
            name="vector_store",
            status="unhealthy",
            details={"error": str(e)},
        ))

    # Check keyword index
    try:
        if retrieval_engine.keyword_index:
            components.append(HealthComponent(
                name="keyword_index",
                status="healthy",
                details={"type": "in_memory"},
            ))
    except Exception as e:
        components.append(HealthComponent(
            name="keyword_index",
            status="unhealthy",
            details={"error": str(e)},
        ))

    settings = get_settings()
    credential_name = "NVIDIA_API_KEY" if settings.llm_provider.lower() == "nvidia" else "OPENAI_API_KEY"
    generation_ready = bool(os.getenv(credential_name)) or settings.test_mode
    components.append(HealthComponent(
        name="generation",
        status="healthy" if generation_ready else "degraded",
        details={"provider": settings.llm_provider, "base_url": settings.llm_base_url, "test_mode": settings.test_mode, "model": settings.llm_primary, "credential_name": credential_name, "credential_configured": bool(os.getenv(credential_name))},
    ))

    # Determine overall status
    statuses = [c.status for c in components]
    if all(s == "healthy" for s in statuses):
        overall = "healthy"
    elif any(s == "unhealthy" for s in statuses):
        overall = "unhealthy"
    else:
        overall = "degraded"

    return HealthResponse(
        status=overall,
        components=components,
    )


@app.get("/corpus/validation", tags=["Admin"])
async def corpus_validation(
    user: CurrentUser = Depends(require_scopes("admin:read")),
    _rate_limit: None = Depends(rate_limit_dependency),
):
    """Validate corpus files before authoritative ingestion."""
    root = get_settings().corpus_root or str(Path(__file__).resolve().parents[2] / "data" / "corpus")
    return validate_corpus(root)


@app.get("/metrics", response_model=MetricsResponse, tags=["Admin"])
async def get_metrics(
    user: CurrentUser = Depends(require_scopes("admin:read")),
    retrieval_engine: RetrievalEngine = Depends(get_retrieval_engine),
    _rate_limit: None = Depends(rate_limit_dependency),
):
    """
    Get system metrics (admin only).
    """
    stats = await retrieval_engine.get_stats()

    return MetricsResponse(
        metrics={
            "retrieval_engine": stats,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


@app.get("/dashboard/metrics", response_model=MetricsResponse, tags=["Dashboard"])
async def get_dashboard_metrics(
    retrieval_engine: RetrievalEngine = Depends(get_retrieval_engine),
):
    """Return non-sensitive aggregates required by the public dashboard."""
    stats = await retrieval_engine.get_stats()
    chunks = list(getattr(retrieval_engine.vector_store, "chunks", {}).values())
    documents = {chunk.document_id for chunk in chunks}
    tiers = {}
    for chunk in chunks:
        tier = chunk.metadata.get("source_authority_tier", "TIER_6")
        tiers[tier] = tiers.get(tier, 0) + 1
    return MetricsResponse(metrics={
        "sources_indexed": len(documents),
        "chunks_indexed": len(chunks),
        "authority_tiers": tiers,
        "retrieval_engine": stats,
        "query_telemetry": stats.get("query_telemetry", {}),
        "test_mode": get_settings().test_mode,
    })


@app.get("/config", response_model=ConfigResponse, tags=["Admin"])
async def get_config(
    user: CurrentUser = Depends(require_scopes("admin:read")),
    _rate_limit: None = Depends(rate_limit_dependency),
):
    """
    Get sanitized system configuration (admin only).
    """
    settings = get_settings()

    # Sanitize sensitive config
    sanitized = {
        "technology": {
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
            "llm_primary": settings.llm_primary,
            "vector_db_type": settings.vector_db_type,
            "graph_db_type": settings.graph_db_type,
        },
        "rag": {
            "retrieval_top_k": getattr(settings, "rag_retrieval_top_k", 10),
            "max_context_tokens": getattr(settings, "rag_max_context_tokens", 8000),
        },
        "deployment": {
            "environment": settings.deployment_environment,
            "replicas": settings.deployment_replicas,
        },
    }

    return ConfigResponse(config=sanitized)


@app.get("/experiments", response_model=ExperimentListResponse, tags=["Admin"])
async def list_experiments(
    user: CurrentUser = Depends(require_scopes("admin:read")),
    _rate_limit: None = Depends(rate_limit_dependency),
):
    """
    List all experiments (admin/researcher only).
    """
    # In production, would query experiment tracking system
    return ExperimentListResponse(
        experiments=[],
        total=0,
    )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/", tags=["Root"])
async def root():
    """API root endpoint."""
    return {
        "name": "IP-SAKTI Sahayak API",
        "version": "1.0.0",
        "description": "Indian Intellectual Property Legal Assistant",
        "docs": "/docs",
        "health": "/health",
    }


# ============================================================
# STARTUP
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "ip_sakti.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
