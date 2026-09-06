# IP-SAKTI Sahayak Finalization Audit

## Scope

Audit date: 2026-09-06

The current repository is `C:\Users\vvars\OneDrive\Desktop\sih rag` at commit `cf2a4f7`.
The historical architecture files are available in Git commit `403ecdb`, but the
`architecture/` directory is absent from the current working tree and ignored by
`.gitignore`.

## Current Runtime Architecture

The FastAPI application creates `RAGPipeline`, `StreamingRAGPipeline`,
`RetrievalEngine`, and `IngestionPipeline` in `ip_sakti/api/app.py`.
`RAGPipeline` is the selected canonical query path. The adapter-based
`PipelineOrchestrator` remains an experimental/research path until it is removed
or explicitly adopted.

## Current Connections

- API to RAG pipeline: connected.
- Query analysis and rewriting: connected.
- Basic jurisdiction detection and metadata filtering: partially connected.
- Ingestion to chunking and authority enrichment: connected.
- Ingestion to the API retrieval index: connected through the shared retrieval engine.
- Dense retrieval: real sentence-transformer provider in normal mode; deterministic mock only in explicit test mode.
- Sparse retrieval: in-memory in the previous runtime.
- Generation: NVIDIA NIM OpenAI-compatible provider in normal mode; explicit extractive test mode.
- Citation stage: repairs absent citation markers and validation remains enforced.
- API citation/evidence mapping: now maps retrieved chunks and source provenance.
- Knowledge graph: separate and not on the main query path.
- Frontend to backend: same-origin API client for query, documents, and health; upload reports unsupported contract instead of simulating success.

## Configuration and Runtime Divergences

The YAML declares Qdrant, Elasticsearch, Neo4j, PostgreSQL, Redis, MLflow,
Prometheus, Jaeger, BGE-M3, and a real LLM. The previous application runtime
instead selected in-memory stores, mock embeddings, template generation, and
NetworkX fallback behavior. Finalization must make this distinction explicit:
production mode fails clearly when required services/models are unavailable;
test mode may use local deterministic substitutes.

## Broken or Missing Components

- Missing retrieval strategy modules exported by `retrieval/strategies/__init__.py` (fixed by exporting only present strategies).
- Ingestion indexing did not call the retrieval engine (fixed).
- No persistent document/job store.
- Document detail returned HTTP 501 (implemented against the current indexed store; durable persistence remains).
- Ingestion status returned a constant pending response (implemented as live in-process job state; durable persistence remains).
- Main RAG citation stage was empty (implemented).
- API responses discarded evidence and citations (implemented).
- Frontend used canned answers, random metrics, and simulated ingestion (removed).
- No current Docker deployment or integration test suite.

## Corpus Risks

The original corpus contained zero-byte PDFs, an empty TRIPS text file, and
duplicate-looking treaty/ABS files. These 29 affected files are now preserved
under `data/quarantine/corpus-invalid-2026-09-06/`; the authoritative
`data/corpus` validates cleanly with 23 files.

## Security and Operations Gaps

- CORS wildcard in the previous API configuration.
- In-memory API keys and rate limits.
- Development JWT key generation.
- Malware scanning placeholder.
- No verified end-to-end trace propagation.
- No persistent ingestion job state.

## Finalization Policy

The production path must never silently fall back to fake embeddings, template
generation, or random UI data. Test mode is explicit and must be visible in
configuration and health output.
