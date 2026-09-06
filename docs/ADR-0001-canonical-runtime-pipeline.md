# ADR-0001: Canonical Runtime Pipeline

## Status

Accepted

## Decision

`ip_sakti.rag.pipeline.RAGPipeline` is the canonical synchronous and streaming
query execution path used by the API. It owns query analysis, rewriting,
retrieval, reranking, context construction, generation, citation validation,
and response metadata.

`PipelineOrchestrator` and the adapter/interface layer remain experimental
research components until they are proven against the same contracts. They must
not be initialized as a second production query path.

## Rationale

The API already depends directly on `RAGPipeline`. Selecting it avoids a broad
rewrite while removing ambiguity. Ingestion and evaluation will call the same
retrieval engine instance or a shared index service rather than creating
independent indexes.

## Consequences

- New query features must enter through `RAGPipeline`.
- Adapter implementations cannot silently replace production services.
- Experimental orchestration must be labeled and tested separately.
- API responses must be derived from `RAGContext`, preserving evidence and
  citation provenance.
