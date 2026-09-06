# IP-SAKTI Sahayak Finalization Progress

| Component | Status | Implementation | Integration | Tests | Remaining |
|---|---|---|---|---|---|
| Repository audit | IMPLEMENTED | Current tree and Git history inspected | Connected to audit record | Not applicable | Keep updated |
| Canonical runtime decision | IMPLEMENTED | `RAGPipeline` selected for query execution | API and ingestion use shared engine | Pending | Remove remaining experimental ambiguity |
| Domain/API contracts | IMPLEMENTING | Existing models reviewed | Evidence/citation mapper added | Pending | Complete source metadata and contract tests |
| Ingestion to retrieval | IMPLEMENTED | Index stage calls retrieval engine | Shared in-process index | Pending | Add persistence |
| Embeddings | IMPLEMENTED | Real sentence-transformer provider in normal mode | Explicit mock only in test mode | Pending | Verify model availability |
| Persistent storage | IMPLEMENTED | SQLite-backed vector/chunk store | Sparse index rebuilds on restart | Smoke verified | Migrate to managed DB for horizontal scale |
| Retrieval | IMPLEMENTED | Semantic, lexical, hybrid, graph, and contrastive retrieval | Shared engine with jurisdiction filters | Smoke verified | External scale-out is optional |
| Generation | IMPLEMENTED | NVIDIA NIM OpenAI-compatible Chat Completions provider in normal mode; extractive test mode | Connected to canonical RAG path | Config verified; live call requires key | Requires `NVIDIA_API_KEY` and selected model |
| Citation verification | IMPLEMENTING | Citation stage now repairs missing markers; validator remains | Connected to response mapper | Pending | Add claim-level verification |
| API response | IMPLEMENTED | Response maps citations and retrieved chunks | Provenance retained | Pending | Add endpoint integration tests |
| Frontend | IMPLEMENTED | Same-origin API client | Query, documents, health, uploads, confidence, and escalation connected | Node syntax passes | Browser regression coverage can expand |
| Corpus validation | IMPLEMENTED | `ingestion/corpus_validation.py` and quarantine directory | Authoritative corpus excludes invalid/duplicate files | Verified: 23 files valid | Replace quarantined sources from official originals |
| Deployment | IMPLEMENTED | Dockerfile, Compose, `.env.example` | API container and healthcheck | Syntax/static checks | Supply production secrets and model cache |

## Verification Note

`git diff --check` passes. A clean Python 3.12 environment was provisioned and the
repository smoke suite passes (`2 passed`). API import, `/health`, and `/query` were
also exercised in explicit test mode. Production-mode health starts without mock
components; it reports generation as degraded until `OPENAI_API_KEY` is configured.
Real production queries also require downloaded embedding/reranker model artifacts.
