# IP-SAKTI Sahayak — Ayurvedic IPR & Regulatory Intelligence Assistant

> **Information, not legal advice.** A retrieval-augmented generation (RAG) system for Indian intellectual property, traditional knowledge, biodiversity (ABS), and AYUSH drug-regulatory law — with a special focus on Ayurvedic formulations.

---

## What This Project Does

IP-SAKTI Sahayak answers IPR questions specific to Ayurveda with **accuracy, source citation, and jurisdictional clarity**. It keeps the **national (India) and international legal layers visibly separate** through an explicit jurisdiction switch, so answers are never conflated.

Because intellectual property for an Ayurvedic product is inseparable from how the product is regulated, the assistant first helps **classify the formulation**:

| Category | IP Posture | Regulatory Path |
|----------|-----------|-----------------|
| Classical / Generic Medicine | TK-barred from patenting (§3(p)); defended via TKDL | D&C Act, First Schedule |
| Patent / Proprietary Medicine | Trademark + trade secret; limited patent scope | D&C Act, proprietary route |
| New / Non-Classical Drug | Genuine patent potential; clinical evidence required | New Drug approval (CDSCO/AYUSH) |
| Phytopharmaceutical | Patentable with botanical evidence | AYUSH phytopharma guidelines |
| Ayurveda-Aahar / Nutraceutical | FSSAI registration; GI possible | FSSAI (Ayurveda-Aahar) |
| Cosmetic | Trademark + design; no drug claims | Cosmetics Rules |

### Coverage

**National (India)**
- Patents Act 1970 & 2024 Rules
- Trade Marks Act 1999
- Geographical Indications Act 1999
- Designs Act 2000
- Copyright Act 1957
- PPVFR Act 2001
- Biological Diversity Act 2002 (as amended 2023) & 2024 Rules
- Drugs & Cosmetics Act 1940
- Drugs & Magic Remedies Act 1954
- FSSAI Act 2006 & Ayurveda-Aahar regulations

**International**
- TRIPS Agreement
- Convention on Biological Diversity & Nagoya Protocol
- WIPO GRATK Treaty (2024)
- PCT, Madrid, Hague, Budapest systems
- Key export-market herbal regimes

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        IP-SAKTI Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│  1. INGESTION LAYER                                             │
│     ├─ Document Loader (PDF, TXT, DOCX, HTML, MD, XML)         │
│     ├─ Text Splitter (RecursiveCharacterTextSplitter)          │
│     ├─ Metadata Extractor (authority tier, jurisdiction)       │
│     └─ Security Validator (malware, polyglot, script scan)     │
│                                                                 │
│  2. EMBEDDING LAYER  [⚠️ MOCK — needs real models]             │
│     ├─ Dense: bge-m3 / e5-large-v2 (planned)                   │
│     ├─ Vector Store: ChromaDB / FAISS (planned)                │
│     └─ Sparse: BM25 (custom in-memory; rank-bm25 planned)      │
│                                                                 │
│  3. KNOWLEDGE GRAPH  [⚠️ NOT IMPLEMENTED]                      │
│     ├─ Entity/Relation Extraction (LLM-based)                  │
│     ├─ Graph Construction (NetworkX)                           │
│     ├─ Community Detection (Leiden algorithm)                  │
│     └─ Community Summarization                                 │
│                                                                 │
│  4. RETRIEVAL LAYER  [⚠️ PARTIAL — hybrid search stubbed]     │
│     ├─ Hybrid Search: BM25 + Vector (RRF fusion)               │
│     ├─ GraphRAG Local / Global Search                          │
│     ├─ Cross-Encoder Reranker (bge-reranker-v2-m3)           │
│     └─ Query Expansion                                         │
│                                                                 │
│  5. GENERATION LAYER  [⚠️ TEMPLATE — needs real LLM]          │
│     ├─ Citation-First Answer Generation                        │
│     ├─ Claim Verification                                      │
│     ├─ Confidence Scoring                                      │
│     └─ Jurisdiction Isolation                                  │
│                                                                 │
│  6. API & UI                                                   │
│     ├─ FastAPI (auth, query, streaming, batch, ingestion)     │
│     ├─ WebSocket / SSE streaming                               │
│     └─ HTML Dashboard (Chat, Pipeline Viz, Files, Metrics)    │
│                                                                 │
│  7. SECURITY & MONITORING  [✅ COMPLETE]                       │
│     ├─ JWT + API Key + MFA (TOTP / WebAuthn)                 │
│     ├─ RBAC + ABAC authorization                               │
│     ├─ Rate limiting (tiered)                                  │
│     ├─ Prompt injection defense                                │
│     ├─ Audit logging (HMAC-signed, tamper-detect)             │
│     └─ Input sanitization & document validation                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
ip_shakti/
├── architecture/               # 9 detailed architecture phase documents
│   ├── phase1-requirements.md
│   ├── phase2-system-architecture.md
│   ├── phase3-rag-architecture.md
│   ├── phase4-ingestion-architecture.md
│   ├── phase5-chunking-strategy.md
│   ├── phase6-retrieval-engine.md
│   ├── phase7-knowledge-graph.md
│   ├── phase8-formulation-classification.md
│   └── phase9-jurisdiction-engine.md
│
├── data/
│   └── corpus/                 # Legal documents (24 sources, ~822K chars)
│       ├── ip_india/acts/      # Patents, TM, GI, Designs, Copyright, PPVFR, D&C, DMR, FSSAI
│       ├── nba/                # Biodiversity Act, Rules, ABS Guidelines, Forms
│       ├── cbd_nagoya/         # CBD 1992, Nagoya Protocol
│       ├── wipo/               # TRIPS, PCT, Madrid, Hague, Budapest, GRATK notes
│       └── sources/            # Metadata JSONs
│
├── ip_sakti/
│   ├── api/
│   │   └── app.py              # ✅ FastAPI application (full endpoints)
│   ├── authority/              # Source authority tier system
│   ├── classification/         # Formulation classification engine
│   ├── config/
│   │   └── loader.py           # ✅ YAML settings with env overrides
│   ├── core/                   # Pydantic models & base types
│   ├── eval/                   # Evaluation framework
│   ├── experiments/            # Experiment tracking
│   ├── generation/
│   │   └── citation_first.py   # ✅ Citation-first generation (template-based)
│   ├── ingestion/
│   │   ├── chunking.py         # ✅ LangChain text splitters wrapper
│   │   ├── loaders.py          # ✅ Multi-format document loaders
│   │   └── pipeline.py         # ✅ Full 6-stage ingestion pipeline
│   ├── jurisdiction/           # Jurisdiction detection & isolation
│   ├── kg/                     # Knowledge graph (placeholder)
│   ├── memory/                 # Conversation memory
│   ├── multilingual/           # Multilingual support framework
│   ├── orchestration/          # Agentic orchestration framework
│   ├── rag/                    # RAG pipeline (stub)
│   ├── retrieval/
│   │   └── retrieval_engine.py # ✅ In-memory retrieval (BM25 + vector hybrid)
│   └── security/
│       └── security_system.py  # ✅ Comprehensive security architecture
│
├── ip-sakti-sahayak.html     # ✅ Frontend dashboard (dark-themed, 4 views)
├── plan.md                     # Master implementation plan (GraphRAG + LangSmith)
├── SESSION_STATE.md            # Session state & progress tracker
├── test_pipeline.py            # Comprehensive test suite
├── test_chunking.py            # Chunking tests
└── test_simple.py              # Basic sanity tests
```

---

## Current Implementation Status

| Component | Status | Library | Notes |
|-----------|--------|---------|-------|
| **Architecture Docs** | ✅ Complete | Markdown | 9 phases, extremely detailed |
| **Document Loaders** | ✅ Complete | `langchain-community` | PDF, TXT, DOCX, HTML, MD, XML |
| **Chunking** | ✅ Complete | `langchain-text-splitters` | Recursive + Fixed-size |
| **Ingestion Pipeline** | ✅ Complete | Custom + LangChain | 6 stages with security validation |
| **Security System** | ✅ Complete | Custom + PyJWT | JWT, API keys, MFA, RBAC, ABAC, rate limits, audit |
| **Citation Generation** | ✅ Framework | Custom | Formatting, verification, claim extraction — uses templates |
| **FastAPI** | ✅ Complete | FastAPI | Auth, query, streaming, batch, ingestion, admin endpoints |
| **Frontend** | ✅ Complete | Vanilla JS/HTML | Chat, Dashboard, Pipeline visualization, Files |
| **Retrieval Engine** | ⚠️ Partial | Custom | In-memory only; hybrid search stubbed |
| **Embeddings** | ❌ Missing | — | Mock provider only; needs `sentence-transformers` |
| **Vector Store** | ❌ Missing | — | In-memory only; needs ChromaDB/FAISS |
| **BM25** | ⚠️ Partial | Custom | Needs `rank-bm25` for production |
| **Reranker** | ❌ Missing | — | Needs `bge-reranker-v2-m3` |
| **Knowledge Graph** | ❌ Missing | — | Needs NetworkX + igraph/Leiden |
| **GraphRAG** | ❌ Missing | — | Local + global search not implemented |
| **LLM Integration** | ❌ Missing | — | Template answers only; needs OpenAI/Ollama |
| **LangGraph** | ❌ Missing | — | Agent orchestration not wired |
| **LangSmith** | ❌ Missing | — | Monitoring, tracing, evaluation not wired |
| **Multilingual** | ❌ Missing | — | Framework exists; needs Bhashini integration |
| **Formulation Classifier** | ⚠️ Framework | — | Rules defined; not wired to chat flow |
| **Jurisdiction Engine** | ⚠️ Framework | — | Detection logic exists; hard filter not enforced |
| **ABS Compliance Helper** | ❌ Missing | — | Not implemented |
| **TKDL / Prior Art** | ❌ Missing | — | Not implemented |
| **Requirements File** | ❌ Missing | — | No `requirements.txt` or `pyproject.toml` |

---

## What Is Left to Do (Priority Order)

### 🔴 Critical — System Will Not Work Without These

1. **`requirements.txt` / `pyproject.toml`**
   - Add all dependencies: `fastapi`, `uvicorn`, `langchain-text-splitters`, `langchain-community`, `pydantic`, `PyYAML`, `PyJWT`, `numpy`, `aiohttp`, `aiofiles`, `pdfplumber`, `beautifulsoup4`, `python-multipart`, `sentence-transformers`, `chromadb`, `rank-bm25`, `networkx`, `python-igraph`, `leidenalg`, `langchain-openai` / `ollama`, `langgraph`, `langsmith`, `cryptography`, `pyotp`

2. **Real Embedding Model**
   - Replace `MockEmbeddingProvider` with `sentence-transformers` (bge-m3 or e5-large-v2)
   - Implement `ip_sakti/embedding/embedder.py`
   - Add batch embedding with progress tracking

3. **Persistent Vector Store**
   - Implement ChromaDB wrapper in `ip_sakti/embedding/vector_store.py`
   - Add FAISS for in-memory speed option
   - Persist embeddings across restarts

4. **Real LLM Integration**
   - Replace template answers in `citation_first.py` with actual LLM calls
   - Add `langchain-openai` or `ollama` wrapper in `ip_sakti/generation/llm.py`
   - Implement streaming generation
   - Wire LLM into the RAG pipeline

5. **RAG Pipeline Wiring**
   - Complete `ip_sakti/rag/pipeline.py` to connect retrieval → citation generation → response
   - Currently the API imports from `rag.pipeline` but this module is likely stubbed
   - Ensure end-to-end query → answer flow works

### 🟠 High Priority — Core Features from Requirements Document

6. **BM25 with `rank-bm25`**
   - Replace custom in-memory BM25 with the battle-tested library
   - Add proper tokenization for legal text (preserve section numbers, citations)

7. **Cross-Encoder Reranker**
   - Implement `ip_sakti/retrieval/reranker.py` using `bge-reranker-v2-m3`
   - Add to retrieval pipeline after hybrid fusion

8. **Jurisdiction Toggle & Isolation**
   - Enforce hard metadata filter BEFORE retrieval fusion
   - Add explicit India / International toggle in API and UI
   - Never mix Indian statutes with treaty text in the same answer

9. **Formulation Classification Flow**
   - Wire the 8-class classifier into the chat flow
   - Ask clarifying questions when product type is unclear
   - Display classification result before answering IP questions

10. **Citation-First Generation with Real LLM**
    - Build structured prompts for IP law
    - Ensure every sentence that makes a legal claim has a citation
    - Add source provenance: chunk → section → document → canonical URL

### 🟡 Medium Priority — Enhanced Capabilities

11. **Knowledge Graph (GraphRAG)**
    - Implement entity/relation extraction using LLM
    - Build NetworkX graph from corpus
    - Add Leiden community detection
    - Implement local search (entity → community → chunks) and global search (map-reduce over communities)

12. **LangGraph Agent Orchestration**
    - Build agent workflow: query understanding → jurisdiction detection → formulation classification → retrieval → generation → verification
    - Add conditional routing based on query type

13. **LangSmith Monitoring**
    - Add tracing to all LLM calls
    - Create evaluation datasets for regression testing
    - Build custom evaluators for citation accuracy and hallucination

14. **ABS Compliance Helper**
    - Add guided workflow for NBA Form I/II
    - Link to Biological Diversity Act sections
    - Point to NBA online filing portal

15. **TKDL / Prior Art Pointer**
    - Add TKDL database integration (or pointer)
    - Help users check if formulation exists in TKDL before filing patents
    - Cross-reference with Section 3(p) patenting bar

### 🟢 Lower Priority — Polish & Scale

16. **Multilingual Support (Bhashini)**
    - Integrate Bhashini API for 12+ Indian languages
    - Add language detection
    - Translate queries and responses

17. **Real Database Backends**
    - Replace in-memory stores with PostgreSQL (relational), Redis (cache), Qdrant/Weaviate (vector), Neo4j (graph)

18. **Evaluation Benchmarks**
    - Create curated test set of 50+ IP law questions with ground-truth answers
    - Measure: citation precision, recall, hallucination rate, abstention rate

19. **Voice Interface**
    - Add speech-to-text for query input
    - Add text-to-speech for responses

20. **Human Escalation Path**
    - Add "Ask a Human IP Facilitator" button when confidence < 0.4
    - Log escalation requests for audit

---

## Quick Start (When Complete)

```bash
# 1. Clone
git clone https://github.com/Vansh-Varshney-07/ip_shakti.git
cd ip_shakti

# 2. Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download legal corpus (or place your own in data/corpus/)
# See data/sources/ for metadata

# 5. Run ingestion
python -m ip_sakti.ingestion.pipeline --corpus data/corpus/

# 6. Start API
uvicorn ip_sakti.api.app:app --host 0.0.0.0 --port 8000 --reload

# 7. Open frontend
# Open ip-sakti-sahayak.html in a browser or serve via static file server
```

---

## Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...           # Or set OLLAMA_BASE_URL for local LLM
DATABASE_URL=postgresql://...   # For persistent storage
REDIS_URL=redis://localhost:6379 # For caching & rate limits

# Optional
ENVIRONMENT=development         # development | staging | production
LOG_LEVEL=info
VECTOR_DB_URL=qdrant://...      # Or chroma, weaviate, etc.
GRAPH_DB_URL=neo4j://...
```

---

## Testing

```bash
# Run all tests
python test_pipeline.py

# Run chunking tests
python test_chunking.py

# Run API tests (when server is running)
curl -X POST http://localhost:8000/query   -H "Content-Type: application/json"   -d '{"query": "What is Section 3(d) of the Patents Act?", "jurisdiction": "INDIA"}'
```

---

## Key Architectural Principles

1. **Jurisdiction Isolation**: Hard metadata filter before fusion — never mix Indian law with treaties
2. **Citation Traceability**: Every claim → evidence chunk → section → document → authoritative source
3. **Authority Hierarchy**: Tier 1 (official statute) > Tier 2 (guidance) > Tier 3 (academic) > Tier 4 (commentary)
4. **Deterministic > Probabilistic**: Legal rules (Schedule E, §3(d)) are deterministic; LLM only for gaps
5. **Abstention over Hallucination**: "I don't have sufficient authoritative evidence" > fabricated answer
6. **Information, Not Legal Advice**: Mandatory disclaimer on every response

---

## License

This is a research and educational project. All legal content is sourced from official Government of India publications and international treaty bodies. The software is provided as-is for the AYUSH community.

> **Disclaimer**: IP-SAKTI Sahayak provides information only. It does not constitute legal advice. Always consult a qualified IP attorney or AYUSH regulatory expert before filing applications or making compliance decisions.

---

## Contributors

- [Vansh Varshney](https://github.com/Vansh-Varshney-07) — Core architecture & implementation

---

*Built for the AYUSH ecosystem — protecting traditional knowledge while enabling legitimate innovation.*
