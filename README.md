# IP-SAKTI Sahayak

An advanced RAG (Retrieval-Augmented Generation) system for Intellectual Property, Ayurveda, Traditional Knowledge, and Biodiversity law. Built with a modular, production-ready architecture supporting multiple retrieval strategies, advanced reranking, citation-first generation, and self-correcting agentic workflows.

The current runnable path is a citation-grounded local deployment: FastAPI serves the authenticated API and web app, SQLite stores the populated vector index, Sentence Transformers provides local embeddings, and NVIDIA NIM provides generation when `IP_SAKTI_TEST_MODE=false`.

## Architecture Overview

```
ip_sakti/
├── core/                   # Core framework (DI container, services, adapters)
├── interfaces/             # Abstract base classes for all components
├── config/                 # Configuration management
├── ingestion/              # Document loading, chunking, embedding
├── retrieval/              # Multi-strategy retrieval (lexical, vector, hybrid)
├── reranker/               # Advanced reranking (cross-encoder, LLM-based)
├── generation/             # Citation-first constrained generation
├── orchestration/          # Pipeline orchestrator, self-correcting agents
├── kg/                     # Knowledge graph integration
├── multimodal/             # Multimodal document processing
├── query_processor/        # Query analysis, rewriting, routing
├── eval/                   # Comprehensive evaluation framework
├── monitoring/             # Production monitoring & observability
├── multilingual/           # Multilingual support (Indic languages)
├── api/                    # FastAPI REST API
└── web/                    # Web frontend (matches reference design)
```

## Key Features

### Core RAG Pipeline
- **Modular Architecture**: Interface-based design with dependency injection
- **Multi-Strategy Retrieval**: BM25, Dense (BGE-M3), Hybrid (RRF), Graph-based
- **Advanced Reranking**: Cross-encoder, LLM-based, Authority-weighted
- **Citation-First Generation**: Grounded answers with verifiable citations
- **Self-Correcting Agents**: Iterative retrieval with reflection and retry

### Domain Specialization
- **IP Law**: Patents, Trademarks, Designs, Copyright, GI
- **Ayurveda**: Formulations, Classical texts, Regulatory
- **Traditional Knowledge**: Prior art, Community knowledge, TKDL
- **Biodiversity/ABS**: CBD, Nagoya Protocol, NBA compliance
- **Formulation triage**: natural-language product facts are routed through classification, procedure, compliance, IP, TK, and ABS retrieval angles rather than requiring an Act number.
- **Contrastive retrieval**: 24 positive semantic probes plus 3 negative/exclusion probes are fused before reranking; negative matches receive a configurable penalty.
- **Graph-assisted retrieval**: the local NetworkX knowledge graph is rehydrated from the persisted corpus and contributes entity evidence alongside semantic and lexical retrieval.

### Production Ready
- **Configuration-Driven**: YAML-based module selection with fallbacks
- **Observability**: Prometheus metrics, structured logging, health checks
- **Caching**: Redis-backed with TTL and invalidation
- **Multilingual**: Hindi/English response routing is enabled locally; authoritative names and citation markers are preserved.
- **Evaluation**: RAGAS, Faithfulness, Answer Relevancy, Context Precision
- **Live telemetry**: dashboard metrics and pipeline graph reflect completed queries in the current API process.
- **Document modes**: uploaded PDFs can either enter the persistent ingestion/indexing pipeline or be used as ephemeral prompt context for one query.
- **Facilitator escalation**: authenticated review requests are queued locally and written to a privacy-conscious audit JSONL stream; no external ticketing service is required.

## Quick Start

### Installation
```bash
# Clone the repository
git clone <repo-url>
cd ip-sakti

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration
Copy and edit the configuration file:
```bash
cp ip_sakti/config/settings.yaml.example ip_sakti/config/settings.yaml
# Edit settings.yaml with your API keys and paths
```

### Run API Server
```bash
python -m ip_sakti.api.app
```
Server runs at `http://localhost:8000`

### Run Web Frontend
```bash
# The web frontend is served by the API server
# Access at http://localhost:8000/web
```

### Ingest Documents
```bash
python -m ip_sakti.ingestion.cli ingest --path ./data/documents
```

### Evaluate System
```bash
python -m ip_sakti.eval.cli evaluate --dataset golden_set.json
```

## Configuration

The system uses `ip_sakti/config/settings.yaml` for configuration:

```yaml
modules:
  embedder:
    default: "bge-m3"
    fallback: "minilm"
    enabled: true
  retriever:
    default: "hybrid"
    fallback: "lexical"
    enabled: true
  reranker:
    default: "cross_encoder"
    fallback: "none"
    enabled: true
  generator:
    default: "citation_constrained"
    fallback: "simple"
    enabled: true

retrieval:
  top_k: 20
  hybrid_alpha: 0.5
  bm25_weight: 0.4
  vector_weight: 0.6

reranking:
  top_k: 10
  cross_encoder_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"

generation:
  model: "gpt-4o-mini"
  temperature: 0.1
  max_tokens: 2048
```

The checked-in runtime configuration uses `sentence-transformers/all-MiniLM-L6-v2` with 384-dimensional vectors for a practical local CPU deployment. Keep credentials in the ignored `.env`; `.env.example` is intentionally blank.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/auth/token` | POST | Obtain a local JWT for the web client |
| `/query` | POST | Authenticated grounded RAG query |
| `/query/upload` | POST | Ask with a PDF as ephemeral prompt context |
| `/escalations` | POST | Queue a local human-facilitator review request |
| `/ingest/upload` | POST | Authenticated upload into the persistent corpus |
| `/documents` | GET | Indexed document summaries |
| `/dashboard/metrics` | GET | Corpus and live query telemetry |
| `/health` | GET | Component health and model readiness |

## Web Interface

The web frontend (`/web`) provides:
- **Chat**: Interactive Q&A with citations
- **Dashboard**: System metrics, latency, corpus stats
- **Pipeline**: Live retrieval pipeline visualization
- **Files**: Document management and ingestion status

## Evaluation

The workflow smoke suite contains 20 deliberately different prompts covering formulation classification, patents, trademarks, GI, ABS, TK, procedures, compliance, international systems, multilingual input, and clarification behavior. It is stored at `docs/prompt_suite.json`.

Run comprehensive evaluation:
```bash
python -m ip_sakti.eval.cli evaluate \
  --dataset data/eval/golden_set.json \
  --metrics faithfulness,answer_relevancy,context_precision,context_recall \
  --output results/eval_report.json
```

## Project Structure Details

### Interfaces (Contracts)
All components implement interfaces from `ip_sakti/interfaces/`:
- `IConfig`, `IChunker`, `IDocumentLoader`, `IEmbedder`
- `IVectorStore`, `IAuthoritySystem`, `IRetriever`
- `IReranker`, `IGenerator`, `IQueryProcessor`
- `IKnowledgeGraph`, `IPipelineOrchestrator`
- `IIngestionPipeline`, `IRAGPipeline`

### Core Services
- **ServiceContainer**: Lightweight DI container
- **AdapterFactory**: Creates implementations from config
- **ServiceLocator**: Global service access

### Retrieval Strategies (`ip_sakti/retrieval/strategies/`)
- `LexicalRetriever`: BM25 with legal structure awareness
- `VectorRetriever`: Dense embeddings with FAISS
- `HybridRetriever`: RRF fusion with configurable weights
- `GraphRetriever`: Knowledge graph traversal

### Agents (`ip_sakti/orchestration/agents/`)
- `SelfCorrectingAgent`: Iterative retrieval-generation-reflection

## Development

### Code Style
```bash
# Format
black ip_sakti/
isort ip_sakti/

# Type check
mypy ip_sakti/

# Lint
ruff ip_sakti/
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ip_sakti --cov-report=html
```

For the focused local checks:
```bash
pytest -q
node --check ip_sakti/web/static/js/app.js
```

## Deployment

### Docker
```bash
docker build -t ip-sakti .
docker run -p 8000:8000 ip-sakti
```

### Environment Variables
```bash
# Required
OPENAI_API_KEY=your_key
# Optional
REDIS_URL=redis://localhost:6379
LOG_LEVEL=INFO
```

## License

Proprietary - IP-SAKTI Project

## Contributing

This is a research project for SIH (Smart India Hackathon). Internal contributions only.
