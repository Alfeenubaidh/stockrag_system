# StockRAG — Architecture

Production-grade Retrieval-Augmented Generation system for financial research.
Answers are grounded in SEC filings (10-K, 10-Q, 8-K) with mandatory citations and disclaimers.

---

## Status legend

| Symbol | Meaning |
|--------|---------|
| ✓ | Exists and integrated |
| ✗ | Missing — not yet built |

---

## File Tree

```
astrorag/
│
├── config/
│   ✓   └── settings.py               # pydantic-settings; reads all config from .env
│
├── data_sources/                      # ✗ Raw document acquisition (not yet a package)
│   ✗   ├── sec_edgar.py              # EDGAR XBRL + full-text search fetcher
│   ✗   ├── earnings_transcripts.py   # PDF / HTML transcript scraper
│   ✗   └── news_feed.py              # RSS / news API ingest adapter
│
├── market_data/                       # Live market context (NEVER stored in vector DB)
│   ✓   ├── prices.py                 # Live price fetch (yfinance / broker API)
│   ✓   ├── ratios.py                 # P/E, EV/EBITDA, debt ratios — fetched at query time
│   ✓   └── market_snapshot.py        # Assembles price + ratios into prompt context block
│
├── ingestion/
│   ✓   ├── models.py                 # RawDocument, ParsedPage, Chunk dataclasses
│   ✓   ├── base_parser.py            # Abstract parser interface
│   ✓   ├── pdf_parser.py             # PyMuPDF extraction; page → ParsedPage
│   ✓   ├── html_parser.py            # BeautifulSoup; EDGAR HTM filings
│   ✓   ├── text_cleaner.py           # Ligature fix, hyphen-break repair, whitespace
│   ✓   ├── section_detector.py       # Risk Factors / MD&A / Notes header detection
│   ✓   ├── chunker.py                # Sliding-window chunker with overlap
│   ✓   ├── chunk_validator.py        # Filters: too short, broken start, reference noise
│   ✓   ├── doc_metadata.py           # DocumentMetadata extractor (ticker, date, accession)
│   ✓   ├── hasher.py                 # SHA-256 content hash for dedup
│   ✓   ├── versioning.py             # Filing version tracking
│   ✓   ├── validator.py              # Pre-ingest document validation
│   ✓   ├── ingest.py                 # Single-document pipeline orchestrator
│   ✓   ├── batch_ingest.py           # Batch pipeline; streams results
│   ✓   ├── scheduler.py              # APScheduler cron: nightly 10-K/10-Q pulls per ticker watchlist
│   ✓   └── dedup.py                  # Re-ingest detection: Qdrant scroll check before fetch
│
├── embeddings/
│   ✓   ├── embedder.py               # SentenceTransformer (all-MiniLM-L6-v2); GPU-aware
│   ✓   └── embedding_version.json    # Tracks model version for re-embed detection
│
├── vector_store/
│   ✓   ├── qdrant_client.py          # QdrantClient factory; fail-fast connectivity check
│   ✓   ├── qdrant_store.py           # Upsert, scroll, delete operations
│   ✓   └── payload_schema.py         # Canonical Qdrant payload field definitions
│
├── retrieval/
│   ✓   ├── retrieval.py              # RetrievalPipeline: embed → filter → rank → rerank
│   ✓   ├── query_parser.py           # Intent classification, ticker extraction, section hint
│   ✓   ├── query_rewriter.py         # Query expansion for multi-query retrieval
│   ✓   ├── reranker.py               # Cross-encoder reranker (sentence-transformers)
│   ✓   ├── ranking_signals.py        # Ticker-match bonus, section-match bonus, staleness penalty
│   ✓   ├── query_interface.py        # Thin orchestration layer; returns list[dict]
│   ✓   ├── hybrid_search.py          # BM25 + dense vector fusion (Reciprocal Rank Fusion)
│   ✓   └── metadata_filters.py       # Fiscal year range, doc_type, and section filters builder
│
├── generation/
│   ✓   ├── generator.py              # Groq LLM call, extraction pre-pass, disclaimer injection
│   ✓   ├── context_builder.py        # Groups chunks by ticker; builds structured prompt context
│   ✓   ├── citation_assembler.py     # Citation format validation and normalization
│   ✓   ├── citation_postprocessor.py # Deterministic citation repair/injection post-LLM
│   ✓   ├── streaming.py              # SSE token streaming; wraps Groq stream=True endpoint
│   ✓   ├── grounding_check.py        # Hallucination guard: claim ↔ chunk entailment check
│   ✓   ├── response_cache.py         # Redis cache keyed on (query_hash, ticker, date_range)
│   ✓   └── prompts/
│   ✓       ├── system_prompt.txt     # LLM persona and citation rules
│   ✓       └── answer_prompt.txt     # Per-query answer template
│
├── api/
│   ✓   ├── main.py                   # FastAPI app; mounts all routers; configures logging
│   ✓   ├── dependencies.py           # Singletons: QdrantClient, EmbeddingPipeline, RetrievalPipeline
│   ✓   └── routers/
│   ✓       ├── health.py             # GET /health
│   ✓       ├── query.py              # POST /query — QueryRequest / QueryResponse
│   ✓       ├── stream.py             # POST /query/stream — SSE endpoint wrapping streaming.py
│   ✓       ├── ingest.py             # POST /ingest — multipart PDF upload
│   ✓       ├── documents.py          # GET /documents — ticker summaries from Qdrant scroll
│   ✓       ├── auth.py               # API key middleware (Bearer token validation)
│   ✓       └── rate_limiter.py       # Per-IP request throttle (slowapi / Redis sliding window)
│
├── observability/
│   ✓   ├── pipeline_observer.py      # Structured event logging: parse / retrieve / generate stages
│   ✓   └── metrics.py                # Prometheus counters/histograms: latency, chunk counts, cache hits
│
├── evaluation/
│   ✓   ├── eval_queries.json         # Ground-truth query → expected citation set
│   ✓   ├── eval.py                   # Retrieval precision/recall scorer
│   ✓   ├── eval_runner.py            # Batch eval runner; writes JSONL results
│   ✓   ├── e2e_eval.py               # End-to-end pipeline eval (retrieval + generation)
│   ✓   ├── results/                  # Timestamped JSONL eval snapshots
│   ✓   └── scorers/
│   ✓       ├── retrieval_scorer.py   # Precision@k, Recall@k, MRR
│   ✓       ├── generation_scorer.py  # Citation coverage, disclaimer presence
│   ✓       └── e2e_scorer.py         # Combined retrieval + generation score
│
├── tests/
│   ✓   ├── test_generation.py
│   ✓   └── test_context_ticker_inference.py
│
├── scripts/
│   ✓   ├── setup_qdrant.py           # Idempotent collection setup (vector_size=384, Cosine)
│   ✓   └── ingest_batch.py           # CLI batch ingest: --dir data/raw/pdfs
│
├── frontend/
│   ✓   ├── server.ts                 # Express dev server; proxies /api/* → FastAPI
│   ✓   ├── vite.config.ts
│   ✓   ├── tailwind.config.js
│   ✓   ├── postcss.config.cjs
│   ✓   └── src/
│   ✓       ├── App.tsx
│   ✓       ├── index.css
│   ✓       ├── main.tsx
│   ✓       ├── components/
│   ✓       │   ├── ChatWindow.tsx    # Message history, auto-scroll, empty state
│   ✓       │   ├── InputBox.tsx      # Textarea + ticker dropdown (AAPL/MSFT/TSLA/NVDA)
│   ✓       │   ├── MessageBubble.tsx # Markdown render, copy, source attribution
│   ✓       │   ├── Sidebar.tsx       # Conversation history, new chat, delete
│   ✓       │   ├── SourcesPanel.tsx  # Collapsible retrieved sources with scores
│   ✓       │   ├── TopBar.tsx        # Nav tabs: Models / Knowledge Base (modal) / API (docs)
│   ✓       │   └── Loader.tsx
│   ✓       ├── services/
│   ✓       │   └── api.ts            # Axios client
│   ✓       ├── store/
│   ✓       │   └── chatStore.ts      # Zustand: messages, conversations, sendMessage(query, ticker)
│   ✓       └── types/
│   ✓           └── index.ts          # QueryRequest, QueryResponse, Message, Citation, Conversation
│
├── infra/
│   ✓   ├── docker-compose.prod.yml   # Production compose: app + qdrant + redis + prometheus + grafana
│   ✓   └── ci/
│   ✓       └── eval_regression.yml   # GitHub Actions: run eval suite on PR; fail if avg_E2E < 0.75
│
├── requirements.txt                   # ✓ Python deps (fastapi, uvicorn, pydantic-settings, qdrant-client, …)
├── Dockerfile                         # ✓ Python 3.11-slim; installs requirements + API deps
├── docker-compose.yml                 # ✓ Dev compose: app + qdrant
└── run_pipeline.py                    # ✓ CLI entry point for manual ingestion runs
```

---

## Data flow

```
User query
    │
    ▼
InputBox.tsx (ticker selected)
    │  POST /api/query/stream  { question, ticker, top_k }
    ▼
Express proxy (frontend/server.ts)
    │  POST http://localhost:8001/query/stream
    ▼
FastAPI (api/routers/stream.py)
    │
    ├─► RetrievalPipeline
    │       QueryParser  →  query expansion (query_rewriter.py)
    │       Qdrant vector search  +  BM25 hybrid (hybrid_search.py)
    │       RankingSignalScorer
    │       CrossEncoderReranker
    │       metadata_filters  (fiscal year / doc_type / section)
    │
    ├─► market_data/  live price + ratios injected into context
    │
    └─► stream_answer (generation/streaming.py)
            context_builder
            Groq LLM  (streaming)
            CitationPostProcessor
            grounding_check
            disclaimer append
                │
                ▼
        SSE token stream
            │
            ▼
    Express  →  text/event-stream
            │
            ▼
    ChatWindow.tsx  +  SourcesPanel.tsx
```

---

## Remaining work

| Component | Reason not yet built |
|-----------|----------------------|
| `data_sources/sec_edgar.py` | EDGAR fetcher for automated filing download |
| `data_sources/earnings_transcripts.py` | Transcript scraper (PDF / HTML) |
| `data_sources/news_feed.py` | RSS / news API ingest adapter |

All other components are built and integrated.
