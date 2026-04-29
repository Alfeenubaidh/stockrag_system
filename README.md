# StockRAG

A production-grade Retrieval-Augmented Generation system for financial research. Ask questions about publicly traded companies — answers are grounded in SEC filings (10-K, 10-Q, 8-K) with source citations, hallucination detection, and live market data injected at query time.

**Frontend:** https://stockrag-system.vercel.app
**Backend API:** https://stockrag-system.onrender.com/docs

---

## Key Capabilities

- Answers grounded in SEC filings with mandatory `[TICKER · DOCTYPE · SECTION]` citations
- Every response carries a financial disclaimer — non-negotiable by design
- Hallucination detection: claims checked against retrieved chunks before serving
- Live market data (price, ratios) fetched at query time and injected into context
- Streaming responses via SSE (`/query/stream`)
- Section-aware retrieval: risk queries routed to Risk Factors, outlook queries to MD&A
- Date-scoped retrieval: Q2 2024 filings are not mixed with 2019 filings

---

## Architecture

```
React + Vite (Vercel)
    │
    │  HTTPS  (VITE_API_URL)
    ▼
FastAPI + Uvicorn (Render)
    ├── /query              dense + BM25 hybrid retrieval → Groq generation
    ├── /query/stream       same pipeline, SSE token stream
    ├── /ingest             upload and index a filing
    └── /documents          list ingested tickers and chunk counts
    │
    ├── Qdrant Cloud        vector store (384-dim, cosine, keyword indexes)
    ├── Groq API            LLM inference (llama-3.1-8b-instant)
    ├── HF Inference API    remote embeddings (all-MiniLM-L6-v2)
    └── yfinance            live market data (price, P/E, EPS)
```

### Backend module layout

```
backend/
├── api/           FastAPI routers, auth, rate limiting
├── ingestion/     PDF/HTML parser → section detection → chunker → validator
├── embeddings/    Local SentenceTransformer or remote HF Inference API
├── vector_store/  Qdrant upsert, collection management, payload indexes
├── retrieval/     Dense search, BM25, RRF fusion, cross-encoder reranker
├── generation/    Groq LLM, citation enforcement, grounding check, streaming
├── market_data/   Live price snapshot (injected at query time, never stored)
├── evaluation/    End-to-end eval harness with per-query R/G/E2E scoring
└── observability/ Prometheus metrics, structured pipeline logging
```

---

## Evaluation Results

| Metric | Score |
|---|---|
| Retrieval (avg_R) | 0.9226 |
| Generation (avg_G) | 0.8710 |
| End-to-End (avg_E2E) | 0.8026 |
| Passing queries | 34 / 36 |

Run eval:

```bash
cd backend
python evaluation/e2e_eval.py
# or a 10-query sample:
python evaluation/e2e_eval.py --sample 10
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 19, Vite, Tailwind CSS, Zustand |
| Hosting (frontend) | Vercel (static build) |
| Backend | FastAPI, Uvicorn, Python 3.11 |
| Hosting (backend) | Render |
| LLM | Groq API — `llama-3.1-8b-instant` |
| Embeddings | HF Inference API — `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | Qdrant Cloud |
| Retrieval | Dense + BM25 hybrid (Reciprocal Rank Fusion) |
| Market Data | yfinance |
| Filing Source | SEC EDGAR public API |
| Rate Limiting | slowapi |
| Metrics | Prometheus + prometheus-fastapi-instrumentator |

---

## Local Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- A [Qdrant Cloud](https://cloud.qdrant.io) cluster (free tier works)
- [Groq API key](https://console.groq.com)
- [Hugging Face token](https://huggingface.co/settings/tokens) with Inference API access

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `backend/.env`:

```env
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
GROQ_API_KEY=your-groq-api-key
HF_API_KEY=your-huggingface-token
USE_REMOTE_EMBEDDINGS=true
```

Set up the Qdrant collection and payload indexes:

```bash
PYTHONPATH=. python scripts/setup_qdrant.py
```

Start the backend:

```bash
PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 10000
```

API docs available at `http://localhost:10000/docs`.

### Frontend

```bash
cd frontend
npm install
```

Create `frontend/.env`:

```env
VITE_API_URL=http://localhost:10000
```

```bash
npm run dev
```

---

## Ingestion

Batch ingest a folder of SEC filing PDFs or HTML files:

```bash
cd backend
PYTHONPATH=. python scripts/ingest_batch.py --dir ../data/raw
```

Files should be named `TICKER_YEAR_DOCTYPE.pdf` (e.g. `AAPL_2024_10-K.pdf`). The pipeline parses, sections, chunks, embeds, and upserts to Qdrant. Duplicate documents are detected and skipped.

---

## API Reference

### POST /query

```json
{ "question": "What are Apple's main risk factors?", "ticker": "AAPL", "top_k": 5 }
```

```json
{
  "answer": "**AAPL**: Apple faces significant competition... [AAPL 10-K · FY2024 · Risk Factors]\n\n---\nThis is not financial advice.",
  "citations": ["[AAPL 10-K · FY2024 · Risk Factors]"],
  "latency_ms": 2100
}
```

### POST /query/stream

Same request body — returns `text/event-stream` SSE token stream. Ends with `data: [DONE]`.

### GET /documents

Returns all ingested tickers with chunk counts, section list, and last filing date.

### POST /ingest

Multipart upload of a single filing with metadata fields (`ticker`, `doc_type`, `filing_date`).

---

## Disclaimer

This system is for informational and research purposes only. Nothing produced by this application constitutes financial advice. Always consult a qualified financial advisor before making investment decisions.

---

## Citation

If you use this project in your research or work, please cite:

```bibtex
@software{stockrag2026,
  author = {Alfeenubaidh},
  title  = {StockRAG: Production Financial RAG System},
  year   = {2026},
  url    = {https://github.com/Alfeenubaidh/stockrag_system}
}
```

---

## License

MIT License

Copyright (c) 2026 Alfeenubaidh

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
