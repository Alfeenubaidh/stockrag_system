# StockRAG

Production RAG system for financial research. Answers questions grounded in SEC filings (10-K, 10-Q, 8-K) with mandatory citations, live market data, and hallucination detection.

**Live:** [stockrag-system.vercel.app](https://stockrag-system.vercel.app) — backend: [stockrag-system.onrender.com](https://stockrag-system.onrender.com)

---

## Architecture

```
frontend/          React + Vite + Tailwind (deployed on Vercel)
backend/
├── api/           FastAPI — /query, /query/stream, /ingest, /documents
├── ingestion/     PDF/HTML parser → section detection → chunking → embed → Qdrant
├── retrieval/     Dense + BM25 hybrid search, RRF, cross-encoder reranking
├── generation/    Groq LLM, citation enforcement, grounding check, SSE streaming
├── market_data/   Live price + ratios via yfinance (injected at query time)
├── embeddings/    Remote mode (HF Inference API) or local SentenceTransformers
├── evaluation/    End-to-end eval harness
└── observability/ Prometheus metrics, structured pipeline logging
```

**Eval:** `avg_R=0.9226` · `avg_G=0.8710` · `avg_E2E=0.8026` · 34/36 passing

---

## Stack

| Layer | Technology |
|---|---|
| LLM | Groq (`llama-3.1-8b-instant`) |
| Embeddings | HF Inference API (`all-MiniLM-L6-v2`, remote) |
| Vector DB | Qdrant Cloud |
| Retrieval | Dense + BM25 hybrid (RRF) |
| Reranker | Cross-encoder (disabled in prod, no torch) |
| Backend | FastAPI + Uvicorn on Render |
| Frontend | React 19 + Tailwind + Zustand on Vercel |
| Market Data | yfinance |
| Filing Source | SEC EDGAR public API |

---

## Local Dev Setup

### Prerequisites

- Python 3.11+
- Node.js 18+

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `backend/.env`:

```env
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
GROQ_API_KEY=your-groq-api-key
HF_API_KEY=your-hf-token
USE_REMOTE_EMBEDDINGS=true
```

Set up Qdrant collection:

```bash
python scripts/setup_qdrant.py
```

Start backend:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 10000
```

API docs: `http://localhost:10000/docs`

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

## Ingest

Batch ingest a folder of PDFs/HTML named `TICKER_YEAR_DOCTYPE.{pdf,html}`:

```bash
cd backend
python scripts/ingest_batch.py --dir ../data/raw
```

Single file:

```bash
python ingest_single.py \
  --ticker AAPL \
  --file path/to/aapl_10k.pdf \
  --doc-type 10-K \
  --filing-date 2024-09-28
```

---

## API

### POST /query

```json
{ "question": "What are Apple's main risk factors?", "ticker": "AAPL", "top_k": 5 }
```

```json
{
  "answer": "**AAPL**: Apple faces... [AAPL 10-K · FY2024 · Risk Factors]\n\n---\nThis is not financial advice.",
  "citations": ["[AAPL 10-K · FY2024 · Risk Factors]"],
  "latency_ms": 2100
}
```

### POST /query/stream

Same request body — returns `text/event-stream` SSE tokens.

### GET /documents

Returns all ingested tickers with chunk counts and sections.

---

## Evaluation

```bash
cd backend
python evaluation/e2e_eval.py
```

Per-query `R`, `G`, `E2E` scores with failure classification. Sample run:

```bash
python evaluation/e2e_eval.py --sample 10
```

---

## Disclaimer

For informational and research purposes only. Nothing in this application constitutes financial advice.
