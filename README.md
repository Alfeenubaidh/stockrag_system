# StockRAG

Production-grade Retrieval-Augmented Generation system for financial research. Answers questions grounded in SEC filings (10-K, 10-Q, 8-K) with mandatory citations, live market data, and hallucination detection.

---

## Demo

Ask questions like:
- *"What are Apple's main risk factors?"*
- *"What is Microsoft's revenue growth over the past year?"*
- *"Compare Tesla and NVDA's operating margins"*

Answers are grounded in retrieved filing chunks with `[TICKER · DOCTYPE · SECTION]` citations and a financial disclaimer on every response.

---

## Architecture

```
frontend/          React + Tailwind chat UI (Express proxy)
backend/
├── api/           FastAPI — /query, /query/stream, /ingest, /documents
├── ingestion/     PDF/HTM parser → section detection → chunking → embed → Qdrant
├── retrieval/     Dense + BM25 hybrid search, MMR reranking, query expansion
├── generation/    Groq LLM, citation enforcement, grounding check, SSE streaming
├── market_data/   Live price + ratios via yfinance (injected at query time)
├── data_sources/  EDGAR auto-fetch, earnings transcripts, news RSS
├── evaluation/    End-to-end eval harness (34/36 passing, avg_E2E=0.80)
└── observability/ Prometheus metrics, structured pipeline logging
infra/             Docker Compose (dev + prod), GitHub Actions CI
```

**Eval results:** `avg_R=0.9226` · `avg_G=0.8710` · `avg_E2E=0.8026` · 34/36 passing

---

## Stack

| Layer | Technology |
|---|---|
| LLM | Groq API (`llama-3.1-8b-instant`) |
| Embeddings | `all-MiniLM-L6-v2` (SentenceTransformers) |
| Vector DB | Qdrant |
| Retrieval | Dense + BM25 hybrid (Reciprocal Rank Fusion) |
| Reranker | Cross-encoder (sentence-transformers) |
| Backend | FastAPI + Uvicorn |
| Frontend | React 19 + Tailwind + Zustand |
| Market Data | yfinance |
| Filing Source | SEC EDGAR public API |

---

## Quickstart

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker (for Qdrant)
- Groq API key — free at [console.groq.com](https://console.groq.com)

### 1. Start Qdrant

```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:v1.9.2
```

### 2. Backend setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy and fill in env:

```bash
cp .env.example .env
# Set GROQ_API_KEY in .env
```

Create Qdrant collection:

```bash
python scripts/setup_qdrant.py
```

Start FastAPI:

```bash
python -m uvicorn api.main:app --port 8001 --env-file .env
```

### 3. Ingest filings

Download a 10-K PDF from SEC EDGAR, then:

```bash
python ingest_single.py \
  --ticker AAPL \
  --file path/to/aapl_10k.pdf \
  --doc-type 10-K \
  --filing-date 2024-09-28
```

Or auto-fetch latest filing from EDGAR:

```bash
python -c "
from data_sources.sec_edgar import fetch_and_ingest
fetch_and_ingest('AAPL', '10-K')
"
```

Or batch ingest a folder of PDFs named `TICKER_YEAR_DOCTYPE.pdf`:

```bash
python scripts/ingest_batch.py --dir data/raw/pdfs
```

### 4. Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

---

## API

FastAPI docs available at `http://localhost:8001/docs`

### POST /query

```json
{
  "question": "What are Apple's main risk factors?",
  "ticker": "AAPL",
  "top_k": 5
}
```

Response:

```json
{
  "answer": "**AAPL**: Apple faces competition... [AAPL 10-K · FY2024 · Risk Factors]\n\n---\nThis is not financial advice.",
  "citations": ["[AAPL 10-K · FY2024 · Risk Factors]"],
  "latency_ms": 2100
}
```

### POST /query/stream

Same request body — returns `text/event-stream` SSE tokens.

### POST /ingest

Multipart form upload:

```bash
curl -X POST http://localhost:8001/ingest \
  -F "file=@aapl_10k.pdf" \
  -F "doc_id=AAPL_2024_10K" \
  -F "ticker=AAPL" \
  -F "doc_type=10-K" \
  -F "filing_date=2024-09-28"
```

### GET /documents

Returns all ingested tickers with chunk counts and sections.

---

## Evaluation

```bash
cd backend
python evaluation/e2e_eval.py
```

Output: per-query `R`, `G`, `E2E` scores with failure classification (`retrieval`, `generation`, `generation_crash`).

Run a 10-query sample:

```bash
python evaluation/e2e_eval.py --sample 10
```

CI eval regression runs automatically on every push via `.github/workflows/eval_regression.yml` — fails if `avg_E2E < 0.75`.

---

## Supported Tickers

Currently ingested: `AAPL` · `MSFT` · `TSLA` · `NVDA` · `GOOGL` · `META` · `AMZN` · `AMD` · `INTC`

Add any ticker by ingesting its 10-K/10-Q PDF or running `fetch_and_ingest(ticker, doc_type)`.

---

## Production Deploy

See `infra/docker-compose.prod.yml` for the full production stack (FastAPI + Qdrant + Redis + Prometheus + Grafana).

For cloud deploy:
- **Backend**: Railway — `infra/railway.toml` is pre-configured
- **Vector DB**: [Qdrant Cloud](https://cloud.qdrant.io) free tier
- **Frontend**: Vercel — set `FASTAPI_URL` env var to your Railway URL

---

## Disclaimer

This system is for informational and research purposes only. Nothing in this application constitutes financial advice. Always consult a qualified financial advisor before making investment decisions.