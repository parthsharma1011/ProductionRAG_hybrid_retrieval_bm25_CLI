# Production RAG — Hybrid Retrieval with BM25 + FAISS

A production-ready Retrieval-Augmented Generation (RAG) CLI that combines dense vector search (FAISS) and sparse keyword search (BM25) via Reciprocal Rank Fusion, with an optional cross-encoder reranker, conversation memory, and query caching — all powered by Google Gemini.

---

## Architecture

```
PDF / TXT files
      │
      ▼
Document Processor  ──►  Chunks (512 tokens, 64 overlap)
      │
      ├──► FAISS Vector Store  (Gemini embeddings, dim=3072)
      └──► BM25 Store          (sparse keyword index)
                │
                ▼
        Hybrid Search (RRF fusion)
                │
                ▼
        Reranker (cosine re-score via Gemini embeddings)
                │
                ▼
        Gemini 2.0 Flash  ◄──  Conversation Memory + Query Cache
                │
                ▼
            Answer (CLI)
```

---

## Features

- **Hybrid Retrieval** — FAISS (semantic) + BM25 (keyword) merged with Reciprocal Rank Fusion (RRF)
- **Reranker** — cosine similarity re-scoring on top hybrid candidates for higher precision
- **Conversation Memory** — retains last 6 turns for multi-turn Q&A
- **Query Cache** — disk-based cache (TTL: 1hr) to skip redundant LLM calls
- **Persistent Index** — FAISS + BM25 indexes saved to `.index/`, reloaded on next run
- **Compare Mode** — side-by-side answers with and without reranker

---

## Project Structure

```
.
├── core/
│   ├── document_processor.py   # PDF/text loading and chunking
│   ├── embeddings.py           # Gemini embedding model wrapper
│   ├── vector_store.py         # FAISS dense store
│   ├── bm25_store.py           # BM25 sparse store
│   ├── hybrid_search.py        # RRF fusion logic
│   ├── reranker.py             # Cosine reranker with comparison table
│   ├── cache.py                # Disk-based query cache
│   └── memory.py               # Sliding window conversation memory
├── rag/
│   └── pipeline.py             # RAGPipeline — orchestrates everything
├── files/                      # Drop your PDFs/TXTs here
├── config.py                   # All hyperparameters in one place
├── main.py                     # CLI entrypoint
└── requirements.txt
```

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Set your Gemini API key**
```bash
# create a .env file in the project root
echo "GEMINI_API_KEY=<your_api_key>" > .env
```

**3. Add your documents**

Drop PDF or `.txt` files into the `files/` folder.

---

## Usage

```bash
# Auto-ingest from files/ folder and start chat
python main.py

# Ingest specific files
python main.py doc1.pdf doc2.txt

# Force re-index (ignore existing index)
python main.py --force-reindex

# Disable reranker
python main.py --no-reranker

# Auto-compare reranker on/off for every query
python main.py --compare
```

**In-chat commands:**

| Command | Description |
|---|---|
| `quit` / `exit` | Exit the CLI |
| `clear` | Clear conversation memory |
| `cache` | Show cache stats |
| `compare <question>` | Run query with and without reranker |
| `reranker on/off` | Toggle reranker at runtime |

---

## Configuration

All parameters are in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `embedding_model` | `gemini-embedding-001` | Embedding model |
| `embedding_dim` | `3072` | Embedding dimensions |
| `chunk_size` | `512` | Tokens per chunk |
| `chunk_overlap` | `64` | Overlap between chunks |
| `top_k_dense` | `10` | FAISS candidates |
| `top_k_sparse` | `10` | BM25 candidates |
| `top_k_hybrid` | `10` | After RRF fusion |
| `top_k_final` | `5` | After reranker |
| `rrf_k` | `60` | RRF constant |
| `llm_model` | `gemini-2.0-flash` | LLM for generation |
| `memory_window` | `6` | Conversation turns kept |
| `cache_ttl` | `3600` | Cache TTL in seconds |

---

## How Hybrid Search Works

1. **Dense search** — query is embedded via Gemini and top-10 chunks are retrieved from FAISS by cosine similarity.
2. **Sparse search** — BM25 scores chunks by keyword overlap and returns top-10.
3. **RRF fusion** — both ranked lists are merged using Reciprocal Rank Fusion:  
   `score(d) = Σ 1 / (k + rank(d))` where `k=60`
4. **Reranker** — top hybrid results are re-scored by cosine similarity between query and chunk embeddings, then the final top-5 are passed to the LLM.

---

## Requirements

- Python 3.9+
- Google Gemini API key ([get one here](https://aistudio.google.com/app/apikey))
