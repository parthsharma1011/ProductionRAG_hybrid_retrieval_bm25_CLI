from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).parent

@dataclass
class Config:
    # Embedding
    embedding_model: str = "models/gemini-embedding-001"
    embedding_dim: int = 3072

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval
    top_k_dense: int = 10          # FAISS candidates
    top_k_sparse: int = 10         # BM25 candidates
    top_k_hybrid: int = 10         # after RRF fusion
    top_k_final: int = 5           # after reranker (or final cut)
    rrf_k: int = 60                # RRF constant

    # Reranker
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # LLM
    llm_model: str = "gemini-2.0-flash"
    max_tokens: int = 1024

    # Memory
    memory_window: int = 6         # last N turns kept

    # Cache
    cache_dir: Path = BASE_DIR / ".cache"
    cache_ttl: int = 3600          # seconds

    # FAISS index path
    index_dir: Path = BASE_DIR / ".index"

cfg = Config()
