from __future__ import annotations
from typing import List, Dict, Any

from config import cfg
from core.vector_store import FAISSVectorStore
from core.bm25_store import BM25Store


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    k: int = cfg.rrf_k,
    top_n: int = cfg.top_k_hybrid,
) -> List[Dict[str, Any]]:
    """Combine dense + sparse via RRF. Returns merged list sorted by RRF score."""
    scores: Dict[str, float] = {}
    docs: Dict[str, Dict[str, Any]] = {}

    for rank, doc in enumerate(dense_results, start=1):
        cid = doc["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        if cid not in docs:
            docs[cid] = dict(doc)

    for rank, doc in enumerate(sparse_results, start=1):
        cid = doc["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        if cid not in docs:
            docs[cid] = dict(doc)
        else:
            # merge bm25 fields
            docs[cid]["bm25_score"] = doc.get("bm25_score")
            docs[cid]["rank_sparse"] = doc.get("rank_sparse")

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    merged = []
    for hybrid_rank, (cid, rrf_score) in enumerate(ranked, start=1):
        doc = dict(docs[cid])
        doc["rrf_score"] = rrf_score
        doc["rank_hybrid"] = hybrid_rank
        merged.append(doc)
    return merged


class HybridSearcher:
    def __init__(self, vector_store: FAISSVectorStore, bm25_store: BM25Store):
        self.vector_store = vector_store
        self.bm25_store = bm25_store

    def search(
        self,
        query: str,
        top_k_dense: int = cfg.top_k_dense,
        top_k_sparse: int = cfg.top_k_sparse,
        top_k_final: int = cfg.top_k_hybrid,
    ) -> List[Dict[str, Any]]:
        dense = self.vector_store.search(query, top_k=top_k_dense)
        sparse = self.bm25_store.search(query, top_k=top_k_sparse)
        return reciprocal_rank_fusion(dense, sparse, top_n=top_k_final)
