from __future__ import annotations
import pickle
from pathlib import Path
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi

from config import cfg
from core.document_processor import Chunk


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


class BM25Store:
    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._metadata: List[Dict[str, Any]] = []

    def add_chunks(self, chunks: List[Chunk]) -> None:
        corpus = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(corpus)
        self._metadata = [{**c.metadata, "chunk_id": c.chunk_id, "text": c.text} for c in chunks]

    def search(self, query: str, top_k: int = cfg.top_k_sparse) -> List[Dict[str, Any]]:
        if self._bm25 is None:
            return []
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_indices = scores.argsort()[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            entry = dict(self._metadata[idx])
            entry["bm25_score"] = float(scores[idx])
            entry["rank_sparse"] = rank
            results.append(entry)
        return results

    def save(self, directory: Path | str) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "bm25.pkl", "wb") as f:
            pickle.dump({"bm25": self._bm25, "metadata": self._metadata}, f)

    def load(self, directory: Path | str) -> None:
        directory = Path(directory)
        with open(directory / "bm25.pkl", "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._metadata = data["metadata"]
