from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any

import faiss
import numpy as np

from config import cfg
from core.document_processor import Chunk
from core.embeddings import get_embedding_model


class FAISSVectorStore:
    def __init__(self, dim: int = cfg.embedding_dim):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)   # cosine (vectors are L2-normalised)
        self.metadata: List[Dict[str, Any]] = []
        self.texts: List[str] = []

    def add_chunks(self, chunks: List[Chunk], batch_size: int = 128, show_progress: bool = True) -> None:
        model = get_embedding_model()
        texts = [c.text for c in chunks]
        from tqdm import tqdm
        for start in tqdm(range(0, len(texts), batch_size), desc="Indexing", disable=not show_progress):
            batch = texts[start: start + batch_size]
            vecs = model.embed(batch)
            self.index.add(vecs)
            for i, c in enumerate(chunks[start: start + batch_size]):
                self.texts.append(c.text)
                self.metadata.append({**c.metadata, "chunk_id": c.chunk_id, "text": c.text})

    def search(self, query: str, top_k: int = cfg.top_k_dense) -> List[Dict[str, Any]]:
        model = get_embedding_model()
        q_vec = model.embed_one(query).reshape(1, -1)
        k = min(top_k, self.index.ntotal)
        if k == 0:
            return []
        scores, indices = self.index.search(q_vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            entry = dict(self.metadata[idx])
            entry["dense_score"] = float(score)
            entry["rank_dense"] = len(results) + 1
            results.append(entry)
        return results

    def save(self, directory: Path | str) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "faiss.index"))
        with open(directory / "metadata.pkl", "wb") as f:
            pickle.dump({"metadata": self.metadata, "texts": self.texts}, f)

    def load(self, directory: Path | str) -> None:
        directory = Path(directory)
        self.index = faiss.read_index(str(directory / "faiss.index"))
        with open(directory / "metadata.pkl", "rb") as f:
            data = pickle.load(f)
        self.metadata = data["metadata"]
        self.texts = data["texts"]

    @property
    def total_chunks(self) -> int:
        return self.index.ntotal