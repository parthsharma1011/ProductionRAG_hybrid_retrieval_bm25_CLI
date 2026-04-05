from __future__ import annotations
import os
from functools import lru_cache
from typing import List

import numpy as np
from google import genai
from config import cfg

_EMBED_MODEL = "models/gemini-embedding-001"


class EmbeddingModel:
    def __init__(self):
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def embed(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        all_vecs = []
        indices = range(0, len(texts), batch_size)
        if show_progress:
            from tqdm import tqdm
            indices = tqdm(indices, desc="Embedding")
        for start in indices:
            batch = texts[start: start + batch_size]
            result = self._client.models.embed_content(
                model=_EMBED_MODEL,
                contents=batch,
            )
            vecs = np.array([e.values for e in result.embeddings], dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.maximum(norms, 1e-9)
            all_vecs.append(vecs)
        return np.vstack(all_vecs)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()
