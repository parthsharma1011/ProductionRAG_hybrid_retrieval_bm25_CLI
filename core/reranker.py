from __future__ import annotations
from typing import List, Dict, Any

import numpy as np
from rich.console import Console
from rich.table import Table

from config import cfg
from core.embeddings import get_embedding_model

console = Console()


class Reranker:
    def __init__(self, model_name: str = cfg.reranker_model):
        # Uses Gemini embeddings (cosine similarity) — no PyTorch required
        self._embed = get_embedding_model()

    def rerank(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        top_k: int = cfg.top_k_final,
        show_comparison: bool = True,
    ) -> List[Dict[str, Any]]:
        if not docs:
            return []

        q_vec = self._embed.embed_one(query)
        texts = [d["text"] for d in docs]
        doc_vecs = self._embed.embed(texts)

        scores = doc_vecs @ q_vec  # cosine sim (vectors are L2-normalised)

        for doc, score in zip(docs, scores):
            doc["rerank_score"] = float(score)

        reranked = sorted(docs, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

        if show_comparison:
            _print_comparison(docs, reranked, query)

        return reranked


def _print_comparison(before: List[Dict], after: List[Dict], query: str) -> None:
    table = Table(title=f"[bold cyan]Reranker Comparison[/] | Query: {query[:80]}", show_lines=True)
    table.add_column("Before Rank", style="yellow", width=12)
    table.add_column("After Rank", style="green", width=10)
    table.add_column("Hybrid Score", width=13)
    table.add_column("Rerank Score", width=13)
    table.add_column("Source", width=15)
    table.add_column("Page", width=6)
    table.add_column("Snippet", width=50)

    before_rank_map = {d["chunk_id"]: i + 1 for i, d in enumerate(before)}

    for after_rank, doc in enumerate(after, start=1):
        cid = doc["chunk_id"]
        b_rank = before_rank_map.get(cid, "?")
        moved = ""
        if isinstance(b_rank, int):
            diff = b_rank - after_rank
            moved = f" [green](+{diff})[/]" if diff > 0 else (f" [red]({diff})[/]" if diff < 0 else "")
        table.add_row(
            f"{b_rank}{moved}",
            str(after_rank),
            f"{doc.get('rrf_score', 0):.4f}",
            f"{doc.get('rerank_score', 0):.4f}",
            doc.get("source", ""),
            str(doc.get("page", "-")),
            doc["text"][:120].replace("\n", " ") + "…",
        )
    console.print(table)
