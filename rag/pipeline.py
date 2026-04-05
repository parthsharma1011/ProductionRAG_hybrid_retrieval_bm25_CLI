from __future__ import annotations
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import os
from google import genai
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from config import cfg
from core.document_processor import Chunk, load_documents
from core.vector_store import FAISSVectorStore
from core.bm25_store import BM25Store
from core.hybrid_search import HybridSearcher
from core.reranker import Reranker
from core.cache import QueryCache
from core.memory import ConversationMemory

console = Console()


def _build_system_prompt() -> str:
    return (
        "You are a knowledgeable assistant. Answer questions based strictly on the provided context. "
        "If the context does not contain enough information, say so explicitly. "
        "Be concise, accurate, and cite the source and page when possible."
    )


def _build_user_prompt(query: str, context_docs: List[Dict], memory: ConversationMemory) -> str:
    ctx_parts = []
    for i, doc in enumerate(context_docs, start=1):
        source = doc.get("source", "unknown")
        page = doc.get("page")
        page_str = f", page {page}" if page else ""
        ctx_parts.append(f"[{i}] ({source}{page_str})\n{doc['text']}")
    context_block = "\n\n---\n\n".join(ctx_parts)

    history = memory.format_for_prompt()
    history_block = f"\n\n{history}\n\n" if history else "\n\n"

    return (
        f"## Context\n{context_block}"
        f"{history_block}"
        f"## Question\n{query}"
    )


class RAGPipeline:
    def __init__(
        self,
        use_reranker: bool = True,
        show_reranker_comparison: bool = True,
    ):
        self.use_reranker = use_reranker
        self.show_reranker_comparison = show_reranker_comparison

        self.vector_store = FAISSVectorStore()
        self.bm25_store = BM25Store()
        self.hybrid_searcher = HybridSearcher(self.vector_store, self.bm25_store)
        self.reranker = Reranker() if use_reranker else None
        self.cache = QueryCache()
        self.memory = ConversationMemory()
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self._system_prompt = _build_system_prompt()

    # ------------------------------------------------------------------ #
    #  Ingestion                                                           #
    # ------------------------------------------------------------------ #

    def ingest(self, paths: List[str | Path], force: bool = False) -> None:
        index_dir = cfg.index_dir
        if not force and (index_dir / "faiss.index").exists():
            console.print("[yellow]Loading existing index...[/]")
            self.vector_store.load(index_dir)
            self.bm25_store.load(index_dir)
            console.print(f"[green]Loaded {self.vector_store.total_chunks} chunks from disk.[/]")
            return

        console.print(f"[cyan]Ingesting {len(paths)} file(s)...[/]")
        chunks = load_documents(paths)
        console.print(f"[cyan]Total chunks: {len(chunks)}[/]")

        self.vector_store.add_chunks(chunks, show_progress=True)
        self.bm25_store.add_chunks(chunks)

        index_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store.save(index_dir)
        self.bm25_store.save(index_dir)
        console.print(f"[green]Indexed {len(chunks)} chunks and saved to {index_dir}[/]")

    # ------------------------------------------------------------------ #
    #  Query                                                               #
    # ------------------------------------------------------------------ #

    def query(self,question: str,use_cache: bool = True,show_sources: bool = True,) -> str:
        t0 = time.perf_counter()

        # 1. Cache check
        cache_key_ctx = str(self.memory.num_turns)
        if use_cache:
            cached = self.cache.get(question, cache_key_ctx)
            if cached:
                console.print("[dim]Cache hit[/]")
                self.memory.add("user", question)
                self.memory.add("assistant", cached)
                return cached

        # 2. Hybrid retrieval
        candidates = self.hybrid_searcher.search(question)

        # 3. Reranker (optional)
        if self.use_reranker and self.reranker:
            final_docs = self.reranker.rerank(
                question,
                candidates,
                top_k=cfg.top_k_final,
                show_comparison=self.show_reranker_comparison,
            )
            retrieval_method = "Hybrid + Reranker"
        else:
            final_docs = candidates[: cfg.top_k_final]
            retrieval_method = "Hybrid (no reranker)"

        # 4. Show sources
        if show_sources:
            _print_sources(final_docs, retrieval_method)

        # 5. Build prompt
        user_prompt = _build_user_prompt(question, final_docs, self.memory)

        # 6. LLM call
        response = self._client.models.generate_content(
            model=cfg.llm_model,
            contents=user_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=self._system_prompt,
                max_output_tokens=cfg.max_tokens,
            ),
        )
        answer = response.text

        # 7. Memory + cache
        self.memory.add("user", question)
        self.memory.add("assistant", answer)
        if use_cache:
            self.cache.set(question, answer, cache_key_ctx)

        elapsed = time.perf_counter() - t0
        console.print(f"[dim]Response time: {elapsed:.2f}s[/]")
        return answer

    # ------------------------------------------------------------------ #
    #  Reranker toggle helpers                                             #
    # ------------------------------------------------------------------ #

    def enable_reranker(self) -> None:
        if self.reranker is None:
            self.reranker = Reranker()
        self.use_reranker = True
        console.print("[green]Reranker ENABLED[/]")

    def disable_reranker(self) -> None:
        self.use_reranker = False
        console.print("[yellow]Reranker DISABLED[/]")

    def compare_reranker(self, question: str) -> None:
        """Run the same query with and without reranker and show both answers."""
        console.rule("[bold]Without Reranker[/]")
        self.disable_reranker()
        answer_no_rerank = self.query(question, use_cache=False, show_sources=True)
        console.print(Panel(Markdown(answer_no_rerank), title="Answer (no reranker)", border_style="yellow"))

        console.rule("[bold]With Reranker[/]")
        self.enable_reranker()
        answer_reranked = self.query(question, use_cache=False, show_sources=True)
        console.print(Panel(Markdown(answer_reranked), title="Answer (with reranker)", border_style="green"))


def _print_sources(docs: List[Dict], method: str) -> None:
    from rich.table import Table
    table = Table(title=f"Retrieved Sources — {method}", show_lines=False)
    table.add_column("#", width=3)
    table.add_column("Source", width=18)
    table.add_column("Page", width=5)
    table.add_column("Words", width=6)
    table.add_column("RRF", width=7)
    table.add_column("Rerank", width=7)
    table.add_column("Snippet", width=60)
    for i, doc in enumerate(docs, start=1):
        table.add_row(
            str(i),
            doc.get("source", ""),
            str(doc.get("page", "-")),
            str(doc.get("word_count", "")),
            f"{doc.get('rrf_score', 0):.4f}",
            f"{doc.get('rerank_score', 0):.4f}" if "rerank_score" in doc else "-",
            doc["text"][:120].replace("\n", " ") + "...",
        )
    console.print(table)
