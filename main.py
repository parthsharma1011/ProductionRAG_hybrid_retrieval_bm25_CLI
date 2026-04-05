#!/usr/bin/env python3
"""Interactive RAG CLI. Usage: python main.py [--no-reranker] file1.pdf file2.txt ..."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from rich.console import Console
from rich.panel import Panel

from rag.pipeline import RAGPipeline

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Production RAG CLI")
    parser.add_argument("files", nargs="*", help="Files to ingest (PDF or text)")
    parser.add_argument("--no-reranker", action="store_true", help="Disable reranker")
    parser.add_argument("--force-reindex", action="store_true", help="Force re-indexing even if index exists")
    parser.add_argument("--compare", action="store_true", help="Compare reranker on/off for each query")
    args = parser.parse_args()

    use_reranker = not args.no_reranker
    pipeline = RAGPipeline(use_reranker=use_reranker)

    if args.files:
        pipeline.ingest(args.files, force=args.force_reindex)
    else:
        # Auto-scan the files/ folder for PDFs
        from config import cfg
        files_dir = Path(__file__).parent / "files"
        auto_files = list(files_dir.glob("*.pdf")) + list(files_dir.glob("*.txt"))
        if auto_files:
            console.print(f"[cyan]Auto-detected {len(auto_files)} file(s) from [bold]files/[/] folder.[/]")
            pipeline.ingest(auto_files, force=args.force_reindex)
        elif (cfg.index_dir / "faiss.index").exists():
            pipeline.ingest([], force=False)
        else:
            console.print("[red]No files provided, no files/ folder found, and no existing index.[/]")
            sys.exit(1)

    console.print(Panel(
        "[bold green]RAG ready![/]\n"
        "Commands: [cyan]quit[/] | [cyan]clear[/] (clear memory) | [cyan]cache[/] (cache stats) | [cyan]compare <question>[/]\n"
        f"Reranker: {'[green]ON[/]' if use_reranker else '[yellow]OFF[/]'}",
        title="Production RAG"
    ))

    while True:
        try:
            question = input("\n[You] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            break
        if question.lower() == "clear":
            pipeline.memory.clear()
            console.print("[dim]Memory cleared.[/]")
            continue
        if question.lower() == "cache":
            console.print(pipeline.cache.stats())
            continue
        if question.lower().startswith("compare "):
            q = question[8:].strip()
            pipeline.compare_reranker(q)
            continue
        if question.lower() == "reranker on":
            pipeline.enable_reranker()
            continue
        if question.lower() == "reranker off":
            pipeline.disable_reranker()
            continue

        if args.compare:
            pipeline.compare_reranker(question)
        else:
            answer = pipeline.query(question)
            console.print(Panel(answer, title="[bold]Answer[/]", border_style="blue"))


if __name__ == "__main__":
    main()
