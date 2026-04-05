#!/usr/bin/env python3
"""
Demo: ingest a PDF and run example queries.
Usage: python demo_pdf.py path/to/document.pdf
       python demo_pdf.py  (uses sample_doc.pdf if it exists)
"""
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from rag.pipeline import RAGPipeline
from core.document_processor import load_pdf

console = Console()


def print_all_words(pdf_path: str) -> None:
    """Extract and print every word from the PDF."""
    chunks = load_pdf(pdf_path)
    all_text = " ".join(c.text for c in chunks)
    words = all_text.split()
    console.print(Rule("[bold yellow]ALL WORDS IN DOCUMENT[/]"))
    console.print(f"[dim]Total words: {len(words)}[/]\n")
    # Print 20 per line for readability
    for i in range(0, len(words), 20):
        console.print(" ".join(words[i:i+20]))
    console.print(Rule())


def main():
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "sample_doc.pdf"
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        console.print(f"[red]File not found: {pdf_path}[/]")
        console.print("[yellow]Creating a minimal sample PDF for demo...[/]")
        _create_sample_pdf(pdf_path)

    console.print(Panel(f"[bold]PDF RAG Demo[/]\nFile: {pdf_path}", border_style="cyan"))

    # Print all words first
    print_all_words(str(pdf_path))

    # Build pipeline with reranker ON
    console.rule("[bold cyan]Ingesting PDF[/]")
    pipeline = RAGPipeline(use_reranker=True, show_reranker_comparison=True)
    pipeline.ingest([str(pdf_path)], force=True)

    demo_questions = [
        "What is this document about?",
        "What are the main topics covered?",
        "Summarize the key points.",
    ]

    for q in demo_questions:
        console.rule(f"[bold]Q: {q}[/]")
        answer = pipeline.query(q, show_sources=True)
        console.print(Panel(answer, title="Answer", border_style="green"))

    # Show reranker comparison for last question
    console.rule("[bold magenta]Reranker Comparison Demo[/]")
    pipeline.compare_reranker(demo_questions[-1])


def _create_sample_pdf(output_path: Path) -> None:
    """Create a minimal sample PDF using only stdlib + a try on reportlab/fpdf2."""
    sample_text = """
    Introduction to Machine Learning

    Machine learning is a subset of artificial intelligence that enables systems to learn and
    improve from experience without being explicitly programmed. It focuses on developing computer
    programs that can access data and use it to learn for themselves.

    Types of Machine Learning

    Supervised Learning: The algorithm learns from labeled training data. Examples include
    classification and regression tasks. Common algorithms are linear regression, decision trees,
    random forests, and support vector machines.

    Unsupervised Learning: The algorithm finds patterns in unlabeled data. Examples include
    clustering and dimensionality reduction. Common algorithms include K-means, DBSCAN, and PCA.

    Reinforcement Learning: The algorithm learns by interacting with an environment and receiving
    rewards or penalties. Used in game playing, robotics, and recommendation systems.

    Deep Learning

    Deep learning uses neural networks with many layers to model complex patterns. Convolutional
    neural networks excel at image recognition. Recurrent neural networks handle sequential data
    like text and time series. Transformer models power modern NLP systems.

    Applications

    Machine learning is applied in healthcare for disease diagnosis, in finance for fraud detection,
    in retail for recommendation systems, in autonomous vehicles, and natural language processing.

    Conclusion

    Machine learning continues to advance rapidly, enabling new applications across every industry.
    Understanding its foundations is essential for any modern software engineer or data scientist.
    """
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        for line in sample_text.strip().split("\n"):
            pdf.multi_cell(0, 8, line.strip())
        pdf.output(str(output_path))
        console.print(f"[green]Created sample PDF: {output_path}[/]")
    except ImportError:
        try:
            from reportlab.pdfgen import canvas as rlcanvas
            from reportlab.lib.pagesizes import letter
            c = rlcanvas.Canvas(str(output_path), pagesize=letter)
            y = 750
            for line in sample_text.strip().split("\n"):
                if y < 50:
                    c.showPage()
                    y = 750
                c.drawString(50, y, line.strip()[:90])
                y -= 15
            c.save()
            console.print(f"[green]Created sample PDF: {output_path}[/]")
        except ImportError:
            # Last resort: create a text file and note it
            txt_path = output_path.with_suffix(".txt")
            txt_path.write_text(sample_text)
            console.print(f"[yellow]fpdf2/reportlab not found. Created text file instead: {txt_path}[/]")
            console.print("[yellow]Install fpdf2: pip install fpdf2[/]")
            # Rename to use the txt file
            import os
            output_path.write_bytes(txt_path.read_bytes())  # won't be real PDF, use as text
            output_path.unlink(missing_ok=True)
            output_path = txt_path


if __name__ == "__main__":
    main()
