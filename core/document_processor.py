from __future__ import annotations
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from pypdf import PdfReader
from config import cfg


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return hashlib.md5(self.text.encode()).hexdigest()[:12]


def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Sentence-aware sliding window chunker."""
    # Split into sentences first (rough)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, current, current_len = [], [], 0

    for sent in sentences:
        words = sent.split()
        if current_len + len(words) > chunk_size and current:
            chunks.append(" ".join(current))
            # keep overlap
            overlap_words = current[-overlap:] if overlap < len(current) else current
            current = overlap_words
            current_len = len(current)
        current.extend(words)
        current_len += len(words)

    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if len(c.split()) >= 10]


def load_pdf(path: str | Path) -> List[Chunk]:
    path = Path(path)
    reader = PdfReader(str(path))
    chunks: List[Chunk] = []
    global_idx = 0

    for page_num, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        if not raw.strip():
            continue
        page_chunks = _split_text(raw, cfg.chunk_size, cfg.chunk_overlap)
        for local_idx, chunk_text in enumerate(page_chunks):
            chunks.append(Chunk(
                text=chunk_text,
                metadata={
                    "source": path.name,
                    "source_path": str(path),
                    "page": page_num,
                    "chunk_index": global_idx,
                    "local_chunk_in_page": local_idx,
                    "word_count": len(chunk_text.split()),
                    "char_count": len(chunk_text),
                }
            ))
            global_idx += 1

    return chunks


def load_text(path: str | Path) -> List[Chunk]:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    raw_chunks = _split_text(text, cfg.chunk_size, cfg.chunk_overlap)
    return [
        Chunk(
            text=c,
            metadata={
                "source": path.name,
                "source_path": str(path),
                "page": None,
                "chunk_index": i,
                "word_count": len(c.split()),
                "char_count": len(c),
            }
        )
        for i, c in enumerate(raw_chunks)
    ]


def load_documents(paths: List[str | Path]) -> List[Chunk]:
    chunks = []
    for p in paths:
        p = Path(p)
        if p.suffix.lower() == ".pdf":
            chunks.extend(load_pdf(p))
        else:
            chunks.extend(load_text(p))
    return chunks