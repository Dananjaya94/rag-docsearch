"""Paragraph-aware chunking that keeps page provenance.

Paragraphs are packed into chunks up to ``max_chars``. A paragraph longer than the budget is
split on sentence boundaries, and as a last resort on whitespace. Consecutive chunks share up to
``overlap_chars`` of trailing text so an answer that straddles a boundary is still retrievable.
Chunks never cross a page boundary, so every citation points at exactly one page.
"""

import re
from dataclasses import dataclass

from app.extract import Page

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WS = re.compile(r"[ \t\r\f\v]+")


@dataclass(frozen=True)
class TextChunk:
    ordinal: int
    page: int | None
    content: str


def normalise(text: str) -> str:
    text = text.replace("­", "")  # soft hyphens from PDF extraction
    text = re.sub(r"-\n(?=\w)", "", text)  # re-join words hyphenated across lines
    lines = [_WS.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _pieces(paragraph: str, max_chars: int) -> list[str]:
    """Break one paragraph into pieces no longer than max_chars."""
    if len(paragraph) <= max_chars:
        return [paragraph]
    out: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(paragraph):
        if len(sentence) <= max_chars:
            out.append(sentence)
            continue
        words, current = sentence.split(), ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if len(candidate) > max_chars and current:
                out.append(current)
                current = word[:max_chars]
            else:
                current = candidate[:max_chars]
        if current:
            out.append(current)
    return out


def _tail(text: str, overlap_chars: int) -> str:
    if overlap_chars <= 0 or len(text) <= overlap_chars:
        return text if overlap_chars > 0 else ""
    tail = text[-overlap_chars:]
    space = tail.find(" ")
    return tail[space + 1 :] if 0 <= space < len(tail) - 1 else tail


def chunk_pages(
    pages: list[Page], max_chars: int = 1200, overlap_chars: int = 200
) -> list[TextChunk]:
    if max_chars < 100:
        raise ValueError("max_chars must be at least 100")
    if not 0 <= overlap_chars < max_chars // 2:
        raise ValueError("overlap_chars must be between 0 and max_chars / 2")

    chunks: list[TextChunk] = []
    for page in pages:
        text = normalise(page.text)
        if not text:
            continue
        pieces: list[str] = []
        for para in _PARAGRAPH_SPLIT.split(text):
            para = " ".join(para.split())
            if para:
                pieces.extend(_pieces(para, max_chars - overlap_chars))

        current = ""
        for piece in pieces:
            candidate = f"{current}\n\n{piece}" if current else piece
            if len(candidate) <= max_chars:
                current = candidate
                continue
            chunks.append(TextChunk(len(chunks), page.number, current))
            overlap = _tail(current, overlap_chars)
            current = f"{overlap} {piece}".strip() if overlap else piece
        if current:
            chunks.append(TextChunk(len(chunks), page.number, current))
    return chunks
