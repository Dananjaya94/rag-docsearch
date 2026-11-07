"""Turn uploaded files into page-level plain text."""

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_TYPES = {
    "application/pdf": "pdf",
    "text/plain": "text",
    "text/markdown": "text",
    "text/x-markdown": "text",
}
SUPPORTED_SUFFIXES = {".pdf": "pdf", ".txt": "text", ".md": "text", ".markdown": "text"}


class UnsupportedFileType(ValueError):
    pass


@dataclass(frozen=True)
class Page:
    number: int | None  # 1-based for paginated formats, None for plain text
    text: str


def detect_kind(filename: str, content_type: str | None) -> str:
    if content_type and content_type in SUPPORTED_TYPES:
        return SUPPORTED_TYPES[content_type]
    suffix = Path(filename).suffix.lower()
    if suffix in SUPPORTED_SUFFIXES:
        return SUPPORTED_SUFFIXES[suffix]
    raise UnsupportedFileType(f"Unsupported file type: {content_type or suffix or 'unknown'}")


def extract_pages(path: Path, kind: str) -> list[Page]:
    if kind == "pdf":
        return _extract_pdf(path)
    if kind == "text":
        return [Page(number=None, text=path.read_text(encoding="utf-8", errors="replace"))]
    raise UnsupportedFileType(kind)


def _extract_pdf(path: Path) -> list[Page]:
    import pymupdf  # imported lazily so the API process starts fast

    pages: list[Page] = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            pages.append(Page(number=i, text=page.get_text("text")))
    return pages
