"""The ingestion job: extract -> chunk -> embed -> store. Runs in the RQ worker."""

import logging
import uuid
from pathlib import Path

from sqlalchemy import delete

from app.chunking import chunk_pages
from app.config import get_settings
from app.db import get_sessionmaker
from app.embeddings import get_embedder
from app.extract import extract_pages
from app.models import Chunk, Document, DocumentStatus

log = logging.getLogger(__name__)
EMBED_BATCH = 64


def upload_path(document_id: uuid.UUID) -> Path:
    return get_settings().upload_dir / str(document_id)


def ingest_document(document_id: str, kind: str) -> None:
    settings = get_settings()
    doc_id = uuid.UUID(document_id)
    Session = get_sessionmaker()

    with Session() as session:
        doc = session.get(Document, doc_id)
        if doc is None:
            log.warning("document %s vanished before ingestion", doc_id)
            return
        doc.status, doc.error = DocumentStatus.PROCESSING, None
        session.commit()

    try:
        pages = extract_pages(upload_path(doc_id), kind)
        chunks = chunk_pages(pages, settings.chunk_max_chars, settings.chunk_overlap_chars)
        if not chunks:
            raise ValueError("No extractable text found (scanned PDFs need OCR, not yet supported)")

        embedder = get_embedder()
        vectors: list[list[float]] = []
        for start in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[start : start + EMBED_BATCH]
            vectors.extend(embedder.embed([c.content for c in batch]))

        with Session() as session, session.begin():
            # Idempotent: a retried job replaces whatever a failed attempt left behind.
            session.execute(delete(Chunk).where(Chunk.document_id == doc_id))
            session.add_all(
                Chunk(
                    document_id=doc_id,
                    ordinal=c.ordinal,
                    page=c.page,
                    content=c.content,
                    embedding=v,
                )
                for c, v in zip(chunks, vectors, strict=True)
            )
            doc = session.get(Document, doc_id)
            doc.status = DocumentStatus.READY
            doc.page_count = sum(1 for p in pages if p.number) or None
            doc.chunk_count = len(chunks)
        log.info("ingested %s: %d chunks", doc_id, len(chunks))
    except Exception as exc:
        log.exception("ingestion failed for %s", doc_id)
        with Session() as session, session.begin():
            doc = session.get(Document, doc_id)
            if doc is not None:
                doc.status, doc.error = DocumentStatus.FAILED, str(exc)[:2000]
        raise
