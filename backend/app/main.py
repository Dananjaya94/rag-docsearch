import hashlib
import logging
import os
import tempfile
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_session
from app.embeddings import get_embedder
from app.extract import UnsupportedFileType, detect_kind
from app.ingestion import upload_path
from app.llm import get_answer_generator
from app.models import Document
from app.queue import enqueue_ingest, get_redis
from app.retrieval import hybrid_search
from app.schemas import (
    AskRequest,
    AskResponse,
    Citation,
    DocumentOut,
    SearchHit,
    SearchRequest,
    SearchResponse,
)

logging.basicConfig(level=logging.INFO)
settings = get_settings()

app = FastAPI(
    title="RAG DocSearch",
    version="0.1.0",
    description="Upload documents, search them with hybrid retrieval, ask grounded questions.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["ops"])
def health(session: Session = Depends(get_session)) -> dict:
    checks = {"database": "ok", "redis": "ok" if settings.ingest_mode == "inline" else None}
    try:
        session.execute(text("select 1"))
    except Exception as exc:
        checks["database"] = f"error: {exc.__class__.__name__}"
    if checks["redis"] is None:
        try:
            get_redis().ping()
            checks["redis"] = "ok"
        except Exception as exc:
            checks["redis"] = f"error: {exc.__class__.__name__}"
    healthy = all(v == "ok" for v in checks.values())
    if not healthy:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=checks)
    return {"status": "ok", **checks}


async def _store_upload(file: UploadFile, dest_dir: Path) -> tuple[Path, str, int]:
    """Stream the upload to disk while hashing it, enforcing the size limit."""
    limit = settings.max_upload_mb * 1024 * 1024
    digest, size = hashlib.sha256(), 0
    dest_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dest_dir, prefix=".upload-")
    try:
        with os.fdopen(fd, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(
                        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        f"File exceeds {settings.max_upload_mb} MB",
                    )
                digest.update(chunk)
                out.write(chunk)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return Path(tmp_name), digest.hexdigest(), size


@app.post("/documents", response_model=DocumentOut, tags=["documents"])
async def upload_document(
    response: Response,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
) -> Document:
    filename = Path(file.filename or "upload").name
    try:
        kind = detect_kind(filename, file.content_type)
    except UnsupportedFileType as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc

    tmp, sha, size = await _store_upload(file, settings.upload_dir)
    if size == 0:
        tmp.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "File is empty")

    existing = session.scalar(select(Document).where(Document.sha256 == sha))
    if existing is not None:
        tmp.unlink(missing_ok=True)
        response.status_code = status.HTTP_200_OK
        return existing

    doc = Document(
        filename=filename,
        content_type=file.content_type or "application/octet-stream",
        size_bytes=size,
        sha256=sha,
    )
    session.add(doc)
    session.flush()
    tmp.replace(upload_path(doc.id))
    session.commit()

    enqueue_ingest(str(doc.id), kind)
    session.refresh(doc)
    response.status_code = status.HTTP_202_ACCEPTED
    return doc


@app.get("/documents", response_model=list[DocumentOut], tags=["documents"])
def list_documents(session: Session = Depends(get_session)) -> list[Document]:
    return list(session.scalars(select(Document).order_by(Document.created_at.desc())))


@app.get("/documents/{document_id}", response_model=DocumentOut, tags=["documents"])
def get_document(document_id: uuid.UUID, session: Session = Depends(get_session)) -> Document:
    doc = session.get(Document, document_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return doc


@app.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["documents"])
def delete_document(document_id: uuid.UUID, session: Session = Depends(get_session)) -> None:
    doc = session.get(Document, document_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    session.delete(doc)
    session.commit()
    upload_path(document_id).unlink(missing_ok=True)


@app.post("/search", response_model=SearchResponse, tags=["retrieval"])
def search(req: SearchRequest, session: Session = Depends(get_session)) -> SearchResponse:
    results = hybrid_search(session, get_embedder(), req.query, req.k, req.document_ids)
    return SearchResponse(
        query=req.query,
        hits=[
            SearchHit(
                chunk_id=r.chunk.id,
                document_id=r.document.id,
                filename=r.document.filename,
                page=r.chunk.page,
                content=r.chunk.content,
                score=round(r.candidate.score, 6),
                vector_rank=r.candidate.vector_rank,
                keyword_rank=r.candidate.keyword_rank,
            )
            for r in results
        ],
    )


@app.post("/ask", response_model=AskResponse, tags=["retrieval"])
def ask(req: AskRequest, session: Session = Depends(get_session)) -> AskResponse:
    results = hybrid_search(session, get_embedder(), req.question, req.k, req.document_ids)
    if not results:
        return AskResponse(
            question=req.question,
            answer="I couldn't find anything relevant in the indexed documents.",
            mode="extractive",
            citations=[],
        )
    generator = get_answer_generator()
    try:
        answer = generator.answer(req.question, results)
    except Exception as exc:
        logging.getLogger(__name__).exception("answer generation failed")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Answer generation failed") from exc
    return AskResponse(
        question=req.question,
        answer=answer,
        mode=generator.mode,
        citations=[
            Citation(
                index=i,
                chunk_id=r.chunk.id,
                document_id=r.document.id,
                filename=r.document.filename,
                page=r.chunk.page,
                snippet=r.chunk.content[:300],
            )
            for i, r in enumerate(results, start=1)
        ],
    )
