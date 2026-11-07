"""Hybrid retrieval: pgvector cosine search + Postgres full-text search, fused with RRF.

Dense vectors catch paraphrases; full-text catches exact identifiers (error codes, account
types, clause numbers) that embeddings tend to blur. Reciprocal Rank Fusion combines the two
rankings without having to calibrate their raw scores against each other.
"""

import uuid
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.embeddings import Embedder
from app.models import Chunk, Document, DocumentStatus

RRF_K = 60  # standard constant from Cormack et al. (2009)


@dataclass
class Candidate:
    chunk_id: int
    vector_rank: int | None = None
    keyword_rank: int | None = None

    @property
    def score(self) -> float:
        return sum(1.0 / (RRF_K + r) for r in (self.vector_rank, self.keyword_rank) if r)


def rrf_fuse(vector_ids: list[int], keyword_ids: list[int], k: int) -> list[Candidate]:
    """Fuse two ranked id lists (best first). Ranks are 1-based."""
    by_id: dict[int, Candidate] = {}
    for rank, cid in enumerate(vector_ids, start=1):
        by_id.setdefault(cid, Candidate(cid)).vector_rank = rank
    for rank, cid in enumerate(keyword_ids, start=1):
        by_id.setdefault(cid, Candidate(cid)).keyword_rank = rank
    return sorted(by_id.values(), key=lambda c: (-c.score, c.chunk_id))[:k]


@dataclass
class RetrievedChunk:
    chunk: Chunk
    document: Document
    candidate: Candidate


def hybrid_search(
    session: Session,
    embedder: Embedder,
    query: str,
    k: int = 8,
    document_ids: list[uuid.UUID] | None = None,
    candidate_pool: int = 40,
) -> list[RetrievedChunk]:
    pool = max(candidate_pool, k * 4)
    ready = select(Document.id).where(Document.status == DocumentStatus.READY)
    if document_ids:
        ready = ready.where(Document.id.in_(document_ids))
    ready_ids = ready.scalar_subquery()

    query_vec = embedder.embed([query])[0]
    vector_ids = list(
        session.scalars(
            select(Chunk.id)
            .where(Chunk.document_id.in_(ready_ids))
            .order_by(Chunk.embedding.cosine_distance(query_vec))
            .limit(pool)
        )
    )

    tsquery = func.websearch_to_tsquery("english", query)
    keyword_ids = list(
        session.scalars(
            select(Chunk.id)
            .where(Chunk.document_id.in_(ready_ids), Chunk.tsv.op("@@")(tsquery))
            .order_by(func.ts_rank_cd(Chunk.tsv, tsquery).desc(), Chunk.id)
            .limit(pool)
        )
    )

    fused = rrf_fuse(vector_ids, keyword_ids, k)
    if not fused:
        return []
    rows = session.execute(
        select(Chunk, Document)
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.id.in_([c.chunk_id for c in fused]))
    ).all()
    by_id = {chunk.id: (chunk, doc) for chunk, doc in rows}
    return [RetrievedChunk(*by_id[c.chunk_id], candidate=c) for c in fused if c.chunk_id in by_id]
