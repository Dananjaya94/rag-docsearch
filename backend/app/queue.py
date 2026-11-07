import contextlib
from functools import lru_cache

from redis import Redis
from rq import Queue, Retry

from app.config import get_settings

QUEUE_NAME = "ingest"


@lru_cache
def get_redis() -> Redis:
    return Redis.from_url(get_settings().redis_url)


def enqueue_ingest(document_id: str, kind: str) -> None:
    settings = get_settings()
    if settings.ingest_mode == "inline":
        from app.ingestion import ingest_document

        # Failures are already recorded on the document row; don't fail the upload request.
        with contextlib.suppress(Exception):
            ingest_document(document_id, kind)
        return
    Queue(QUEUE_NAME, connection=get_redis()).enqueue(
        "app.ingestion.ingest_document",
        document_id,
        kind,
        job_timeout=900,
        retry=Retry(max=2, interval=[10, 60]),
    )
