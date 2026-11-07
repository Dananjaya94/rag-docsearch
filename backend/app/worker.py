"""RQ worker entrypoint: `python -m app.worker`."""

import logging

from rq import Worker

from app.embeddings import get_embedder
from app.queue import QUEUE_NAME, get_redis


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    get_embedder()  # load the model once, before taking jobs
    Worker([QUEUE_NAME], connection=get_redis()).work(with_scheduler=True)


if __name__ == "__main__":
    main()
