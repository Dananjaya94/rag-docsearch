# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models
WORKDIR /app

# CPU-only torch keeps the image ~1.5 GB smaller than the default CUDA wheels.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY pyproject.toml ./
RUN mkdir app && pip install ".[embeddings]" && rm -rf app

COPY app ./app
COPY migrations ./migrations
COPY alembic.ini ./
RUN pip install --no-deps . \
    && useradd --create-home --uid 10001 docsearch \
    && mkdir -p /data/uploads /models && chown -R docsearch /data /models
USER docsearch

EXPOSE 8000
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
