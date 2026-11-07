import uuid
from types import SimpleNamespace

import fakeredis
from rq import Queue

from app import queue as queue_mod
from app.config import get_settings
from app.llm import SYSTEM_PROMPT, ClaudeAnswerGenerator
from app.retrieval import Candidate, RetrievedChunk


def _retrieved(content: str, page: int | None, filename: str = "policy.pdf") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=SimpleNamespace(id=1, page=page, content=content),
        document=SimpleNamespace(id=uuid.uuid4(), filename=filename),
        candidate=Candidate(1, vector_rank=1),
    )


def test_claude_generator_sends_numbered_sources_and_returns_text():
    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                content=[
                    SimpleNamespace(type="text", text=" 14 days [1]. "),
                    SimpleNamespace(type="thinking", text="ignored"),
                ]
            )

    gen = ClaudeAnswerGenerator.__new__(ClaudeAnswerGenerator)
    gen._client, gen._model, gen._max_tokens = SimpleNamespace(messages=FakeMessages()), "m", 256

    answer = gen.answer("How much leave?", [_retrieved("14 days of annual leave", page=1)])

    assert answer == "14 days [1]."
    assert captured["system"] == SYSTEM_PROMPT
    assert captured["model"] == "m"
    prompt = captured["messages"][0]["content"]
    assert '<source n="1" file="policy.pdf, page 1">' in prompt
    assert prompt.endswith("Question: How much leave?")


def test_queue_mode_enqueues_job_with_retry(monkeypatch):
    fake = fakeredis.FakeRedis()
    monkeypatch.setattr(queue_mod, "get_redis", lambda: fake)
    monkeypatch.setattr(get_settings(), "ingest_mode", "queue")

    queue_mod.enqueue_ingest("abc", "pdf")

    jobs = Queue(queue_mod.QUEUE_NAME, connection=fake).jobs
    assert len(jobs) == 1
    assert jobs[0].func_name == "app.ingestion.ingest_document"
    assert jobs[0].args == ("abc", "pdf")
    assert jobs[0].retries_left == 2
