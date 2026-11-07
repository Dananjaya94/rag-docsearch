from pathlib import Path

from fastapi.testclient import TestClient


def _upload(client: TestClient, path: Path, content_type: str = "application/pdf"):
    with path.open("rb") as fh:
        return client.post("/documents", files={"file": (path.name, fh, content_type)})


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_upload_ingests_pdf_and_tracks_pages(client, sample_pdf):
    res = _upload(client, sample_pdf)
    assert res.status_code == 202
    doc = client.get(f"/documents/{res.json()['id']}").json()
    assert doc["status"] == "ready", doc["error"]
    assert doc["page_count"] == 3
    assert doc["chunk_count"] >= 3


def test_duplicate_upload_is_deduplicated(client, sample_pdf):
    first = _upload(client, sample_pdf)
    second = _upload(client, sample_pdf)
    assert second.status_code == 200
    assert second.json()["id"] == first.json()["id"]
    assert len(client.get("/documents").json()) == 1


def test_rejects_unsupported_and_empty_files(client, tmp_path):
    exe = tmp_path / "tool.exe"
    exe.write_bytes(b"MZ")
    assert _upload(client, exe, "application/octet-stream").status_code == 415
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    assert _upload(client, empty, "text/plain").status_code == 400


def test_search_finds_the_right_page(client, sample_pdf):
    _upload(client, sample_pdf)
    hits = client.post("/search", json={"query": "how many days of annual leave", "k": 3}).json()[
        "hits"
    ]
    assert hits[0]["page"] == 1
    assert "14 days" in hits[0]["content"]


def test_keyword_half_catches_exact_codes(client, sample_pdf):
    _upload(client, sample_pdf)
    hits = client.post("/search", json={"query": "RW-404", "k": 3}).json()["hits"]
    assert hits[0]["page"] == 3
    assert hits[0]["keyword_rank"] == 1


def test_ask_without_llm_returns_extractive_answer_with_citations(client, sample_pdf):
    _upload(client, sample_pdf)
    body = client.post("/ask", json={"question": "Who approves expense claims over 50,000?"}).json()
    assert body["mode"] == "extractive"
    assert body["citations"][0]["page"] == 2
    assert "department head" in body["answer"]


def test_ask_with_no_documents(client):
    body = client.post("/ask", json={"question": "anything?"}).json()
    assert body["citations"] == []


def test_failed_ingestion_is_reported(client, tmp_path):
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"%PDF-1.4 not really a pdf")
    res = _upload(client, bad)
    doc = client.get(f"/documents/{res.json()['id']}").json()
    assert doc["status"] == "failed"
    assert doc["error"]


def test_delete_removes_document_and_chunks(client, sample_pdf):
    doc_id = _upload(client, sample_pdf).json()["id"]
    assert client.delete(f"/documents/{doc_id}").status_code == 204
    assert client.get(f"/documents/{doc_id}").status_code == 404
    assert client.post("/search", json={"query": "leave"}).json()["hits"] == []


def test_search_can_be_scoped_to_documents(client, sample_pdf, tmp_path):
    notes = tmp_path / "notes.md"
    notes.write_text("# Notes\n\nAnnual leave planning for the offsite in March.")
    _upload(client, sample_pdf)
    notes_id = _upload(client, notes, "text/markdown").json()["id"]
    hits = client.post(
        "/search", json={"query": "annual leave", "document_ids": [notes_id]}
    ).json()["hits"]
    assert hits and all(h["document_id"] == notes_id for h in hits)
