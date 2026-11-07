import os
import tempfile
from pathlib import Path

# Configure before any app module is imported (settings are cached).
os.environ.setdefault(
    "DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/docsearch_test"
)
os.environ["INGEST_MODE"] = "inline"
os.environ["EMBEDDER"] = "hashing"
os.environ["UPLOAD_DIR"] = tempfile.mkdtemp(prefix="docsearch-uploads-")
os.environ.pop("ANTHROPIC_API_KEY", None)

import pytest  # noqa: E402
from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.db import get_engine  # noqa: E402

BACKEND = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def migrated_db():
    cfg = Config(str(BACKEND / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND / "migrations"))
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    yield
    get_engine().dispose()


@pytest.fixture(autouse=True)
def clean_tables():
    yield
    with get_engine().begin() as conn:
        conn.execute(text("TRUNCATE chunks, documents RESTART IDENTITY CASCADE"))


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    return TestClient(app)


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    import pymupdf

    pages = [
        "Leave Policy\n\nPermanent employees are entitled to 14 days of annual leave per year. "
        "Unused annual leave may be carried forward for up to six months.",
        "Expense Claims\n\nClaims must be submitted within 30 days with original receipts. "
        "Claims above LKR 50,000 require approval from a department head.",
        "Remote Work\n\nStaff may work remotely two days per week with manager approval. "
        "Error code RW-404 means the VPN token has expired.",
    ]
    path = tmp_path / "handbook.pdf"
    doc = pymupdf.open()
    for body in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), body, fontsize=11)
    doc.save(path)
    doc.close()
    return path
