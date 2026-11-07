import math

from app.embeddings import HashingEmbedder
from app.retrieval import RRF_K, rrf_fuse


def test_rrf_rewards_agreement_between_rankers():
    fused = rrf_fuse(vector_ids=[1, 2, 3], keyword_ids=[3, 4], k=10)
    assert fused[0].chunk_id == 3  # ranked by both lists
    assert fused[0].score == 1 / (RRF_K + 3) + 1 / (RRF_K + 1)
    assert {c.chunk_id for c in fused} == {1, 2, 3, 4}


def test_rrf_respects_k_and_is_deterministic():
    a = rrf_fuse([5, 6, 7, 8], [], k=2)
    b = rrf_fuse([5, 6, 7, 8], [], k=2)
    assert [c.chunk_id for c in a] == [5, 6] == [c.chunk_id for c in b]


def test_rrf_handles_empty_inputs():
    assert rrf_fuse([], [], k=5) == []


def _cos(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_hashing_embedder_is_normalised_and_tracks_overlap():
    emb = HashingEmbedder(384)
    q, near, far = emb.embed(
        ["annual leave carry forward", "annual leave may be carried forward", "VPN token expired"]
    )
    assert math.isclose(_cos(q, q), 1.0, rel_tol=1e-9)
    assert _cos(q, near) > _cos(q, far)
    assert emb.embed(["same text"]) == emb.embed(["same text"])
