import pytest

from app.chunking import chunk_pages, normalise
from app.extract import Page


def test_short_pages_become_one_chunk_each_and_keep_page_numbers():
    chunks = chunk_pages([Page(1, "Alpha paragraph."), Page(2, "Beta paragraph.")])
    assert [(c.page, c.content) for c in chunks] == [
        (1, "Alpha paragraph."),
        (2, "Beta paragraph."),
    ]
    assert [c.ordinal for c in chunks] == [0, 1]


def test_paragraphs_are_packed_up_to_budget():
    para = "word " * 40  # ~200 chars
    text = "\n\n".join([para.strip()] * 10)
    chunks = chunk_pages([Page(1, text)], max_chars=500, overlap_chars=0)
    assert len(chunks) > 1
    assert all(len(c.content) <= 500 for c in chunks)


def test_long_paragraph_is_split_and_overlaps():
    sentences = " ".join(f"Sentence number {i} talks about topic {i}." for i in range(80))
    chunks = chunk_pages([Page(1, sentences)], max_chars=400, overlap_chars=80)
    assert len(chunks) > 3
    assert all(len(c.content) <= 400 for c in chunks)
    # The start of each chunk repeats text from the end of the previous one.
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.content.split()[0] in prev.content


def test_unbroken_text_is_hard_wrapped():
    chunks = chunk_pages([Page(None, "x" * 5000)], max_chars=300, overlap_chars=0)
    assert all(len(c.content) <= 300 for c in chunks)


def test_empty_pages_are_skipped():
    assert chunk_pages([Page(1, "   \n\n "), Page(2, "")]) == []


def test_normalise_rejoins_hyphenated_words_and_collapses_spaces():
    assert normalise("docu-\nment   management­") == "document management"


@pytest.mark.parametrize("max_chars,overlap", [(50, 0), (500, 300), (500, -1)])
def test_invalid_parameters(max_chars, overlap):
    with pytest.raises(ValueError):
        chunk_pages([Page(1, "text")], max_chars=max_chars, overlap_chars=overlap)
