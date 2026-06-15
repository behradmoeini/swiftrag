from swiftrag.chunking import chunk_text


def test_short_text_single_chunk():
    text = "A short sentence."
    assert chunk_text(text, chunk_size=100, chunk_overlap=10) == [text]


def test_empty_text():
    assert chunk_text("", chunk_size=100) == []
    assert chunk_text("   \n  ", chunk_size=100) == []


def test_long_text_splits_with_overlap():
    sentences = " ".join(f"Sentence number {i} has some words." for i in range(200))
    chunks = chunk_text(sentences, chunk_size=32, chunk_overlap=8)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


def test_paragraphs_preserved_when_small():
    text = "Para one is here.\n\nPara two is here.\n\nPara three is here."
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=10)
    assert len(chunks) == 1
