import pytest

from swiftrag import RAG
from swiftrag.exceptions import ConfigurationError, EmptyCorpusError
from swiftrag.persistence import is_safe_file


def _docs():
    return [
        {"text": "Apples are a kind of fruit.", "metadata": {"source": "a"}},
        {"text": "Bananas are yellow fruit.", "metadata": {"source": "b"}},
        {"text": "Carrots are vegetables.", "metadata": {"source": "c"}},
    ]


def test_delete_by_metadata_removes_matching_chunks():
    rag = RAG(documents=_docs())
    assert len(rag) == 3
    removed = rag.delete({"source": "b"})
    assert removed == 1
    assert len(rag) == 2
    sources = {s.metadata["source"] for s in rag.retrieve("fruit", top_k=5)}
    assert "b" not in sources


def test_delete_by_predicate():
    rag = RAG(documents=_docs())
    removed = rag.delete(lambda chunk: "vegetable" in chunk.text.lower())
    assert removed == 1
    assert len(rag) == 2


def test_delete_without_filter_raises():
    rag = RAG(documents=_docs())
    with pytest.raises(ConfigurationError):
        rag.delete(None)


def test_delete_everything_then_query_raises():
    rag = RAG(documents="Only one document here.")
    rag.delete(lambda c: True)
    assert len(rag) == 0
    with pytest.raises(EmptyCorpusError):
        rag.retrieve("anything")


def test_delete_allows_reinserting_same_text():
    # dedup must forget deleted chunks so the same text can be re-added.
    rag = RAG(documents={"text": "unique line", "metadata": {"source": "x"}})
    rag.delete({"source": "x"})
    rag.add({"text": "unique line", "metadata": {"source": "y"}})
    assert len(rag) == 1
    assert rag.retrieve("unique", top_k=1)[0].metadata["source"] == "y"


def test_update_replaces_matching_documents():
    rag = RAG(documents=_docs())
    rag.update({"text": "Bananas are now green.", "metadata": {"source": "b"}}, where={"source": "b"})
    assert len(rag) == 3
    hits = rag.retrieve("bananas", top_k=1)
    assert "green" in hits[0].text


def test_safe_save_load_roundtrip(tmp_path):
    rag = RAG(documents=_docs(), use_hybrid=True)
    p = tmp_path / "index.swiftrag"
    rag.save(p)  # default format is "safe"
    assert is_safe_file(p)  # zip magic, not pickle

    loaded = RAG.load(p)
    assert len(loaded) == 3
    assert loaded.use_hybrid is True
    assert loaded.retrieve("carrots", top_k=1)[0].metadata["source"] == "c"


def test_safe_format_contains_no_pickle(tmp_path):
    rag = RAG(documents="content")
    p = tmp_path / "idx.zip"
    rag.save(p, format="safe")
    with open(p, "rb") as f:
        head = f.read(2)
    assert head == b"PK"  # zip, never pickle's b"\x80"


def test_pickle_format_still_supported(tmp_path):
    rag = RAG(documents=_docs())
    p = tmp_path / "legacy.pkl"
    rag.save(p)  # .pkl -> pickle automatically
    assert not is_safe_file(p)
    loaded = RAG.load(p)
    assert len(loaded) == 3


def test_unknown_save_format_raises(tmp_path):
    rag = RAG(documents="content")
    with pytest.raises(ConfigurationError):
        rag.save(tmp_path / "x.bin", format="bogus")
