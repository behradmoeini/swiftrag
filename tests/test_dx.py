import pytest

from swiftrag import RAG, ConfigurationError


def test_invalid_chunk_size():
    with pytest.raises(ConfigurationError):
        RAG(chunk_size=0)


def test_invalid_top_k():
    with pytest.raises(ConfigurationError):
        RAG(top_k=0)


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ConfigurationError):
        RAG(chunk_size=100, chunk_overlap=100)


def test_from_files_reads_directory(tmp_path):
    (tmp_path / "a.txt").write_text("Apples are a kind of fruit.", encoding="utf-8")
    (tmp_path / "b.md").write_text("Bananas are yellow fruit.", encoding="utf-8")
    (tmp_path / "ignore.bin").write_bytes(b"\x00\x01\x02")

    rag = RAG.from_files(tmp_path)
    assert len(rag) == 2
    sources = rag.retrieve("banana", top_k=1)
    assert sources[0].metadata["source"].endswith("b.md")


def test_from_files_single_file(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("A single note about cats.", encoding="utf-8")
    rag = RAG.from_files(f)
    assert len(rag) == 1


def test_from_files_missing_path():
    with pytest.raises(FileNotFoundError):
        RAG.from_files("does/not/exist.txt")


def test_format_sources():
    rag = RAG(documents={"text": "Tigers are large cats.", "metadata": {"source": "zoo"}})
    resp = rag.query("tigers")
    formatted = resp.format_sources()
    assert "[1]" in formatted
    assert "zoo" in formatted


def test_save_load_roundtrip_version(tmp_path):
    rag = RAG(documents="Persisted content for round trip.")
    p = tmp_path / "idx.pkl"
    rag.save(p)
    loaded = RAG.load(p)
    assert len(loaded) == len(rag)
