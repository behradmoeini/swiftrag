import pytest

from swiftrag import RAG, loaders
from swiftrag.exceptions import DependencyError


def test_html_to_text_strips_tags_and_scripts():
    html = (
        "<html><head><style>.x{color:red}</style></head>"
        "<body><h1>Title</h1><p>Hello <b>world</b>.</p>"
        "<script>var secret = 1;</script></body></html>"
    )
    text = loaders.html_to_text(html)
    assert "Title" in text
    assert "Hello" in text and "world" in text
    assert "secret" not in text  # script contents removed


def test_html_to_text_unescapes_entities():
    assert "Tom & Jerry" in loaders.html_to_text("<p>Tom &amp; Jerry</p>")


def test_extension_registry_covers_rich_types():
    for ext in (".pdf", ".docx", ".html", ".htm"):
        assert ext in loaders.EXTENSION_LOADERS


def test_load_file_dispatches_plain_text(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("plain text body", encoding="utf-8")
    assert loaders.load_file(f) == "plain text body"


def test_load_file_dispatches_html(tmp_path):
    f = tmp_path / "page.html"
    f.write_text("<p>Penguins are flightless birds.</p>", encoding="utf-8")
    out = loaders.load_file(f)
    assert "Penguins" in out and "<p>" not in out


def test_from_files_indexes_html(tmp_path):
    (tmp_path / "page.html").write_text(
        "<html><body><h1>Birds</h1><p>Penguins are flightless birds.</p></body></html>",
        encoding="utf-8",
    )
    (tmp_path / "note.txt").write_text("Cats are mammals.", encoding="utf-8")

    rag = RAG.from_files(tmp_path)
    assert len(rag) >= 2
    sources = rag.retrieve("penguins", top_k=1)
    assert sources[0].metadata["source"].endswith("page.html")


def test_load_pdf_without_pypdf_raises_dependency_error(tmp_path):
    try:
        import pypdf  # noqa: F401

        pytest.skip("pypdf is installed; missing-dependency path not exercised")
    except ImportError:
        pass

    f = tmp_path / "doc.pdf"
    f.write_bytes(b"%PDF-1.4 not a real pdf")
    with pytest.raises(DependencyError):
        loaders.load_pdf(f)
