"""Optional document loaders: PDF, DOCX, HTML, and web URLs.

The core library reads plain text only. These loaders pull text out of richer
formats and the web. Heavy parsers are imported lazily and kept behind the
``loaders`` extra so the numpy-only core stays tiny:

    pip install 'swiftrag[loaders]'

HTML and URL loading work with no extra dependencies (a stdlib fallback strips
tags); installing ``beautifulsoup4`` simply improves extraction quality. PDF
needs ``pypdf`` and DOCX needs ``python-docx``.
"""

from __future__ import annotations

import html as _html_std
import importlib
import io
import re
import urllib.request
from collections.abc import Callable
from pathlib import Path

from .exceptions import DependencyError

#: Sent on URL requests; some servers reject the default urllib agent.
_USER_AGENT = "swiftrag/0.1 (+https://github.com/behradmoeini/swiftrag)"

_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style|noscript|template|head)\b[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_BR_RE = re.compile(r"(?i)<br\s*/?>")
_BLOCK_END_RE = re.compile(r"(?i)</(p|div|h[1-6]|li|tr|section|article|header|footer)>")
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RUNS_RE = re.compile(r"[ \t]+")


def _require(module_name: str, package: str, extra: str):
    """Import an optional module or raise a friendly :class:`DependencyError`."""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise DependencyError(package, extra) from exc


def _collapse_whitespace(text: str) -> str:
    lines = [_WS_RUNS_RE.sub(" ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def _html_to_text_fallback(markup: str) -> str:
    """Dependency-free HTML-to-text (used when beautifulsoup4 isn't installed)."""
    markup = _SCRIPT_STYLE_RE.sub(" ", markup)
    markup = _BR_RE.sub("\n", markup)
    markup = _BLOCK_END_RE.sub("\n", markup)
    text = _TAG_RE.sub(" ", markup)
    return _collapse_whitespace(_html_std.unescape(text))


def html_to_text(markup: str) -> str:
    """Extract readable text from an HTML string.

    Uses ``beautifulsoup4`` when available for robust parsing, otherwise falls
    back to a fast stdlib-only tag stripper.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return _html_to_text_fallback(markup)
    soup = BeautifulSoup(markup, "html.parser")
    for tag in soup(["script", "style", "noscript", "template", "head"]):
        tag.decompose()
    return _collapse_whitespace(soup.get_text(separator="\n"))


def _pdf_bytes_to_text(data: bytes) -> str:
    pypdf = _require("pypdf", "pypdf", "loaders")
    reader = pypdf.PdfReader(io.BytesIO(data))
    parts = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text)
    return "\n\n".join(parts)


def load_pdf(path: str | Path) -> str:
    """Extract text from a PDF file (requires ``pypdf``)."""
    return _pdf_bytes_to_text(Path(path).read_bytes())


def load_docx(path: str | Path) -> str:
    """Extract text from a ``.docx`` file (requires ``python-docx``)."""
    docx = _require("docx", "python-docx", "loaders")
    document = docx.Document(str(path))
    parts = [p.text for p in document.paragraphs if p.text.strip()]
    for table in getattr(document, "tables", []):
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n\n".join(parts)


def load_html(path: str | Path, *, encoding: str = "utf-8") -> str:
    """Read an HTML file from disk and extract its text."""
    return html_to_text(Path(path).read_text(encoding=encoding, errors="replace"))


def _fetch(url: str, *, timeout: float, user_agent: str) -> tuple[bytes, str]:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (caller-supplied URL)
        content_type = resp.headers.get("Content-Type", "")
        return resp.read(), content_type


def load_url(
    url: str,
    *,
    timeout: float = 30.0,
    user_agent: str = _USER_AGENT,
) -> str:
    """Fetch ``url`` and extract readable text.

    PDFs (by content type or ``.pdf`` suffix) are parsed with ``pypdf``;
    everything else is treated as HTML.
    """
    data, content_type = _fetch(url, timeout=timeout, user_agent=user_agent)
    if "application/pdf" in content_type.lower() or url.lower().split("?")[0].endswith(".pdf"):
        return _pdf_bytes_to_text(data)
    return html_to_text(data.decode("utf-8", errors="replace"))


#: Maps a file suffix to the loader that handles it.
EXTENSION_LOADERS: dict[str, Callable[[Path], str]] = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".html": load_html,
    ".htm": load_html,
}


def load_file(path: str | Path, *, encoding: str = "utf-8") -> str:
    """Read one file, dispatching to a rich loader by extension when needed.

    Falls back to reading the file as plain text for unknown extensions.
    """
    p = Path(path)
    loader = EXTENSION_LOADERS.get(p.suffix.lower())
    if loader is not None:
        return loader(p)
    return p.read_text(encoding=encoding)


__all__ = [
    "EXTENSION_LOADERS",
    "html_to_text",
    "load_docx",
    "load_file",
    "load_html",
    "load_pdf",
    "load_url",
]
