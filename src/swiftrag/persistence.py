"""Safe, pickle-free persistence for a swiftrag index.

An index is saved as a single zip file containing ``meta.json`` (chunks +
config) and ``matrix.npy`` (the embedding matrix). Unlike pickle, this format
contains no executable payload, so it is safe to share and load from untrusted
sources. The legacy pickle format is still supported by :meth:`RAG.load` for
backward compatibility.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from .types import Chunk

#: First bytes of a zip archive; used to sniff the safe format on load.
SAFE_MAGIC = b"PK\x03\x04"

#: Bumped when the on-disk safe format changes incompatibly.
SAFE_FORMAT_VERSION = 1


def is_safe_file(path: str | Path) -> bool:
    """Return True if ``path`` looks like a swiftrag safe (zip) index."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == SAFE_MAGIC
    except OSError:
        return False


def save_safe(
    path: str | Path,
    matrix: np.ndarray | None,
    chunks: list[Chunk],
    config: dict[str, Any],
) -> None:
    """Write a pickle-free index to ``path``."""
    meta = {
        "format": "swiftrag-safe",
        "format_version": SAFE_FORMAT_VERSION,
        "config": config,
        "chunks": [
            {
                "text": c.text,
                "doc_id": c.doc_id,
                "chunk_index": c.chunk_index,
                "metadata": c.metadata,
            }
            for c in chunks
        ],
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("meta.json", json.dumps(meta, ensure_ascii=False))
        if matrix is not None:
            buf = io.BytesIO()
            np.save(buf, np.ascontiguousarray(matrix), allow_pickle=False)
            zf.writestr("matrix.npy", buf.getvalue())


def load_safe(path: str | Path) -> dict[str, Any]:
    """Load an index written by :func:`save_safe`."""
    with zipfile.ZipFile(path, "r") as zf:
        meta = json.loads(zf.read("meta.json").decode("utf-8"))
        matrix = None
        if "matrix.npy" in zf.namelist():
            matrix = np.load(io.BytesIO(zf.read("matrix.npy")), allow_pickle=False)

    chunks = [
        Chunk(
            text=d["text"],
            doc_id=d["doc_id"],
            chunk_index=d["chunk_index"],
            metadata=d.get("metadata", {}),
        )
        for d in meta.get("chunks", [])
    ]
    return {
        "matrix": matrix,
        "chunks": chunks,
        "config": meta.get("config", {}),
        "format_version": meta.get("format_version", 1),
    }


__all__ = ["SAFE_FORMAT_VERSION", "is_safe_file", "load_safe", "save_safe"]
