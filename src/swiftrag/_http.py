"""Minimal stdlib HTTP helpers.

Used by the dependency-free Ollama backend so a fully local embedding + LLM
stack works without pulling in ``requests`` or a vendor SDK.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Iterator
from typing import Any


def _request(url: str, payload: dict[str, Any], timeout: float, headers: dict[str, str] | None):
    body = json.dumps(payload).encode("utf-8")
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    return urllib.request.Request(url, data=body, headers=hdrs, method="POST")


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: float = 60.0,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """POST ``payload`` as JSON and parse the JSON response."""
    req = _request(url, payload, timeout, headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (configured host)
        return json.loads(resp.read().decode("utf-8"))


def stream_json_lines(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: float = 120.0,
    headers: dict[str, str] | None = None,
) -> Iterator[dict[str, Any]]:
    """POST ``payload`` and yield one parsed object per NDJSON response line."""
    req = _request(url, payload, timeout, headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (configured host)
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if line:
                yield json.loads(line)


__all__ = ["post_json", "stream_json_lines"]
