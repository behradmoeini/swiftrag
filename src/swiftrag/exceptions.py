"""Exception hierarchy for swiftrag."""

from __future__ import annotations


class SwiftRagError(Exception):
    """Base class for all swiftrag errors."""


class ConfigurationError(SwiftRagError):
    """Raised when the RAG pipeline is misconfigured (bad provider string, etc.)."""


class DependencyError(SwiftRagError):
    """Raised when an optional dependency is required but not installed."""

    def __init__(self, package: str, extra: str) -> None:
        super().__init__(
            f"This feature needs the '{package}' package. "
            f"Install it with:  pip install 'swiftrag[{extra}]'"
        )
        self.package = package
        self.extra = extra


class EmptyCorpusError(SwiftRagError):
    """Raised when a query is issued against an empty index."""
