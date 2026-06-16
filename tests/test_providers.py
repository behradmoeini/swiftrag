import pytest

from swiftrag.embeddings import OllamaEmbeddings, resolve_embeddings
from swiftrag.exceptions import ConfigurationError, DependencyError
from swiftrag.llms import OllamaLLM, resolve_llm


def test_resolve_ollama_embeddings_no_network():
    emb = resolve_embeddings("ollama:nomic-embed-text")
    assert isinstance(emb, OllamaEmbeddings)
    assert emb.model == "nomic-embed-text"


def test_resolve_ollama_llm_no_network():
    llm = resolve_llm("ollama:llama3")
    assert isinstance(llm, OllamaLLM)
    assert llm.model == "llama3"


def test_ollama_defaults_to_localhost():
    assert resolve_embeddings("ollama").host == "http://localhost:11434"


def test_ollama_host_from_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://example.test:1234/")
    assert resolve_llm("ollama").host == "http://example.test:1234"


def test_cohere_missing_dependency_raises():
    try:
        import cohere  # noqa: F401

        pytest.skip("cohere is installed; missing-dependency path not exercised")
    except ImportError:
        pass
    with pytest.raises(DependencyError):
        resolve_embeddings("cohere:embed-english-v3.0")


def test_gemini_missing_dependency_raises():
    try:
        import google.generativeai  # noqa: F401

        pytest.skip("google-generativeai is installed; missing-dependency path not exercised")
    except ImportError:
        pass
    with pytest.raises(DependencyError):
        resolve_llm("gemini:gemini-1.5-flash")
    with pytest.raises(DependencyError):
        resolve_embeddings("gemini:text-embedding-004")


def test_unknown_providers_still_raise():
    with pytest.raises(ConfigurationError):
        resolve_embeddings("bogus:model")
    with pytest.raises(ConfigurationError):
        resolve_llm("bogus:model")
