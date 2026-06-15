from swiftrag import RAG
from swiftrag.chunking import count_tokens


def test_count_tokens_nonzero():
    assert count_tokens("hello world") > 0
    assert count_tokens("") == 0


def test_dedup_skips_identical_chunks():
    rag = RAG()
    rag.add("Repeated sentence here.")
    rag.add("Repeated sentence here.")  # identical -> skipped
    assert len(rag) == 1


def test_dedup_can_be_disabled():
    rag = RAG(dedup=False)
    rag.add("Repeated sentence here.")
    rag.add("Repeated sentence here.")
    assert len(rag) == 2


def test_clear_resets_index():
    rag = RAG(documents="Some content to index here.")
    assert len(rag) > 0
    rag.clear()
    assert len(rag) == 0
    assert not rag._seen_hashes
    rag.add("After clear we can add again.")
    assert len(rag) == 1


def test_max_context_tokens_limits_sources_in_prompt():
    docs = [f"Document number {i} discussing distinct topic {i}." for i in range(10)]
    rag = RAG(documents=docs, top_k=10, max_context_tokens=20)
    sources = rag.retrieve("topic", top_k=10)
    fitted = rag._fit_to_budget(sources)
    assert 0 < len(fitted) < len(sources)


def test_budget_keeps_at_least_one_source():
    rag = RAG(documents="A very long document " * 200, top_k=5, max_context_tokens=1)
    sources = rag.retrieve("document", top_k=5)
    assert len(rag._fit_to_budget(sources)) >= 1
