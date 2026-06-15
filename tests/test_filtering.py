from swiftrag import RAG


def _rag():
    return RAG(
        documents=[
            {"text": "Cats are small domesticated feline animals.", "metadata": {"topic": "cats"}},
            {"text": "Dogs are loyal canine companions.", "metadata": {"topic": "dogs"}},
            {"text": "Falcons are fast birds of prey.", "metadata": {"topic": "birds"}},
        ],
        top_k=3,
    )


def test_where_dict_filters_by_metadata():
    rag = _rag()
    results = rag.retrieve("animal", where={"topic": "dogs"})
    assert results
    assert all(r.metadata["topic"] == "dogs" for r in results)


def test_where_callable_filters():
    rag = _rag()
    results = rag.retrieve("animal", where=lambda c: c.metadata.get("topic") in {"cats", "birds"})
    assert {r.metadata["topic"] for r in results} <= {"cats", "birds"}


def test_min_score_drops_low_matches():
    rag = _rag()
    high = rag.retrieve("loyal canine dogs", top_k=3, min_score=0.99)
    # With a very high threshold few/no chunks survive.
    assert all(r.score >= 0.99 for r in high)


def test_min_score_filters_everything_when_impossible():
    rag = _rag()
    assert rag.retrieve("dogs", top_k=3, min_score=2.0) == []


def test_query_cache_reuses_embedding():
    rag = _rag()
    rag.retrieve("dogs")
    assert "dogs" in rag._query_cache
    cached = rag._query_cache["dogs"]
    rag.retrieve("dogs")
    assert rag._query_cache["dogs"] is cached


def test_query_cache_cleared_on_add():
    rag = _rag()
    rag.retrieve("dogs")
    assert rag._query_cache
    rag.add("New document about hamsters.")
    assert not rag._query_cache
