import asyncio

from swiftrag import RAG, RAGResponse

KB = """
Paris is the capital of France.
Tokyo is the capital of Japan.
Cairo is the capital of Egypt.
"""


def test_query_many_returns_one_response_each():
    rag = RAG(documents=KB, llm_model=lambda p: "ok")
    responses = rag.query_many(
        ["capital of France", "capital of Japan", "capital of Egypt"]
    )
    assert len(responses) == 3
    assert all(isinstance(r, RAGResponse) for r in responses)
    assert all(r.sources for r in responses)


def test_query_many_empty():
    rag = RAG(documents=KB)
    assert rag.query_many([]) == []


def test_retrieve_many_matches_individual_retrieve():
    rag = RAG(documents=KB, top_k=1)
    batch = rag.retrieve_many(["France", "Japan"], top_k=1)
    assert len(batch) == 2
    assert "France" in batch[0][0].text
    assert "Japan" in batch[1][0].text


def test_aquery_runs():
    rag = RAG(documents=KB, llm_model=lambda p: "async-answer")
    resp = asyncio.run(rag.aquery("capital of France"))
    assert resp.answer == "async-answer"
    assert resp.sources


def test_astream_yields_tokens():
    rag = RAG(documents=KB, llm_model=lambda p: "hello world")

    async def collect():
        return [tok async for tok in rag.astream("capital of France")]

    tokens = asyncio.run(collect())
    assert "".join(tokens) == "hello world"
