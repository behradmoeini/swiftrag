import pytest

from swiftrag import RAG, EmptyCorpusError, RAGResponse

KNOWLEDGE = """
The Eiffel Tower is located in Paris, France. It is 330 metres tall.
The Great Wall of China is over 21,000 kilometres long.
Mount Everest is the tallest mountain on Earth at 8,849 metres.
"""


def test_offline_pipeline_runs_without_keys():
    rag = RAG(documents=KNOWLEDGE)  # hash embeddings + echo LLM
    resp = rag.query("How tall is the Eiffel Tower?")
    assert isinstance(resp, RAGResponse)
    assert resp.sources
    assert "330" in resp.answer or "Eiffel" in resp.answer


def test_retrieve_orders_by_relevance():
    rag = RAG(documents=KNOWLEDGE, top_k=1)
    sources = rag.retrieve("How long is the Great Wall of China?")
    assert "Great Wall" in sources[0].text


def test_custom_callable_llm():
    rag = RAG(documents=KNOWLEDGE, llm_model=lambda prompt: "ANSWER")
    assert rag.query("anything").answer == "ANSWER"


def test_list_and_dict_documents():
    rag = RAG(
        documents=[
            "Plain string document about cats.",
            {"text": "Dict document about dogs.", "metadata": {"src": "kb"}},
        ]
    )
    assert len(rag) >= 2
    sources = rag.retrieve("dogs", top_k=1)
    assert sources[0].metadata.get("src") == "kb"


def test_empty_corpus_raises():
    rag = RAG()
    with pytest.raises(EmptyCorpusError):
        rag.query("hello")


def test_add_is_chainable():
    rag = RAG()
    rag.add("first").add("second")
    assert len(rag) >= 2


def test_save_and_load(tmp_path):
    rag = RAG(documents=KNOWLEDGE)
    path = tmp_path / "index.pkl"
    rag.save(path)
    loaded = RAG.load(path)
    assert len(loaded) == len(rag)
    assert loaded.retrieve("Everest", top_k=1)[0].text
