import numpy as np

from swiftrag import RAG
from swiftrag.lexical import BM25, tokenize


def test_bm25_ranks_exact_term_first():
    bm = BM25(["the cat sat on the mat", "a dog ran in the park"])
    scores = bm.scores("cat")
    assert scores[0] > scores[1]
    assert scores[0] > 0


def test_bm25_unknown_term_scores_zero():
    bm = BM25(["alpha beta", "gamma delta"])
    assert np.all(bm.scores("zzzznotpresent") == 0)


def test_tokenize_lowercases_and_splits():
    assert tokenize("Hello, World! 123") == ["hello", "world", "123"]


def test_hybrid_finds_rare_exact_term():
    docs = [
        "The quarterly revenue report covers sales performance.",
        "Our internal project is code-named Zorblax42 and ships in Q3.",
        "Customer support handles refunds and returns.",
    ]
    rag = RAG(documents=docs, use_hybrid=True, top_k=1)
    hits = rag.retrieve("Zorblax42")
    assert hits[0].text == docs[1]


def test_hybrid_can_be_overridden_per_call():
    rag = RAG(documents=["alpha content", "beta content"], use_hybrid=False, top_k=1)
    hits = rag.retrieve("alpha", hybrid=True)
    assert hits[0].text == "alpha content"


def test_hybrid_respects_metadata_filter():
    rag = RAG(
        documents=[
            {"text": "red apples grow on tall trees", "metadata": {"src": "a"}},
            {"text": "green apples are tasty fruit", "metadata": {"src": "b"}},
        ],
        use_hybrid=True,
        top_k=5,
    )
    hits = rag.retrieve("apples", where={"src": "b"})
    assert len(hits) == 1
    assert hits[0].metadata["src"] == "b"


def test_reranker_hook_reorders_results():
    # Reranker that reverses whatever retrieval returned.
    rag = RAG(
        documents=["first doc about cats", "second doc about cats", "third doc about cats"],
        reranker=lambda q, results: list(reversed(results)),
        top_k=3,
    )
    baseline = RAG(
        documents=["first doc about cats", "second doc about cats", "third doc about cats"],
        top_k=3,
    )
    reranked = rag.retrieve("cats")
    normal = baseline.retrieve("cats")
    assert [r.text for r in reranked] == [r.text for r in reversed(normal)]


def test_hybrid_survives_save_load(tmp_path):
    rag = RAG(documents=["alpha unique token", "beta other token"], use_hybrid=True)
    p = tmp_path / "idx.pkl"
    rag.save(p)
    loaded = RAG.load(p)
    assert loaded.use_hybrid is True
    hits = loaded.retrieve("alpha", top_k=1)
    assert hits[0].text == "alpha unique token"
