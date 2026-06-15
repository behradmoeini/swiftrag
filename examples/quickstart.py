"""swiftrag quickstart.

Run offline (no API key needed):
    python examples/quickstart.py

Run with a real model:
    pip install "swiftrag[openai]"
    set OPENAI_API_KEY=sk-...        # Windows
    export OPENAI_API_KEY=sk-...     # macOS/Linux
    Then uncomment the "real model" block below.
"""

from swiftrag import RAG

knowledge = """
swiftrag is a tiny Python library for Retrieval-Augmented Generation.
You give it your documents as a string and a model, and it builds a RAG LLM.
It supports OpenAI, Anthropic, and local sentence-transformers models.
The core has only one dependency: numpy.
"""

# --- Offline mode (default): hashing embeddings + extractive answer -----------
rag = RAG(documents=knowledge)
print(rag.query("What dependency does the swiftrag core have?"))
print("-" * 60)

# --- Real model (uncomment after installing extras + setting your key) --------
# rag = RAG(
#     documents=knowledge,
#     embedding_model="openai:text-embedding-3-small",
#     llm_model="openai:gpt-4o-mini",
# )
# response = rag.query("Which model providers does swiftrag support?")
# print(response.answer)
# for src in response.sources:
#     print(f"  source (score={src.score:.3f}): {src.text[:60]}...")
#
# # Streaming:
# for token in rag.stream("Summarize swiftrag in one sentence."):
#     print(token, end="", flush=True)
