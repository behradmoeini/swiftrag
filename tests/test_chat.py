from swiftrag import RAG, Conversation
from swiftrag.llms import LLMProvider


def test_chat_returns_conversation():
    rag = RAG(documents="Some content about cats.")
    chat = rag.chat()
    assert isinstance(chat, Conversation)


def test_history_accumulates_and_resets():
    rag = RAG(documents="Apples are a fruit. Carrots are a vegetable.")
    chat = rag.chat()
    chat.ask("Tell me about apples.")
    chat.ask("What about carrots?")
    assert len(chat.history) == 4  # 2 user + 2 assistant
    assert chat.history[0] == {"role": "user", "content": "Tell me about apples."}
    chat.reset()
    assert chat.history == []


def test_offline_answer_uses_retrieved_context():
    rag = RAG(documents=["Penguins are flightless birds.", "Lions live in Africa."])
    chat = rag.chat()
    resp = chat.ask("Tell me about penguins.")
    assert "Penguins" in resp.answer
    assert resp.sources


def test_offline_skips_rewrite_and_keeps_raw_query():
    # With the offline EchoLLM, rewriting is skipped, so the search query
    # is the raw follow-up (not a rewritten standalone form).
    rag = RAG(documents="Bananas are yellow.")
    chat = rag.chat()
    chat.ask("first question")
    resp = chat.ask("bananas?")
    assert resp.query == "bananas?"


class _ScriptedLLM(LLMProvider):
    """Rewrites to a fixed standalone query; otherwise returns a canned answer."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def generate(self, messages, **kwargs) -> str:
        content = messages[-1]["content"]
        self.seen.append(content)
        if "Standalone question:" in content:
            return "What is the capital of France?"
        return "Paris."


def test_followup_is_rewritten_before_retrieval():
    llm = _ScriptedLLM()
    rag = RAG(documents="France is a country in Europe.", llm_model=llm)
    chat = rag.chat()

    first = chat.ask("What's the capital of France?")
    assert first.query == "What's the capital of France?"  # no history yet -> no rewrite

    second = chat.ask("And its population?")
    # The follow-up was condensed into the standalone query used for retrieval.
    assert second.query == "What is the capital of France?"
    # But the recorded history keeps the user's original wording.
    assert chat.history[2] == {"role": "user", "content": "And its population?"}


def test_rewrite_can_be_disabled():
    llm = _ScriptedLLM()
    rag = RAG(documents="France is a country.", llm_model=llm)
    chat = rag.chat(rewrite_queries=False)
    chat.ask("What's the capital of France?")
    resp = chat.ask("And its population?")
    assert resp.query == "And its population?"  # rewriting disabled


def test_stream_records_history():
    rag = RAG(documents="Tigers are big cats.")
    chat = rag.chat()
    tokens = list(chat.stream("Tell me about tigers."))
    assert "".join(tokens)  # produced some text
    assert len(chat.history) == 2
