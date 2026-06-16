"""Conversational (multi-turn) RAG.

:class:`Conversation` wraps a :class:`~swiftrag.core.RAG` and keeps chat history
so follow-up questions work naturally. Before retrieval it rewrites a follow-up
("and how tall is it?") into a standalone query using the conversation context,
which is what makes retrieval work across turns. Rewriting needs a real LLM, so
it is skipped automatically in offline mode (the :class:`~swiftrag.llms.EchoLLM`).
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING

from .llms import EchoLLM
from .types import RAGResponse

if TYPE_CHECKING:
    from .core import RAG, Where

Message = dict[str, str]

_REWRITE_SYSTEM = (
    "You rewrite a user's follow-up question into a standalone question, using "
    "the conversation only for context (e.g. resolving pronouns and references). "
    "Output ONLY the rewritten question, with no preamble or quotes."
)


class Conversation:
    """A stateful, multi-turn view over a :class:`RAG` index."""

    def __init__(
        self,
        rag: RAG,
        *,
        rewrite_queries: bool = True,
        max_history_turns: int | None = 6,
        system_prompt: str | None = None,
    ) -> None:
        self.rag = rag
        self.rewrite_queries = rewrite_queries
        self.max_history_turns = max_history_turns
        self.system_prompt = system_prompt or rag.system_prompt
        self._history: list[Message] = []

    @property
    def history(self) -> list[Message]:
        """A copy of the conversation so far, as ``{"role", "content"}`` dicts."""
        return list(self._history)

    def reset(self) -> Conversation:
        """Clear the conversation history (keeps the underlying index)."""
        self._history.clear()
        return self

    # ------------------------------------------------------------ internals
    def _history_window(self) -> list[Message]:
        if self.max_history_turns is None:
            return list(self._history)
        return self._history[-self.max_history_turns * 2 :]

    def _format_history(self) -> str:
        return "\n".join(f"{m['role'].title()}: {m['content']}" for m in self._history_window())

    def _should_rewrite(self) -> bool:
        return bool(self.rewrite_queries and self._history) and not isinstance(self.rag.llm, EchoLLM)

    def _standalone_query(self, question: str) -> str:
        """Condense a follow-up into a standalone query (best-effort)."""
        if not self._should_rewrite():
            return question
        messages = [
            {"role": "system", "content": _REWRITE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Conversation:\n{self._format_history()}\n\n"
                    f"Follow-up question: {question}\n\nStandalone question:"
                ),
            },
        ]
        try:
            rewritten = self.rag.llm.generate(messages).strip()
        except Exception:
            return question
        return rewritten or question

    def _build_messages(self, question: str, sources) -> list[Message]:
        # Reuse RAG's context-packed user message, then splice in history and
        # the conversation's system prompt.
        base = self.rag._build_messages(question, sources)
        system = {"role": "system", "content": self.system_prompt}
        return [system, *self._history_window(), base[-1]]

    def _record(self, question: str, answer: str) -> None:
        self._history.append({"role": "user", "content": question})
        self._history.append({"role": "assistant", "content": answer})

    # --------------------------------------------------------------- public
    def ask(
        self,
        question: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
        where: Where | None = None,
        hybrid: bool | None = None,
        **llm_kwargs,
    ) -> RAGResponse:
        """Answer ``question`` in the context of the conversation so far."""
        search_query = self._standalone_query(question)
        sources = self.rag.retrieve(
            search_query, top_k=top_k, min_score=min_score, where=where, hybrid=hybrid
        )
        messages = self._build_messages(question, sources)
        answer = self.rag.llm.generate(messages, **llm_kwargs)
        self._record(question, answer)
        return RAGResponse(answer=answer, sources=sources, query=search_query)

    def stream(
        self,
        question: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
        where: Where | None = None,
        hybrid: bool | None = None,
        **llm_kwargs,
    ) -> Iterator[str]:
        """Like :meth:`ask`, but yields answer tokens. History updates at the end."""
        search_query = self._standalone_query(question)
        sources = self.rag.retrieve(
            search_query, top_k=top_k, min_score=min_score, where=where, hybrid=hybrid
        )
        messages = self._build_messages(question, sources)
        tokens: list[str] = []
        for token in self.rag.llm.stream(messages, **llm_kwargs):
            tokens.append(token)
            yield token
        self._record(question, "".join(tokens))

    async def aask(self, question: str, **kwargs) -> RAGResponse:
        """Async wrapper around :meth:`ask` (runs in a worker thread)."""
        return await asyncio.to_thread(self.ask, question, **kwargs)

    async def astream(self, question: str, **kwargs) -> AsyncIterator[str]:
        """Async streaming variant of :meth:`stream`."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        sentinel = object()

        def _worker() -> None:
            try:
                for token in self.stream(question, **kwargs):
                    loop.call_soon_threadsafe(queue.put_nowait, token)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        threading.Thread(target=_worker, daemon=True).start()
        while True:
            item = await queue.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    def __repr__(self) -> str:
        return f"Conversation(turns={len(self._history) // 2}, rag={self.rag!r})"


__all__ = ["Conversation"]
