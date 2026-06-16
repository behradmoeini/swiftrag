"""LLM providers.

A provider turns a list of chat messages into an answer (and optionally streams
it). Built-in providers:

* ``openai``    -> OpenAI / OpenAI-compatible chat completions (needs ``openai``).
* ``anthropic`` -> Claude messages API (needs ``anthropic``).
* ``echo``      -> offline stub that returns the prompt's context. Lets the full
  pipeline run with no keys (tests/demos), and is selected automatically when no
  real model is configured.

Any callable ``fn(prompt: str) -> str`` or object with ``.generate(messages)``
is accepted directly.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, Callable

from ._retry import DEFAULT_MAX_RETRIES, retry_call
from .exceptions import ConfigurationError, DependencyError

Message = dict[str, str]


class LLMProvider(ABC):
    """Abstract chat LLM backend."""

    #: Token usage from the most recent :meth:`generate` call, when available.
    last_usage: dict[str, Any] = {}

    @abstractmethod
    def generate(self, messages: list[Message], **kwargs) -> str:
        """Return the assistant's full reply for ``messages``."""

    def stream(self, messages: list[Message], **kwargs) -> Iterator[str]:
        """Yield reply tokens. Default: emit the full reply at once."""
        yield self.generate(messages, **kwargs)


class EchoLLM(LLMProvider):
    """Offline stub. Returns an extractive answer from the provided context.

    This keeps swiftrag fully functional without any API key: retrieval still
    runs, and the "answer" is the most relevant retrieved context. Swap in a
    real provider for generative answers.
    """

    def generate(self, messages: list[Message], **kwargs) -> str:
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        marker = "Context:"
        if marker in user:
            context = user.split(marker, 1)[1]
            context = context.split("Question:", 1)[0].strip()
            if context:
                return (
                    "[swiftrag offline mode, no LLM configured] "
                    "Most relevant context:\n\n" + context
                )
        return "[swiftrag offline mode] No context retrieved."


class OpenAILLM(LLMProvider):
    """OpenAI / OpenAI-compatible chat completions."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise DependencyError("openai", "openai") from e
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.last_usage = {}
        self._client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )

    def _params(self, messages, kwargs):
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
        }
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if max_tokens:
            params["max_tokens"] = max_tokens
        return params

    def generate(self, messages: list[Message], **kwargs) -> str:
        resp = retry_call(
            lambda: self._client.chat.completions.create(**self._params(messages, kwargs)),
            retries=self.max_retries,
        )
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.last_usage = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
                "model": self.model,
            }
        return resp.choices[0].message.content or ""

    def stream(self, messages: list[Message], **kwargs) -> Iterator[str]:
        stream = retry_call(
            lambda: self._client.chat.completions.create(
                stream=True, **self._params(messages, kwargs)
            ),
            retries=self.max_retries,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class AnthropicLLM(LLMProvider):
    """Anthropic Claude messages API."""

    def __init__(
        self,
        model: str = "claude-3-5-haiku-latest",
        *,
        api_key: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise DependencyError("anthropic", "anthropic") from e
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.last_usage = {}
        self._client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    @staticmethod
    def _split(messages: list[Message]):
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        convo = [m for m in messages if m["role"] != "system"]
        return system, convo

    def generate(self, messages: list[Message], **kwargs) -> str:
        system, convo = self._split(messages)
        resp = retry_call(
            lambda: self._client.messages.create(
                model=self.model,
                system=system or None,
                messages=convo,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            ),
            retries=self.max_retries,
        )
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.last_usage = {
                "prompt_tokens": getattr(usage, "input_tokens", None),
                "completion_tokens": getattr(usage, "output_tokens", None),
                "model": self.model,
            }
        return "".join(block.text for block in resp.content if block.type == "text")

    def stream(self, messages: list[Message], **kwargs) -> Iterator[str]:
        system, convo = self._split(messages)
        with self._client.messages.stream(
            model=self.model,
            system=system or None,
            messages=convo,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        ) as stream:
            yield from stream.text_stream


class OllamaLLM(LLMProvider):
    """Chat completions via a local (or remote) Ollama server. No extra deps.

    Point at a different host with ``OLLAMA_HOST`` or the ``host`` argument.
    """

    def __init__(
        self,
        model: str = "llama3",
        *,
        host: str | None = None,
        timeout: float = 120.0,
        temperature: float = 0.2,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self.model = model
        self.host = (host or os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.max_retries = max_retries
        self.last_usage = {}

    def _payload(self, messages, kwargs, *, stream: bool):
        return {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": kwargs.get("temperature", self.temperature)},
        }

    def generate(self, messages: list[Message], **kwargs) -> str:
        from ._http import post_json

        data = retry_call(
            lambda: post_json(
                f"{self.host}/api/chat",
                self._payload(messages, kwargs, stream=False),
                timeout=self.timeout,
            ),
            retries=self.max_retries,
        )
        self.last_usage = {
            "prompt_tokens": data.get("prompt_eval_count"),
            "completion_tokens": data.get("eval_count"),
            "model": self.model,
        }
        return data.get("message", {}).get("content", "")

    def stream(self, messages: list[Message], **kwargs) -> Iterator[str]:
        from ._http import stream_json_lines

        for obj in stream_json_lines(
            f"{self.host}/api/chat",
            self._payload(messages, kwargs, stream=True),
            timeout=self.timeout,
        ):
            token = obj.get("message", {}).get("content")
            if token:
                yield token
            if obj.get("done"):
                break


class GeminiLLM(LLMProvider):
    """Google Gemini chat (needs ``google-generativeai``)."""

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        *,
        api_key: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise DependencyError("google-generativeai", "gemini") from e
        genai.configure(api_key=api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
        self._genai = genai
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.last_usage = {}

    def _prepare(self, messages: list[Message], kwargs):
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        prompt = "\n\n".join(m["content"] for m in messages if m["role"] != "system")
        config: dict = {"temperature": kwargs.get("temperature", self.temperature)}
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if max_tokens:
            config["max_output_tokens"] = max_tokens
        model = self._genai.GenerativeModel(self.model, system_instruction=system or None)
        return model, prompt, config

    def generate(self, messages: list[Message], **kwargs) -> str:
        model, prompt, config = self._prepare(messages, kwargs)
        resp = retry_call(
            lambda: model.generate_content(prompt, generation_config=config),
            retries=self.max_retries,
        )
        usage = getattr(resp, "usage_metadata", None)
        if usage is not None:
            self.last_usage = {
                "prompt_tokens": getattr(usage, "prompt_token_count", None),
                "completion_tokens": getattr(usage, "candidates_token_count", None),
                "total_tokens": getattr(usage, "total_token_count", None),
                "model": self.model,
            }
        return resp.text or ""

    def stream(self, messages: list[Message], **kwargs) -> Iterator[str]:
        model, prompt, config = self._prepare(messages, kwargs)
        for chunk in model.generate_content(prompt, generation_config=config, stream=True):
            if getattr(chunk, "text", None):
                yield chunk.text


class CallableLLM(LLMProvider):
    """Wrap a plain ``fn(prompt: str) -> str`` callable as a provider."""

    def __init__(self, fn: Callable[[str], str]) -> None:
        self._fn = fn

    def generate(self, messages: list[Message], **kwargs) -> str:
        prompt = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        return self._fn(prompt)


def resolve_llm(spec, **kwargs) -> LLMProvider:
    """Build an :class:`LLMProvider` from a spec.

    ``spec`` may be ``None`` (-> offline :class:`EchoLLM`), an
    :class:`LLMProvider`, a callable, an object with ``.generate``, or a string
    ``"provider:model"`` such as ``"openai:gpt-4o-mini"`` or
    ``"anthropic:claude-3-5-sonnet-latest"``.
    """
    if spec is None:
        return EchoLLM()
    if isinstance(spec, LLMProvider):
        return spec
    if callable(spec) and not isinstance(spec, str):
        return CallableLLM(spec)
    if hasattr(spec, "generate"):
        return spec
    if not isinstance(spec, str):
        raise ConfigurationError(f"Unsupported LLM spec: {spec!r}")

    provider, _, model = spec.partition(":")
    provider = provider.strip().lower()
    model = model.strip()

    if provider in ("openai", "azure", "oai"):
        return OpenAILLM(model or "gpt-4o-mini", **kwargs)
    if provider in ("anthropic", "claude"):
        return AnthropicLLM(model or "claude-3-5-haiku-latest", **kwargs)
    if provider in ("ollama",):
        return OllamaLLM(model or "llama3", **kwargs)
    if provider in ("gemini", "google", "googleai"):
        return GeminiLLM(model or "gemini-1.5-flash", **kwargs)
    if provider in ("echo", "offline", "none"):
        return EchoLLM()
    raise ConfigurationError(
        f"Unknown LLM provider '{provider}'. Use one of: openai, anthropic, ollama, "
        "gemini, echo, or pass a custom provider/callable."
    )


__all__ = [
    "LLMProvider",
    "EchoLLM",
    "OpenAILLM",
    "AnthropicLLM",
    "OllamaLLM",
    "GeminiLLM",
    "CallableLLM",
    "resolve_llm",
]
