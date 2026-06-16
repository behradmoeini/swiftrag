"""Retry helper with exponential backoff and jitter.

Network calls to model providers fail transiently (timeouts, rate limits, 5xx).
A small, dependency-free retry wrapper makes the pipeline far more robust in
production without callers having to think about it.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger("swiftrag")

T = TypeVar("T")

#: Default number of retries (i.e. up to N+1 total attempts) for provider calls.
DEFAULT_MAX_RETRIES = 2


def retry_call(
    fn: Callable[[], T],
    *,
    retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    sleep: Callable[[float], None] = time.sleep,
    rng: Callable[[], float] = random.random,
) -> T:
    """Call ``fn`` and retry on failure with exponential backoff + jitter.

    Args:
        fn: Zero-arg callable to invoke.
        retries: Number of retries after the first attempt. ``0`` disables retrying.
        base_delay: Base backoff in seconds; doubles each attempt.
        max_delay: Upper bound on the (pre-jitter) backoff.
        exceptions: Exception types that trigger a retry.
        sleep: Injectable sleep (for tests).
        rng: Injectable random in ``[0, 1)`` for jitter (for tests).
    """
    attempt = 0
    while True:
        try:
            return fn()
        except exceptions as exc:
            if attempt >= retries:
                raise
            delay = min(max_delay, base_delay * (2**attempt)) + rng() * base_delay
            logger.warning(
                "swiftrag: call failed (%s); retrying %d/%d in %.2fs",
                exc,
                attempt + 1,
                retries,
                delay,
            )
            sleep(delay)
            attempt += 1


__all__ = ["DEFAULT_MAX_RETRIES", "retry_call"]
