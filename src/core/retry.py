"""Bounded retry for transient filesystem failures.

The pipeline writes results over runs lasting hours or days, usually onto
network storage. A momentary ``OSError`` -- an NFS or SMB timeout, a reconnect,
an antivirus holding a handle open on Windows -- used to be indistinguishable
from a permanent failure: the chunk was counted as failed and its candidates
were lost for good.

Only operations that are safe to repeat belong here. Writing the candidate
buffer qualifies because the buffer is not cleared until the write returns, and
so does the checkpoint because it is written to a temp file and renamed.
"""
from __future__ import annotations

import logging
import random
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: Errors worth retrying. A ValueError or a TypeError is a bug and repeating it
#: only wastes time; these are the ones that come from the storage layer.
TRANSIENT_ERRORS = (OSError, TimeoutError)


def with_retry(
    operation: Callable[[], T],
    *,
    description: str,
    attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 8.0,
) -> T:
    """Run *operation*, retrying transient failures with backoff and jitter.

    Re-raises the last error once the attempts are exhausted: a retry that keeps
    failing is a real failure and must still be visible to the caller.
    """
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except TRANSIENT_ERRORS as e:
            last_error = e
            if attempt == attempts:
                break
            # Jitter keeps several processes writing to the same share from
            # retrying in lockstep.
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay *= 0.5 + random.random()
            logger.warning(
                "%s failed (attempt %d/%d): %s. Retrying in %.1fs",
                description, attempt, attempts, e, delay,
            )
            time.sleep(delay)
    logger.error("%s failed after %d attempts: %s", description, attempts, last_error)
    raise last_error  # type: ignore[misc]
