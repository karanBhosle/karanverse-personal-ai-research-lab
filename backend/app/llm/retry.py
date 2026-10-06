import logging
import random
import time
from collections.abc import Callable

from app.llm.exceptions import LLMError, LLMRateLimitError, LLMTimeoutError

logger = logging.getLogger(__name__)


def execute_with_retry(
    operation: Callable[[], object],
    *,
    max_retries: int,
    base_backoff_seconds: float,
    retry_on: tuple[type[Exception], ...] = (LLMRateLimitError, LLMTimeoutError),
) -> object:
    attempt = 0
    while True:
        try:
            return operation()
        except retry_on as exc:
            if attempt >= max_retries:
                raise
            sleep_seconds = base_backoff_seconds * (2**attempt)
            if isinstance(exc, LLMRateLimitError) and exc.retry_after_seconds is not None:
                sleep_seconds = max(sleep_seconds, exc.retry_after_seconds)
            sleep_seconds += random.uniform(0, 0.25)
            logger.warning(
                "LLM call failed (%s); retrying in %.2fs (attempt %d/%d)",
                exc,
                sleep_seconds,
                attempt + 1,
                max_retries,
            )
            time.sleep(sleep_seconds)
            attempt += 1
        except LLMError:
            raise
