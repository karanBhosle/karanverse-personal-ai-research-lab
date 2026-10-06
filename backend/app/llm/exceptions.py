class LLMError(Exception):
    """Base LLM provider error."""


class LLMConfigurationError(LLMError):
    """Missing or invalid provider configuration."""


class LLMTimeoutError(LLMError):
    """Request exceeded configured timeout."""


class LLMRateLimitError(LLMError):
    """Provider rate limit exceeded."""

    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class LLMProviderResponseError(LLMError):
    """Non-retryable provider response error."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMStructuredOutputError(LLMError):
    """Failed to parse structured model output."""
