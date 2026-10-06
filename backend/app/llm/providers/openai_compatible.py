import json
import logging
from collections.abc import Iterator
from typing import Any

import httpx

from app.llm.exceptions import (
    LLMProviderResponseError,
    LLMRateLimitError,
    LLMStructuredOutputError,
    LLMTimeoutError,
)
from app.llm.provider import LLMProvider
from app.llm.structured import build_structured_request, parse_structured_output
from app.llm.types import LLMMessage, LLMRequest, LLMResponse, LLMStreamChunk, LLMUsage

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI-style chat completions API (OmniRouter and similar gateways)."""

    def __init__(
        self,
        *,
        provider_name: str,
        base_url: str,
        api_key: str,
        default_model: str,
        timeout_seconds: float,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self._provider_name = provider_name
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._default_model = default_model
        self._timeout_seconds = timeout_seconds
        self._default_headers = default_headers or {}

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def default_model(self) -> str:
        return self._default_model

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self._default_headers)
        return headers

    def _payload(self, request: LLMRequest, *, stream: bool) -> dict[str, Any]:
        model = request.model or self._default_model
        payload: dict[str, Any] = {
            "model": model,
            "messages": [message.model_dump() for message in request.messages],
            "temperature": request.temperature,
            "stream": stream,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.require_json_object:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _handle_response_errors(self, response: httpx.Response) -> None:
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            retry_seconds = float(retry_after) if retry_after else None
            raise LLMRateLimitError(
                "Provider rate limit exceeded",
                retry_after_seconds=retry_seconds,
            )
        if response.status_code >= 500:
            raise LLMRateLimitError(f"Provider temporary error ({response.status_code})")
        if response.status_code >= 400:
            raise LLMProviderResponseError(
                f"Provider request failed ({response.status_code}): {response.text}",
                status_code=response.status_code,
            )

    def generate(self, request: LLMRequest) -> LLMResponse:
        url = f"{self._base_url}/chat/completions"
        payload = self._payload(request, stream=False)
        try:
            with httpx.Client(timeout=self._timeout_seconds) as client:
                response = client.post(url, headers=self._headers(), json=payload)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("LLM request timed out") from exc
        except httpx.HTTPError as exc:
            raise LLMProviderResponseError(f"HTTP error calling LLM provider: {exc}") from exc

        self._handle_response_errors(response)
        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderResponseError("Malformed provider response payload") from exc

        usage_raw = data.get("usage") or {}
        usage = LLMUsage(
            prompt_tokens=usage_raw.get("prompt_tokens"),
            completion_tokens=usage_raw.get("completion_tokens"),
            total_tokens=usage_raw.get("total_tokens"),
        )
        return LLMResponse(
            content=content or "",
            model=str(data.get("model") or payload["model"]),
            provider=self.provider_name,
            usage=usage,
            raw=data,
        )

    def generate_structured(self, request: LLMRequest, response_model: type) -> Any:
        structured_request = build_structured_request(request, response_model)
        try:
            response = self.generate(structured_request)
            return parse_structured_output(response.content, response_model)
        except LLMStructuredOutputError:
            retry_request = structured_request.model_copy(
                update={
                    "temperature": min(structured_request.temperature, 0.1),
                    "messages": [
                        *structured_request.messages,
                        LLMMessage(
                            role="user",
                            content=(
                                "Your previous reply was not valid JSON. "
                                "Return ONLY one JSON object matching the schema. No other text."
                            ),
                        ),
                    ],
                }
            )
            response = self.generate(retry_request)
            return parse_structured_output(response.content, response_model)

    def stream_generate(self, request: LLMRequest) -> Iterator[LLMStreamChunk]:
        url = f"{self._base_url}/chat/completions"
        payload = self._payload(request, stream=True)
        try:
            with httpx.Client(timeout=self._timeout_seconds) as client:
                with client.stream("POST", url, headers=self._headers(), json=payload) as response:
                    self._handle_response_errors(response)
                    for line in response.iter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line.removeprefix("data: ").strip()
                        if data_str == "[DONE]":
                            yield LLMStreamChunk(content_delta="", done=True)
                            break
                        try:
                            event = json.loads(data_str)
                            delta = event["choices"][0]["delta"].get("content") or ""
                        except (KeyError, IndexError, json.JSONDecodeError, TypeError):
                            logger.debug("Skipping unparseable stream chunk: %s", data_str)
                            continue
                        if delta:
                            yield LLMStreamChunk(content_delta=delta, done=False)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("LLM streaming request timed out") from exc
        except httpx.HTTPError as exc:
            raise LLMProviderResponseError(f"HTTP error streaming from provider: {exc}") from exc
