import json
import logging
from collections.abc import Iterator
from typing import Any

import httpx

from app.llm.exceptions import LLMProviderResponseError, LLMRateLimitError, LLMTimeoutError
from app.llm.provider import LLMProvider
from app.llm.structured import build_structured_request, parse_structured_output
from app.llm.types import LLMRequest, LLMResponse, LLMStreamChunk

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """Local Ollama chat API."""

    def __init__(self, *, base_url: str, default_model: str, timeout_seconds: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._timeout_seconds = timeout_seconds

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def default_model(self) -> str:
        return self._default_model

    def _payload(self, request: LLMRequest, *, stream: bool) -> dict[str, Any]:
        return {
            "model": request.model or self._default_model,
            "messages": [message.model_dump() for message in request.messages],
            "stream": stream,
            "options": {"temperature": request.temperature},
        }

    def generate(self, request: LLMRequest) -> LLMResponse:
        url = f"{self._base_url}/api/chat"
        payload = self._payload(request, stream=False)
        try:
            with httpx.Client(timeout=self._timeout_seconds) as client:
                response = client.post(url, json=payload)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("Ollama request timed out") from exc
        except httpx.HTTPError as exc:
            raise LLMProviderResponseError(f"HTTP error calling Ollama: {exc}") from exc

        if response.status_code == 429:
            raise LLMRateLimitError("Ollama rate limit exceeded")
        if response.status_code >= 400:
            raise LLMProviderResponseError(
                f"Ollama request failed ({response.status_code}): {response.text}",
                status_code=response.status_code,
            )

        data = response.json()
        content = data.get("message", {}).get("content", "")
        return LLMResponse(
            content=content,
            model=str(payload["model"]),
            provider=self.provider_name,
            raw=data,
        )

    def generate_structured(self, request: LLMRequest, response_model: type) -> Any:
        structured_request = build_structured_request(request, response_model)
        response = self.generate(structured_request)
        return parse_structured_output(response.content, response_model)

    def stream_generate(self, request: LLMRequest) -> Iterator[LLMStreamChunk]:
        url = f"{self._base_url}/api/chat"
        payload = self._payload(request, stream=True)
        try:
            with httpx.Client(timeout=self._timeout_seconds) as client:
                with client.stream("POST", url, json=payload) as response:
                    if response.status_code >= 400:
                        raise LLMProviderResponseError(
                            f"Ollama stream failed ({response.status_code})",
                            status_code=response.status_code,
                        )
                    for line in response.iter_lines():
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                            if event.get("done"):
                                yield LLMStreamChunk(content_delta="", done=True)
                                break
                            delta = event.get("message", {}).get("content") or ""
                        except json.JSONDecodeError:
                            continue
                        if delta:
                            yield LLMStreamChunk(content_delta=delta, done=False)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("Ollama streaming request timed out") from exc
        except httpx.HTTPError as exc:
            raise LLMProviderResponseError(f"HTTP error streaming from Ollama: {exc}") from exc
