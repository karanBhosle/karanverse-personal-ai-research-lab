import json
import logging
from collections.abc import Iterator
from typing import Any

import httpx

from app.llm.exceptions import LLMProviderResponseError, LLMRateLimitError, LLMTimeoutError
from app.llm.provider import LLMProvider
from app.llm.structured import build_structured_request, parse_structured_output
from app.llm.types import LLMMessage, LLMRequest, LLMResponse, LLMStreamChunk

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """Google Gemini REST API provider."""

    def __init__(
        self,
        *,
        api_key: str,
        default_model: str,
        timeout_seconds: float,
    ) -> None:
        self._api_key = api_key
        self._default_model = default_model
        self._timeout_seconds = timeout_seconds

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def default_model(self) -> str:
        return self._default_model

    def _to_gemini_contents(self, messages: list[LLMMessage]) -> tuple[str | None, list[dict[str, Any]]]:
        system_instruction = None
        contents: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "system":
                system_instruction = message.content
                continue
            role = "user" if message.role == "user" else "model"
            contents.append({"role": role, "parts": [{"text": message.content}]})
        return system_instruction, contents

    def _endpoint(self, model: str, stream: bool) -> str:
        action = "streamGenerateContent" if stream else "generateContent"
        return (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:{action}?key={self._api_key}"
        )

    def _handle_response_errors(self, response: httpx.Response) -> None:
        if response.status_code == 429:
            raise LLMRateLimitError("Gemini rate limit exceeded")
        if response.status_code >= 500:
            raise LLMRateLimitError(f"Gemini temporary error ({response.status_code})")
        if response.status_code >= 400:
            raise LLMProviderResponseError(
                f"Gemini request failed ({response.status_code}): {response.text}",
                status_code=response.status_code,
            )

    def generate(self, request: LLMRequest) -> LLMResponse:
        model = request.model or self._default_model
        system_instruction, contents = self._to_gemini_contents(request.messages)
        payload: dict[str, Any] = {"contents": contents}
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        generation_config: dict[str, Any] = {"temperature": request.temperature}
        if request.max_tokens is not None:
            generation_config["maxOutputTokens"] = request.max_tokens
        payload["generationConfig"] = generation_config

        try:
            with httpx.Client(timeout=self._timeout_seconds) as client:
                response = client.post(self._endpoint(model, stream=False), json=payload)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("Gemini request timed out") from exc
        except httpx.HTTPError as exc:
            raise LLMProviderResponseError(f"HTTP error calling Gemini: {exc}") from exc

        self._handle_response_errors(response)
        data = response.json()
        try:
            content = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderResponseError("Malformed Gemini response payload") from exc

        return LLMResponse(content=content or "", model=model, provider=self.provider_name, raw=data)

    def generate_structured(self, request: LLMRequest, response_model: type) -> Any:
        structured_request = build_structured_request(request, response_model)
        response = self.generate(structured_request)
        return parse_structured_output(response.content, response_model)

    def stream_generate(self, request: LLMRequest) -> Iterator[LLMStreamChunk]:
        model = request.model or self._default_model
        system_instruction, contents = self._to_gemini_contents(request.messages)
        payload: dict[str, Any] = {"contents": contents}
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        payload["generationConfig"] = {"temperature": request.temperature}

        try:
            with httpx.Client(timeout=self._timeout_seconds) as client:
                with client.stream("POST", self._endpoint(model, stream=True), json=payload) as response:
                    self._handle_response_errors(response)
                    for line in response.iter_lines():
                        if not line:
                            continue
                        try:
                            event = json.loads(line.decode() if isinstance(line, bytes) else line)
                            text = event["candidates"][0]["content"]["parts"][0]["text"]
                        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                            continue
                        if text:
                            yield LLMStreamChunk(content_delta=text, done=False)
            yield LLMStreamChunk(content_delta="", done=True)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("Gemini streaming request timed out") from exc
        except httpx.HTTPError as exc:
            raise LLMProviderResponseError(f"HTTP error streaming from Gemini: {exc}") from exc
