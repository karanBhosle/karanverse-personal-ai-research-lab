import json
import re

from pydantic import BaseModel, ValidationError

from app.llm.exceptions import LLMStructuredOutputError
from app.llm.types import LLMMessage, LLMRequest

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL | re.IGNORECASE)


def build_structured_request(request: LLMRequest, response_model: type[BaseModel]) -> LLMRequest:
    schema_json = json.dumps(response_model.model_json_schema(), indent=2)
    system_prompt = (
        "You are a JSON API. Respond with a single JSON object only—no markdown, no code fences, "
        "no explanation before or after the JSON. The object must validate against this JSON Schema:\n"
        f"{schema_json}"
    )
    messages = [LLMMessage(role="system", content=system_prompt), *request.messages]
    return request.model_copy(update={"messages": messages, "require_json_object": True})


def extract_json_text(content: str) -> str:
    """Best-effort extraction when models prepend prose or wrap JSON in fences."""
    text = content.strip()
    if not text:
        raise ValueError("empty response")

    fence = _JSON_BLOCK_RE.search(text)
    if fence:
        return fence.group(1).strip()

    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in response")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError("unbalanced JSON object in response")


def parse_structured_output(content: str, response_model: type[BaseModel]) -> BaseModel:
    try:
        json_text = extract_json_text(content)
    except ValueError as exc:
        raise LLMStructuredOutputError(f"Failed to parse structured output: {exc}") from exc
    try:
        return response_model.model_validate_json(json_text)
    except (ValidationError, ValueError) as exc:
        raise LLMStructuredOutputError(f"Failed to parse structured output: {exc}") from exc
