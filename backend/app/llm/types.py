from collections.abc import Iterator
from typing import Any, Literal

from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMRequestObservability(BaseModel):
    agent_name: str | None = None
    research_id: str | None = None
    source: str | None = None
    retrieval_method: str | None = None


class LLMRequest(BaseModel):
    messages: list[LLMMessage]
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    model: str | None = None
    observability: LLMRequestObservability | None = None
    require_json_object: bool = Field(
        default=False,
        description="Request OpenAI-compatible JSON object response_format when supported.",
    )


class LLMUsage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class LLMResponse(BaseModel):
    content: str
    model: str
    provider: str
    usage: LLMUsage | None = None
    raw: dict[str, Any] | None = None


class LLMStreamChunk(BaseModel):
    content_delta: str
    done: bool = False
