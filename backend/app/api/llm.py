import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.config import get_settings
from app.config.settings import Settings
from app.llm.exceptions import LLMConfigurationError, LLMError
from app.llm.factory import get_llm_provider, llm_configuration_status
from app.llm.provider import LLMProvider
from app.llm.types import LLMMessage, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm", tags=["llm"])


class LLMConfigResponse(BaseModel):
    provider: str
    credentials_configured: bool
    model: str | None
    ready: bool
    detail: str | None = None


class LLMGenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    system_prompt: str | None = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)


@router.get("/config", response_model=LLMConfigResponse)
def get_llm_config(settings: Settings = Depends(get_settings)) -> LLMConfigResponse:
    status = llm_configuration_status(settings)
    return LLMConfigResponse(**status)


@router.post("/generate", response_model=LLMResponse)
def generate_text(
    body: LLMGenerateRequest,
    settings: Settings = Depends(get_settings),
    provider: LLMProvider = Depends(get_llm_provider),
) -> LLMResponse:
    if not llm_configuration_status(settings)["ready"]:
        raise HTTPException(status_code=503, detail="LLM provider is not configured")

    messages = []
    if body.system_prompt:
        messages.append(LLMMessage(role="system", content=body.system_prompt))
    messages.append(LLMMessage(role="user", content=body.prompt))

    try:
        return provider.generate(
            LLMRequest(
                messages=messages,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
            )
        )
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMError as exc:
        logger.exception("LLM generation failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
