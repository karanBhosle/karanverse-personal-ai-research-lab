from functools import lru_cache

from app.config.settings import Settings, get_settings
from app.llm.exceptions import LLMConfigurationError
from app.llm.provider import LLMProvider
from app.llm.providers.gemini import GeminiProvider
from app.llm.providers.ollama import OllamaProvider
from app.llm.providers.openai_compatible import OpenAICompatibleProvider
from app.llm.observing_provider import ObservingLLMProvider
from app.llm.resilient import ResilientLLMProvider


def _openai_compatible_headers(settings: Settings, *, provider: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    if provider == "openrouter":
        app_name = settings.openrouter_app_name or settings.app_name
        site_url = settings.openrouter_site_url
        if app_name:
            headers["X-Title"] = app_name
        if site_url:
            headers["HTTP-Referer"] = site_url
        return headers

    if settings.omnirouter_app_name:
        headers["X-Title"] = settings.omnirouter_app_name
    if settings.omnirouter_app_url:
        headers["HTTP-Referer"] = settings.omnirouter_app_url
    return headers


def _build_provider(settings: Settings) -> LLMProvider:
    provider = settings.llm_provider.lower()

    if provider == "openrouter":
        if not settings.openrouter_api_key:
            raise LLMConfigurationError("OPENROUTER_API_KEY is not set")
        return OpenAICompatibleProvider(
            provider_name="openrouter",
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
            default_model=settings.openrouter_model,
            timeout_seconds=settings.llm_timeout_seconds,
            default_headers=_openai_compatible_headers(settings, provider="openrouter"),
        )

    if provider == "omnirouter":
        if not settings.omnirouter_api_key:
            raise LLMConfigurationError("OMNIROUTER_API_KEY is not set")
        if not settings.omnirouter_base_url:
            raise LLMConfigurationError("OMNIROUTER_BASE_URL is not set")
        return OpenAICompatibleProvider(
            provider_name="omnirouter",
            base_url=settings.omnirouter_base_url,
            api_key=settings.omnirouter_api_key,
            default_model=settings.omnirouter_model,
            timeout_seconds=settings.llm_timeout_seconds,
            default_headers=_openai_compatible_headers(settings, provider="omnirouter"),
        )

    if provider == "gemini":
        if not settings.gemini_api_key:
            raise LLMConfigurationError("GEMINI_API_KEY is not set")
        return GeminiProvider(
            api_key=settings.gemini_api_key,
            default_model=settings.gemini_model,
            timeout_seconds=settings.llm_timeout_seconds,
        )

    if provider == "ollama":
        return OllamaProvider(
            base_url=settings.ollama_base_url,
            default_model=settings.ollama_model,
            timeout_seconds=settings.llm_timeout_seconds,
        )

    raise LLMConfigurationError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")


def create_llm_provider(settings: Settings | None = None) -> LLMProvider:
    cfg = settings or get_settings()
    inner = _build_provider(cfg)
    resilient = ResilientLLMProvider(
        inner,
        max_retries=cfg.llm_max_retries,
        base_backoff_seconds=cfg.llm_retry_backoff_seconds,
    )
    return ObservingLLMProvider(resilient)


@lru_cache
def get_llm_provider() -> LLMProvider:
    return create_llm_provider(get_settings())


def llm_configuration_status(settings: Settings | None = None) -> dict[str, object]:
    cfg = settings or get_settings()
    status: dict[str, object] = {
        "provider": cfg.llm_provider,
        "credentials_configured": False,
        "model": None,
        "ready": False,
        "detail": None,
    }
    try:
        provider = create_llm_provider(cfg)
        status["credentials_configured"] = True
        status["model"] = provider.default_model
        status["ready"] = True
    except LLMConfigurationError as exc:
        status["detail"] = str(exc)
    return status
