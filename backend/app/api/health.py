from fastapi import APIRouter

from app.config import get_settings
from app.knowledge_graph.service import get_knowledge_graph_service
from app.observability.service import get_observability_service

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str | bool]:
    settings = get_settings()
    kg = get_knowledge_graph_service()
    obs = get_observability_service()
    langfuse_configured = bool(settings.langfuse_public_key and settings.langfuse_secret_key)
    return {
        "status": "ok",
        "service": settings.app_name,
        "environment": settings.app_env,
        "neo4j_enabled": settings.neo4j_enabled,
        "neo4j_connected": kg.enabled,
        "langfuse_enabled": settings.langfuse_enabled,
        "langfuse_configured": langfuse_configured,
        "langfuse_active": obs.enabled,
        "api_auth_enabled": settings.api_auth_enabled,
        "external_llm_allow_private_knowledge": settings.external_llm_allow_private_knowledge,
        "research_allow_private_knowledge": settings.research_allow_private_knowledge,
    }
