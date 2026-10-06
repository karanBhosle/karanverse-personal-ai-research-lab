from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.llm import router as llm_router
from app.api.research import router as research_router
from app.api.knowledge import router as knowledge_router
from app.api.search import router as search_router

__all__ = [
    "documents_router",
    "health_router",
    "knowledge_router",
    "llm_router",
    "research_router",
    "search_router",
]
