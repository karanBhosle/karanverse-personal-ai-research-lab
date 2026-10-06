import re

from app.config.settings import Settings
from app.models.evidence import Evidence
from app.models.knowledge_source import KnowledgeScope
from app.retrieval.knowledge_scope import infer_knowledge_scopes

_PRIVATE_SCOPES = frozenset({KnowledgeScope.PERSONAL_KNOWLEDGE, KnowledgeScope.PROJECT})

_LITERATURE_QUERY_MAX_LEN = 400
_MULTILINE_RE = re.compile(r"[\r\n]+")


def is_private_knowledge_scope(scope: str | KnowledgeScope | None) -> bool:
    if scope is None:
        return False
    try:
        parsed = scope if isinstance(scope, KnowledgeScope) else KnowledgeScope(str(scope))
    except ValueError:
        return False
    return parsed in _PRIVATE_SCOPES


def is_external_llm_provider(settings: Settings) -> bool:
    return settings.llm_provider.lower() != "ollama"


def include_private_in_research_retrieval(settings: Settings, *, user_opt_in: bool) -> bool:
    return bool(settings.research_allow_private_knowledge and user_opt_in)


def include_private_in_llm_context(settings: Settings, *, user_opt_in: bool) -> bool:
    if not settings.research_allow_private_knowledge or not user_opt_in:
        return False
    if is_external_llm_provider(settings):
        return bool(settings.external_llm_allow_private_knowledge)
    return True


def research_retrieval_scopes(
    question: str,
    settings: Settings,
    *,
    user_allow_private: bool,
    allow_mixed: bool = False,
) -> set[KnowledgeScope]:
    if not include_private_in_research_retrieval(settings, user_opt_in=user_allow_private):
        return {KnowledgeScope.PUBLIC_RESEARCH}
    return infer_knowledge_scopes(question, allow_mixed=allow_mixed)


def sanitize_literature_search_query(query: str) -> str:
    """Strip multi-line / verbose text before sending queries to public literature APIs."""
    cleaned = _MULTILINE_RE.sub(" ", query.strip())
    cleaned = re.sub(r"\s+", " ", cleaned)
    if len(cleaned) > _LITERATURE_QUERY_MAX_LEN:
        cleaned = cleaned[:_LITERATURE_QUERY_MAX_LEN].rsplit(" ", 1)[0]
    return cleaned or "research"


def filter_evidence_for_llm(evidence: list[Evidence], *, allow_private: bool) -> list[Evidence]:
    if allow_private:
        return evidence
    return [
        item
        for item in evidence
        if not is_private_knowledge_scope(item.source_metadata.knowledge_scope)
    ]
