import re

from app.models.knowledge_source import KnowledgeScope

_PERSONAL_PATTERNS = [
    re.compile(r"\bmy (projects?|notes?|knowledge|documentation|reports?|learnings?)\b", re.I),
    re.compile(r"\b(i('ve| have) (learned|written|built|implemented))\b", re.I),
    re.compile(r"\bwhat have i learned\b", re.I),
    re.compile(r"\bpreviously learned\b", re.I),
    re.compile(r"\bwhich of my\b", re.I),
    re.compile(r"\bconnect my knowledge\b", re.I),
    re.compile(r"\bmy (experience|work) (with|on)\b", re.I),
]

_PUBLIC_PATTERNS = [
    re.compile(r"\b(literature|openalex|arxiv|published papers?|state of the art)\b", re.I),
    re.compile(r"\bpublic research\b", re.I),
]

_EXPLICIT_MIX_PATTERNS = [
    re.compile(r"\b(both|combine|mixed|all sources|public and personal)\b", re.I),
]


def infer_knowledge_scopes(query: str, *, allow_mixed: bool = False) -> set[KnowledgeScope]:
    """Infer which knowledge scopes a query should search (no mixing by default)."""
    if allow_mixed or any(p.search(query) for p in _EXPLICIT_MIX_PATTERNS):
        return set(KnowledgeScope)

    if any(p.search(query) for p in _PERSONAL_PATTERNS):
        return {KnowledgeScope.PERSONAL_KNOWLEDGE, KnowledgeScope.PROJECT}

    if any(p.search(query) for p in _PUBLIC_PATTERNS):
        return {KnowledgeScope.PUBLIC_RESEARCH}

    # Default: public research corpus only (avoid blending private notes into generic queries).
    return {KnowledgeScope.PUBLIC_RESEARCH}


def chunk_matches_scopes(metadata: dict, allowed: set[KnowledgeScope]) -> bool:
    if not allowed:
        return True
    raw = metadata.get("knowledge_scope")
    if raw is None:
        return KnowledgeScope.PUBLIC_RESEARCH in allowed
    try:
        scope = KnowledgeScope(str(raw))
    except ValueError:
        return False
    return scope in allowed


def filter_results_by_scope(results: list, allowed: set[KnowledgeScope]) -> list:
    if not allowed or len(allowed) == len(KnowledgeScope):
        return results
    filtered = [r for r in results if chunk_matches_scopes(getattr(r, "metadata", {}) or {}, allowed)]
    return filtered
