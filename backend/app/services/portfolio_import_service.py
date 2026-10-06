import json
import logging
from pathlib import Path

from app.config.paths import get_project_root
from app.models.knowledge_source import DocumentKind, KnowledgeScope
from app.services.ingestion_service import IngestionService

logger = logging.getLogger(__name__)


def default_portfolio_seed_path() -> Path:
    return get_project_root() / "data" / "personal" / "portfolio_seed.json"


def _project_markdown(project: dict) -> str:
    tech = ", ".join(project.get("technologies") or [])
    topics = ", ".join(project.get("topics") or [])
    return (
        f"# Project: {project.get('name', 'Untitled')}\n\n"
        f"{project.get('description', '').strip()}\n\n"
        f"**Technologies:** {tech}\n\n"
        f"**Topics:** {topics}\n"
    )


def _learning_note_markdown(note: dict) -> str:
    return f"# Learning note: {note.get('topic', 'Topic')}\n\n{note.get('content', '').strip()}\n"


def import_portfolio_seed(
    ingestion: IngestionService,
    seed_path: Path | None = None,
) -> list[str]:
    path = seed_path or default_portfolio_seed_path()
    if not path.is_file():
        raise FileNotFoundError(f"Portfolio seed not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    imported_ids: list[str] = []

    profile = payload.get("profile") or {}
    profile_md = (
        f"# {profile.get('name', 'Karan Bhosle')} — {profile.get('role', 'Data Scientist')}\n\n"
        f"{profile.get('summary', '').strip()}\n\n"
        f"**Focus areas:** {', '.join(profile.get('focus_areas') or [])}\n"
    )
    doc = ingestion.ingest_text_document(
        title="Personal profile",
        text=profile_md,
        filename="portfolio_profile.md",
        knowledge_scope=KnowledgeScope.PERSONAL_KNOWLEDGE,
        document_kind=DocumentKind.PORTFOLIO,
        tags=["portfolio", "profile"],
    )
    imported_ids.append(doc.document_id)

    for project in payload.get("projects") or []:
        project_id = str(project.get("id") or project.get("name") or "project")
        pdoc = ingestion.ingest_text_document(
            title=str(project.get("name") or project_id),
            text=_project_markdown(project),
            filename=f"project_{project_id}.md",
            knowledge_scope=KnowledgeScope.PROJECT,
            document_kind=DocumentKind.PROJECT_DOCUMENTATION,
            project_id=project_id,
            tags=["portfolio", "project"] + list(project.get("topics") or []),
        )
        imported_ids.append(pdoc.document_id)

    for note in payload.get("learning_notes") or []:
        topic = str(note.get("topic") or "note")
        ndoc = ingestion.ingest_text_document(
            title=f"Learning note: {topic}",
            text=_learning_note_markdown(note),
            filename=f"learning_{topic.lower().replace(' ', '_')}.md",
            knowledge_scope=KnowledgeScope.PERSONAL_KNOWLEDGE,
            document_kind=DocumentKind.TECHNICAL_NOTE,
            tags=["learning", topic.lower()],
        )
        imported_ids.append(ndoc.document_id)

    logger.info("Imported %d portfolio documents from %s", len(imported_ids), path)
    return imported_ids
