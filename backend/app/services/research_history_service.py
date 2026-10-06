import json
import logging
import uuid
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from app.config.paths import get_research_history_dir
from app.config.settings import Settings, get_settings
from app.models.critique import CritiqueResult
from app.models.evidence import Citation, Evidence
from app.models.research_evidence import ResearchEvidenceBundle
from app.models.research_history import (
    ResearchHistoryRecord,
    ResearchHistorySummary,
    ResearchHistoryContextSnippet,
    ResearchSourceEntry,
)
from app.models.research_plan import ResearchPlan
from app.models.research_report import ResearchReport
from app.security.identifiers import validate_safe_identifier
from app.security.paths import resolve_path_under
from app.services.research_history_retrieval import retrieve_relevant_history

logger = logging.getLogger(__name__)


class ResearchHistoryService:
    def __init__(self, history_dir: Path) -> None:
        self.history_dir = history_dir
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def _record_path(self, research_id: str) -> Path:
        validate_safe_identifier(research_id, field="research_id")
        return resolve_path_under(self.history_dir, f"{research_id}.json")

    def list_records(self) -> list[ResearchHistoryRecord]:
        records: list[ResearchHistoryRecord] = []
        for path in sorted(self.history_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                records.append(ResearchHistoryRecord.model_validate(payload))
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning("Skipping invalid research history file %s: %s", path, exc)
        return records

    def get_record(self, research_id: str) -> ResearchHistoryRecord | None:
        path = self._record_path(research_id)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return ResearchHistoryRecord.model_validate(payload)

    def list_summaries(self) -> list[ResearchHistorySummary]:
        summaries: list[ResearchHistorySummary] = []
        for record in self.list_records():
            summaries.append(
                ResearchHistorySummary(
                    research_id=record.research_id,
                    question=record.question,
                    created_at=record.created_at,
                    research_objective=record.plan.research_objective,
                    conclusion_count=len(record.conclusions),
                    unresolved_question_count=len(record.unresolved_questions),
                )
            )
        return summaries

    @staticmethod
    def _build_sources(plan: ResearchPlan, bundle: ResearchEvidenceBundle | None) -> list[ResearchSourceEntry]:
        sources: list[ResearchSourceEntry] = []
        for name in plan.required_sources:
            sources.append(ResearchSourceEntry(source=name, title=name))
        if bundle:
            for paper in bundle.papers:
                sources.append(
                    ResearchSourceEntry(
                        source=paper.source,
                        external_id=paper.external_id,
                        title=paper.title,
                        url=str(paper.landing_page_url) if paper.landing_page_url else None,
                    )
                )
        deduped: dict[tuple[str, str | None], ResearchSourceEntry] = {}
        for entry in sources:
            deduped[(entry.source, entry.external_id)] = entry
        return list(deduped.values())

    @staticmethod
    def _build_conclusions(report: ResearchReport) -> list[str]:
        conclusions = [finding.statement for finding in report.key_findings]
        if report.executive_summary and not conclusions:
            conclusions = [report.executive_summary]
        return conclusions

    def save_run(
        self,
        *,
        question: str,
        plan: ResearchPlan,
        bundle: ResearchEvidenceBundle | None,
        critique: CritiqueResult | None,
        report: ResearchReport,
        research_id: str | None = None,
    ) -> ResearchHistoryRecord:
        research_id = research_id or str(uuid.uuid4())
        evidence: list[Evidence] = bundle.evidence if bundle else []
        citations: list[Citation] = bundle.citations if bundle else []
        unresolved = list(
            dict.fromkeys(
                (bundle.unresolved_questions if bundle else [])
                + report.open_questions
                + (critique.additional_research_queries if critique else [])
            )
        )
        record = ResearchHistoryRecord(
            research_id=research_id,
            question=question.strip(),
            created_at=datetime.now(UTC),
            plan=plan,
            sources=self._build_sources(plan, bundle),
            evidence=evidence,
            report=report,
            citations=citations,
            conclusions=self._build_conclusions(report),
            unresolved_questions=unresolved,
            critique=critique,
        )
        path = self._record_path(research_id)
        path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Saved research history research_id=%s", research_id)
        return record

    def retrieve_relevant(
        self,
        query: str,
        *,
        top_k: int = 3,
        exclude_research_id: str | None = None,
    ) -> list[ResearchHistoryContextSnippet]:
        records = self.list_records()
        return retrieve_relevant_history(
            records,
            query,
            top_k=top_k,
            exclude_research_id=exclude_research_id,
        )


@lru_cache
def get_research_history_service() -> ResearchHistoryService:
    settings = get_settings()
    return ResearchHistoryService(settings.research_history_dir)


def create_research_history_service(settings: Settings) -> ResearchHistoryService:
    return ResearchHistoryService(settings.research_history_dir)
