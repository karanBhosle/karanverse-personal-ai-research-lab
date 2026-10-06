import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ExpectedSource(BaseModel):
    """Document-level relevance label (optional if chunk_ids are provided)."""

    document_id: str | None = None
    filename: str | None = None
    external_id: str | None = None


class EvaluationItem(BaseModel):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_sources: list[ExpectedSource] = Field(default_factory=list)
    expected_evidence: list[str] = Field(
        default_factory=list,
        description="Relevant chunk_id values for IR metrics.",
    )
    reference_answer: str | None = None
    retrieval_sub_queries: list[str] = Field(
        default_factory=list,
        description="Sub-questions for agentic retrieval; defaults to [question] when empty.",
    )
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("expected_evidence", mode="before")
    @classmethod
    def _normalize_chunk_ids(cls, value: Any) -> list[str]:
        if value is None:
            return []
        return [str(item).strip() for item in value if str(item).strip()]


class EvaluationDataset(BaseModel):
    version: str = "1.0"
    name: str = "evaluation"
    description: str = ""
    items: list[EvaluationItem] = Field(min_length=1)

    @classmethod
    def load(cls, path: Path) -> "EvaluationDataset":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls.model_validate(raw)

    def content_hash(self) -> str:
        payload = self.model_dump(mode="json")
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return digest[:16]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
