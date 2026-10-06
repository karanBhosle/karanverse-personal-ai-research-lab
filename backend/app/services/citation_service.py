import re
from pathlib import Path

from app.models.document import Chunk
from app.models.evidence import Citation, Evidence, RetrievalMethod
from app.models.search import HybridSearchResult, RerankedSearchResult, SearchResult
from app.retrieval.tokenizer import tokenize
from app.services.evidence_factory import (
    evidence_from_hybrid_result,
    evidence_from_reranked_result,
    evidence_from_search_result,
)
from app.services.evidence_resolver import ChunkResolutionError, load_chunk_from_processed

_CITATION_LABEL_RE = re.compile(r"^\[(\d+)\]$")


class CitationService:
    """
    Register retrieval hits as numbered evidence and map claims to citations.

    Citation labels use inline form: [1], [2], [3]
    """

    def __init__(self, processed_dir: Path | None = None) -> None:
        self.processed_dir = processed_dir
        self._evidence_by_id: dict[str, Evidence] = {}
        self._evidence_by_chunk: dict[str, Evidence] = {}
        self._citations_by_number: dict[int, Citation] = {}
        self._next_number = 1

    @property
    def evidence_items(self) -> list[Evidence]:
        return list(self._evidence_by_id.values())

    @property
    def citations(self) -> list[Citation]:
        return sorted(self._citations_by_number.values(), key=lambda c: c.citation_number)

    @staticmethod
    def format_citation_label(citation_number: int) -> str:
        if citation_number < 1:
            raise ValueError("citation_number must be >= 1")
        return f"[{citation_number}]"

    def register_evidence(self, evidence: Evidence) -> Citation:
        if evidence.chunk_id in self._evidence_by_chunk:
            existing = self._evidence_by_chunk[evidence.chunk_id]
            for citation in self._citations_by_number.values():
                if citation.evidence_id == existing.evidence_id:
                    return citation

        self._evidence_by_id[evidence.evidence_id] = evidence
        self._evidence_by_chunk[evidence.chunk_id] = evidence

        number = self._next_number
        self._next_number += 1
        citation = Citation(
            citation_number=number,
            label=self.format_citation_label(number),
            evidence_id=evidence.evidence_id,
        )
        self._citations_by_number[number] = citation
        return citation

    def register_search_results(
        self,
        results: list[SearchResult],
        *,
        method: RetrievalMethod,
    ) -> list[Citation]:
        citations: list[Citation] = []
        for result in results:
            evidence = evidence_from_search_result(
                result,
                method=method,
                processed_dir=self.processed_dir,
            )
            citations.append(self.register_evidence(evidence))
        return citations

    def register_hybrid_results(self, results: list[HybridSearchResult]) -> list[Citation]:
        citations: list[Citation] = []
        for result in results:
            evidence = evidence_from_hybrid_result(result, processed_dir=self.processed_dir)
            citations.append(self.register_evidence(evidence))
        return citations

    def register_reranked_results(self, results: list[RerankedSearchResult]) -> list[Citation]:
        citations: list[Citation] = []
        for result in results:
            evidence = evidence_from_reranked_result(result, processed_dir=self.processed_dir)
            citations.append(self.register_evidence(evidence))
        return citations

    def get_evidence(self, evidence_id: str) -> Evidence:
        try:
            return self._evidence_by_id[evidence_id]
        except KeyError as exc:
            raise KeyError(f"Unknown evidence_id: {evidence_id}") from exc

    def get_evidence_by_chunk(self, chunk_id: str) -> Evidence:
        try:
            return self._evidence_by_chunk[chunk_id]
        except KeyError as exc:
            raise KeyError(f"Unknown chunk_id: {chunk_id}") from exc

    def resolve_citation_label(self, label: str) -> Evidence:
        match = _CITATION_LABEL_RE.match(label.strip())
        if not match:
            raise ValueError(f"Invalid citation label: {label}")
        number = int(match.group(1))
        return self.resolve_citation_number(number)

    def resolve_citation_number(self, citation_number: int) -> Evidence:
        try:
            citation = self._citations_by_number[citation_number]
        except KeyError as exc:
            raise KeyError(f"Unknown citation number: {citation_number}") from exc
        return self.get_evidence(citation.evidence_id)

    def resolve_to_source_chunk(self, evidence: Evidence) -> Chunk:
        if self.processed_dir is None:
            raise ChunkResolutionError("processed_dir is not configured for chunk resolution")
        chunk = load_chunk_from_processed(self.processed_dir, evidence.document_id, evidence.chunk_id)
        if chunk.text != evidence.text:
            raise ChunkResolutionError(
                f"Evidence text does not match stored chunk {evidence.chunk_id} in {evidence.document_id}"
            )
        return chunk

    def map_claim_to_citations(
        self,
        claim: str,
        *,
        max_citations: int = 3,
        min_token_overlap: int = 2,
    ) -> list[Citation]:
        """
        Map a claim to supporting citations using deterministic token overlap (no LLM).

        Returns citations ordered by overlap strength.
        """
        claim_tokens = set(tokenize(claim))
        if not claim_tokens:
            return []

        scored: list[tuple[int, Citation]] = []
        for citation in self.citations:
            evidence = self.get_evidence(citation.evidence_id)
            overlap = claim_tokens.intersection(tokenize(evidence.text))
            if len(overlap) < min_token_overlap:
                continue
            scored.append((len(overlap), citation))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            citation.model_copy(update={"claim_text": claim})
            for _, citation in scored[:max_citations]
        ]

    def attach_citations_to_text(self, text: str, citations: list[Citation]) -> str:
        if not citations:
            return text
        labels = "".join(citation.label for citation in citations)
        return f"{text}{labels}"
