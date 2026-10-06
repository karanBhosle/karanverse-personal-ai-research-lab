from dataclasses import dataclass, field

from app.models.search import HybridSearchResult, RankContribution, SearchResult


@dataclass
class _FusedState:
    chunk_id: str
    document_id: str = ""
    text: str = ""
    metadata: dict = field(default_factory=dict)
    bm25_score: float | None = None
    vector_score: float | None = None
    bm25_rank: int | None = None
    vector_rank: int | None = None
    bm25_rrf: float = 0.0
    vector_rrf: float = 0.0

    @property
    def final_score(self) -> float:
        return self.bm25_rrf + self.vector_rrf


def _rrf_contribution(rank: int, rrf_k: int, weight: float) -> float:
    """Standard RRF term: weight / (k + rank), rank is 1-indexed."""
    if rank < 1:
        raise ValueError("rank must be >= 1")
    return weight / (rrf_k + rank)


def reciprocal_rank_fusion(
    bm25_results: list[SearchResult],
    vector_results: list[SearchResult],
    *,
    top_k: int,
    rrf_k: int = 60,
    bm25_weight: float = 1.0,
    vector_weight: float = 1.0,
) -> list[HybridSearchResult]:
    """
    Fuse two ranked lists with weighted Reciprocal Rank Fusion.

    final_score(chunk) = bm25_weight / (k + bm25_rank) + vector_weight / (k + vector_rank)
    (missing from a list contributes 0 for that retriever)
    """
    fused: dict[str, _FusedState] = {}

    def _touch(result: SearchResult) -> _FusedState:
        state = fused.get(result.chunk_id)
        if state is None:
            state = _FusedState(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                text=result.text,
                metadata=dict(result.metadata),
            )
            fused[result.chunk_id] = state
        if not state.document_id:
            state.document_id = result.document_id
        if not state.text:
            state.text = result.text
        if not state.metadata:
            state.metadata = dict(result.metadata)
        return state

    for rank, hit in enumerate(bm25_results, start=1):
        state = _touch(hit)
        state.bm25_score = hit.score
        state.bm25_rank = rank
        state.bm25_rrf = _rrf_contribution(rank, rrf_k, bm25_weight)

    for rank, hit in enumerate(vector_results, start=1):
        state = _touch(hit)
        state.vector_score = hit.score
        state.vector_rank = rank
        state.vector_rrf = _rrf_contribution(rank, rrf_k, vector_weight)

    ordered = sorted(fused.values(), key=lambda item: item.final_score, reverse=True)[:top_k]

    return [
        HybridSearchResult(
            chunk_id=state.chunk_id,
            document_id=state.document_id,
            text=state.text,
            final_score=state.final_score,
            bm25_score=state.bm25_score,
            vector_score=state.vector_score,
            rank_contribution=RankContribution(
                bm25_rank=state.bm25_rank,
                vector_rank=state.vector_rank,
                bm25_rrf=state.bm25_rrf,
                vector_rrf=state.vector_rrf,
            ),
            metadata=state.metadata,
        )
        for state in ordered
    ]
