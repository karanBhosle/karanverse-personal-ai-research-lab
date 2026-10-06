from app.models.evidence import Evidence
from app.retrieval.tokenizer import tokenize


def _token_set(text: str) -> set[str]:
    return set(tokenize(text))


def pairwise_jaccard(a: Evidence, b: Evidence) -> float:
    tokens_a = _token_set(a.text)
    tokens_b = _token_set(b.text)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union else 0.0


def mean_max_jaccard(old_evidence: list[Evidence], new_evidence: list[Evidence]) -> float:
    """How similar new evidence is to prior evidence (higher = more overlap)."""
    if not old_evidence or not new_evidence:
        return 0.0
    scores: list[float] = []
    for new_item in new_evidence:
        best = max(pairwise_jaccard(old_item, new_item) for old_item in old_evidence)
        scores.append(best)
    return sum(scores) / len(scores)


def novel_evidence_ids(
    old_evidence: list[Evidence],
    new_evidence: list[Evidence],
    *,
    novelty_threshold: float = 0.35,
) -> list[str]:
    """New evidence items with low overlap to any prior excerpt."""
    novel: list[str] = []
    for new_item in new_evidence:
        best = max((pairwise_jaccard(old_item, new_item) for old_item in old_evidence), default=0.0)
        if best < novelty_threshold:
            novel.append(new_item.evidence_id)
    return novel
