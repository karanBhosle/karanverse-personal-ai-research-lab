import re

from pydantic import BaseModel, Field

from app.retrieval.tokenizer import tokenize


class GeneratedAnswerEval(BaseModel):
    answer: str
    cited_chunk_ids: list[str] = Field(default_factory=list)


class AnswerQualityMetrics(BaseModel):
    faithfulness: float = Field(ge=0.0, le=1.0)
    answer_relevance: float = Field(ge=0.0, le=1.0)
    citation_correctness: float = Field(ge=0.0, le=1.0)
    citation_completeness: float = Field(ge=0.0, le=1.0)


def _token_coverage(hypothesis: str, reference: str) -> float:
    ref_tokens = set(tokenize(reference))
    if not ref_tokens:
        return 0.0
    hyp_tokens = set(tokenize(hypothesis))
    return len(ref_tokens & hyp_tokens) / len(ref_tokens)


def _answer_sentences(answer: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", answer.strip())
    return [part.strip() for part in parts if part.strip()]


def faithfulness_score(answer: str, evidence_texts: list[str]) -> float:
    """Fraction of answer sentences with token overlap to at least one evidence excerpt."""
    sentences = _answer_sentences(answer)
    if not sentences:
        return 0.0
    if not evidence_texts:
        return 0.0
    evidence_tokens = [set(tokenize(text)) for text in evidence_texts if text.strip()]
    if not evidence_tokens:
        return 0.0
    supported = 0
    for sentence in sentences:
        sent_tokens = set(tokenize(sentence))
        if not sent_tokens:
            continue
        if any(len(sent_tokens & ev_tokens) >= max(2, len(sent_tokens) // 4) for ev_tokens in evidence_tokens):
            supported += 1
    return supported / len(sentences)


def evaluate_answer(
    generated: GeneratedAnswerEval,
    *,
    reference_answer: str | None,
    expected_evidence_chunk_ids: set[str],
    evidence_text_by_chunk: dict[str, str],
) -> AnswerQualityMetrics:
    evidence_texts = [
        evidence_text_by_chunk[chunk_id]
        for chunk_id in generated.cited_chunk_ids
        if chunk_id in evidence_text_by_chunk
    ]
    faithfulness = faithfulness_score(generated.answer, evidence_texts)
    relevance = _token_coverage(generated.answer, reference_answer or "") if reference_answer else 0.0

    cited = [chunk_id for chunk_id in generated.cited_chunk_ids if chunk_id]
    if cited:
        correct = sum(1 for chunk_id in cited if chunk_id in expected_evidence_chunk_ids)
        citation_correctness = correct / len(cited)
    else:
        citation_correctness = 0.0

    if expected_evidence_chunk_ids:
        cited_expected = sum(1 for chunk_id in expected_evidence_chunk_ids if chunk_id in set(cited))
        citation_completeness = cited_expected / len(expected_evidence_chunk_ids)
    else:
        citation_completeness = 0.0

    return AnswerQualityMetrics(
        faithfulness=round(faithfulness, 4),
        answer_relevance=round(relevance, 4),
        citation_correctness=round(citation_correctness, 4),
        citation_completeness=round(citation_completeness, 4),
    )
