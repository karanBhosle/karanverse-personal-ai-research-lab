from dataclasses import dataclass

from app.config.settings import Settings
from app.models.evidence import Evidence
from app.models.research_evidence import ResearchEvidenceBundle
from app.retrieval.tokenizer import tokenize


@dataclass(frozen=True, slots=True)
class DeterministicBundleAnalysis:
    duplicate_groups: list[tuple[str, ...]]
    low_quality_evidence_ids: list[str]
    uncovered_sub_questions: list[str]
    deterministic_sufficient: bool
    flags: list[str]


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def find_duplicate_evidence_groups(
    evidence: list[Evidence],
    *,
    min_jaccard: float = 0.85,
) -> list[tuple[str, ...]]:
    tokenized = [(item.evidence_id, set(tokenize(item.text))) for item in evidence]
    groups: list[tuple[str, ...]] = []
    used: set[str] = set()

    for i, (eid_a, tokens_a) in enumerate(tokenized):
        if eid_a in used:
            continue
        group = [eid_a]
        for j in range(i + 1, len(tokenized)):
            eid_b, tokens_b = tokenized[j]
            if eid_b in used:
                continue
            if eid_a == eid_b:
                continue
            if _jaccard(tokens_a, tokens_b) >= min_jaccard:
                group.append(eid_b)
        if len(group) > 1:
            groups.append(tuple(group))
            used.update(group)
    return groups


def analyze_bundle(bundle: ResearchEvidenceBundle, settings: Settings) -> DeterministicBundleAnalysis:
    flags: list[str] = []
    duplicate_groups = find_duplicate_evidence_groups(
        bundle.evidence,
        min_jaccard=settings.critic_duplicate_jaccard_threshold,
    )
    if duplicate_groups:
        flags.append("duplicate_evidence_detected")

    low_quality: list[str] = []
    for item in bundle.evidence:
        if item.retrieval_score < settings.critic_low_retrieval_score_threshold:
            low_quality.append(item.evidence_id)
    if low_quality:
        flags.append("low_retrieval_scores")

    covered_sub_questions = {
        score.sub_question
        for score in bundle.retrieval_scores
        if score.sub_question and score.final_score is not None
    }
    uncovered = [
        sq
        for sq in bundle.plan.sub_questions
        if sq not in covered_sub_questions and sq in bundle.unresolved_questions
    ]
    if bundle.unresolved_questions:
        flags.append("unresolved_sub_questions")
        uncovered = list(dict.fromkeys(bundle.unresolved_questions + uncovered))

    min_evidence = settings.critic_min_evidence_count
    min_papers = settings.critic_min_external_papers
    sufficient = True
    if len(bundle.evidence) < min_evidence:
        sufficient = False
        flags.append("insufficient_local_evidence")
    if len(bundle.papers) < min_papers:
        sufficient = False
        flags.append("insufficient_external_literature")
    if bundle.unresolved_questions:
        sufficient = False

    return DeterministicBundleAnalysis(
        duplicate_groups=duplicate_groups,
        low_quality_evidence_ids=low_quality,
        uncovered_sub_questions=uncovered,
        deterministic_sufficient=sufficient,
        flags=flags,
    )
