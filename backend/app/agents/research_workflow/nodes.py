from app.agents.critic.analysis import analyze_bundle
from app.agents.critic.context import build_critic_context
from app.agents.critic.prompts import build_critic_system_prompt, build_critic_user_prompt
from app.agents.critic.validation import merge_critique
from app.agents.research_agent.nodes import (
    run_document_retrieval,
    run_hybrid_retrieval,
    run_literature_search,
    run_reranking,
    run_sub_question_generation,
)
from app.agents.research_planner.graph import compile_research_planner
from app.services.research_history_retrieval import format_snippets_for_planner
from app.agents.research_workflow.deps import ResearchWorkflowDeps
from app.agents.research_workflow.state import ResearchWorkflowState
from app.agents.synthesizer.citations import build_citation_index
from app.agents.synthesizer.context import build_synthesizer_context
from app.agents.synthesizer.prompts import build_synthesizer_system_prompt, build_synthesizer_user_prompt
from app.agents.synthesizer.validation import assemble_report
from app.llm.types import LLMMessage, LLMRequest, LLMRequestObservability
from app.models.critique import CritiqueLLMOutput
from app.models.evidence import Evidence
from app.models.research_evidence import ResearchEvidenceBundle
from app.models.research_report import SynthesizerLLMOutput
from app.security.knowledge_guard import include_private_in_llm_context
from app.services.citation_service import CitationService
from app.agents.research_agent.nodes import run_evidence_collection


def _agent_state(state: ResearchWorkflowState) -> dict:
    return dict(state)


def run_load_history_context(state: ResearchWorkflowState, deps: ResearchWorkflowDeps) -> dict:
    use_history = state.get("use_research_history", True)
    allow_private_llm = include_private_in_llm_context(
        deps.settings,
        user_opt_in=bool(state.get("allow_private_knowledge")),
    )
    if use_history and not allow_private_llm:
        return {
            "history_context_text": "",
            "history_context_snippets": [],
            "_trace_message": "prior research context withheld (private data policy for external LLM)",
        }
    if not use_history:
        return {
            "history_context_text": "",
            "history_context_snippets": [],
            "_trace_message": "research history context disabled",
        }
    top_k = state.get("history_top_k")
    if top_k is None:
        top_k = deps.settings.research_history_context_top_k
    if top_k <= 0:
        return {
            "history_context_text": "",
            "history_context_snippets": [],
            "_trace_message": "research history top_k=0",
        }
    snippets = deps.history_service.retrieve_relevant(state["question"], top_k=top_k)
    return {
        "history_context_text": format_snippets_for_planner(snippets),
        "history_context_snippets": [snippet.model_dump(mode="json") for snippet in snippets],
        "_trace_message": f"loaded {len(snippets)} prior research snippet(s)",
    }


def run_planner(state: ResearchWorkflowState, deps: ResearchWorkflowDeps) -> dict:
    sources = deps.source_registry.list_sources()
    graph = compile_research_planner(
        deps.llm,
        available_sources=sources,
        temperature=deps.settings.research_planner_temperature,
    )
    result = graph.invoke(
        {
            "question": state["question"],
            "plan": None,
            "history_context": state.get("history_context_text") or None,
        }
    )
    plan = result["plan"]
    return {
        "plan": plan,
        "_trace_message": f"plan created with {len(plan.sub_questions)} sub-questions",
    }


def run_source_search(state: ResearchWorkflowState, deps: ResearchWorkflowDeps) -> dict:
    agent_deps = deps.agent_deps(plan_question=lambda _q: state["plan"])
    out = run_literature_search(_agent_state(state), agent_deps)
    return {
        **out,
        "_trace_message": f"source search returned {len(out.get('papers', []))} papers",
    }


def run_sub_questions_node(state: ResearchWorkflowState, deps: ResearchWorkflowDeps) -> dict:
    agent_deps = deps.agent_deps(plan_question=lambda _q: state["plan"])
    out = run_sub_question_generation(_agent_state(state), agent_deps)
    return {
        **out,
        "_trace_message": f"sub-questions prepared count={len(out.get('sub_questions', []))}",
    }


def run_evidence_retrieval(state: ResearchWorkflowState, deps: ResearchWorkflowDeps) -> dict:
    agent_deps = deps.agent_deps(plan_question=lambda _q: state["plan"])
    out = run_document_retrieval(_agent_state(state), agent_deps)
    return {
        **out,
        "_trace_message": f"document retrieval candidates={len(out.get('document_candidates', []))}",
    }


def run_hybrid_retrieval_node(state: ResearchWorkflowState, deps: ResearchWorkflowDeps) -> dict:
    agent_deps = deps.agent_deps(plan_question=lambda _q: state["plan"])
    if not state.get("sub_questions"):
        state = {**state, **run_sub_question_generation(_agent_state(state), agent_deps)}
    out = run_hybrid_retrieval(_agent_state(state), agent_deps)
    return {
        **out,
        "_trace_message": f"hybrid retrieval for {len(out.get('hybrid_candidates_by_sub_question', {}))} sub-questions",
    }


def run_reranking_node(state: ResearchWorkflowState, deps: ResearchWorkflowDeps) -> dict:
    agent_deps = deps.agent_deps(plan_question=lambda _q: state["plan"])
    out = run_reranking(_agent_state(state), agent_deps)
    total = sum(len(v) for v in out.get("reranked_by_sub_question", {}).values())
    return {**out, "_trace_message": f"reranking produced {total} hits"}


def run_evidence_collection_node(state: ResearchWorkflowState, deps: ResearchWorkflowDeps) -> dict:
    agent_deps = deps.agent_deps(plan_question=lambda _q: state["plan"])
    out = run_evidence_collection(_agent_state(state), agent_deps)
    bundle = out["bundle"]
    prior = state.get("bundle")
    if prior is not None:
        bundle = _merge_bundles(prior, bundle, deps.settings.data_processed_dir)
    return {
        "bundle": bundle,
        "_trace_message": f"evidence bundle size={len(bundle.evidence)} papers={len(bundle.papers)}",
    }


def _merge_bundles(
    previous: ResearchEvidenceBundle,
    current: ResearchEvidenceBundle,
    processed_dir,
) -> ResearchEvidenceBundle:
    papers_map = {(p.source, p.external_id): p for p in previous.papers}
    for paper in current.papers:
        papers_map[(paper.source, paper.external_id)] = paper

    evidence_map: dict[str, Evidence] = {e.evidence_id: e for e in previous.evidence}
    for item in current.evidence:
        evidence_map[item.evidence_id] = item

    scores = previous.retrieval_scores + current.retrieval_scores
    unresolved = list(dict.fromkeys(previous.unresolved_questions + current.unresolved_questions))

    merged_evidence = list(evidence_map.values())
    citation_service = CitationService(processed_dir=processed_dir)
    for ev in merged_evidence:
        citation_service.register_evidence(ev)

    metadata_seen: set[tuple[str, str]] = set()
    merged_metadata = []
    for meta in previous.source_metadata + current.source_metadata:
        key = (meta.filename, meta.source_type)
        if key in metadata_seen:
            continue
        metadata_seen.add(key)
        merged_metadata.append(meta)

    return ResearchEvidenceBundle(
        question=current.question,
        plan=current.plan,
        papers=list(papers_map.values()),
        evidence=merged_evidence,
        citations=citation_service.citations,
        source_metadata=merged_metadata,
        retrieval_scores=scores,
        unresolved_questions=unresolved,
    )


def run_critic_node(state: ResearchWorkflowState, deps: ResearchWorkflowDeps) -> dict:
    bundle = state["bundle"]
    analysis = analyze_bundle(bundle, deps.settings)
    allow_private = include_private_in_llm_context(
        deps.settings,
        user_opt_in=bool(state.get("allow_private_knowledge")),
    )
    context = build_critic_context(
        bundle,
        max_excerpt_chars=deps.settings.critic_max_excerpt_chars,
        allow_private_knowledge=allow_private,
    )
    request = LLMRequest(
        messages=[
            LLMMessage(role="system", content=build_critic_system_prompt()),
            LLMMessage(role="user", content=build_critic_user_prompt(context)),
        ],
        temperature=deps.settings.critic_temperature,
        observability=LLMRequestObservability(agent_name="critic"),
    )
    llm_output = deps.llm.generate_structured(request, CritiqueLLMOutput)
    critique = merge_critique(bundle, analysis, llm_output)
    return {
        "critique": critique,
        "_trace_message": f"critic sufficient={critique.evidence_sufficient}",
    }


def run_additional_research(state: ResearchWorkflowState, deps: ResearchWorkflowDeps) -> dict:
    critique = state["critique"]
    iteration = int(state.get("iteration") or 0) + 1
    queries = list(dict.fromkeys(critique.additional_research_queries))
    sub_questions = list(dict.fromkeys((state.get("sub_questions") or []) + bundle_unresolved(state)))
    return {
        "iteration": iteration,
        "active_search_queries": queries,
        "sub_questions": sub_questions or state.get("sub_questions") or [],
        "_trace_message": f"additional research iteration={iteration} queries={len(queries)}",
    }


def bundle_unresolved(state: ResearchWorkflowState) -> list[str]:
    bundle = state.get("bundle")
    if bundle is None:
        return []
    return bundle.unresolved_questions


def run_synthesizer_node(state: ResearchWorkflowState, deps: ResearchWorkflowDeps) -> dict:
    bundle = state["bundle"]
    critique = state["critique"]
    plan = state["plan"]
    question = state["question"]
    index = build_citation_index(bundle)
    allow_private = include_private_in_llm_context(
        deps.settings,
        user_opt_in=bool(state.get("allow_private_knowledge")),
    )
    context = build_synthesizer_context(
        question=question,
        plan=plan,
        bundle=bundle,
        critique=critique,
        citation_index=index,
        max_excerpt_chars=deps.settings.synthesizer_max_excerpt_chars,
        allow_private_knowledge=allow_private,
    )
    request = LLMRequest(
        messages=[
            LLMMessage(role="system", content=build_synthesizer_system_prompt()),
            LLMMessage(role="user", content=build_synthesizer_user_prompt(context)),
        ],
        temperature=deps.settings.synthesizer_temperature,
        observability=LLMRequestObservability(agent_name="synthesis"),
    )
    llm_output = deps.llm.generate_structured(request, SynthesizerLLMOutput)
    report = assemble_report(
        question=question,
        plan=plan,
        bundle=bundle,
        critique=critique,
        index=index,
        llm_output=llm_output,
    )
    return {"report": report, "_trace_message": "research report synthesized"}

