import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.llm.exceptions import LLMConfigurationError, LLMError
from app.llm.factory import llm_configuration_status
from pydantic import BaseModel, Field

from app.models.rag import ResearchAnswerRequest, ResearchAnswerResponse
from app.models.critique import CritiqueRequest, CritiqueResult
from app.models.research_history import (
    ResearchHistoryListResponse,
    ResearchHistoryRecord,
    ResearchWorkflowResponse,
)
from app.models.research_report import ResearchReport, ResearchSynthesizeRequest
from app.models.research_state import ResearchWorkflowRequest
from app.models.research_update import ResearchUpdate, ResearchUpdateRequest
from app.models.research_evidence import ResearchCollectRequest, ResearchEvidenceBundle
from app.models.research_plan import ResearchPlanRequest, ResearchPlanResponse
from app.models.research_paper import ResearchPaper
from app.services.rag_service import GroundedRAGService, get_grounded_rag_service
from app.services.critic_service import CriticAgentService, get_critic_agent_service
from app.services.research_synthesizer_service import (
    ResearchSynthesizerService,
    get_research_synthesizer_service,
)
from app.services.research_agent_service import ResearchAgentService, get_research_agent_service
from app.services.research_history_service import ResearchHistoryService, get_research_history_service
from app.services.research_update_service import ResearchUpdateService, get_research_update_service
from app.services.research_workflow_service import ResearchWorkflowService, get_research_workflow_service
from app.services.research_planner_service import ResearchPlannerService, get_research_planner_service
from app.sources.exceptions import ResearchSourceError, UnknownResearchSourceError
from app.sources.registry import SourceRegistry, get_source_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/research", tags=["research"])


@router.get("/history", response_model=ResearchHistoryListResponse)
def list_research_history(
    history: ResearchHistoryService = Depends(get_research_history_service),
) -> ResearchHistoryListResponse:
    items = history.list_summaries()
    return ResearchHistoryListResponse(items=items, count=len(items))


@router.post("/update", response_model=ResearchUpdate)
def research_update(
    body: ResearchUpdateRequest,
    update_service: ResearchUpdateService = Depends(get_research_update_service),
) -> ResearchUpdate:
    llm_status = llm_configuration_status()
    if not llm_status["ready"]:
        raise HTTPException(
            status_code=503,
            detail=f"LLM provider is not configured: {llm_status.get('detail')}",
        )
    try:
        return update_service.run(body)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMError as exc:
        logger.exception("Research update failed during LLM call")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ResearchSourceError as exc:
        logger.exception("Research update failed during literature search")
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/{research_id}", response_model=ResearchHistoryRecord)
def get_research_history_record(
    research_id: str,
    history: ResearchHistoryService = Depends(get_research_history_service),
) -> ResearchHistoryRecord:
    record = history.get_record(research_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Research run not found: {research_id}")
    return record


@router.get("/sources")
def list_research_sources(registry: SourceRegistry = Depends(get_source_registry)) -> dict[str, list[str]]:
    return {"sources": registry.list_sources()}


@router.post("/stream")
def stream_research_workflow(
    body: ResearchWorkflowRequest,
    workflow: ResearchWorkflowService = Depends(get_research_workflow_service),
) -> StreamingResponse:
    llm_status = llm_configuration_status()
    if not llm_status["ready"]:
        raise HTTPException(
            status_code=503,
            detail=f"LLM provider is not configured: {llm_status.get('detail')}",
        )

    def event_generator():
        try:
            for event in workflow.stream_run(
                body.question,
                evidence_top_k=body.evidence_top_k,
                max_iterations=body.max_iterations,
                use_research_history=body.use_research_history,
                history_top_k=body.history_top_k,
                allow_private_knowledge=body.allow_private_knowledge,
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except LLMConfigurationError as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        except LLMError as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("", response_model=ResearchWorkflowResponse)
def run_research_workflow(
    body: ResearchWorkflowRequest,
    workflow: ResearchWorkflowService = Depends(get_research_workflow_service),
) -> ResearchWorkflowResponse:
    llm_status = llm_configuration_status()
    if not llm_status["ready"]:
        raise HTTPException(
            status_code=503,
            detail=f"LLM provider is not configured: {llm_status.get('detail')}",
        )

    try:
        return workflow.run(
            body.question,
            evidence_top_k=body.evidence_top_k,
            max_iterations=body.max_iterations,
            use_research_history=body.use_research_history,
            history_top_k=body.history_top_k,
            allow_private_knowledge=body.allow_private_knowledge,
        )
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMError as exc:
        logger.exception("Research workflow failed during LLM call")
        raise HTTPException(status_code=502, detail=str(exc)) from exc


class ExternalResearchSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=50)
    source: str = Field(default="openalex", min_length=1)


class ExternalResearchSearchResponse(BaseModel):
    query: str
    source: str
    count: int
    papers: list[ResearchPaper]


@router.post("/search", response_model=ExternalResearchSearchResponse)
def search_external_literature(
    body: ExternalResearchSearchRequest,
    registry: SourceRegistry = Depends(get_source_registry),
) -> ExternalResearchSearchResponse:
    try:
        papers = registry.search_papers(body.source, body.query, limit=body.limit)
    except UnknownResearchSourceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ResearchSourceError as exc:
        logger.exception("External literature search failed source=%s", body.source)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ExternalResearchSearchResponse(
        query=body.query,
        source=body.source.strip().lower(),
        count=len(papers),
        papers=papers,
    )


@router.post("/plan", response_model=ResearchPlanResponse)
def create_research_plan(
    body: ResearchPlanRequest,
    planner: ResearchPlannerService = Depends(get_research_planner_service),
) -> ResearchPlanResponse:
    llm_status = llm_configuration_status()
    if not llm_status["ready"]:
        raise HTTPException(
            status_code=503,
            detail=f"LLM provider is not configured: {llm_status.get('detail')}",
        )

    try:
        plan = planner.create_plan(body.question)
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMError as exc:
        logger.exception("Research planner failed during LLM call")
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ResearchPlanResponse(plan=plan)


@router.post("/collect", response_model=ResearchEvidenceBundle)
def collect_research_evidence(
    body: ResearchCollectRequest,
    agent: ResearchAgentService = Depends(get_research_agent_service),
) -> ResearchEvidenceBundle:
    llm_status = llm_configuration_status()
    if not llm_status["ready"]:
        raise HTTPException(
            status_code=503,
            detail=f"LLM provider is not configured: {llm_status.get('detail')}",
        )

    try:
        return agent.collect_evidence(body.question, evidence_top_k=body.evidence_top_k)
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMError as exc:
        logger.exception("Research agent failed during planning LLM call")
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/critique", response_model=CritiqueResult)
def critique_research_evidence(
    body: CritiqueRequest,
    critic: CriticAgentService = Depends(get_critic_agent_service),
) -> CritiqueResult:
    llm_status = llm_configuration_status()
    if not llm_status["ready"]:
        raise HTTPException(
            status_code=503,
            detail=f"LLM provider is not configured: {llm_status.get('detail')}",
        )

    try:
        return critic.critique(body.bundle)
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMError as exc:
        logger.exception("Critic agent failed during LLM call")
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/synthesize", response_model=ResearchReport)
def synthesize_research_report(
    body: ResearchSynthesizeRequest,
    synthesizer: ResearchSynthesizerService = Depends(get_research_synthesizer_service),
) -> ResearchReport:
    llm_status = llm_configuration_status()
    if not llm_status["ready"]:
        raise HTTPException(
            status_code=503,
            detail=f"LLM provider is not configured: {llm_status.get('detail')}",
        )

    try:
        return synthesizer.synthesize(
            question=body.question,
            plan=body.plan,
            bundle=body.bundle,
            critique=body.critique,
        )
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMError as exc:
        logger.exception("Research synthesizer failed during LLM call")
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/answer", response_model=ResearchAnswerResponse)
def research_answer(
    body: ResearchAnswerRequest,
    rag_service: GroundedRAGService = Depends(get_grounded_rag_service),
) -> ResearchAnswerResponse:
    llm_status = llm_configuration_status()
    if not llm_status["ready"]:
        raise HTTPException(
            status_code=503,
            detail=f"LLM provider is not configured: {llm_status.get('detail')}",
        )

    try:
        return rag_service.answer(
            body.question,
            top_k=body.top_k,
            allow_private_knowledge=body.allow_private_knowledge,
        )
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMError as exc:
        logger.exception("Grounded RAG failed during LLM call")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
