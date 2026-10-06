from contextvars import ContextVar

from app.observability.types import ResearchTraceHandle

_current_research_trace: ContextVar[ResearchTraceHandle | None] = ContextVar(
    "current_research_trace",
    default=None,
)


def get_current_research_trace() -> ResearchTraceHandle | None:
    return _current_research_trace.get()


def set_current_research_trace(handle: ResearchTraceHandle | None):
    return _current_research_trace.set(handle)


def reset_current_research_trace(token) -> None:
    _current_research_trace.reset(token)
