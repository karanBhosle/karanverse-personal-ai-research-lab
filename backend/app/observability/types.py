from dataclasses import dataclass, field
from typing import Any


@dataclass
class ResearchTraceHandle:
    trace_id: str | None = None
    research_id: str | None = None
    user_question: str | None = None
    _root: Any = None


@dataclass
class AgentStepHandle:
    agent_name: str
    span: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
