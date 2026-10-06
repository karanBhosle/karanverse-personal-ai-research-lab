from unittest.mock import patch

import pytest

from app.observability.backends import NoOpObservabilityBackend, create_observability_backend
from app.observability.service import create_observability_service, get_observability_service


def test_create_backend_without_credentials_is_noop():
    backend = create_observability_backend(
        enabled=True,
        public_key=None,
        secret_key=None,
        host=None,
    )
    assert isinstance(backend, NoOpObservabilityBackend)
    assert backend.enabled is False


def test_create_backend_when_disabled_is_noop():
    backend = create_observability_backend(
        enabled=False,
        public_key="pk",
        secret_key="sk",
        host="https://cloud.langfuse.com",
    )
    assert isinstance(backend, NoOpObservabilityBackend)


def test_observability_service_disabled_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    from app.config.settings import get_settings

    get_settings.cache_clear()
    get_observability_service.cache_clear()
    service = create_observability_service(get_settings())
    assert service.enabled is False


def test_langfuse_init_failure_falls_back_to_noop():
    with patch("app.observability.backends.LangfuseObservabilityBackend", side_effect=RuntimeError("boom")):
        backend = create_observability_backend(
            enabled=True,
            public_key="pk-test",
            secret_key="sk-test",
            host="https://cloud.langfuse.com",
        )
    assert isinstance(backend, NoOpObservabilityBackend)


def test_research_trace_lifecycle_noop():
    backend = NoOpObservabilityBackend()
    from app.observability.service import ObservabilityService

    service = ObservabilityService(backend)
    handle = service.start_research_trace(question="What is RAG?", research_id="rid-1")
    token = service.bind_research_trace(handle)
    try:
        with service.run_agent_step(handle, agent_name="planner", input_payload={"q": "x"}):
            pass
        service.complete_research_trace(handle, research_id="rid-1", report_summary="done")
    finally:
        service.unbind_research_trace(token)

