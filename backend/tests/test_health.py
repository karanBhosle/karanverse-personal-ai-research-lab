from fastapi.testclient import TestClient

from app.main import create_app


def test_health_returns_ok():
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "service" in body
    assert "environment" in body
    assert body["langfuse_enabled"] is False
    assert body["langfuse_configured"] is False
    assert body["langfuse_active"] is False
