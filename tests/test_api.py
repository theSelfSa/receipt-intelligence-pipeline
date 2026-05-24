from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
import app.main as main_module


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    async def _noop(*_args, **_kwargs) -> None:
        return None

    async def _healthy(*_args, **_kwargs) -> bool:
        return True

    settings = main_module.get_settings()
    settings.upload_dir = tmp_path

    monkeypatch.setattr(main_module, "init_database", _noop)
    monkeypatch.setattr(main_module, "close_database", _noop)
    monkeypatch.setattr(main_module, "check_database_health", _healthy)
    monkeypatch.setattr(main_module, "check_redis_health", _healthy)

    with TestClient(app) as test_client:
        yield test_client


def test_health_returns_ok_when_database_is_available(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok", "redis": "ok"}


def test_health_returns_degraded_when_database_is_unavailable(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _unhealthy(*_args, **_kwargs) -> bool:
        return False

    monkeypatch.setattr(main_module, "check_database_health", _unhealthy)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "degraded", "database": "down", "redis": "ok"}


def test_health_returns_degraded_when_redis_is_unavailable(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _unhealthy(*_args, **_kwargs) -> bool:
        return False

    monkeypatch.setattr(main_module, "check_redis_health", _unhealthy)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "degraded", "database": "ok", "redis": "down"}


def test_metrics_endpoint_returns_prometheus_payload(client: TestClient) -> None:
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "receipt_upload_total" in response.text
