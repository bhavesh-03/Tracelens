"""Security and privacy behavior for the HTTP ingest API."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from tracelens.config import TraceLensConfig
from tracelens.store import connect, load_trace


def _configure_server(monkeypatch, tmp_path: Path, **overrides):
    import tracelens.server as server

    cfg = TraceLensConfig(
        db_path=str(tmp_path / "server.db"),
        api_key="test-api-key",
        allowed_origins=("https://console.example.com",),
        **overrides,
    )
    conn = connect(cfg.db_path)
    monkeypatch.setattr(server, "_cfg", cfg)
    monkeypatch.setattr(server, "_conn", conn)
    monkeypatch.setattr(server, "_last_retention_cleanup", 0.0)
    return server, conn


def test_api_requires_key_and_limits_origins(monkeypatch, tmp_path: Path) -> None:
    server, conn = _configure_server(monkeypatch, tmp_path)
    client = TestClient(server.app)

    assert client.get("/v1/health").status_code == 200
    assert client.get("/v1/traces").status_code == 401
    assert client.get("/v1/traces", headers={"X-TraceLens-API-Key": "wrong"}).status_code == 401
    authorized = client.get("/v1/traces", headers={"X-TraceLens-API-Key": "test-api-key"})
    assert authorized.status_code == 200

    denied = client.options("/v1/traces", headers={"Origin": "https://untrusted.example"})
    assert denied.status_code == 403
    allowed = client.options("/v1/traces", headers={"Origin": "https://console.example.com"})
    assert allowed.status_code == 204
    assert allowed.headers["access-control-allow-origin"] == "https://console.example.com"
    conn.close()


def test_ingest_redacts_content_before_storage(monkeypatch, tmp_path: Path) -> None:
    server, conn = _configure_server(monkeypatch, tmp_path)
    client = TestClient(server.app)
    headers = {"X-TraceLens-API-Key": "test-api-key"}

    response = client.post(
        "/v1/spans",
        headers=headers,
        json={
            "trace_id": "private-trace",
            "span_id": "root",
            "agent_name": "Router",
            "span_type": "router",
            "input_text": "Email alice@example.com",
            "output_text": "password=hunter2",
        },
    )
    assert response.status_code == 200

    finalized = client.post(
        "/v1/traces/private-trace/finalize",
        headers=headers,
        json={
            "query": "Email alice@example.com",
            "final_answer": "password=hunter2",
            "run_diagnosis": False,
        },
    )
    assert finalized.status_code == 200

    stored = load_trace(conn, "private-trace")
    assert "alice@example.com" not in stored["query"]
    assert "hunter2" not in stored["final_answer"]
    assert "hunter2" not in stored["steps"][0]["io"]["output_text"]
    conn.close()
