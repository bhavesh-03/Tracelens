"""HTTP client helper for sending spans to the TraceLens ingest API.

Use this when:
  - You want to decouple your agent code from the tracelens package entirely
  - Your agent runs in a different process or container than TraceLens
  - You need to send spans from non-Python code (the API accepts plain JSON)

Usage:
    from tracelens.integrations.http_client import TraceLensHTTPClient

    client = TraceLensHTTPClient(endpoint="http://localhost:4318")
    
    client.push_span(
        trace_id="trace_abc",
        span_id="span_001",
        agent_name="RouterAgent",
        span_type="router",
        input_text=user_query,
        output_text=routing_decision,
    )
    
    client.finalize(
        trace_id="trace_abc",
        query=user_query,
        final_answer=final_answer,
    )
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import httpx


class TraceLensHTTPClient:
    """Lightweight HTTP client for pushing spans to the TraceLens ingest API.

    Thread-safe. Uses a persistent httpx client for connection reuse.
    All methods are synchronous — see AsyncTraceLensHTTPClient for async usage.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:4318",
        timeout: float = 5.0,
        project_name: str = "default",
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.project_name = project_name
        self._client = httpx.Client(timeout=timeout)

    def push_span(
        self,
        trace_id: str,
        agent_name: str,
        span_id: str | None = None,
        span_type: str = "agent",
        parent_span_ids: list[str] | None = None,
        input_text: str = "",
        output_text: str = "",
        tool_name: str | None = None,
        tool_args: dict[str, Any] | None = None,
        tool_output: str | None = None,
        model: str | None = None,
        start_time_ms: float | None = None,
        end_time_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        project_name: str | None = None,
    ) -> str:
        """Send one span to the ingest API. Returns the span_id.

        Fire-and-forget: any network error is caught and logged, never raised.
        This ensures instrumentation never crashes your agent.
        """
        now_ms = time.time() * 1000
        sid = span_id or f"span_{uuid.uuid4().hex[:8]}"

        payload = {
            "trace_id": trace_id,
            "span_id": sid,
            "project_name": project_name or self.project_name,
            "agent_name": agent_name,
            "span_type": span_type,
            "parent_span_ids": parent_span_ids or [],
            "input_text": input_text,
            "output_text": output_text,
            "tool_name": tool_name,
            "tool_args": tool_args or {},
            "tool_output": tool_output,
            "model": model,
            "start_time_ms": start_time_ms or now_ms,
            "end_time_ms": end_time_ms or now_ms,
            "metadata": metadata or {},
            "tags": tags or [],
        }

        try:
            resp = self._client.post(f"{self.endpoint}/v1/spans", json=payload)
            resp.raise_for_status()
        except Exception as e:
            # Never crash the agent — just log and continue
            import logging
            logging.getLogger(__name__).warning(f"TraceLens span push failed: {e}")

        return sid

    def finalize(
        self,
        trace_id: str,
        query: str,
        final_answer: str,
        expected_answer: str | None = None,
        tags: list[str] | None = None,
        run_diagnosis: bool = True,
    ) -> dict:
        """Finalize a trace and trigger background diagnosis.

        Returns the API response dict. Raises httpx.HTTPError on failure.
        """
        payload = {
            "query": query,
            "final_answer": final_answer,
            "expected_answer": expected_answer,
            "tags": tags or [],
            "run_diagnosis": run_diagnosis,
        }
        resp = self._client.post(
            f"{self.endpoint}/v1/traces/{trace_id}/finalize",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "TraceLensHTTPClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class AsyncTraceLensHTTPClient:
    """Async HTTP client for pushing spans — use in async agent frameworks."""

    def __init__(
        self,
        endpoint: str = "http://localhost:4318",
        timeout: float = 5.0,
        project_name: str = "default",
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.project_name = project_name
        self._client = httpx.AsyncClient(timeout=timeout)

    async def push_span(self, **kwargs: Any) -> str:
        """Async version of TraceLensHTTPClient.push_span."""
        now_ms = time.time() * 1000
        sid = kwargs.pop("span_id", None) or f"span_{uuid.uuid4().hex[:8]}"
        payload = {
            "trace_id": kwargs["trace_id"],
            "span_id": sid,
            "project_name": kwargs.get("project_name", self.project_name),
            "agent_name": kwargs["agent_name"],
            "span_type": kwargs.get("span_type", "agent"),
            "parent_span_ids": kwargs.get("parent_span_ids", []),
            "input_text": kwargs.get("input_text", ""),
            "output_text": kwargs.get("output_text", ""),
            "tool_name": kwargs.get("tool_name"),
            "tool_args": kwargs.get("tool_args", {}),
            "tool_output": kwargs.get("tool_output"),
            "model": kwargs.get("model"),
            "start_time_ms": kwargs.get("start_time_ms", now_ms),
            "end_time_ms": kwargs.get("end_time_ms", now_ms),
            "metadata": kwargs.get("metadata", {}),
            "tags": kwargs.get("tags", []),
        }
        try:
            resp = await self._client.post(f"{self.endpoint}/v1/spans", json=payload)
            resp.raise_for_status()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"TraceLens async span push failed: {e}")
        return sid

    async def finalize(self, trace_id: str, query: str, final_answer: str, **kwargs: Any) -> dict:
        """Async finalize."""
        payload = {
            "query": query,
            "final_answer": final_answer,
            "expected_answer": kwargs.get("expected_answer"),
            "tags": kwargs.get("tags", []),
            "run_diagnosis": kwargs.get("run_diagnosis", True),
        }
        resp = await self._client.post(
            f"{self.endpoint}/v1/traces/{trace_id}/finalize",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncTraceLensHTTPClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()
