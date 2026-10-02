"""Tests for HTTP client delivery behavior."""

import httpx
import pytest

from tracelens.integrations.http_client import TraceLensHTTPClient


def test_push_span_raises_by_default_on_delivery_failure(monkeypatch) -> None:
    client = TraceLensHTTPClient()
    monkeypatch.setattr(
        client._client,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(httpx.ConnectError("offline")),
    )

    with pytest.raises(httpx.ConnectError):
        client.push_span(trace_id="t1", agent_name="agent")
    client.close()


def test_best_effort_delivery_is_explicit_opt_in(monkeypatch) -> None:
    client = TraceLensHTTPClient(raise_on_error=False)
    monkeypatch.setattr(
        client._client,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(httpx.ConnectError("offline")),
    )

    assert client.push_span(trace_id="t1", agent_name="agent", span_id="s1") == "s1"
    client.close()
