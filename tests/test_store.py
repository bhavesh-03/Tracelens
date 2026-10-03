"""Tests for tracelens.store — SQLite round-trip and query operations."""

from __future__ import annotations

from pathlib import Path

import pytest

from tracelens.schema import Diagnosis, StepAttribution, StepIO, Trace, TraceStep
from tracelens.store import (
    connect,
    list_traces,
    load_diagnosis,
    load_trace,
    purge_expired_data,
    save_diagnosis,
    save_span,
    save_trace,
)


def _make_trace(trace_id: str = "t1") -> Trace:
    return Trace(
        trace_id=trace_id,
        query="test query",
        final_answer="test answer",
        expected_answer="expected",
        tags=["test", "unit"],
        steps=[
            TraceStep(
                step_id="s1",
                agent_name="router",
                step_type="router",
                parent_step_id=None,
                io=StepIO(input_text="q", output_text="agent_a"),
            ),
            TraceStep(
                step_id="s2",
                agent_name="agent_a",
                step_type="agent",
                parent_step_id="s1",
                io=StepIO(
                    input_text="q",
                    output_text="answer",
                    tool_name="search",
                    tool_args={"query": "test"},
                    tool_output="search results",
                ),
            ),
        ],
    )


class TestSaveAndLoad:
    def test_round_trip(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = connect(db_path)

        trace = _make_trace()
        row_id = save_trace(conn, trace)
        assert row_id is not None

        loaded = load_trace(conn, "t1")
        assert loaded["trace_id"] == "t1"
        assert loaded["query"] == "test query"
        assert loaded["final_answer"] == "test answer"
        assert loaded["expected_answer"] == "expected"
        assert len(loaded["steps"]) == 2
        conn.close()

    def test_step_data_preserved(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = connect(db_path)
        save_trace(conn, _make_trace())

        loaded = load_trace(conn, "t1")
        step = loaded["steps"][1]  # agent_a step
        assert step["agent_name"] == "agent_a"
        assert step["io"]["tool_name"] == "search"
        assert step["io"]["tool_output"] == "search results"
        conn.close()

    def test_repeated_save_is_idempotent(self, tmp_path: Path) -> None:
        conn = connect(tmp_path / "test.db")
        trace = _make_trace()
        first_id = save_trace(conn, trace)
        second_id = save_trace(conn, trace)
        assert first_id == second_id
        assert len(load_trace(conn, "t1")["steps"]) == 2
        conn.close()

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = connect(db_path)
        with pytest.raises(ValueError, match="not found"):
            load_trace(conn, "nonexistent")
        conn.close()

    def test_redacts_sensitive_content_before_persistence(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = connect(db_path)
        trace = _make_trace()
        trace.query = "Contact alice@example.com with password=hunter2"
        trace.steps[0].io.output_text = "Authorization: Bearer secret-token-value"

        save_trace(conn, trace)

        loaded = load_trace(conn, "t1")
        assert "alice@example.com" not in loaded["query"]
        assert "hunter2" not in loaded["query"]
        assert "secret-token-value" not in loaded["steps"][0]["io"]["output_text"]
        conn.close()


class TestListTraces:
    def test_list_returns_all(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = connect(db_path)

        save_trace(conn, _make_trace("t1"))
        save_trace(conn, _make_trace("t2"))

        traces = list_traces(conn)
        assert len(traces) == 2
        conn.close()

    def test_list_empty(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = connect(db_path)
        traces = list_traces(conn)
        assert traces == []
        conn.close()


class TestRetention:
    def test_purges_expired_traces_and_buffered_spans(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = connect(db_path)
        save_trace(conn, _make_trace("expired"))
        save_span(
            conn,
            {
                "trace_id": "buffered",
                "span_id": "buffered-step",
                "agent_name": "agent",
            },
        )
        with conn:
            conn.execute("UPDATE traces SET created_at = '2000-01-01T00:00:00+00:00'")
            conn.execute("UPDATE span_buffer SET received_at = '2000-01-01T00:00:00+00:00'")

        counts = purge_expired_data(conn, retention_days=30)

        assert counts == {"traces": 1, "buffered_spans": 1}
        assert list_traces(conn) == []
        conn.close()


class TestDiagnosisPersistence:
    def test_save_and_load_diagnosis(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = connect(db_path)
        save_trace(conn, _make_trace("t1"))

        diagnosis = Diagnosis(
            trace_id="t1",
            root_cause_step=StepAttribution(
                step_id="s2",
                agent_name="agent_a",
                step_type="agent",
                attribution_score=0.85,
            ),
            summary="Agent A hallucinated the answer",
            diagnosed_at="2026-07-30T12:00:00Z",
        )
        save_diagnosis(conn, diagnosis)

        loaded = load_diagnosis(conn, "t1")
        assert loaded is not None
        assert loaded["root_cause_step_id"] == "s2"
        assert loaded["root_cause_agent"] == "agent_a"
        assert loaded["attribution_score"] == 0.85
        assert loaded["summary"] == "Agent A hallucinated the answer"
        conn.close()

    def test_list_uses_latest_diagnosis_once(self, tmp_path: Path) -> None:
        conn = connect(tmp_path / "test.db")
        save_trace(conn, _make_trace("t1"))
        for score in (0.2, 0.8):
            save_diagnosis(
                conn,
                Diagnosis(
                    trace_id="t1",
                    root_cause_step=StepAttribution(
                        step_id="s2", agent_name="agent_a", step_type="agent",
                        attribution_score=score,
                    ),
                    diagnosed_at=f"2026-01-0{int(score * 10)}T00:00:00+00:00",
                ),
            )
        traces = list_traces(conn)
        assert len(traces) == 1
        assert traces[0]["attribution_score"] == 0.8
        conn.close()

    def test_no_diagnosis_returns_none(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = connect(db_path)
        save_trace(conn, _make_trace("t1"))
        assert load_diagnosis(conn, "t1") is None
        conn.close()
