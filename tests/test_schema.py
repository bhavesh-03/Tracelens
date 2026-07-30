"""Tests for tracelens.schema — trace, step, and claim data model validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tracelens.schema import (
    Claim,
    Diagnosis,
    StepAttribution,
    StepIO,
    Trace,
    TraceStep,
)


def _make_step(
    step_id: str = "step_000",
    agent_name: str = "test_agent",
    step_type: str = "agent",
    parent_step_id: str | None = None,
    output_text: str = "test output",
) -> TraceStep:
    """Helper to create a valid TraceStep with minimal boilerplate."""
    return TraceStep(
        step_id=step_id,
        agent_name=agent_name,
        step_type=step_type,
        parent_step_id=parent_step_id,
        io=StepIO(input_text="test input", output_text=output_text),
    )


class TestTraceStepValidation:
    def test_valid_step(self) -> None:
        step = _make_step()
        assert step.step_id == "step_000"
        assert step.agent_name == "test_agent"

    def test_empty_step_id_raises(self) -> None:
        with pytest.raises(ValidationError, match="step_id"):
            _make_step(step_id="")

    def test_empty_agent_name_raises(self) -> None:
        with pytest.raises(ValidationError, match="agent_name"):
            _make_step(agent_name="")

    def test_whitespace_step_id_raises(self) -> None:
        with pytest.raises(ValidationError, match="step_id"):
            _make_step(step_id="   ")

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs"):
            TraceStep(
                step_id="s1",
                agent_name="a",
                step_type="agent",
                io=StepIO(),
                bogus_field="should fail",
            )


class TestTraceValidation:
    def test_valid_trace(self) -> None:
        trace = Trace(
            trace_id="t1",
            query="test query",
            final_answer="test answer",
            steps=[_make_step()],
        )
        assert trace.trace_id == "t1"
        assert len(trace.steps) == 1

    def test_empty_trace_id_raises(self) -> None:
        with pytest.raises(ValidationError, match="trace_id"):
            Trace(
                trace_id="",
                query="q",
                final_answer="a",
                steps=[_make_step()],
            )

    def test_no_steps_raises(self) -> None:
        with pytest.raises(ValidationError, match="at least one step"):
            Trace(
                trace_id="t1",
                query="q",
                final_answer="a",
                steps=[],
            )

    def test_multiple_roots_raises(self) -> None:
        with pytest.raises(ValidationError, match="root steps"):
            Trace(
                trace_id="t1",
                query="q",
                final_answer="a",
                steps=[
                    _make_step(step_id="s1"),  # root 1
                    _make_step(step_id="s2"),  # root 2 (also no parent)
                ],
            )

    def test_invalid_parent_reference_raises(self) -> None:
        with pytest.raises(ValidationError, match="does not exist"):
            Trace(
                trace_id="t1",
                query="q",
                final_answer="a",
                steps=[
                    _make_step(step_id="s1"),
                    _make_step(step_id="s2", parent_step_id="nonexistent"),
                ],
            )

    def test_valid_parent_chain(self) -> None:
        trace = Trace(
            trace_id="t1",
            query="q",
            final_answer="a",
            steps=[
                _make_step(step_id="s1"),
                _make_step(step_id="s2", parent_step_id="s1"),
                _make_step(step_id="s3", parent_step_id="s1"),
            ],
        )
        assert len(trace.steps) == 3

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs"):
            Trace(
                trace_id="t1",
                query="q",
                final_answer="a",
                steps=[_make_step()],
                bogus="fail",
            )


class TestTraceProperties:
    def test_root_step(self) -> None:
        trace = Trace(
            trace_id="t1",
            query="q",
            final_answer="a",
            steps=[
                _make_step(step_id="root"),
                _make_step(step_id="child", parent_step_id="root"),
            ],
        )
        assert trace.root_step.step_id == "root"

    def test_leaf_steps(self) -> None:
        trace = Trace(
            trace_id="t1",
            query="q",
            final_answer="a",
            steps=[
                _make_step(step_id="root"),
                _make_step(step_id="leaf1", parent_step_id="root"),
                _make_step(step_id="leaf2", parent_step_id="root"),
            ],
        )
        leaf_ids = [s.step_id for s in trace.leaf_steps]
        assert set(leaf_ids) == {"leaf1", "leaf2"}

    def test_get_step(self) -> None:
        trace = Trace(
            trace_id="t1",
            query="q",
            final_answer="a",
            steps=[_make_step(step_id="s1")],
        )
        assert trace.get_step("s1").step_id == "s1"

    def test_get_step_missing_raises(self) -> None:
        trace = Trace(
            trace_id="t1",
            query="q",
            final_answer="a",
            steps=[_make_step(step_id="s1")],
        )
        with pytest.raises(ValueError, match="not found"):
            trace.get_step("nonexistent")


class TestClaimDataclass:
    def test_claim_to_dict(self) -> None:
        claim = Claim(
            claim_id="c1",
            text="Revenue was $80B",
            source_step_id="s1",
            grounded=True,
            verdict="grounded",
            evidence="Revenue reported at $80B in Q1",
            confidence=0.95,
        )
        d = claim.to_dict()
        assert d["claim_id"] == "c1"
        assert d["grounded"] is True
        assert d["confidence"] == 0.95

    def test_claim_defaults(self) -> None:
        claim = Claim(claim_id="c1", text="test", source_step_id="s1")
        assert claim.grounded is None
        assert claim.verdict is None
        assert claim.confidence == 0.0


class TestDiagnosis:
    def test_diagnosis_without_root_cause(self) -> None:
        d = Diagnosis(trace_id="t1")
        assert not d.has_root_cause
        assert d.to_dict()["root_cause_step_id"] is None

    def test_diagnosis_with_root_cause(self) -> None:
        root = StepAttribution(
            step_id="s3",
            agent_name="security_auditor",
            step_type="agent",
            attribution_score=0.91,
        )
        d = Diagnosis(trace_id="t1", root_cause_step=root, all_steps=[root])
        assert d.has_root_cause
        assert d.to_dict()["root_cause_agent"] == "security_auditor"
        assert d.to_dict()["root_cause_score"] == 0.91
