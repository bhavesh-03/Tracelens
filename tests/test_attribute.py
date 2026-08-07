"""Tests for the causal attribution engine."""

from unittest.mock import patch

import pytest

from tracelens.attribute import compute_p_ungrounded, diagnose_trace
from tracelens.config import TraceLensConfig
from tracelens.schema import Claim, StepIO, Trace, TraceStep


def test_compute_p_ungrounded() -> None:
    # Grounded with high confidence -> low probability of being ungrounded
    c1 = Claim(claim_id="1", text="x", source_step_id="s1", verdict="grounded", confidence=0.9)
    assert compute_p_ungrounded(c1) == pytest.approx(0.1)

    # Ungrounded with high confidence -> high probability of being ungrounded
    c2 = Claim(claim_id="2", text="x", source_step_id="s1", verdict="ungrounded", confidence=0.8)
    assert compute_p_ungrounded(c2) == pytest.approx(0.8)

    # Uncertain
    c3 = Claim(claim_id="3", text="x", source_step_id="s1", verdict="uncertain", confidence=0.5)
    assert compute_p_ungrounded(c3) == pytest.approx(0.5)

@patch("tracelens.attribute.verify_claim_ensemble")
@patch("tracelens.attribute.decompose_into_claims")
def test_diagnose_trace_linear_hallucination(mock_decompose, mock_verify) -> None:
    """Test a linear A -> B -> C trace where B hallucinates."""
    
    # 3 Steps: Root -> Agent -> Tool
    # Wait, DAG is parent-child.
    # Coordinator (s1) -> Linter (s2)
    s2 = TraceStep(
        step_id="s2", agent_name="Linter", step_type="agent", parent_step_id="s1",
        io=StepIO(input_text="code", output_text="Linter says syntax is bad")
    )
    s1 = TraceStep(
        step_id="s1", agent_name="Coordinator", step_type="router", parent_step_id=None,
        io=StepIO(input_text="code", output_text="Final: Syntax is bad")
    )
    trace = Trace(trace_id="t1", query="code", final_answer="bad", steps=[s1, s2])

    config = TraceLensConfig(attribution_threshold=0.4)

    # Mock decompose: return 1 claim per step
    def fake_decompose(text, step_id, config=None):
        return [Claim(claim_id=f"{step_id}_c1", text="Syntax is bad", source_step_id=step_id)]
    mock_decompose.side_effect = fake_decompose

    # Mock verify: 
    # Linter (s2) hallucinates (it had no tool output, just guessed)
    # Coordinator (s1) is innocent (it just copied Linter)
    def fake_verify(claim, step, trace_steps, config):
        if claim.source_step_id == "s2":
            claim.verdict = "ungrounded"
            claim.confidence = 0.95
        else: # s1
            claim.verdict = "grounded"
            claim.confidence = 0.99
        return claim
    mock_verify.side_effect = fake_verify

    diagnosis = diagnose_trace(trace, config)
    
    # s2 (Linter) should be the root cause
    assert diagnosis.root_cause_step is not None
    assert diagnosis.root_cause_step.step_id == "s2"
    assert diagnosis.root_cause_step.agent_name == "Linter"
    
    # Check attribution scores
    s1_attr = next(a for a in diagnosis.all_steps if a.step_id == "s1")
    s2_attr = next(a for a in diagnosis.all_steps if a.step_id == "s2")
    
    # s1 score: expected ungrounded = 1.0 - 0.99 = 0.01. Ratio = 0.01. Impact = 0. Score = 0.01
    assert s1_attr.attribution_score < 0.1
    
    # s2 score: expected ungrounded = 0.95. Ratio = 0.95. Impact = 0.0. Score = 0.95 * 1.0 = 0.95
    assert s2_attr.attribution_score == 0.95
    assert s2_attr.novel_claim_ratio == 0.95
    assert len(s2_attr.novel_claims) == 1
