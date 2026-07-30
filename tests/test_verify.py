"""Tests for step-level entailment verification."""

import json
from unittest.mock import MagicMock, patch

from tracelens.config import TraceLensConfig
from tracelens.schema import Claim, StepIO, TraceStep
from tracelens.verify import verify_claim


def _make_mock_step() -> TraceStep:
    return TraceStep(
        step_id="step_000",
        agent_name="tool_agent",
        step_type="tool",
        parent_step_id=None,
        io=StepIO(
            input_text="Check if file exists",
            output_text="File does not exist.",
            tool_output="ERROR: File not found",
        ),
        timestamp_ms=0.0,
        duration_ms=10.0,
        metadata={}
    )

@patch("tracelens.verify.litellm.completion")
def test_verify_grounded_claim(mock_completion: MagicMock) -> None:
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps({
        "verdict": "grounded",
        "evidence": "ERROR: File not found",
        "confidence": 0.95
    })
    mock_completion.return_value = mock_response

    claim = Claim(claim_id="c1", text="The file is missing.", source_step_id="step_001")
    parent = _make_mock_step()
    config = TraceLensConfig()

    verified_claim = verify_claim(claim, parent, config)
    
    assert verified_claim.verdict == "grounded"
    assert verified_claim.grounded is True
    assert verified_claim.confidence == 0.95
    assert "File not found" in verified_claim.evidence

@patch("tracelens.verify.litellm.completion")
def test_verify_ungrounded_claim(mock_completion: MagicMock) -> None:
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps({
        "verdict": "ungrounded",
        "evidence": "None",
        "confidence": 0.99
    })
    mock_completion.return_value = mock_response

    claim = Claim(
        claim_id="c2",
        text="The file exists and contains a password.",
        source_step_id="step_001"
    )
    parent = _make_mock_step()
    config = TraceLensConfig()

    verified_claim = verify_claim(claim, parent, config)
    
    assert verified_claim.verdict == "ungrounded"
    assert verified_claim.grounded is False
    assert verified_claim.confidence == 0.99

@patch("tracelens.verify.litellm.completion")
def test_verify_fallback_on_error(mock_completion: MagicMock) -> None:
    mock_completion.side_effect = Exception("Network timeout")

    claim = Claim(claim_id="c3", text="Random claim.", source_step_id="step_001")
    parent = _make_mock_step()
    config = TraceLensConfig()

    verified_claim = verify_claim(claim, parent, config)
    
    assert verified_claim.verdict == "uncertain"
    assert verified_claim.grounded is None
    assert verified_claim.confidence == 0.0
    assert "Network timeout" in verified_claim.evidence
