"""Tests for the claim decomposition engine."""

from unittest.mock import MagicMock, patch

from tracelens.claims import decompose_into_claims
from tracelens.config import TraceLensConfig


def test_empty_text_returns_empty_list() -> None:
    config = TraceLensConfig()
    claims = decompose_into_claims("   ", "step_1", config)
    assert claims == []

@patch("tracelens.claims.litellm.completion")
def test_decompose_valid_json(mock_completion: MagicMock) -> None:
    # Setup mock response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"claims": ["fact 1", "fact 2"]}'
    mock_completion.return_value = mock_response

    config = TraceLensConfig()
    claims = decompose_into_claims("Some text", "step_001", config)

    assert len(claims) == 2
    assert claims[0].claim_id == "step_001_claim_000"
    assert claims[0].text == "fact 1"
    assert claims[0].source_step_id == "step_001"
    
    assert claims[1].claim_id == "step_001_claim_001"
    assert claims[1].text == "fact 2"
    assert claims[1].source_step_id == "step_001"

@patch("tracelens.claims.litellm.completion")
def test_decompose_with_markdown_blocks(mock_completion: MagicMock) -> None:
    # Setup mock response with markdown JSON block (which LLMs sometimes do)
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '```json\n{"claims": ["fact 1"]}\n```'
    mock_completion.return_value = mock_response

    config = TraceLensConfig()
    claims = decompose_into_claims("text", "step_001", config)

    assert len(claims) == 1
    assert claims[0].text == "fact 1"

@patch("tracelens.claims.litellm.completion")
def test_decompose_max_claims_limit(mock_completion: MagicMock) -> None:
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    # Return 5 claims
    mock_response.choices[0].message.content = '{"claims": ["1", "2", "3", "4", "5"]}'
    mock_completion.return_value = mock_response

    # Configure limit to 3
    config = TraceLensConfig(max_claims_per_step=3)
    claims = decompose_into_claims("text", "step_001", config)

    assert len(claims) == 3
    assert claims[-1].text == "3"

@patch("tracelens.claims.litellm.completion")
def test_decompose_fallback_on_error(mock_completion: MagicMock) -> None:
    # Force an exception
    mock_completion.side_effect = Exception("API offline")

    config = TraceLensConfig()
    claims = decompose_into_claims("This is the raw text.", "step_999", config)

    # Should fallback to a single claim containing the raw text
    assert len(claims) == 1
    assert claims[0].claim_id == "step_999_claim_fallback"
    assert claims[0].text == "This is the raw text."
    assert claims[0].source_step_id == "step_999"
