"""Tests for trace-data redaction."""

from tracelens.config import TraceLensConfig
from tracelens.privacy import redact_text, redact_value


def test_redacts_common_secrets_and_emails() -> None:
    text = (
        "Email alice@example.com; password=hunter2; "
        "Authorization: Bearer secret-token-value"
    )
    redacted = redact_text(text, TraceLensConfig())

    assert "alice@example.com" not in redacted
    assert "hunter2" not in redacted
    assert "secret-token-value" not in redacted


def test_redacts_nested_payloads() -> None:
    payload = {"metadata": {"owner": "alice@example.com"}, "items": ["sk-abcdefghijk01234567890"]}
    redacted = redact_value(payload, TraceLensConfig())

    assert redacted["metadata"]["owner"] == "[REDACTED_EMAIL]"
    assert redacted["items"] == ["[REDACTED]"]


def test_redaction_can_be_disabled() -> None:
    text = "Email alice@example.com"
    assert redact_text(text, TraceLensConfig(redact_sensitive_data=False)) == text
