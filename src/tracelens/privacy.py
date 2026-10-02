"""Redaction helpers for protecting trace content before persistence."""

from __future__ import annotations

import re
from typing import Any

from tracelens.config import TraceLensConfig
from tracelens.schema import Trace

_REDACTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL),
    re.compile(r"\bAIza[\w-]{35}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{12,}\b"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|secret|password)"
        r"\b\s*[:=]\s*[^\s,;]+"
    ),
)
_EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)


def redact_text(value: str, config: TraceLensConfig) -> str:
    """Remove common secrets and email addresses from a text value."""
    if not config.redact_sensitive_data:
        return value

    redacted = value
    for pattern in _REDACTION_PATTERNS:
        redacted = pattern.sub(config.redaction_replacement, redacted)
    return _EMAIL_PATTERN.sub("[REDACTED_EMAIL]", redacted)


def redact_value(value: Any, config: TraceLensConfig) -> Any:
    """Recursively redact strings in JSON-compatible payloads."""
    if isinstance(value, str):
        return redact_text(value, config)
    if isinstance(value, dict):
        return {str(key): redact_value(item, config) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_value(item, config) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_value(item, config) for item in value)
    return value


def redact_trace(trace: Trace, config: TraceLensConfig) -> Trace:
    """Return a validated copy of ``trace`` with sensitive text redacted."""
    if not config.redact_sensitive_data:
        return trace
    return Trace.model_validate(redact_value(trace.model_dump(), config))
