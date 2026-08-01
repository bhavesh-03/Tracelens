"""TraceLens configuration — loads tracelens.toml from the working directory.

Uses tomllib (stdlib, Python 3.11+) for TOML parsing and plain dataclasses
for the config object. Pydantic is reserved for user-facing trace schemas
where validation error messages matter — config is internal and loaded once
at startup.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

_DEFAULT_CONFIG_NAME = "tracelens.toml"


@dataclass(frozen=True, slots=True)
class CostConfig:
    """Per-token cost estimates for reporting."""

    input_per_million: float = 0.075
    output_per_million: float = 0.30


@dataclass(frozen=True, slots=True)
class TraceLensConfig:
    """All TraceLens settings, loaded from tracelens.toml."""

    judge_model: str = "gemini/gemini-2.5-flash"
    max_claims_per_step: int = 20
    attribution_threshold: float = 0.7
    db_path: str = "tracelens.db"
    temperature: float = 0.1
    max_concurrent_verifications: int = 1
    verification_timeout_s: int = 60
    nli_ensemble_votes: int = 3
    nli_min_agreement: float = 0.67
    costs: CostConfig = field(default_factory=CostConfig)

    def validate(self) -> None:
        """Fail fast on obviously wrong config values."""
        if self.max_claims_per_step < 1:
            raise ValueError(
                f"max_claims_per_step must be >= 1, got {self.max_claims_per_step}"
            )
        if not 0.0 < self.attribution_threshold <= 1.0:
            raise ValueError(
                f"attribution_threshold must be in (0, 1], got {self.attribution_threshold}"
            )
        if self.max_concurrent_verifications < 1:
            raise ValueError(
                "max_concurrent_verifications must be >= 1, "
                f"got {self.max_concurrent_verifications}"
            )
        if not 1 <= self.nli_ensemble_votes <= 10:
            raise ValueError(
                f"nli_ensemble_votes must be in [1, 10], got {self.nli_ensemble_votes}"
            )
        if not 0.5 <= self.nli_min_agreement <= 1.0:
            raise ValueError(
                f"nli_min_agreement must be in [0.5, 1.0], got {self.nli_min_agreement}"
            )
        if self.verification_timeout_s < 1:
            raise ValueError(
                f"verification_timeout_s must be >= 1, got {self.verification_timeout_s}"
            )


def load_config(path: str | Path | None = None) -> TraceLensConfig:
    """Load config from a TOML file; falls back to defaults if not found.

    The search order is:
    1. Explicit `path` argument
    2. `tracelens.toml` in the current working directory
    3. All-defaults config (no file needed)
    """
    if path is None:
        path = Path.cwd() / _DEFAULT_CONFIG_NAME

    config_path = Path(path)
    if not config_path.exists():
        cfg = TraceLensConfig()
        cfg.validate()
        return cfg

    with open(config_path, "rb") as f:
        raw = tomllib.load(f)

    tl = raw.get("tracelens", {})
    costs_raw = raw.get("costs", {})

    costs = CostConfig(
        input_per_million=costs_raw.get("input_per_million", 0.075),
        output_per_million=costs_raw.get("output_per_million", 0.30),
    )

    cfg = TraceLensConfig(
        judge_model=tl.get("judge_model", "gemini/gemini-2.5-flash"),
        max_claims_per_step=tl.get("max_claims_per_step", 20),
        attribution_threshold=tl.get("attribution_threshold", 0.7),
        db_path=tl.get("db_path", "tracelens.db"),
        temperature=tl.get("temperature", 0.1),
        max_concurrent_verifications=tl.get("max_concurrent_verifications", 1),
        verification_timeout_s=tl.get("verification_timeout_s", 60),
        nli_ensemble_votes=tl.get("nli_ensemble_votes", 3),
        nli_min_agreement=tl.get("nli_min_agreement", 0.67),
        costs=costs,
    )
    cfg.validate()
    return cfg
