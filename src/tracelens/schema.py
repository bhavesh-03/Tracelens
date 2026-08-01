"""Pydantic v2 models for Trace, TraceStep, StepIO, Claim, and diagnosis results.

These are the user-facing data models validated on trace ingestion. Internal
result types (StepAttribution, Diagnosis) are dataclasses — same split as
EvalGate uses between EvalCase (Pydantic) and TrialResult (dataclass).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Trace input models (user-facing, validated on ingestion)
# ---------------------------------------------------------------------------


class StepIO(BaseModel):
    """The input received and output produced by one execution step."""

    model_config = ConfigDict(extra="ignore")

    input_text: str = ""
    output_text: str = ""
    model: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] = Field(default_factory=dict)
    tool_output: str | None = None


class TraceStep(BaseModel):
    """One node in the execution DAG — represents an agent call, tool call,
    or any processing step in a multi-agent pipeline.

    Attributes:
        step_id: Unique identifier within this trace (e.g., "step_003").
        agent_name: Which agent or component ran this step.
        step_type: Category of step for filtering and display.
        parent_step_id: The step that invoked this one (None for root).
        io: The input/output pair for this step.
        timestamp_ms: Unix epoch milliseconds when this step started.
        duration_ms: Wall-clock time this step took.
        metadata: Arbitrary key-value pairs (token counts, model params, etc.).
    """

    model_config = ConfigDict(extra="ignore")

    step_id: str
    agent_name: str
    step_type: Literal["router", "agent", "tool", "synthesizer", "llm_call", "custom"]
    parent_step_id: str | None = None
    io: StepIO = Field(default_factory=StepIO)
    timestamp_ms: float = 0.0
    duration_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("step_id")
    @classmethod
    def valid_step_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("step_id must not be empty")
        return v

    @field_validator("agent_name")
    @classmethod
    def valid_agent_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("agent_name must not be empty")
        return v


class Trace(BaseModel):
    """A complete execution trace — one user query through the multi-agent pipeline.

    Attributes:
        trace_id: Globally unique identifier for this trace.
        query: The original user query that started the execution.
        final_answer: The system's final response to the user.
        steps: Ordered list of execution steps (chronological).
        expected_answer: Ground truth answer (if known) for accuracy checking.
        tags: Free-form labels for filtering (e.g., ["production", "regression"]).
    """

    model_config = ConfigDict(extra="ignore")

    trace_id: str
    project_name: str = "default"
    query: str
    final_answer: str
    steps: list[TraceStep]
    expected_answer: str | None = None
    tags: list[str] = Field(default_factory=list)

    @field_validator("trace_id")
    @classmethod
    def valid_trace_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("trace_id must not be empty")
        return v

    @field_validator("steps")
    @classmethod
    def at_least_one_step(cls, v: list[TraceStep]) -> list[TraceStep]:
        if not v:
            raise ValueError("a trace must contain at least one step")
        return v

    @model_validator(mode="after")
    def parent_ids_are_valid(self) -> Trace:
        """Every non-root step's parent_step_id must reference an existing step."""
        valid_ids = {s.step_id for s in self.steps}
        for step in self.steps:
            if step.parent_step_id is not None and step.parent_step_id not in valid_ids:
                raise ValueError(
                    f"step {step.step_id!r} references parent {step.parent_step_id!r} "
                    f"which does not exist in this trace"
                )
        return self

    @model_validator(mode="after")
    def exactly_one_root(self) -> Trace:
        """Exactly one step must have parent_step_id=None (the root)."""
        roots = [s for s in self.steps if s.parent_step_id is None]
        if len(roots) == 0:
            raise ValueError("trace has no root step (a step with parent_step_id=None)")
        if len(roots) > 1:
            root_ids = [r.step_id for r in roots]
            raise ValueError(
                f"trace has {len(roots)} root steps ({root_ids}), expected exactly 1"
            )
        return self

    def get_step(self, step_id: str) -> TraceStep:
        """Look up a step by its ID. Raises ValueError if not found."""
        for s in self.steps:
            if s.step_id == step_id:
                return s
        raise ValueError(f"step {step_id!r} not found in trace {self.trace_id!r}")

    @property
    def root_step(self) -> TraceStep:
        """The single root step (parent_step_id is None)."""
        for s in self.steps:
            if s.parent_step_id is None:
                return s
        raise ValueError("no root step found")  # unreachable after validation

    @property
    def leaf_steps(self) -> list[TraceStep]:
        """Steps that are not the parent of any other step."""
        parent_ids = {s.parent_step_id for s in self.steps}
        return [s for s in self.steps if s.step_id not in parent_ids]

    def to_json_file(self, path: str | Path) -> Path:
        """Serialize trace to a formatted JSON file."""
        from pathlib import Path as _Path
        p = _Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return p

    @classmethod
    def from_json_file(cls, path: str | Path) -> Trace:
        """Load and validate a Trace from a JSON file."""
        from pathlib import Path as _Path
        p = _Path(path)
        if not p.exists():
            raise ValueError(f"Trace file {path} not found")
        return cls.model_validate_json(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Claim models (populated by the claim decomposition engine)
# ---------------------------------------------------------------------------


@dataclass
class Claim:
    """An atomic factual statement extracted from a step's output.

    Attributes:
        claim_id: Unique identifier (e.g., "step_003_claim_002").
        text: The atomic claim text.
        source_step_id: Which step produced this claim.
        grounded: Whether this claim is supported by the parent step's output.
        verdict: Verification result from the entailment checker.
        evidence: The specific parent text that supports or contradicts.
        confidence: Verification confidence score (0.0 to 1.0).
    """

    claim_id: str
    text: str
    source_step_id: str
    grounded: bool | None = None
    verdict: Literal["grounded", "ungrounded", "uncertain"] | None = None
    evidence: str | None = None
    confidence: float = 0.0
    agreement_score: float = 0.0
    vote_breakdown: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "source_step_id": self.source_step_id,
            "grounded": self.grounded,
            "verdict": self.verdict,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "agreement_score": self.agreement_score,
            "vote_breakdown": self.vote_breakdown,
        }


# ---------------------------------------------------------------------------
# Diagnosis result types (internal, populated by attribution engine)
# ---------------------------------------------------------------------------


@dataclass
class StepAttribution:
    """Attribution score for one step in the execution DAG.

    Attributes:
        step_id: Which step this attribution is for.
        agent_name: The agent that ran this step.
        step_type: Category of step.
        attribution_score: 0.0 (innocent) to 1.0 (definitive root cause).
        novel_claim_ratio: Fraction of this step's claims that are ungrounded.
        downstream_impact: Fraction of the final answer that depends on this step.
        novel_claims: List of claims introduced by this step without parent support.
        total_claims: Total number of claims decomposed from this step's output.
    """

    step_id: str
    agent_name: str
    step_type: str
    attribution_score: float = 0.0
    novel_claim_ratio: float = 0.0
    downstream_impact: float = 0.0
    novel_claims: list[Claim] = field(default_factory=list)
    total_claims: int = 0


@dataclass
class Diagnosis:
    """Complete diagnostic result for one trace.

    Attributes:
        trace_id: The trace that was diagnosed.
        root_cause_step: The step with the highest attribution score.
        all_steps: All steps ranked by attribution score (descending).
        summary: Human-readable one-paragraph explanation.
        diagnosed_at: ISO timestamp of when the diagnosis was computed.
    """

    trace_id: str
    root_cause_step: StepAttribution | None = None
    all_steps: list[StepAttribution] = field(default_factory=list)
    summary: str = ""
    diagnosed_at: str = ""

    @property
    def has_root_cause(self) -> bool:
        """True if a step was identified as the root cause."""
        return self.root_cause_step is not None

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "root_cause_step_id": (
                self.root_cause_step.step_id if self.root_cause_step else None
            ),
            "root_cause_agent": (
                self.root_cause_step.agent_name if self.root_cause_step else None
            ),
            "root_cause_score": (
                self.root_cause_step.attribution_score if self.root_cause_step else None
            ),
            "summary": self.summary,
            "diagnosed_at": self.diagnosed_at,
            "steps": [
                {
                    "step_id": s.step_id,
                    "agent_name": s.agent_name,
                    "attribution_score": s.attribution_score,
                    "novel_claim_ratio": s.novel_claim_ratio,
                    "downstream_impact": s.downstream_impact,
                    "total_claims": s.total_claims,
                    "novel_claims": [c.to_dict() for c in s.novel_claims],
                }
                for s in self.all_steps
            ],
        }
