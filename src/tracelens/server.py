"""TraceLens HTTP Ingest API — FastAPI server.

Provides a fire-and-forget HTTP endpoint so any language (Python, Node.js,
Go, Java) can push spans to TraceLens without importing the SDK directly.

Endpoints:
    POST /v1/spans                     — buffer one agent span
    POST /v1/traces/{trace_id}/finalize — assemble + diagnose a trace
    GET  /v1/traces                    — list recent traces (used by dashboard)
    GET  /v1/traces/{trace_id}         — full trace + diagnosis
    GET  /v1/health                    — liveness check

Run via CLI:
    tracelens serve --port 4318

Or directly:
    uvicorn tracelens.server:app --port 4318 --reload

Port 4318 intentionally mirrors the OpenTelemetry OTLP HTTP port.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from tracelens.config import load_config
from tracelens.schema import StepIO, Trace, TraceStep
from tracelens.store import (
    connect,
    flush_span_buffer,
    list_traces,
    load_diagnosis,
    load_trace,
    save_diagnosis,
    save_span,
    save_trace,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="TraceLens Ingest API",
    description="Push agent spans to TraceLens for causal root-cause analysis.",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Dependency: shared config + DB connection
# ---------------------------------------------------------------------------

_cfg = None
_conn = None


def _get_cfg():
    global _cfg
    if _cfg is None:
        _cfg = load_config()
    return _cfg


def _get_conn():
    global _conn
    if _conn is None:
        _conn = connect(_get_cfg().db_path)
    return _conn


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class SpanPayload(BaseModel):
    """One agent execution span sent from the SDK or any HTTP client."""

    trace_id: str = Field(..., description="Shared across all spans in one user request")
    span_id: str = Field(..., description="Unique ID for this specific agent step")
    project_name: str = Field("default", description="Project name for dashboard filtering")
    agent_name: str = Field(..., description="Display name of the agent or component")
    span_type: str = Field(
        "agent",
        description="One of: router, agent, tool, synthesizer, llm_call, custom",
    )
    parent_span_ids: list[str] = Field(
        default_factory=list,
        description="IDs of parent spans (supports fan-in with multiple parents)",
    )
    input_text: str = Field("", description="What the agent received as input")
    output_text: str = Field("", description="What the agent produced as output")
    tool_name: str | None = Field(None, description="Tool called, if any")
    tool_args: dict[str, Any] = Field(default_factory=dict)
    tool_output: str | None = Field(None, description="Raw tool output")
    model: str | None = Field(None, description="LLM model used")
    start_time_ms: float = Field(
        default_factory=lambda: time.time() * 1000,
        description="Unix epoch milliseconds when this span started",
    )
    end_time_ms: float = Field(
        default_factory=lambda: time.time() * 1000,
        description="Unix epoch milliseconds when this span ended",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


class FinalizePayload(BaseModel):
    """Marks a trace as complete and triggers background diagnosis."""

    query: str = Field(..., description="The original user query")
    final_answer: str = Field(..., description="The pipeline's final response")
    expected_answer: str | None = Field(None, description="Ground truth, if known")
    tags: list[str] = Field(default_factory=list)
    run_diagnosis: bool = Field(
        True,
        description="Set to false to skip LLM diagnosis (just save the trace)",
    )


class SpanResponse(BaseModel):
    status: str
    span_id: str
    trace_id: str


class FinalizeResponse(BaseModel):
    status: str
    trace_id: str
    step_count: int
    diagnosis_status: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/v1/health")
async def health():
    """Liveness check."""
    return {"status": "ok", "version": "0.2.0"}


@app.post("/v1/spans", response_model=SpanResponse)
async def ingest_span(span: SpanPayload):
    """Buffer one agent span. Returns immediately (< 5ms).

    The span is stored in the span_buffer table until the trace is finalized.
    Spans may arrive out of order — that's fine, they're assembled at finalize time.
    """
    conn = _get_conn()
    save_span(conn, span.model_dump())
    logger.info(f"Buffered span {span.span_id} for trace {span.trace_id}")
    return SpanResponse(status="buffered", span_id=span.span_id, trace_id=span.trace_id)


@app.post("/v1/traces/{trace_id}/finalize", response_model=FinalizeResponse)
async def finalize_trace(
    trace_id: str,
    payload: FinalizePayload,
    background: BackgroundTasks,
):
    """Assemble all buffered spans into a Trace and trigger diagnosis.

    The endpoint returns immediately after saving the trace.
    Diagnosis runs in the background (non-blocking).
    """
    conn = _get_conn()

    # Flush buffered spans
    spans = flush_span_buffer(conn, trace_id)
    if not spans:
        raise HTTPException(
            status_code=404,
            detail=f"No buffered spans found for trace_id={trace_id!r}. "
            f"Send spans via POST /v1/spans first.",
        )

    # Reconstruct TraceStep objects from raw span dicts
    steps: list[TraceStep] = []
    for s in spans:
        parent_ids = json.loads(s.get("parent_span_ids_json") or "[]")
        step = TraceStep(
            step_id=s["span_id"],
            agent_name=s["agent_name"],
            step_type=s.get("span_type", "agent"),
            parent_span_ids=parent_ids,
            io=StepIO(
                input_text=s.get("input_text") or "",
                output_text=s.get("output_text") or "",
                tool_name=s.get("tool_name"),
                tool_args=json.loads(s.get("tool_args_json") or "{}"),
                tool_output=s.get("tool_output"),
                model=s.get("model"),
            ),
            timestamp_ms=s.get("start_time_ms") or 0.0,
            duration_ms=(
                (s.get("end_time_ms") or 0.0) - (s.get("start_time_ms") or 0.0)
            ),
            metadata=json.loads(s.get("metadata_json") or "{}"),
        )
        steps.append(step)

    # Get project_name from first span
    project_name = spans[0].get("project_name", "default")
    all_tags = list({t for s in spans for t in json.loads(s.get("tags_json") or "[]")})
    all_tags.extend(payload.tags)

    trace = Trace(
        trace_id=trace_id,
        project_name=project_name,
        query=payload.query,
        final_answer=payload.final_answer,
        expected_answer=payload.expected_answer,
        steps=steps,
        tags=list(set(all_tags)),
    )

    save_trace(conn, trace)
    logger.info(f"Saved trace {trace_id} with {len(steps)} steps")

    # Schedule background diagnosis
    diag_status = "skipped"
    if payload.run_diagnosis:
        background.add_task(_run_diagnosis_background, trace_id)
        diag_status = "queued"

    return FinalizeResponse(
        status="saved",
        trace_id=trace_id,
        step_count=len(steps),
        diagnosis_status=diag_status,
    )


async def _run_diagnosis_background(trace_id: str) -> None:
    """Background task: load trace → diagnose → save diagnosis."""
    from tracelens.attribute import diagnose_trace_async

    try:
        cfg = _get_cfg()
        conn = connect(cfg.db_path)
        trace_dict = load_trace(conn, trace_id)
        trace = Trace.model_validate(trace_dict)
        logger.info(f"Starting background diagnosis for {trace_id}")
        diagnosis = await diagnose_trace_async(trace, cfg)
        save_diagnosis(conn, diagnosis)
        logger.info(f"Diagnosis complete for {trace_id}: {diagnosis.summary}")
    except Exception as e:
        logger.error(f"Background diagnosis failed for {trace_id}: {e}", exc_info=True)


@app.get("/v1/traces")
async def list_traces_api(
    project: str | None = None,
    limit: int = 50,
):
    """List recent traces with diagnosis summary."""
    conn = _get_conn()
    traces = list_traces(conn, limit=limit)
    if project:
        traces = [t for t in traces if t.get("project_name") == project]
    return {"traces": traces, "count": len(traces)}


@app.get("/v1/traces/{trace_id}")
async def get_trace_api(trace_id: str):
    """Return a full trace with its diagnosis."""
    conn = _get_conn()
    try:
        trace_dict = load_trace(conn, trace_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    diagnosis = load_diagnosis(conn, trace_id)
    return {"trace": trace_dict, "diagnosis": diagnosis}
