"""SQLite persistence for traces, steps, and diagnoses.

Same design as EvalGate's store.py: hand-written DDL, one transaction per
save, idempotent schema creation on every connect().
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from tracelens.schema import Diagnosis, Trace

_DDL = """
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY,
    trace_id TEXT UNIQUE NOT NULL,
    query TEXT NOT NULL,
    final_answer TEXT NOT NULL,
    expected_answer TEXT,
    created_at TEXT NOT NULL,
    tags_json TEXT,
    metadata_json TEXT
);
CREATE TABLE IF NOT EXISTS steps (
    id INTEGER PRIMARY KEY,
    trace_id TEXT NOT NULL REFERENCES traces(trace_id),
    step_id TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    step_type TEXT NOT NULL,
    parent_step_id TEXT,
    input_text TEXT,
    output_text TEXT,
    tool_name TEXT,
    tool_args_json TEXT,
    tool_output TEXT,
    model TEXT,
    timestamp_ms REAL,
    duration_ms REAL,
    metadata_json TEXT
);
CREATE TABLE IF NOT EXISTS diagnoses (
    id INTEGER PRIMARY KEY,
    trace_id TEXT NOT NULL REFERENCES traces(trace_id),
    diagnosed_at TEXT NOT NULL,
    root_cause_step_id TEXT,
    root_cause_agent TEXT,
    attribution_score REAL,
    summary TEXT,
    step_scores_json TEXT
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open (creating if needed) the TraceLens DB and ensure the schema exists."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_DDL)
    return conn


def save_trace(conn: sqlite3.Connection, trace: Trace) -> int:
    """Persist a trace and all its steps. Returns the trace row ID.

    Uses a single transaction — a crash mid-save leaves no partial trace.
    """
    now = datetime.now(UTC).isoformat()

    with conn:
        cursor = conn.execute(
            "INSERT INTO traces (trace_id, query, final_answer, expected_answer, "
            "created_at, tags_json, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                trace.trace_id,
                trace.query,
                trace.final_answer,
                trace.expected_answer,
                now,
                json.dumps(trace.tags),
                "{}",
            ),
        )
        trace_row_id = cursor.lastrowid

        for step in trace.steps:
            conn.execute(
                "INSERT INTO steps (trace_id, step_id, agent_name, step_type, "
                "parent_step_id, input_text, output_text, tool_name, tool_args_json, "
                "tool_output, model, timestamp_ms, duration_ms, metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    trace.trace_id,
                    step.step_id,
                    step.agent_name,
                    step.step_type,
                    step.parent_step_id,
                    step.io.input_text,
                    step.io.output_text,
                    step.io.tool_name,
                    json.dumps(step.io.tool_args),
                    step.io.tool_output,
                    step.io.model,
                    step.timestamp_ms,
                    step.duration_ms,
                    json.dumps(step.metadata),
                ),
            )

    return trace_row_id


def load_trace(conn: sqlite3.Connection, trace_id: str) -> dict:
    """Load a trace and its steps from the DB. Returns a dict.

    Raises ValueError if the trace_id does not exist.
    """
    row = conn.execute(
        "SELECT * FROM traces WHERE trace_id = ?", (trace_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"Trace {trace_id!r} not found")

    trace_dict = dict(row)
    trace_dict["tags"] = json.loads(trace_dict.get("tags_json") or "[]")

    step_rows = conn.execute(
        "SELECT * FROM steps WHERE trace_id = ? ORDER BY timestamp_ms", (trace_id,)
    ).fetchall()
    trace_dict["steps"] = [dict(sr) for sr in step_rows]

    return trace_dict


def save_diagnosis(conn: sqlite3.Connection, diagnosis: Diagnosis) -> int:
    """Persist a diagnosis result. Returns the diagnosis row ID."""
    with conn:
        cursor = conn.execute(
            "INSERT INTO diagnoses (trace_id, diagnosed_at, root_cause_step_id, "
            "root_cause_agent, attribution_score, summary, step_scores_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                diagnosis.trace_id,
                diagnosis.diagnosed_at,
                diagnosis.root_cause_step.step_id if diagnosis.root_cause_step else None,
                diagnosis.root_cause_step.agent_name if diagnosis.root_cause_step else None,
                (
                    diagnosis.root_cause_step.attribution_score
                    if diagnosis.root_cause_step
                    else None
                ),
                diagnosis.summary,
                json.dumps(diagnosis.to_dict()["steps"]),
            ),
        )
    return cursor.lastrowid


def load_diagnosis(conn: sqlite3.Connection, trace_id: str) -> dict | None:
    """Load the most recent diagnosis for a trace. Returns None if not diagnosed."""
    row = conn.execute(
        "SELECT * FROM diagnoses WHERE trace_id = ? ORDER BY diagnosed_at DESC LIMIT 1",
        (trace_id,),
    ).fetchone()
    if row is None:
        return None
    return dict(row)


def list_traces(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    """List all stored traces, most recent first."""
    rows = conn.execute(
        "SELECT t.*, d.root_cause_step_id, d.root_cause_agent, d.attribution_score "
        "FROM traces t LEFT JOIN diagnoses d ON t.trace_id = d.trace_id "
        "ORDER BY t.created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]
