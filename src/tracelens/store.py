"""SQLite persistence for traces, steps, span_buffer, and diagnoses.

Design decisions:
- connection pool via threading.local() — safe for multi-threaded Streamlit
- All DDL is idempotent (CREATE IF NOT EXISTS + ADD COLUMN IF NOT EXISTS)
- Indexes on hot query paths (project_name, created_at, trace_id)
- span_buffer table for the HTTP ingest API (spans arrive before finalize)
- Schema version table for future migrations
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

from tracelens.schema import Diagnosis, Trace

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY,
    trace_id TEXT UNIQUE NOT NULL,
    project_name TEXT NOT NULL DEFAULT 'default',
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
    parent_span_ids_json TEXT,
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

CREATE TABLE IF NOT EXISTS span_buffer (
    id INTEGER PRIMARY KEY,
    trace_id TEXT NOT NULL,
    span_id TEXT NOT NULL UNIQUE,
    parent_span_ids_json TEXT,
    project_name TEXT NOT NULL DEFAULT 'default',
    agent_name TEXT NOT NULL,
    span_type TEXT NOT NULL DEFAULT 'agent',
    input_text TEXT,
    output_text TEXT,
    tool_name TEXT,
    tool_args_json TEXT,
    tool_output TEXT,
    model TEXT,
    start_time_ms REAL,
    end_time_ms REAL,
    metadata_json TEXT,
    tags_json TEXT,
    received_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_traces_project   ON traces(project_name);
CREATE INDEX IF NOT EXISTS idx_traces_created   ON traces(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_steps_trace      ON steps(trace_id);
CREATE INDEX IF NOT EXISTS idx_diagnoses_trace  ON diagnoses(trace_id);
CREATE INDEX IF NOT EXISTS idx_buffer_trace     ON span_buffer(trace_id);
"""

CURRENT_SCHEMA_VERSION = 2


# ---------------------------------------------------------------------------
# Thread-local connection pool
# ---------------------------------------------------------------------------

class _ConnectionPool:
    """Thread-local SQLite connection pool.

    SQLite connections are not thread-safe. Rather than passing one shared
    connection everywhere (which breaks under Streamlit's multi-thread model),
    each thread gets its own connection to the same DB file.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._local = threading.local()

    def get(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.executescript(_DDL)
            _ensure_schema_version(conn)
            self._local.conn = conn
        return self._local.conn

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None


# Module-level registry of pools keyed by resolved db_path
_pools: dict[str, _ConnectionPool] = {}
_pool_lock = threading.Lock()


def _get_pool(db_path: str | Path) -> _ConnectionPool:
    key = str(Path(db_path).resolve())
    with _pool_lock:
        if key not in _pools:
            _pools[key] = _ConnectionPool(key)
        return _pools[key]


def _ensure_schema_version(conn: sqlite3.Connection) -> None:
    """Insert the current schema version if not already set."""
    row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    if row is None:
        with conn:
            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (?)",
                (CURRENT_SCHEMA_VERSION,),
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def connect(db_path: str | Path) -> sqlite3.Connection:
    """Return a thread-local SQLite connection, creating the schema if needed.

    Safe to call from any thread — each thread gets its own connection to
    the same database file. WAL mode allows concurrent reads during writes.
    """
    return _get_pool(db_path).get()


def save_trace(conn: sqlite3.Connection, trace: Trace) -> int:
    """Persist a trace and all its steps. Returns the trace row ID.

    Uses a single transaction — a crash mid-save leaves no partial trace.
    Silently ignores duplicate trace_ids (idempotent).
    """
    now = datetime.now(UTC).isoformat()

    with conn:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO traces "
            "(trace_id, project_name, query, final_answer, expected_answer, "
            "created_at, tags_json, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                trace.trace_id,
                trace.project_name,
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
            # support both old single-parent and new multi-parent schemas
            parent_ids = getattr(step, "parent_span_ids", None)
            if parent_ids is None:
                parent_ids = [step.parent_step_id] if step.parent_step_id else []

            conn.execute(
                "INSERT OR IGNORE INTO steps "
                "(trace_id, step_id, agent_name, step_type, parent_step_id, "
                "parent_span_ids_json, input_text, output_text, tool_name, "
                "tool_args_json, tool_output, model, timestamp_ms, duration_ms, "
                "metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    trace.trace_id,
                    step.step_id,
                    step.agent_name,
                    step.step_type,
                    step.parent_step_id,
                    json.dumps(parent_ids),
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

    return trace_row_id or 0


def load_trace(conn: sqlite3.Connection, trace_id: str) -> dict:
    """Load a trace and its steps from the DB. Returns a dict.

    The returned dict is compatible with Trace.model_validate().
    Raises ValueError if the trace_id does not exist.
    """
    row = conn.execute(
        "SELECT * FROM traces WHERE trace_id = ?", (trace_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"Trace {trace_id!r} not found")

    raw = dict(row)
    trace_dict = {
        "trace_id": raw["trace_id"],
        "project_name": raw.get("project_name", "default"),
        "query": raw["query"],
        "final_answer": raw["final_answer"],
        "expected_answer": raw.get("expected_answer"),
        "tags": json.loads(raw.get("tags_json") or "[]"),
    }

    step_rows = conn.execute(
        "SELECT * FROM steps WHERE trace_id = ? ORDER BY timestamp_ms", (trace_id,)
    ).fetchall()

    steps = []
    for sr in step_rows:
        sd = dict(sr)

        # Restore multi-parent list
        raw_ids = sd.get("parent_span_ids_json")
        if raw_ids:
            parent_span_ids = json.loads(raw_ids)
        else:
            parent_span_ids = (
                [sd["parent_step_id"]] if sd.get("parent_step_id") else []
            )

        # Build a clean step dict with only the fields TraceStep expects
        clean_step = {
            "step_id": sd["step_id"],
            "agent_name": sd["agent_name"],
            "step_type": sd["step_type"],
            "parent_step_id": sd.get("parent_step_id"),
            "parent_span_ids": parent_span_ids,
            "io": {
                "input_text": sd.get("input_text") or "",
                "output_text": sd.get("output_text") or "",
                "tool_name": sd.get("tool_name"),
                "tool_args": json.loads(sd.get("tool_args_json") or "{}"),
                "tool_output": sd.get("tool_output"),
                "model": sd.get("model"),
            },
            "timestamp_ms": sd.get("timestamp_ms", 0.0),
            "duration_ms": sd.get("duration_ms", 0.0),
            "metadata": json.loads(sd.get("metadata_json") or "{}"),
        }
        steps.append(clean_step)

    trace_dict["steps"] = steps
    return trace_dict


def save_diagnosis(conn: sqlite3.Connection, diagnosis: Diagnosis) -> int:
    """Persist a diagnosis result. Returns the diagnosis row ID."""
    with conn:
        cursor = conn.execute(
            "INSERT INTO diagnoses "
            "(trace_id, diagnosed_at, root_cause_step_id, root_cause_agent, "
            "attribution_score, summary, step_scores_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
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
    """List traces most-recent-first with diagnosis summary joined in."""
    rows = conn.execute(
        "SELECT t.*, d.root_cause_step_id, d.root_cause_agent, d.attribution_score "
        "FROM traces t LEFT JOIN diagnoses d ON t.trace_id = d.trace_id "
        "ORDER BY t.created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Span buffer API (used by HTTP ingest server)
# ---------------------------------------------------------------------------

def save_span(conn: sqlite3.Connection, span: dict) -> None:
    """Buffer an incoming span before the trace is finalized."""
    now = datetime.now(UTC).isoformat()
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO span_buffer "
            "(trace_id, span_id, parent_span_ids_json, project_name, agent_name, "
            "span_type, input_text, output_text, tool_name, tool_args_json, "
            "tool_output, model, start_time_ms, end_time_ms, metadata_json, "
            "tags_json, received_at) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                span["trace_id"],
                span["span_id"],
                json.dumps(span.get("parent_span_ids", [])),
                span.get("project_name", "default"),
                span["agent_name"],
                span.get("span_type", "agent"),
                span.get("input_text", ""),
                span.get("output_text", ""),
                span.get("tool_name"),
                json.dumps(span.get("tool_args", {})),
                span.get("tool_output"),
                span.get("model"),
                span.get("start_time_ms", 0.0),
                span.get("end_time_ms", 0.0),
                json.dumps(span.get("metadata", {})),
                json.dumps(span.get("tags", [])),
                now,
            ),
        )


def flush_span_buffer(conn: sqlite3.Connection, trace_id: str) -> list[dict]:
    """Read and delete all buffered spans for a trace_id. Returns them as dicts."""
    rows = conn.execute(
        "SELECT * FROM span_buffer WHERE trace_id = ? ORDER BY start_time_ms",
        (trace_id,),
    ).fetchall()
    spans = [dict(r) for r in rows]
    with conn:
        conn.execute("DELETE FROM span_buffer WHERE trace_id = ?", (trace_id,))
    return spans
