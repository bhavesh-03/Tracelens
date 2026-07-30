# TraceLens — Engineering Journal

Append-only. After every phase: date, key decisions, bugs + root cause, surprises.

---

## Phase 0: Scaffold

**Key decisions:**

- **`hatchling` as build backend.** Same as EvalGate — lightweight, `uv`-friendly, clean
  src-layout support with no config beyond
  `[tool.hatch.build.targets.wheel] packages = ["src/tracelens"]`.

- **`tomllib` for TOML parsing (stdlib, Python 3.12+).** No extra dependency needed.

- **Dataclasses for `TraceLensConfig`, Pydantic v2 for user-facing trace schemas.**
  Same split as EvalGate: Pydantic's validation errors matter for user-facing schemas
  (Trace, TraceStep), but config is internal and loaded once at startup.

- **`networkx` for DAG operations.** Standard library for graph algorithms in Python.
  Provides topological sort, ancestor/descendant queries, shortest path, and cycle
  detection — all needed by the attribution engine.

- **Rate-limit-aware defaults.** `max_concurrent_verifications=1` because Gemini free
  tier has a 15 req/min cap. Same lesson EvalGate learned in Phase 4 addendum.

- **Trace capture SDK uses `ContextVar` for automatic parent tracking.** When a
  decorated function calls another decorated function, the SDK automatically knows
  that the inner call is a child of the outer call. This is the same mechanism
  Python's `contextvars` module was designed for — async-safe implicit state.

**Bugs encountered:** None in this phase.

**Surprises:** None yet.
