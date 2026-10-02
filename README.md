<h1 align="center">TraceLens</h1>

<p align="center">
  Find where unsupported claims enter a multi-agent AI workflow.
</p>

<p align="center">
  <img alt="Python 3.12 or later" src="https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white">
  <img alt="Version 0.1.0" src="https://img.shields.io/badge/version-0.1.0-6E56CF">
  <img alt="Status alpha" src="https://img.shields.io/badge/status-alpha-F59E0B">
</p>

---

TraceLens captures a multi-agent run, maps the information flow between its steps,
then uses claim-level verification to rank the step most likely to have introduced
an unsupported statement. It is designed to make an investigation faster: inspect
the trace, review the evidence, and see a clear hypothesis for where to start.

> TraceLens is an early-stage diagnostic assistant. Its attribution score is a
> ranked hypothesis for human review—not a substitute for ground truth or a
> guarantee of causality.

## What it does

- Captures agent, LLM, tool, router, and synthesizer steps with timing and metadata.
- Represents sequential, fan-out, and fan-in workflows as a directed trace graph.
- Breaks outputs into factual claims and checks them against recorded parent evidence.
- Uses multiple NLI judge votes to make a claim verdict less dependent on one model call.
- Ranks suspect steps and shows the supporting evidence in a Streamlit dashboard.
- Supports a Python SDK, HTTP ingestion API, CLI, and initial LangChain, CrewAI, and AutoGen hooks.

## Quick start

**Requirements:** Python 3.12+ and a provider API key for the configured judge model.

```bash
git clone https://github.com/bhavesh-03/Tracelens.git
cd Tracelens
uv sync

# The default judge model is Gemini Flash.
export GOOGLE_API_KEY="your_key_here"
```

Create and save a trace:

```python
from tracelens.capture import TraceLensCapture
from tracelens.store import connect, save_trace

query = "Why is the checkout service failing?"
tracer = TraceLensCapture(project_name="checkout-agent")

with tracer.step("Router", step_type="router", input_text=query) as step:
    step.output_text = "Route to the payments specialist."

with tracer.step("PaymentsAgent", step_type="agent", input_text="Investigate checkout.") as step:
    step.output_text = "The payment gateway returned a timeout."
    step.model = "gemini-2.5-flash"

trace = tracer.finalize(
    query=query,
    final_answer="Checkout is failing because the payment gateway timed out.",
)
save_trace(connect("tracelens.db"), trace)
```

Diagnose it and open the dashboard:

```bash
uv run tracelens diagnose <trace_id>
uv run tracelens dashboard
# Dashboard: http://localhost:8501
```

Try the included end-to-end checkout incident example:

```bash
# Capture, redact, and persist a realistic trace without calling an LLM.
uv run python examples/checkout_incident_demo.py

# With a configured judge API key, run full claim verification and ranking.
uv run python examples/checkout_incident_demo.py --diagnose
```

## How a diagnosis works

```text
User query
    │
    ▼
Router ──► Specialist agent ──► Tool / research ──► Synthesizer ──► Final answer
                                                       │
                                                       ▼
                                  claim extraction + evidence verification + ranking
                                                       │
                                                       ▼
                                           reviewable root-cause hypothesis
```

For every step, TraceLens extracts atomic factual claims from its output and
checks each claim against the immediate parent evidence recorded in the trace.
The final ranking combines the estimated unsupported-claim rate, claim-content
matching, and whether a step can reach a final-answer leaf in the recorded graph.

## Use it your way

### Python SDK

The context manager in the quick start is best when you control the agent code.
There is also a decorator for regular or async functions, plus `add_step()` for
framework-managed or manually reconstructed traces.

### Framework hooks

```python
from tracelens.integrations import TraceLensCallbackHandler, instrument_crew

# LangChain
handler = TraceLensCallbackHandler(tracer)
chain.invoke({"input": "..."}, config={"callbacks": [handler]})

# CrewAI
instrument_crew(my_crew, tracer)
```

AutoGen instrumentation and synchronous/asynchronous HTTP clients are available
under `tracelens.integrations` as well.

### HTTP ingestion

Run the local ingest service:

```bash
uv run tracelens serve --host 127.0.0.1 --port 4318
```

Send spans to `POST /v1/spans`, then finalize the trace with
`POST /v1/traces/{trace_id}/finalize`. Interactive endpoint documentation is
available at `http://127.0.0.1:4318/docs`.

**Security note:** the current HTTP service is intended for trusted local or
private-network development. To accept remote requests, set `TRACELENS_API_KEY`
and send it as `X-TraceLens-API-Key`. Configure exact browser origins with
`allowed_origins`; requests are size-limited and sensitive values are redacted
before persistence by default.

## Commands

| Command | Purpose |
|---|---|
| `uv run tracelens ingest trace.json` | Validate and store a trace JSON file. |
| `uv run tracelens diagnose <trace_id>` | Run claim verification and attribution. |
| `uv run tracelens report` | List stored traces and diagnosis summaries. |
| `uv run tracelens dashboard` | Start the live trace explorer. |
| `uv run tracelens serve` | Start the HTTP ingest API. |
| `uv run pytest -q` | Run the test suite. |

## Configuration

`tracelens.toml` controls the judge model, claim limit, attribution threshold,
database path, ensemble vote count, and judge settings. The defaults use
`gemini/gemini-2.5-flash` and write to `tracelens.db`.

Set `retention_days` to automatically remove old traces and buffered spans during
ingestion, or run `uv run tracelens purge --older-than-days 30` for an explicit,
confirmed cleanup.

## Project status

TraceLens is an **alpha** project. The core capture, graph, diagnosis, dashboard,
HTTP API, and initial integrations are in place. The next important work is
production safety (authentication, privacy controls, durable ingestion), diagnostic
evaluation against labelled traces, and CI/deployment support.

See [PROJECT_REVIEW.md](PROJECT_REVIEW.md) for the detailed engineering and product
review, including priorities and a recommended roadmap.

## Repository layout

```text
src/tracelens/
├── capture.py       Python instrumentation SDK
├── schema.py        Trace and diagnosis data models
├── dag.py           Information-flow graph builder
├── claims.py        Claim extraction
├── verify.py        Ensemble NLI verification
├── attribute.py     Claim-origin ranking
├── store.py         SQLite persistence
├── server.py        HTTP ingest API
└── dashboard/       Streamlit trace explorer

examples/            Runnable multi-agent examples
tests/               Unit tests
```

## Contributing

This project is early and feedback is welcome. Before opening a change, please run:

```bash
uv run pytest -q
uv run ruff check src tests
```
