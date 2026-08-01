# TraceLens

**Automated causal root-cause diagnostics for multi-agent AI systems.**

When a multi-agent pipeline produces a wrong answer, TraceLens captures the execution trace, builds a graph of how information flowed between agents, and automatically identifies **which agent introduced the hallucination** — with an ensemble-verified confidence score and human-readable evidence.

> Think of it like this: Langfuse and Phoenix *show* you logs. TraceLens *tells* you what broke and why.

---

## How It Works

```
Your Chatbot                         TraceLens
────────────                         ─────────────────────────────────────
User: "Fix my bug"
  │
  ├──> RouterAgent runs         ──> Captured as step_000
  ├──> CodeReviewAgent runs     ──> Captured as step_001  
  ├──> SecurityAgent runs       ──> Captured as step_002 ← hallucinates
  └──> Synthesizer responds     ──> Captured as step_003
                                        │
                                        ▼
                                  Ensemble NLI Judge
                                  (3 independent votes per claim)
                                        │
                                        ▼
                               Dashboard auto-updates →
                               SecurityAgent highlighted 🔴
                               "Known firmware bug" = ungrounded
                               NLI votes: {ungrounded: 2, grounded: 1}
```

---

## Quick Start

```bash
# 1. Install
git clone <repo-url> && cd tracelens
uv sync && source .venv/bin/activate

# 2. Set your API key (used by the NLI judge)
echo "GOOGLE_API_KEY=your_key" > .env

# 3. Instrument your agent
python examples/customer_support_study.py

# 4. Launch the dashboard
tracelens dashboard
# → Open http://localhost:8501 (auto-refreshes every 5 seconds)
```

---

## SDK — 30-second Integration

```python
from tracelens.capture import TraceLensCapture
from tracelens.store import connect, save_trace

# One tracer per user request
tracer = TraceLensCapture(project_name="my_chatbot")

# Wrap each agent with a context manager
with tracer.step("RouterAgent", step_type="router", input_text=user_query) as io:
    io.output_text = my_router(user_query)

with tracer.step("AnswerAgent", step_type="agent", input_text=io.output_text) as io:
    io.output_text = my_agent(io.output_text)
    io.model = "gemini-2.5-flash"

# Save to DB
trace = tracer.finalize(query=user_query, final_answer=io.output_text)
save_trace(connect("tracelens.db"), trace)
```

**→ See [DOCUMENTATION.md](./DOCUMENTATION.md) for the full SDK guide, all integration patterns, CLI reference, and configuration options.**

---

## Commands

```bash
# Diagnose a stored trace (runs ensemble NLI + attribution engine)
tracelens diagnose <trace_id>

# View all stored diagnoses
tracelens report

# Ingest a trace from a JSON file
tracelens ingest trace.json

# Launch the live dashboard
tracelens dashboard

# Run tests
python -m pytest -v
```

---

## Architecture

```
src/tracelens/
├── schema.py       Trace, TraceStep, StepIO, Claim, Diagnosis data models
├── capture.py      Instrumentation SDK (decorator + context manager + manual API)
├── dag.py          DAG builder (networkx DiGraph)
├── claims.py       LLM-based claim decomposition engine
├── verify.py       Ensemble NLI judge (majority vote, calibrated confidence)
├── attribute.py    Bayesian causal attribution scoring engine
├── store.py        SQLite persistence
├── config.py       tracelens.toml loader
├── cli.py          Typer CLI (ingest, diagnose, report, dashboard)
└── dashboard/
    ├── dashboard.py    Streamlit app with 5-second auto-refresh
    └── components.py   pyvis graph, Gantt timeline, diagnosis panels

examples/
├── code_reviewer/          Multi-agent code review system
├── defect_study.py         Proves TraceLens catches hallucinations mathematically
└── customer_support_study.py  Batch trace generation with mixed healthy/defective
```

---

## Project Status

| Phase | Description | Status |
|---|---|---|
| 0 | Scaffold — pyproject, config, CLI | ✅ Done |
| 1 | Trace schema & capture SDK | ✅ Done |
| 2 | Multi-agent code reviewer example | ✅ Done |
| 3 | SQLite store & DAG builder | ✅ Done |
| 4 | Claim decomposition engine | ✅ Done |
| 5 | NLI entailment verification | ✅ Done |
| 6 | Causal attribution scoring | ✅ Done |
| 7 | CLI diagnostic commands | ✅ Done |
| 8 | Defect injection study | ✅ Done |
| 9 | Streamlit dashboard | ✅ Done |
| 3B | Ensemble NLI (non-determinism fix) | ✅ Done |
| 3A | Fixed attribution formula (leaf-bias) | ✅ Done |
| 4A | pyvis interactive graph | ✅ Done |
| 4B | Live auto-refresh dashboard | ✅ Done |
| **Next** | **HTTP Ingest API + Multi-parent DAG** | 🔲 Planned |
| **Next** | **LangChain / AutoGen / CrewAI integrations** | 🔲 Planned |

---

## Documentation

| Document | Description |
|---|---|
| [DOCUMENTATION.md](./DOCUMENTATION.md) | Full SDK guide — integration patterns, CLI reference, schema, how the engine works |
| [tracelens.toml](./tracelens.toml) | Annotated configuration file with all available options |
| [examples/](./examples/) | Working code examples you can run directly |
