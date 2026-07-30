# TraceLens

Automated causal root-cause diagnostics for multi-agent AI systems.

When a multi-agent pipeline produces a wrong answer, TraceLens captures the
execution trace, builds a directed acyclic graph (DAG) of agent steps, and
automatically pinpoints **which agent introduced the error** — with a
confidence score and human-readable explanation.

Think of it as a diagnostic engine: Langfuse/Phoenix *show* you logs,
TraceLens *tells* you what broke and why.

---

## Setup

```bash
# Clone and enter the repo
git clone <your-repo-url>
cd tracelens

# Install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate

# Add your Google API key (get one from aistudio.google.com)
echo "GOOGLE_API_KEY=your_key_here" > .env
```

---

## Commands

**Run the code reviewer agent interactively:**
```bash
python -m examples.code_reviewer.chat
```

**Diagnose a stored trace:**
```bash
tracelens diagnose --trace-id <id>
```

**View diagnostic report:**
```bash
tracelens report
```

**Run all tests (no network, no API calls needed):**
```bash
python -m pytest -v
```

**Launch the interactive dashboard:**
```bash
tracelens dashboard
```

---

## Architecture

```
tracelens/
├── src/tracelens/           # The framework (library + CLI)
│   ├── schema.py            # Trace, TraceStep, Claim data models
│   ├── capture.py           # Instrumentation SDK (decorators/context managers)
│   ├── dag.py               # DAG builder (networkx)
│   ├── claims.py            # Claim decomposition engine
│   ├── verify.py            # Step-level entailment verification
│   ├── attribute.py         # Causal attribution scoring
│   ├── store.py             # SQLite persistence
│   ├── config.py            # tracelens.toml loader
│   └── cli.py               # Typer CLI
├── examples/code_reviewer/  # Subject agent (what gets diagnosed)
│   ├── agent.py             # Multi-agent code reviewer
│   ├── fixtures/            # Pre-recorded responses
│   └── chat.py              # Interactive REPL
├── study/                   # Defect injection study
│   ├── injections.py        # Synthetic defect catalog
│   └── run_study.py         # Precision/recall measurement
├── dashboard/
│   └── app.py               # Streamlit DAG viewer
└── tests/
```

---

## Project Status

| Phase | Description | Status |
|---|---|---|
| 0 | Scaffold — pyproject, config, CLI stub | ✅ Done |
| 1 | Trace schema & capture SDK | ✅ Done |
| 2 | Subject agent (multi-agent code reviewer) | ✅ Done |
| 3 | Trace store (SQLite) & DAG builder | ✅ Done |
| 4 | Claim decomposition engine | ✅ Done |
| 5 | Step-level entailment verification | ✅ Done |
| 6 | Causal attribution scoring | ✅ Done |
| 7 | CLI diagnostic commands | ✅ Done |
| 8 | Defect injection study | ✅ Done |
| 9 | Interactive DAG dashboard & release polish | ⬜ Planned |
