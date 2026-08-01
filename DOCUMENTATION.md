# TraceLens — SDK & Integration Documentation

> **TraceLens** is an automated causal diagnostics engine for multi-agent AI systems.
> It captures execution traces from your agents, builds a graph of how information
> flowed between them, and mathematically identifies which agent introduced a
> hallucination or error — with a confidence score and evidence.

---

## Table of Contents

1. [Concepts](#1-concepts)
2. [Quick Start](#2-quick-start)
3. [SDK Reference — `TraceLensCapture`](#3-sdk-reference--tracelens-capture)
   - [Decorator API](#31-decorator-api)
   - [Context Manager API](#32-context-manager-api)
   - [Manual Step API](#33-manual-step-api)
   - [Finalizing a Trace](#34-finalizing-a-trace)
4. [Saving and Diagnosing Traces](#4-saving-and-diagnosing-traces)
5. [Integration Patterns](#5-integration-patterns)
   - [Simple Sequential Pipeline](#51-simple-sequential-pipeline)
   - [Router + Multi-Agent Pipeline](#52-router--multi-agent-pipeline)
   - [Tool-Calling Agents](#53-tool-calling-agents)
   - [LiteLLM Integration](#54-litellm-integration)
6. [CLI Reference](#6-cli-reference)
7. [Dashboard](#7-dashboard)
8. [Configuration (`tracelens.toml`)](#8-configuration-tracelenstoml)
9. [Schema Reference](#9-schema-reference)
10. [How the Diagnostic Engine Works](#10-how-the-diagnostic-engine-works)
11. [What's Next (Roadmap)](#11-whats-next-roadmap)

---

## 1. Concepts

Before integrating TraceLens, understand three key concepts:

| Concept | What It Is | Analogy |
|---|---|---|
| **Trace** | One complete user request through your pipeline | A single HTTP request in a web server |
| **Step** | One agent or tool call within a trace | A function call in a stack trace |
| **Diagnosis** | TraceLens's computed root-cause report | A debugger's stack frame analysis |

### How a Trace Is Built

A trace is a **directed graph** of steps. Each step has:
- An **input** (what the agent received)
- An **output** (what the agent said)
- A **parent** (which step invoked it)

```
User Query
    │
    ▼
RouterAgent ──────────────────────────────────┐
    │                                          │
    ▼                                          ▼
BillingAgent ──> BillingTool         TechSupportAgent ──> DiagnosticTool
    │                                          │
    └──────────────┬────────────────────────────┘
                   ▼
             SynthesizerAgent
                   │
                   ▼
            Final Answer
```

Each arrow is a parent → child relationship. This is your **Execution DAG**.

TraceLens walks this graph backwards from the final answer, decomposes every
agent's output into atomic claims, and checks each claim against the evidence
the agent actually had. Claims that are ungrounded (hallucinated) increase
that agent's **attribution score**.

---

## 2. Quick Start

### Installation

```bash
git clone <your-repo-url>
cd tracelens
uv sync
source .venv/bin/activate
```

### Set Your API Key

TraceLens uses Gemini Flash as the NLI judge by default. Get a key from
[aistudio.google.com](https://aistudio.google.com) and set it:

```bash
echo "GOOGLE_API_KEY=your_key_here" > .env
```

Or export it directly:

```bash
export GOOGLE_API_KEY=your_key_here
```

### Instrument Your First Agent

```python
from tracelens.capture import TraceLensCapture
from tracelens.store import connect, save_trace

# Create one tracer per user request
tracer = TraceLensCapture(project_name="my_chatbot")

# Wrap each agent call
with tracer.step("RouterAgent", step_type="router", input_text=user_query) as io:
    routing_decision = my_router(user_query)
    io.output_text = routing_decision

with tracer.step("AnswerAgent", step_type="agent", input_text=routing_decision) as io:
    answer = my_agent(routing_decision)
    io.output_text = answer

# Finalize — creates a validated Trace object
trace = tracer.finalize(query=user_query, final_answer=answer)

# Save to database
conn = connect("tracelens.db")
save_trace(conn, trace)
```

### Launch the Dashboard

```bash
tracelens dashboard
# → Open http://localhost:8501
```

---

## 3. SDK Reference — `TraceLensCapture`

Import:
```python
from tracelens.capture import TraceLensCapture
```

### Initializing the Tracer

```python
tracer = TraceLensCapture(
    trace_id=None,        # auto-generated if not provided (recommended)
    project_name="default"  # used for filtering in the dashboard
)
```

Create **one `TraceLensCapture` instance per user request**. Do not reuse
a tracer across multiple requests — each request must be its own trace.

---

### 3.1 Decorator API

Best for simple functions with a clear input → output shape.

```python
tracer = TraceLensCapture(project_name="my_project")

@tracer.trace_step(agent_name="RouterAgent", step_type="router")
def route_query(query: str) -> str:
    # ... your routing logic ...
    return selected_agent_name

@tracer.trace_step(agent_name="TechAgent", step_type="agent", model="gemini-2.5-flash")
async def tech_agent(query: str) -> str:
    # ... async agent logic ...
    return answer
```

> **How it works:** The decorator captures `str(args[0])` as `input_text` and
> `str(return_value)` as `output_text`. It works for both `def` and `async def`.

> **Limitation:** The decorator can only capture the first argument as input.
> If your function takes a complex object, use the context manager API instead.

---

### 3.2 Context Manager API

Gives you full control over input, output, tool calls, and metadata.

```python
with tracer.step(
    agent_name="BillingAgent",
    step_type="agent",
    input_text=f"User query: {query}\nCustomer ID: {customer_id}",
    model="gemini-2.5-flash",
    metadata={"customer_tier": "premium"},
) as io:
    # io is a StepIO object — mutate it freely
    tool_result = billing_api.get_invoice(customer_id)
    io.tool_name = "get_invoice"
    io.tool_args = {"customer_id": customer_id}
    io.tool_output = str(tool_result)
    
    answer = llm.generate(f"Invoice data: {tool_result}\nQuery: {query}")
    io.output_text = answer
```

**`StepIO` fields you can set inside the context manager:**

| Field | Type | Description |
|---|---|---|
| `output_text` | `str` | Agent's final output (most important) |
| `tool_name` | `str \| None` | Name of tool called, if any |
| `tool_args` | `dict` | Arguments passed to the tool |
| `tool_output` | `str \| None` | Raw output from the tool |
| `model` | `str \| None` | LLM model used (e.g. `"gemini-2.5-flash"`) |

---

### 3.3 Manual Step API

Use this when you cannot wrap the code (e.g., when instrumenting a framework
that manages its own execution, or when building steps from log files).

```python
# Add a step manually — returns the step_id
step_id = tracer.add_step(
    agent_name="SecurityAgent",
    step_type="agent",
    input_text=input_text,
    output_text=output_text,
    parent_step_id=previous_step_id,   # set explicitly
    model="gpt-4o",
    tool_name="code_scanner",
    tool_args={"file": "main.py"},
    tool_output=scanner_result,
    duration_ms=1234.5,
    metadata={"scan_version": "2.1"},
)
```

> **When to use this:** When you have a fan-out (one step spawns multiple parallel
> agents), you must use `add_step` manually and set `parent_step_id` explicitly
> for each parallel agent. The decorator/context manager auto-detects parent from
> the Python call stack, which breaks with `asyncio.gather`.

---

### 3.4 Finalizing a Trace

After all steps are captured, call `finalize()` to get a validated `Trace` object:

```python
trace = tracer.finalize(
    query="What is wrong with my internet?",   # the original user query
    final_answer="Your modem shows 0% packet loss.",  # the system's response
    expected_answer=None,          # optional: ground truth for accuracy scoring
    tags=["production", "ticket-4521"],  # free-form tags for dashboard filtering
)
```

`finalize()` validates the trace structure (no broken parent references, exactly
one root step) and raises `pydantic.ValidationError` if anything is wrong.

> **Important:** Call `finalize()` exactly once per tracer. It raises `ValueError`
> if called a second time.

---

## 4. Saving and Diagnosing Traces

### Save to Database

```python
from tracelens.store import connect, save_trace

conn = connect("tracelens.db")   # creates DB and schema if not exists
save_trace(conn, trace)
```

### Run Diagnosis (CLI)

```bash
tracelens diagnose <trace_id>
# → Runs ensemble NLI on all agent outputs
# → Prints root cause + hallucinated claims
# → Saves diagnosis to DB for dashboard viewing
```

### Run Diagnosis (Python API)

```python
from tracelens.attribute import diagnose_trace
from tracelens.config import load_config
from tracelens.store import save_diagnosis

cfg = load_config()   # loads tracelens.toml
diagnosis = diagnose_trace(trace, cfg)
save_diagnosis(conn, diagnosis)

print(f"Root cause: {diagnosis.root_cause_step.agent_name}")
print(f"Score: {diagnosis.root_cause_step.attribution_score:.3f}")
for claim in diagnosis.root_cause_step.novel_claims:
    print(f"  Hallucinated claim: {claim.text}")
    print(f"  NLI votes: {claim.vote_breakdown}")
```

---

## 5. Integration Patterns

### 5.1 Simple Sequential Pipeline

The most common pattern: agents call each other in sequence.

```python
from tracelens.capture import TraceLensCapture
from tracelens.store import connect, save_trace

def handle_user_request(user_query: str) -> str:
    tracer = TraceLensCapture(project_name="my_chatbot")

    # Step 1: Router
    with tracer.step("Router", step_type="router", input_text=user_query) as io:
        destination = classify_intent(user_query)
        io.output_text = destination

    # Step 2: Specialist Agent
    with tracer.step("SpecialistAgent", step_type="agent", input_text=destination) as io:
        response = call_llm(system_prompt=SPECIALIST_PROMPT, user=destination)
        io.output_text = response
        io.model = "gemini-2.5-flash"

    final_answer = response
    trace = tracer.finalize(query=user_query, final_answer=final_answer)

    # Save to DB (fire and forget — doesn't block response)
    conn = connect("tracelens.db")
    save_trace(conn, trace)

    return final_answer
```

---

### 5.2 Router + Multi-Agent Pipeline

When a router dispatches to multiple agents. You **must** use `add_step` to
correctly set the parent for each agent.

```python
def handle_request(query: str) -> str:
    tracer = TraceLensCapture(project_name="multi_agent")

    # 1. Router — the root step
    router_id = tracer.add_step(
        agent_name="RouterAgent",
        step_type="router",
        input_text=query,
        output_text="Dispatching to BillingAgent and TechAgent",
        parent_step_id=None,  # root
    )

    # 2. Two parallel agents — both children of the router
    billing_id = tracer.add_step(
        agent_name="BillingAgent",
        step_type="agent",
        input_text=f"Query: {query}",
        output_text=billing_agent_result,
        parent_step_id=router_id,          # ← parent is the router
    )

    tech_id = tracer.add_step(
        agent_name="TechAgent",
        step_type="agent",
        input_text=f"Query: {query}",
        output_text=tech_agent_result,
        parent_step_id=router_id,          # ← same parent (fan-out)
    )

    # 3. Synthesizer merges both
    # Currently TraceLens supports one parent per step.
    # Use the primary input parent for now:
    synth_id = tracer.add_step(
        agent_name="Synthesizer",
        step_type="synthesizer",
        input_text=f"Billing: {billing_agent_result}\nTech: {tech_agent_result}",
        output_text=final_answer,
        parent_step_id=billing_id,         # primary parent (phase 2 will support multiple)
    )

    trace = tracer.finalize(query=query, final_answer=final_answer)
    save_trace(connect("tracelens.db"), trace)
    return final_answer
```

---

### 5.3 Tool-Calling Agents

Record tool calls so TraceLens can verify claims against the *actual tool output*
(not just the agent's stated input):

```python
with tracer.step(
    "WeatherAgent",
    step_type="agent",
    input_text=user_query,
    model="gpt-4o",
) as io:
    # Call your tool
    weather_data = weather_api.get(city="London")
    
    # Record the tool call in the step
    io.tool_name = "weather_api.get"
    io.tool_args = {"city": "London"}
    io.tool_output = json.dumps(weather_data)   # ← this becomes the NLI evidence
    
    # Agent uses tool result to generate answer
    answer = llm.complete(f"Weather data: {weather_data}\nQuestion: {user_query}")
    io.output_text = answer
```

> **Why this matters:** TraceLens's NLI engine uses `tool_output` as the primary
> evidence when verifying claims from tool-using agents. If you skip recording
> tool calls, the judge will check claims against the agent's `input_text` instead,
> which may produce false positives.

---

### 5.4 LiteLLM Integration

If you already use LiteLLM for all your LLM calls, you can auto-capture model
and timing metadata:

```python
import litellm
from tracelens.capture import TraceLensCapture

tracer = TraceLensCapture(project_name="litellm_app")

with tracer.step("MyAgent", step_type="agent", input_text=query) as io:
    response = litellm.completion(
        model="gemini/gemini-2.5-flash",
        messages=[{"role": "user", "content": query}]
    )
    output = response.choices[0].message.content
    
    # Capture model metadata
    io.model = response.model
    io.output_text = output
    io.metadata = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "cost_usd": response._hidden_params.get("response_cost", 0),
    }
```

---

## 6. CLI Reference

All commands accept `--config <path>` to specify a custom `tracelens.toml`.

### `tracelens ingest <file.json>`

Import a trace from a JSON file into the database.

```bash
tracelens ingest trace_20240801.json
tracelens ingest traces/run_42.json --config /path/to/tracelens.toml
```

The JSON file must be a valid serialized `Trace` object (use `trace.model_dump_json()`).

---

### `tracelens diagnose <trace_id>`

Run the causal attribution engine on a stored trace.

```bash
tracelens diagnose trace_abc123
tracelens diagnose trace_abc123 --verbose   # shows per-step debug logs
```

This command:
1. Loads the trace from the database
2. Decomposes every agent's output into atomic claims (LLM call)
3. Verifies each claim against parent context using ensemble NLI (3 LLM calls per claim)
4. Computes attribution scores and saves the diagnosis

**Cost estimate:** ~3-5 LLM calls per step × 3 NLI votes = ~9-15 calls per step.
With Gemini Flash at $0.075/million tokens, a 5-step trace ≈ $0.001-0.003.

---

### `tracelens report`

Print a summary of all stored diagnoses.

```bash
tracelens report                              # list all traces
tracelens report --trace-id trace_abc123      # full report for one trace
```

---

### `tracelens dashboard`

Launch the interactive Streamlit dashboard.

```bash
tracelens dashboard
# → Opens http://localhost:8501
# → Auto-refreshes every 5 seconds
```

---

## 7. Dashboard

Open **http://localhost:8501** after running `tracelens dashboard`.

### What You See

| Panel | What It Shows |
|---|---|
| **Sidebar** | Project filter, live trace feed with status icons (🔴🟢⏳) |
| **Metrics bar** | Project, step count, status, root cause agent |
| **Execution Graph** | Interactive pyvis graph — hover nodes for details, red = hallucinator |
| **Causal Attribution** | Per-agent attribution scores + NLI ensemble vote breakdown |
| **Timeline** | Gantt chart of agent execution (shows parallelism) |
| **Step Inspector** | Raw input/output for any step + tool call details |

### Auto-Refresh

The dashboard refreshes every **5 seconds automatically**. When you run your
chatbot and a new trace is saved to `tracelens.db`, it will appear in the
sidebar within 5 seconds — no manual F5 needed.

### Reading the Execution Graph

Each node in the graph is one agent step:
- 🔴 **Red node** = Root cause (highest attribution score, above threshold)
- 🟠 **Orange node** = Suspicious (score > 0.05 but below threshold)
- 🟢 **Green node** = Innocent (evaluated and cleared)
- 🔵 **Blue node** = Tool call (not evaluated for hallucination)
- ⚪ **Grey node** = Not yet diagnosed

Hover any node to see: attribution score, novel claim ratio, propagation to
final answer, and the first hallucinated claim text.

---

## 8. Configuration (`tracelens.toml`)

Place this file in your project root (same directory where you run `tracelens`):

```toml
[tracelens]
judge_model = "gemini/gemini-2.5-flash"  # any litellm-supported model

# Attribution
attribution_threshold = 0.15   # flag as root cause if score > this
                                # adjust lower to catch more subtle hallucinations

# Claim decomposition
max_claims_per_step = 20        # cap to control cost

# Ensemble NLI — the core non-determinism correction
nli_ensemble_votes = 3          # 1=fast (non-deterministic), 3=recommended, 5=high-confidence
nli_min_agreement   = 0.67      # 0.67 = 2/3 judges must agree; below this → "uncertain"

# Judge behavior
temperature = 0.3               # slightly above 0 for ensemble diversity
max_concurrent_verifications = 1  # increase if you have a paid API tier

# Storage
db_path = "tracelens.db"        # relative to CWD; use absolute path for reliability

[costs]
input_per_million  = 0.075      # USD — used for cost reporting only
output_per_million = 0.30
```

### Using a Different LLM Judge

TraceLens uses [LiteLLM](https://litellm.ai) internally, so any model it
supports works out of the box:

```toml
judge_model = "openai/gpt-4o-mini"       # cheap OpenAI option
judge_model = "anthropic/claude-3-haiku" # fast Anthropic option
judge_model = "ollama/llama3.1"          # local, free, slower
```

---

## 9. Schema Reference

### `Trace`

```python
Trace(
    trace_id="trace_abc123",           # unique ID (auto-generated by TraceLensCapture)
    project_name="my_chatbot",         # used for dashboard filtering
    query="What is wrong?",            # original user input
    final_answer="Your modem is fine", # system's final response
    steps=[...],                       # list of TraceStep objects
    expected_answer=None,              # optional ground truth
    tags=["production", "bug-report"], # free-form labels
)
```

### `TraceStep`

```python
TraceStep(
    step_id="step_001",               # unique within this trace
    agent_name="TechSupportAgent",    # display name in dashboard
    step_type="agent",                # one of: router, agent, tool, synthesizer, llm_call, custom
    parent_step_id="step_000",        # None for the root step only
    io=StepIO(
        input_text="...",             # what the agent received
        output_text="...",            # what the agent said
        tool_name="run_diagnostics",  # if a tool was called
        tool_args={"id": "123"},      # tool arguments
        tool_output='{"status":"ok"}',# raw tool output (key for NLI evidence)
        model="gemini-2.5-flash",     # LLM model used
    ),
    timestamp_ms=1722497000000.0,     # Unix epoch milliseconds
    duration_ms=1234.5,               # wall-clock time
    metadata={"tokens": 847},         # arbitrary key-value pairs
)
```

### `Diagnosis`

```python
Diagnosis(
    trace_id="trace_abc123",
    root_cause_step=StepAttribution(  # None if no hallucination detected
        step_id="step_002",
        agent_name="TechSupportAgent",
        attribution_score=0.312,       # 0.0 (innocent) to ~1.0 (definitive)
        novel_claim_ratio=0.667,       # P(hallucinated) from ensemble NLI
        downstream_impact=0.468,       # P(propagated to final answer)
        novel_claims=[Claim(
            text="There is a known firmware bug.",
            verdict="ungrounded",
            confidence=0.89,
            agreement_score=0.67,      # 2/3 judges agreed
            vote_breakdown={"ungrounded": 2, "grounded": 1},
            evidence="None",           # no supporting evidence found
        )],
    ),
    all_steps=[...],   # all steps ranked by attribution score
    summary="Agent 'TechSupportAgent' (step: step_002) is the root cause...",
    diagnosed_at="2024-08-01T09:05:00+00:00",
)
```

---

## 10. How the Diagnostic Engine Works

### Step 1 — Claim Decomposition

TraceLens sends each agent's output text to an LLM with a structured prompt
that extracts atomic factual claims:

```
Input:  "Your modem shows 0% packet loss. There is a known firmware bug."
Output: [
  "The modem shows 0% packet loss.",
  "There is a known firmware bug in the modem."
]
```

### Step 2 — Ensemble NLI Verification

For each claim, TraceLens calls the NLI judge **N times** (default: 3) and
takes a majority vote:

```
Claim: "There is a known firmware bug in the modem."
Evidence: '{"status": "online", "uptime_hours": 2, "packet_loss": "0%"}'

Vote 1: ungrounded (confidence 0.94) ← no firmware info in tool output
Vote 2: ungrounded (confidence 0.91)
Vote 3: grounded   (confidence 0.60) ← hallucinated spuriously

Majority: ungrounded (2/3 = 0.67 agreement)
Calibrated confidence: mean(0.94, 0.91, 0.60) × 0.67 = 0.81 × 0.67 = 0.54
```

### Step 3 — Attribution Scoring

```
p_hallucinated = 0.54   (from ensemble NLI above)

p_propagated   = overlap(step_claims, final_answer_claims) / len(final_answer_claims)
               = 0.60   (60% of final answer traces back to this step)

attribution_score = p_hallucinated × (0.5 + 0.5 × p_propagated)
                  = 0.54 × (0.5 + 0.5 × 0.60)
                  = 0.54 × 0.80
                  = 0.432
```

### Step 4 — Root Cause Selection

The step with the highest `attribution_score` above `attribution_threshold`
(default: 0.15) is flagged as the root cause. If no step clears the threshold,
the trace is diagnosed as healthy.

### Why This Formula Is Correct

The old formula (`novel_ratio × (1 + descendants)`) penalised root agents because
they have the most descendants. The new formula uses `p_propagated` — the fraction
of the **final answer** that traces back to this step. Leaf agents that directly
produce the final answer get high `p_propagated`; root agents whose output is
transformed many times get low `p_propagated`. This ensures hallucinations close
to the output surface score highest.

---

## 11. What's Next (Roadmap)

Current status after Phase 4:

| Phase | Feature | Status |
|---|---|---|
| 0-4 (original) | Trace schema, capture SDK, DAG builder, store, CLI | ✅ Done |
| 5 | Claim decomposition engine | ✅ Done |
| 6 | Causal attribution scoring | ✅ Done |
| 7 | CLI diagnostic commands | ✅ Done |
| 8 | Defect injection study & sample projects | ✅ Done |
| 9 | Streamlit dashboard | ✅ Done |
| 3B | Ensemble NLI (non-determinism correction) | ✅ Done |
| 3A | Fixed attribution formula (leaf-bias) | ✅ Done |
| 4A | pyvis interactive graph | ✅ Done |
| 4B | Auto-refresh (live dashboard) | ✅ Done |
| **Next: Phase 2** | **Multi-parent DAG (fan-out/fan-in support)** | 🔲 Planned |
| **Next: Phase 1** | **HTTP Ingest API (FastAPI `/v1/spans`)** | 🔲 Planned |
| **Next: Phase 5** | **LangChain / AutoGen / CrewAI integrations** | 🔲 Planned |
| **Next: Phase 6** | **Async diagnosis pipeline** | 🔲 Planned |
| **Next: Phase 7** | **Storage hardening (connection pool, indexes)** | 🔲 Planned |

See [implementation_plan.md](./implementation_plan.md) for detailed specs of the upcoming phases.

---

*TraceLens v0.2.0 — Built with litellm, Pydantic, networkx, Streamlit, pyvis*
