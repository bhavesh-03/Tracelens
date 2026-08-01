"""TraceLens Dashboard — Live Multi-Agent Execution Monitor.

Auto-refreshes every 5 seconds. When you run a chatbot or multi-agent pipeline
that is instrumented with TraceLens, new traces will automatically appear here
without any manual refresh.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

# Ensure the tracelens package is importable when running directly via streamlit
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tracelens.config import load_config
from tracelens.dashboard.components import (
    render_dag,
    render_diagnosis_panel,
    render_step_details,
    render_timeline,
    render_trace_summary,
)
from tracelens.schema import Trace
from tracelens.store import connect, list_traces, load_diagnosis, load_trace

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TraceLens | Multi-Agent Diagnostics",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Auto-refresh — the key to "live" updates without manual page reload
# ---------------------------------------------------------------------------
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=5000, key="live_refresh")  # refresh every 5 seconds
except ImportError:
    pass  # graceful degradation if not installed

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #111827;
    }
    [data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }
    
    /* Main content */
    .main .block-container {
        padding-top: 1.5rem;
        max-width: 1400px;
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background: #1f2937;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Live indicator */
    .live-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #22c55e;
        animation: pulse 1.5s infinite;
        margin-right: 6px;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }
    
    h1, h2, h3 { color: #f9fafb !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# DB connection — cached at the resource level (connection reuse, not data reuse)
# ---------------------------------------------------------------------------
@st.cache_resource
def _get_connection():
    cfg = load_config()
    return cfg, connect(cfg.db_path)

cfg, conn = _get_connection()

# ---------------------------------------------------------------------------
# SIDEBAR — Project filter + live trace list
# ---------------------------------------------------------------------------
st.sidebar.markdown("""
<div style="padding: 0.5rem 0 1rem 0;">
    <h2 style="margin: 0; font-size: 1.4rem; color: white;">🔬 TraceLens</h2>
    <p style="margin: 0; font-size: 0.8rem; color: #9ca3af;">Multi-Agent Causal Diagnostics</p>
</div>
""", unsafe_allow_html=True)

# Always fetch fresh — this runs on every 5-second auto-refresh cycle
all_traces = list_traces(conn, limit=100)

# Live indicator
n_undiagnosed = sum(1 for t in all_traces if t.get("attribution_score") is None)
st.sidebar.markdown(
    f'<span class="live-dot"></span> **Live** — {len(all_traces)} traces total'
    + (f", {n_undiagnosed} pending diagnosis" if n_undiagnosed else ""),
    unsafe_allow_html=True,
)
st.sidebar.divider()

if not all_traces:
    st.warning(
        "⚠️ No traces found. Instrument your app with `TraceLensCapture` "
        "and run it to see traces appear here automatically."
    )
    st.code("""
from tracelens.capture import TraceLensCapture
from tracelens.store import connect, save_trace

tracer = TraceLensCapture(project_name="my_project")

with tracer.step("RouterAgent", step_type="router", input_text=query) as io:
    io.output_text = route(query)

trace = tracer.finalize(query=query, final_answer=final_answer)
save_trace(connect("tracelens.db"), trace)
    """, language="python")
    st.stop()

# Project filter
projects = sorted({t.get("project_name", "default") for t in all_traces})
selected_project = st.sidebar.selectbox("📂 Project", ["All Projects"] + projects)

filtered = all_traces
if selected_project != "All Projects":
    filtered = [t for t in filtered if t.get("project_name") == selected_project]

# Trace list
st.sidebar.markdown("**Recent Traces**")
trace_options: dict[str, str] = {}
for t in filtered[:30]:  # show last 30
    score = t.get("attribution_score")
    root = t.get("root_cause_agent")
    tid = t["trace_id"]

    if score is None:
        status_icon = "⏳"
        detail = "not diagnosed"
    elif root:
        status_icon = "🔴"
        detail = f"{root} ({score:.2f})"
    else:
        status_icon = "🟢"
        detail = f"healthy ({score:.2f})"

    short_id = tid[:14] + "..."
    label = f"{status_icon} {short_id}\n   {detail}"
    trace_options[label] = tid

selected_label = st.sidebar.radio(
    "Select trace:", list(trace_options.keys()), label_visibility="collapsed"
)

# ---------------------------------------------------------------------------
# MAIN CONTENT
# ---------------------------------------------------------------------------
if not selected_label:
    st.info("Select a trace from the sidebar.")
    st.stop()

trace_id = trace_options[selected_label]

try:
    trace_dict = load_trace(conn, trace_id)
    trace = Trace.model_validate(trace_dict)
    diagnosis_dict = load_diagnosis(conn, trace_id)

    # Reconstruct Diagnosis object from raw DB row
    from tracelens.schema import Claim, Diagnosis, StepAttribution

    diagnosis = None
    if diagnosis_dict:
        all_steps = []
        if diagnosis_dict.get("step_scores_json"):
            for s in json.loads(diagnosis_dict["step_scores_json"]):
                claims = [Claim(**c) for c in s.get("novel_claims", [])]
                all_steps.append(StepAttribution(
                    step_id=s["step_id"],
                    agent_name=s["agent_name"],
                    step_type=s.get("step_type", "agent"),
                    attribution_score=s["attribution_score"],
                    novel_claim_ratio=s["novel_claim_ratio"],
                    downstream_impact=s["downstream_impact"],
                    novel_claims=claims,
                    total_claims=s.get("total_claims", len(claims)),
                ))

        root_cause = None
        root_id = diagnosis_dict.get("root_cause_step_id")
        if root_id:
            root_cause = next((s for s in all_steps if s.step_id == root_id), None)

        diagnosis = Diagnosis(
            trace_id=trace_id,
            root_cause_step=root_cause,
            all_steps=all_steps,
            summary=diagnosis_dict.get("summary", ""),
        )

except Exception as e:
    import traceback
    st.error(f"Error loading trace `{trace_id}`: {e}")
    st.code(traceback.format_exc(), language="python")
    st.stop()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    f"## Trace Explorer  "
    f"<span style='font-size:0.75rem; color:#9ca3af; font-weight:400;'>"
    f"`{trace.trace_id}` · {trace.project_name} · {trace_dict.get('created_at', '')[:19]}"
    f"</span>",
    unsafe_allow_html=True,
)

if not diagnosis:
    st.warning(
        "⚠️ This trace has not been diagnosed yet. "
        "Run `tracelens diagnose <trace_id>` to compute attribution scores."
    )

# Summary metrics banner
render_trace_summary(trace, diagnosis)
st.divider()

# ---------------------------------------------------------------------------
# Two-column layout: Left = Analysis | Right = Graph
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1.1, 0.9], gap="large")

with col_left:
    if diagnosis:
        render_diagnosis_panel(diagnosis)
    else:
        st.info("Run diagnosis to see the causal attribution analysis here.")

    st.divider()
    render_timeline(trace)
    st.divider()
    render_step_details(trace)

with col_right:
    render_dag(trace, diagnosis)
