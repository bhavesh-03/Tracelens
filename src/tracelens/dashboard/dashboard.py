"""Main entrypoint for the TraceLens Streamlit Dashboard."""

import sys
from pathlib import Path

import streamlit as st

# Ensure TraceLens is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tracelens.config import load_config
from tracelens.dashboard.components import (
    render_dag,
    render_diagnosis_panel,
    render_step_details,
    render_trace_summary,
)
from tracelens.schema import Trace
from tracelens.store import connect, list_traces, load_diagnosis, load_trace

st.set_page_config(
    page_title="TraceLens | Multi-Agent Diagnostics",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Optional: Add custom CSS for a more premium look
st.markdown("""
<style>
    .reportview-container {
        background: #f8f9fa;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    h1, h2, h3 {
        color: #1e1e1e;
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Load configuration and DB connection
@st.cache_resource
def get_db_connection():
    # In a real app, config path might be passed via env vars
    cfg = load_config()
    return cfg, connect(cfg.db_path)

cfg, conn = get_db_connection()

# --- SIDEBAR ---
st.sidebar.title("🔬 TraceLens")
st.sidebar.markdown("Automated Causal Diagnostics")

st.sidebar.header("Filter Traces")
all_traces = list_traces(conn)

if not all_traces:
    st.warning("No traces found in the database. Run `tracelens ingest` or a subject app.")
    st.stop()

# Extract unique projects and tags
projects = sorted(list(set([t.get("project_name", "default") for t in all_traces])))
selected_project = st.sidebar.selectbox("Project", ["All"] + projects)

# Filter by project
filtered_traces = all_traces
if selected_project != "All":
    filtered_traces = [t for t in filtered_traces if t.get("project_name") == selected_project]
    
# Trace Selection
st.sidebar.header("Recent Traces")
trace_options = {}
for t in filtered_traces:
    # Format: trace_123... (project) - Score: X
    score = t.get("attribution_score")
    score_str = f"| Score: {score:.2f}" if score is not None else ""
    label = f"{t['trace_id'][:12]}... {score_str}"
    trace_options[label] = t['trace_id']

selected_label = st.sidebar.radio("Select a Trace:", list(trace_options.keys()))

# --- MAIN AREA ---
if selected_label:
    trace_id = trace_options[selected_label]
    
    # Load data
    try:
        trace_dict = load_trace(conn, trace_id)
        trace = Trace.model_validate(trace_dict)
        diagnosis_dict = load_diagnosis(conn, trace_id)
        
        from tracelens.schema import Diagnosis, StepAttribution, Claim
        import json
        
        diagnosis = None
        if diagnosis_dict:
            all_steps = []
            if diagnosis_dict.get("step_scores_json"):
                steps_data = json.loads(diagnosis_dict["step_scores_json"])
                for s in steps_data:
                    claims = [Claim(**c) for c in s.get("novel_claims", [])]
                    attr = StepAttribution(
                        step_id=s["step_id"],
                        agent_name=s["agent_name"],
                        step_type=s.get("step_type", "agent"),
                        attribution_score=s["attribution_score"],
                        novel_claim_ratio=s["novel_claim_ratio"],
                        downstream_impact=s["downstream_impact"],
                        novel_claims=claims
                    )
                    all_steps.append(attr)
                    
            root_cause = None
            root_id = diagnosis_dict.get("root_cause_step_id")
            if root_id:
                root_cause = next((s for s in all_steps if s.step_id == root_id), None)
                
            diagnosis = Diagnosis(
                trace_id=trace_id,
                root_cause_step=root_cause,
                all_steps=all_steps,
                summary=diagnosis_dict.get("summary", "")
            )
        
    except Exception as e:
        import traceback
        st.error(f"Error loading trace: {e}")
        st.code(traceback.format_exc(), language="python")
        st.stop()

    st.title("Trace Explorer")
    st.caption(f"ID: `{trace.trace_id}` | Date: `{trace_dict.get('created_at', 'Unknown')}`")
    
    # Top Summary Banner
    render_trace_summary(trace, diagnosis)
    st.divider()
    
    # Main Dashboard Body
    col1, col2 = st.columns([1.2, 1.0])
    
    with col1:
        if diagnosis:
            render_diagnosis_panel(diagnosis)
        else:
            st.warning("⚠️ This trace has not been diagnosed yet. Run `tracelens diagnose <trace_id>` via CLI to compute causal attribution scores.")
            st.info("The DAG below will render in default colors because attribution scores are missing.")
            
        st.divider()
        render_step_details(trace)
        
    with col2:
        render_dag(trace, diagnosis)
