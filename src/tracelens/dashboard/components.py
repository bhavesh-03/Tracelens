"""Reusable UI components for the TraceLens Streamlit Dashboard."""

import json

import graphviz
import streamlit as st

from tracelens.dag import build_dag
from tracelens.schema import Diagnosis, StepAttribution, Trace


def render_trace_summary(trace: Trace, diagnosis: Diagnosis | None):
    """Renders the top summary banner."""
    st.markdown("### Trace Details")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Project", trace.project_name)
    col2.metric("Total Steps", len(trace.steps))
    
    status = "✅ Healthy"
    if diagnosis and diagnosis.root_cause_step:
        status = "❌ Hallucination Detected"
    col3.metric("Status", status)
    
    st.markdown("#### User Query")
    st.info(trace.query)
    
    st.markdown("#### Final Answer")
    st.success(trace.final_answer)

def render_dag(trace: Trace, diagnosis: Diagnosis | None):
    """Renders the execution graph using Graphviz."""
    st.markdown("### Execution DAG")
    
    try:
        dag = build_dag(trace)
    except Exception as e:
        st.error(f"Failed to build DAG: {e}")
        return

    attr_map: dict[str, StepAttribution] = {}
    if diagnosis:
        for attr in diagnosis.all_steps:
            attr_map[attr.step_id] = attr
            
    root_cause_id = diagnosis.root_cause_step.step_id if diagnosis and diagnosis.root_cause_step else None

    # Render steps in chronological order using native Streamlit elements
    for i, step in enumerate(trace.steps):
        node_id = step.step_id
        
        # Determine the color/status based on attribution score
        if node_id == root_cause_id:
            score = attr_map[node_id].attribution_score
            st.error(f"🚨 **{step.agent_name}** ({step.step_type}) — **Root Cause (Score: {score:.2f})**")
        elif node_id in attr_map and attr_map[node_id].attribution_score > 0.05:
            score = attr_map[node_id].attribution_score
            st.warning(f"⚠️ **{step.agent_name}** ({step.step_type}) — Suspicious (Score: {score:.2f})")
        elif diagnosis:
            st.success(f"✅ **{step.agent_name}** ({step.step_type}) — Innocent")
        elif step.step_type == "tool":
            st.info(f"🔧 **{step.agent_name}** (tool)")
        else:
            st.info(f"🤖 **{step.agent_name}** ({step.step_type})")
            
        # Draw arrow pointing down
        if i < len(trace.steps) - 1:
            st.markdown("<div style='text-align: center; color: #888; font-size: 24px; line-height: 0.5;'>↓</div><br/>", unsafe_allow_html=True)

def render_diagnosis_panel(diagnosis: Diagnosis):
    """Renders the mathematical diagnosis panel."""
    st.markdown("### Causal Attribution Analysis")
    
    if not diagnosis.root_cause_step:
        st.success("No ungrounded claims or hallucinations detected. The trace is healthy.")
        return
        
    st.error(f"**Root Cause Detected:** Agent '{diagnosis.root_cause_step.agent_name}'")
    st.write(diagnosis.summary)
    
    with st.expander("View Mathematical Scoring Breakdown"):
        # Format the all_steps list for a dataframe
        data = []
        for attr in diagnosis.all_steps:
            data.append({
                "Agent": attr.agent_name,
                "Score": round(attr.attribution_score, 3),
                "Novel Claim Ratio": round(attr.novel_claim_ratio, 3),
                "Impact": round(attr.downstream_impact, 3)
            })
        st.dataframe(data, use_container_width=True)
        
    if diagnosis.root_cause_step.novel_claims:
        st.markdown("#### Hallucinated Claims Identified")
        for claim in diagnosis.root_cause_step.novel_claims:
            st.warning(
                f"**Claim:** \"{claim.text}\"\n\n"
                f"**Judge Verdict:** {claim.verdict} (Confidence: {claim.confidence:.2f})\n\n"
                f"**Available Evidence:** {claim.evidence}"
            )

def render_step_details(trace: Trace):
    """Renders an interactive step explorer."""
    st.markdown("### Step Explorer")
    
    step_options = {f"{s.agent_name} ({s.step_id})": s for s in trace.steps}
    selected = st.selectbox("Select a step to view raw inputs/outputs:", list(step_options.keys()))
    
    if selected:
        step = step_options[selected]
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Input Context")
            st.code(step.io.input_text or "None", language="text")
            
        with col2:
            st.markdown("#### Output Response")
            st.code(step.io.output_text or "None", language="text")
            
        if step.io.tool_name:
            st.markdown("#### Tool Call")
            st.write(f"**Tool:** `{step.io.tool_name}`")
            st.json(step.io.tool_args)
            st.markdown("**Tool Output:**")
            st.code(step.io.tool_output or "None", language="text")
            
        with st.expander("Raw Step JSON"):
            st.json(json.loads(step.model_dump_json()))
