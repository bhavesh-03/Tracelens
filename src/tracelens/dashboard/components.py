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

    # Extract attribution data for quick lookup
    attr_map: dict[str, StepAttribution] = {}
    if diagnosis:
        for attr in diagnosis.all_steps:
            attr_map[attr.step_id] = attr
            
    root_cause_id = diagnosis.root_cause_step.step_id if diagnosis and diagnosis.root_cause_step else None

    # Create Graphviz graph
    dot = graphviz.Digraph(engine="dot")
    dot.attr(rankdir="TB", size="10,10")
    
    # Define nodes
    for step in trace.steps:
        node_id = step.step_id
        label_lines = [
            f"<<B>{step.agent_name}</B>",
            f"<I>({step.step_type})</I>"
        ]
        
        # Color styling
        fillcolor = "white"
        fontcolor = "black"
        shape = "box"
        style = "filled,rounded"
        
        if step.step_type == "tool":
            shape = "ellipse"
            fillcolor = "#f0f2f6"
            
        if node_id == root_cause_id:
            fillcolor = "#ff4b4b" # Streamlit Red
            fontcolor = "white"
            label_lines.append(f"<BR/><B>Score: {attr_map[node_id].attribution_score:.2f}</B>")
        elif node_id in attr_map and attr_map[node_id].attribution_score > 0.05:
            # Minor score / innocent but involved
            fillcolor = "#ffa421" # Streamlit Orange
            fontcolor = "white"
        elif diagnosis:
            # Evaluated and cleared
            fillcolor = "#00c04b" # Streamlit Green
            fontcolor = "white"
            
        label = "<" + "<BR/>".join(label_lines) + ">"
        
        dot.node(
            node_id,
            label=label,
            shape=shape,
            style=style,
            fillcolor=fillcolor,
            fontcolor=fontcolor,
            penwidth="2" if node_id == root_cause_id else "1"
        )
        
    # Define edges (Parent -> Child flow)
    for step in trace.steps:
        if step.parent_step_id:
            dot.edge(step.parent_step_id, step.step_id, color="#888888")
            
    st.graphviz_chart(dot, use_container_width=True)

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
