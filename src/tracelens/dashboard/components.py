"""Reusable UI components for the TraceLens Streamlit Dashboard."""

from __future__ import annotations

import json
import os
import tempfile

import streamlit as st
import streamlit.components.v1 as components

from tracelens.dag import build_dag
from tracelens.schema import Diagnosis, StepAttribution, Trace


def render_trace_summary(trace: Trace, diagnosis: Diagnosis | None) -> None:
    """Renders the top summary banner with metrics."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Project", trace.project_name)
    col2.metric("Steps", len(trace.steps))

    if diagnosis:
        if diagnosis.root_cause_step:
            score = diagnosis.root_cause_step.attribution_score
            col3.metric("Status", "❌ Hallucination", delta=f"Score {score:.2f}", delta_color="inverse")
            col4.metric("Root Cause", diagnosis.root_cause_step.agent_name)
        else:
            col3.metric("Status", "✅ Healthy")
            col4.metric("Root Cause", "None")
    else:
        col3.metric("Status", "⏳ Not Diagnosed")
        col4.metric("Root Cause", "—")

    with st.expander("📝 Query & Final Answer", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**User Query**")
            st.info(trace.query)
        with c2:
            st.markdown("**Final Answer**")
            st.success(trace.final_answer)


def render_dag(trace: Trace, diagnosis: Diagnosis | None) -> None:
    """Renders an interactive execution graph using PyVis."""
    st.markdown("### 🗺️ Execution Graph")

    try:
        build_dag(trace)  # Validate the DAG before rendering
    except Exception as e:
        st.error(f"Failed to build execution graph: {e}")
        return

    attr_map: dict[str, StepAttribution] = {}
    if diagnosis:
        for attr in diagnosis.all_steps:
            attr_map[attr.step_id] = attr

    root_cause_id = (
        diagnosis.root_cause_step.step_id
        if diagnosis and diagnosis.root_cause_step
        else None
    )

    try:
        from pyvis.network import Network

        net = Network(
            height="480px",
            width="100%",
            directed=True,
            bgcolor="#0e1117",
            font_color="#ffffff",
        )

        # Physics layout: hierarchical top-down for clean agent flow
        net.set_options("""{
          "layout": {
            "hierarchical": {
              "enabled": true,
              "direction": "UD",
              "sortMethod": "directed",
              "levelSeparation": 120,
              "nodeSpacing": 160
            }
          },
          "physics": { "enabled": false },
          "nodes": {
            "shape": "box",
            "borderWidth": 2,
            "borderWidthSelected": 4,
            "font": { "size": 13, "face": "Inter, Arial, sans-serif" },
            "shadow": { "enabled": true, "size": 6, "x": 2, "y": 2 }
          },
          "edges": {
            "arrows": { "to": { "enabled": true, "scaleFactor": 1.2 } },
            "color": { "color": "#555555", "highlight": "#ffffff" },
            "smooth": { "type": "cubicBezier", "forceDirection": "vertical" },
            "width": 2
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 100
          }
        }""")

        for step in trace.steps:
            sid = step.step_id
            attr = attr_map.get(sid)
            score = attr.attribution_score if attr else 0.0
            n_novel = len(attr.novel_claims) if attr else 0

            # Node colours
            if sid == root_cause_id:
                bg, border = "#c0392b", "#ff6b6b"
                emoji = "🚨"
            elif attr and score > 0.05:
                bg, border = "#d35400", "#ffa95a"
                emoji = "⚠️"
            elif diagnosis and attr:
                bg, border = "#1a5c2a", "#27ae60"
                emoji = "✅"
            elif step.step_type == "tool":
                bg, border = "#1a3a5c", "#3498db"
                emoji = "🔧"
            else:
                bg, border = "#2c2c3e", "#7f8c8d"
                emoji = "🤖"

            label = f"{emoji} {step.agent_name}\n({step.step_type})"
            if attr:
                label += f"\nScore: {score:.3f}"

            # Tooltip with full details
            tooltip_lines = [
                f"<b>{step.agent_name}</b>",
                f"Type: {step.step_type}",
                f"Step ID: {step.step_id}",
            ]
            if attr:
                tooltip_lines += [
                    f"Attribution Score: {score:.4f}",
                    f"Novel Claim Ratio: {attr.novel_claim_ratio:.4f}",
                    f"Propagation to Answer: {attr.downstream_impact:.4f}",
                    f"Novel Claims: {n_novel}",
                ]
            if n_novel > 0 and attr:
                first_claim = attr.novel_claims[0]
                tooltip_lines.append(
                    f'<br/><i>Sample hallucination:</i><br/>"{first_claim.text[:80]}..."'
                )
            tooltip = "<br/>".join(tooltip_lines)

            net.add_node(
                sid,
                label=label,
                color={"background": bg, "border": border, "highlight": {"background": "#f0f0f0", "border": border}},
                title=tooltip,
                margin=10,
            )

        # Add edges
        for step in trace.steps:
            parent_ids = getattr(step, "parent_span_ids", None)
            if not parent_ids:
                pid = getattr(step, "parent_step_id", None)
                parent_ids = [pid] if pid else []

            for pid in parent_ids:
                net.add_edge(pid, step.step_id)

        # Write to temp file and render inline
        with tempfile.NamedTemporaryFile(
            suffix=".html", delete=False, mode="w", encoding="utf-8"
        ) as f:
            tmppath = f.name
            net.save_graph(tmppath)

        html_content = open(tmppath, encoding="utf-8").read()
        os.unlink(tmppath)
        components.html(html_content, height=500, scrolling=False)

    except Exception as e:
        st.error(f"Graph rendering failed: {e}")
        # Fallback: plain text list
        for step in trace.steps:
            attr = attr_map.get(step.step_id)
            score = attr.attribution_score if attr else 0.0
            if step.step_id == root_cause_id:
                st.error(f"🚨 **{step.agent_name}** — Root Cause (Score: {score:.3f})")
            elif attr and score > 0.05:
                st.warning(f"⚠️ **{step.agent_name}** — Suspicious (Score: {score:.3f})")
            elif diagnosis:
                st.success(f"✅ **{step.agent_name}** — Innocent")
            else:
                st.info(f"🤖 **{step.agent_name}** ({step.step_type})")


def render_timeline(trace: Trace) -> None:
    """Renders a Gantt-style execution timeline showing parallel agent runs."""
    st.markdown("### ⏱️ Execution Timeline")

    import plotly.express as px
    import pandas as pd
    from datetime import datetime, timezone

    data = []
    for step in trace.steps:
        start_dt = datetime.fromtimestamp(step.timestamp_ms / 1000, tz=timezone.utc)
        # Ensure a minimum visible bar width of 100ms for very fast steps
        duration_ms = max(step.duration_ms, 100)
        end_dt = datetime.fromtimestamp((step.timestamp_ms + duration_ms) / 1000, tz=timezone.utc)
        data.append({
            "Agent": f"{step.agent_name} ({step.step_type})",
            "Start": start_dt,
            "Finish": end_dt,
            "Step ID": step.step_id,
            "Duration (ms)": round(step.duration_ms, 1),
        })

    if not data:
        st.warning("No timing data available for this trace.")
        return

    df = pd.DataFrame(data)

    fig = px.timeline(
        df,
        x_start="Start",
        x_end="Finish",
        y="Agent",
        color="Agent",
        hover_data=["Step ID", "Duration (ms)"],
        title="Agent Execution Timeline",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
        height=280,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#333333")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)


def render_diagnosis_panel(diagnosis: Diagnosis) -> None:
    """Renders the mathematical diagnosis panel with ensemble NLI details."""
    st.markdown("### 🔬 Causal Attribution Analysis")

    if not diagnosis.root_cause_step:
        st.success(
            "✅ No hallucinations detected. All agent claims are grounded in their inputs."
        )
    else:
        st.error(
            f"**Root Cause Detected:** `{diagnosis.root_cause_step.agent_name}` "
            f"— Attribution Score: **{diagnosis.root_cause_step.attribution_score:.3f}**"
        )
        st.markdown(f"> {diagnosis.summary}")

    # Attribution breakdown table
    with st.expander("📊 View Full Attribution Breakdown", expanded=bool(diagnosis.root_cause_step)):
        data = []
        for attr in diagnosis.all_steps:
            is_root = (
                diagnosis.root_cause_step
                and attr.step_id == diagnosis.root_cause_step.step_id
            )
            row = {
                "": "🚨" if is_root else "",
                "Agent": attr.agent_name,
                "Type": attr.step_type,
                "Attribution Score": round(attr.attribution_score, 4),
                "P(Hallucinated)": round(attr.novel_claim_ratio, 4),
                "P(Propagated to Answer)": round(attr.downstream_impact, 4),
                "Novel Claims": len(attr.novel_claims),
                "Total Claims": attr.total_claims,
            }
            data.append(row)
        st.dataframe(data, use_container_width=True, hide_index=True)

    # Show hallucinated claims with ensemble NLI details
    if diagnosis.root_cause_step and diagnosis.root_cause_step.novel_claims:
        st.markdown("#### 🚩 Hallucinated Claims (Identified by Ensemble NLI)")
        for i, claim in enumerate(diagnosis.root_cause_step.novel_claims, 1):
            with st.expander(
                f"Claim #{i}: \"{claim.text[:80]}{'...' if len(claim.text) > 80 else ''}\"",
                expanded=(i == 1),
            ):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Verdict**")
                    st.code(claim.verdict or "unknown", language="text")
                    st.markdown("**Calibrated Confidence**")
                    st.progress(claim.confidence)
                    st.caption(f"{claim.confidence:.1%}")

                with c2:
                    st.markdown("**NLI Ensemble Vote Breakdown**")
                    if claim.vote_breakdown:
                        st.json(claim.vote_breakdown)
                        st.caption(f"Agreement: {claim.agreement_score:.0%}")
                    else:
                        st.caption("No ensemble data (single-vote mode)")

                st.markdown("**Full Claim Text**")
                st.info(f'"{claim.text}"')

                st.markdown("**Evidence Available to Agent**")
                st.warning(claim.evidence or "No supporting evidence found in parent outputs.")


def render_step_details(trace: Trace) -> None:
    """Renders an interactive step explorer with raw I/O inspection."""
    st.markdown("### 🔍 Step Inspector")

    step_options = {
        f"{s.agent_name} ({s.step_type}) [{s.step_id}]": s for s in trace.steps
    }
    selected = st.selectbox(
        "Select a step to inspect:",
        list(step_options.keys()),
        key="step_inspector_select",
    )

    if not selected:
        return

    step = step_options[selected]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**📥 Input Context**")
        st.code(step.io.input_text or "(empty)", language="text")
    with c2:
        st.markdown("**📤 Output Response**")
        st.code(step.io.output_text or "(empty)", language="text")

    if step.io.tool_name:
        st.markdown("**🔧 Tool Call**")
        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown(f"`{step.io.tool_name}`")
            st.json(step.io.tool_args or {})
        with tc2:
            st.markdown("**Tool Output:**")
            st.code(step.io.tool_output or "(empty)", language="text")

    meta_cols = st.columns(3)
    meta_cols[0].metric("Duration", f"{step.duration_ms:.0f} ms")
    meta_cols[1].metric("Model", step.io.model or "—")
    meta_cols[2].metric("Parent Step", step.parent_step_id or "Root")

    with st.expander("Raw Step JSON"):
        st.json(json.loads(step.model_dump_json()))
