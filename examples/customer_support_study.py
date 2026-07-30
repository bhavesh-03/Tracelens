"""
Customer Support Multi-Agent Defect Study.

This script simulates a multi-agent customer support system.
It generates a batch of traces (both healthy and defective) to populate
the TraceLens database for dashboard visualization.

Project: 'customer_support'
Agents:
- RouterAgent
- BillingAgent
- TechSupportAgent
"""

import asyncio
import json
import uuid
from datetime import datetime, UTC


from tracelens.schema import TraceStep, StepIO
from tracelens.store import connect, save_trace
from tracelens.cli import diagnose


def _generate_trace(is_defective: bool, index: int) -> tuple[str, list[TraceStep]]:
    trace_id = f"cs_trace_{index}_{uuid.uuid4().hex[:6]}"
    
    # 1. Router Agent
    router_input = "My internet keeps dropping every 10 minutes and my bill is high."
    router_output = "Routing to TechSupportAgent for the connection issue."
    
    router_step = TraceStep(
        step_id=f"step_{uuid.uuid4().hex[:6]}",
        agent_name="RouterAgent",
        step_type="router",
        parent_step_id=None,
        io=StepIO(input_text=router_input, output_text=router_output),
        timestamp_ms=datetime.now(UTC).timestamp() * 1000,
        duration_ms=150.0,
    )
    
    # 2. TechSupportAgent - Tool Call
    tool_input = "Running diagnostics on customer modem."
    tool_output = '{"status": "online", "uptime_hours": 2, "packet_loss": "0%"}'
    
    tool_step = TraceStep(
        step_id=f"step_{uuid.uuid4().hex[:6]}",
        agent_name="DiagnosticTool",
        step_type="tool",
        parent_step_id=router_step.step_id,
        io=StepIO(
            input_text=tool_input, 
            output_text=tool_output,
            tool_name="run_diagnostics",
            tool_args={"customer_id": "12345"}
        ),
        timestamp_ms=datetime.now(UTC).timestamp() * 1000 + 200,
        duration_ms=500.0,
    )
    
    # 3. TechSupportAgent - Final Answer
    tech_input = f"User query: {router_input}\nTool data: {tool_output}"
    
    if is_defective:
        # Hallucination: Claims a firmware bug that was never in the tool output
        tech_output = (
            "I checked your modem. It shows as online, but there is a known firmware "
            "bug causing drops every 10 minutes. We will send a technician to replace it."
        )
    else:
        # Healthy: Grounded in tool output
        tech_output = (
            "I ran diagnostics on your modem. It currently shows as online with 0% packet loss. "
            "However, since it only has 2 hours of uptime, I recommend keeping an eye on it. "
            "Please call back if it drops again."
        )
        
    tech_step = TraceStep(
        step_id=f"step_{uuid.uuid4().hex[:6]}",
        agent_name="TechSupportAgent",
        step_type="agent",
        parent_step_id=tool_step.step_id,
        io=StepIO(input_text=tech_input, output_text=tech_output),
        timestamp_ms=datetime.now(UTC).timestamp() * 1000 + 1000,
        duration_ms=1200.0,
    )
    
    trace_id = f"cs_trace_{index}_{uuid.uuid4().hex[:6]}"
    return trace_id, router_input, [router_step, tool_step, tech_step]

async def run_study():
    print("Generating Customer Support traces...")
    
    conn = connect("tracelens.db")
    
    # Generate 5 traces: 3 healthy, 2 defective
    scenarios = [False, False, True, False, True]
    
    for i, is_defective in enumerate(scenarios):
        print(f"\n--- Generating Trace {i+1} (Defective: {is_defective}) ---")
        
        trace_id, query, steps = _generate_trace(is_defective, i+1)
        final_answer = steps[-1].io.output_text
        
        from tracelens.schema import Trace
        trace = Trace(
            trace_id=trace_id,
            project_name="customer_support",
            query=query,
            final_answer=final_answer,
            steps=steps,
            tags=["demo", "batch_gen", "defective" if is_defective else "healthy"]
        )
        
        # Save to DB
        save_trace(conn, trace)
        print(f"Saved trace {trace.trace_id}")
        
        # Run diagnosis engine on it
        print("Diagnosing...")
        import typer
        try:
            diagnose(trace_id=trace.trace_id, config=None, verbose=False)
        except typer.Exit:
            pass
            
    print("\n✅ Study complete! Open the dashboard to see the 'customer_support' project.")

if __name__ == "__main__":
    asyncio.run(run_study())
