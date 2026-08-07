"""CrewAI integration for TraceLens."""

from __future__ import annotations

from typing import Any

from tracelens.capture import TraceLensCapture


def instrument_crew(crew: Any, tracer: TraceLensCapture) -> None:
    """Instrument a CrewAI crew to capture traces.
    
    This function modifies the crew's agents to use step callbacks, capturing
    tool usage and reasoning steps.
    """
    
    def _create_step_callback(agent_name: str, original_callback: Any = None) -> Any:
        def callback(agent_action: Any) -> None:
            # We use add_step to record the action
            input_text = getattr(agent_action, "tool_input", "")
            tool_name = getattr(agent_action, "tool", None)
            output_text = getattr(agent_action, "log", getattr(agent_action, "result", ""))
            
            tracer.add_step(
                agent_name=agent_name,
                step_type="tool" if tool_name else "agent",
                input_text=str(input_text),
                output_text=str(output_text),
                tool_name=tool_name,
            )
            
            if original_callback:
                original_callback(agent_action)
        return callback
    
    # Instrument each agent's step_callback
    for agent in crew.agents:
        agent.step_callback = _create_step_callback(
            agent_name=agent.role,
            original_callback=agent.step_callback,
        )
