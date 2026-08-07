"""AutoGen integration for TraceLens."""

from __future__ import annotations

from typing import Any

from tracelens.capture import TraceLensCapture


def instrument_autogen_agent(agent: Any, tracer: TraceLensCapture) -> None:
    """Instrument a pyautogen ConversableAgent to capture replies.
    
    Wraps the generate_reply function to log the incoming message and the generated reply.
    """
    original_generate_reply = agent.generate_reply
    
    def wrapped_generate_reply(messages=None, sender=None, **kwargs) -> Any:
        input_text = ""
        if messages:
            input_text = messages[-1].get("content", "")
        elif agent.chat_messages and sender in agent.chat_messages:
            input_text = agent.chat_messages[sender][-1].get("content", "")
            
        with tracer.step(agent_name=agent.name, step_type="agent", input_text=str(input_text)) as io:
            reply = original_generate_reply(messages=messages, sender=sender, **kwargs)
            io.output_text = str(reply) if reply else ""
            return reply
            
    agent.generate_reply = wrapped_generate_reply
