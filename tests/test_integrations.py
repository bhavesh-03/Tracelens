"""Tests for third-party integrations."""

from unittest.mock import MagicMock

from tracelens.capture import TraceLensCapture
from tracelens.integrations.autogen import instrument_autogen_agent
from tracelens.integrations.crewai import instrument_crew
from tracelens.integrations.langchain import TraceLensCallbackHandler


def test_langchain_callback():
    tracer = TraceLensCapture()
    handler = TraceLensCallbackHandler(tracer)
    
    # Simulate chain start
    handler.on_chain_start({"name": "MyChain"}, {"input": "hello"}, run_id="run1")
    # Simulate tool start
    handler.on_tool_start({"name": "SearchTool"}, "search query", run_id="run2", parent_run_id="run1")
    # Simulate tool end
    handler.on_tool_end("search result", run_id="run2")
    # Simulate chain end
    handler.on_chain_end({"output": "final answer"}, run_id="run1")
    
    assert tracer.step_count == 2
    trace = tracer.finalize(query="hello", final_answer="final answer")
    
    tool_step = trace.get_step(handler._step_ids["run2"])
    chain_step = trace.get_step(handler._step_ids["run1"])
    
    assert tool_step.agent_name == "SearchTool"
    assert tool_step.step_type == "tool"
    assert "search query" in tool_step.io.input_text
    assert "search result" in tool_step.io.output_text
    assert tool_step.parent_step_id == chain_step.step_id
    
    assert chain_step.agent_name == "MyChain"
    assert chain_step.step_type == "router"


def test_crewai_instrumentation():
    tracer = TraceLensCapture()
    
    # Mock Crew and Agent
    agent1 = MagicMock()
    agent1.role = "Researcher"
    agent1.step_callback = None
    
    crew = MagicMock()
    crew.agents = [agent1]
    
    instrument_crew(crew, tracer)
    
    # Agent should now have a step_callback
    assert agent1.step_callback is not None
    
    # Simulate agent action
    mock_action = MagicMock()
    mock_action.tool_input = "query"
    mock_action.tool = "SearchTool"
    mock_action.log = "Found something"
    
    agent1.step_callback(mock_action)
    
    assert tracer.step_count == 1
    trace = tracer.finalize(query="q", final_answer="a")
    step = trace.steps[0]
    
    assert step.agent_name == "Researcher"
    assert step.step_type == "tool"
    assert step.io.tool_name == "SearchTool"
    assert step.io.input_text == "query"
    assert step.io.output_text == "Found something"


def test_autogen_instrumentation():
    tracer = TraceLensCapture()
    
    # Mock AutoGen Agent
    agent = MagicMock()
    agent.name = "Assistant"
    
    # Mock original generate_reply
    def mock_generate_reply(messages=None, sender=None, **kwargs):
        return "I can help with that."
    
    agent.generate_reply = mock_generate_reply
    
    instrument_autogen_agent(agent, tracer)
    
    # Call the wrapped function
    reply = agent.generate_reply(messages=[{"content": "Help me"}])
    
    assert reply == "I can help with that."
    assert tracer.step_count == 1
    
    trace = tracer.finalize(query="q", final_answer="a")
    step = trace.steps[0]
    
    assert step.agent_name == "Assistant"
    assert step.step_type == "agent"
    assert step.io.input_text == "Help me"
    assert step.io.output_text == "I can help with that."
