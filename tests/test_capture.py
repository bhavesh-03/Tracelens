"""Tests for tracelens.capture — trace capture SDK."""

from __future__ import annotations

import pytest

from tracelens.capture import TraceLensCapture


class TestDecorator:
    """Tests for the @trace_step decorator on sync functions."""

    def test_basic_capture(self) -> None:
        tracer = TraceLensCapture(trace_id="test_trace")

        @tracer.trace_step(agent_name="router", step_type="router")
        def route(query: str) -> str:
            return "agent_a"

        @tracer.trace_step(agent_name="agent_a", step_type="agent")
        def handle(query: str) -> str:
            return "the answer is 42"

        route("what is the meaning of life?")
        handle("what is the meaning of life?")

        assert tracer.step_count == 2

    def test_finalize_produces_valid_trace(self) -> None:
        tracer = TraceLensCapture(trace_id="test_trace")

        @tracer.trace_step(agent_name="root", step_type="router")
        def root_fn(query: str) -> str:
            child_fn(query)
            return "done"

        @tracer.trace_step(agent_name="child", step_type="agent")
        def child_fn(query: str) -> str:
            return "child result"

        root_fn("test query")

        trace = tracer.finalize(
            query="test query",
            final_answer="done",
        )

        assert trace.trace_id == "test_trace"
        assert len(trace.steps) == 2

        root_step = trace.get_step("step_000")
        child_step = trace.get_step("step_001")

        assert root_step.agent_name == "root"
        assert root_step.parent_step_id is None
        assert child_step.agent_name == "child"
        assert child_step.parent_step_id == "step_000"

    def test_nested_parent_tracking(self) -> None:
        """Nested calls should automatically track parent-child relationships."""
        tracer = TraceLensCapture(trace_id="nested_test")

        @tracer.trace_step(agent_name="coordinator", step_type="router")
        def coordinator(query: str) -> str:
            linter(query)
            security(query)
            return "review complete"

        @tracer.trace_step(agent_name="linter", step_type="agent")
        def linter(query: str) -> str:
            return "no lint issues"

        @tracer.trace_step(agent_name="security", step_type="agent")
        def security(query: str) -> str:
            return "no security issues"

        coordinator("review this code")

        trace = tracer.finalize(query="review this code", final_answer="review complete")

        assert len(trace.steps) == 3

        # Step ID assigned at start time:
        # step_000 = coordinator (starts first, parent = None)
        # step_001 = linter (starts second, parent = step_000)
        # step_002 = security (starts third, parent = step_000)
        coord_step = trace.get_step("step_000")
        linter_step = trace.get_step("step_001")
        security_step = trace.get_step("step_002")

        assert coord_step.agent_name == "coordinator"
        assert coord_step.parent_step_id is None

        assert linter_step.agent_name == "linter"
        assert linter_step.parent_step_id == "step_000"

        assert security_step.agent_name == "security"
        assert security_step.parent_step_id == "step_000"

    def test_finalize_twice_raises(self) -> None:
        tracer = TraceLensCapture()

        @tracer.trace_step(agent_name="a", step_type="agent")
        def fn(q: str) -> str:
            return "ok"

        fn("q")
        tracer.finalize(query="q", final_answer="ok")

        with pytest.raises(ValueError, match="already been finalized"):
            tracer.finalize(query="q", final_answer="ok")

    def test_return_value_preserved(self) -> None:
        """The decorator must not alter the function's return value."""
        tracer = TraceLensCapture()

        @tracer.trace_step(agent_name="a", step_type="agent")
        def fn(q: str) -> str:
            return "exact_return_value"

        result = fn("q")
        assert result == "exact_return_value"


class TestManualAddStep:
    """Tests for the manual add_step() API."""

    def test_add_step(self) -> None:
        tracer = TraceLensCapture(trace_id="manual_test")

        root_id = tracer.add_step(
            agent_name="router",
            step_type="router",
            input_text="user query",
            output_text="selected: agent_b",
        )

        tracer.add_step(
            agent_name="agent_b",
            step_type="agent",
            input_text="user query",
            output_text="the answer is 42",
            parent_step_id=root_id,
        )

        trace = tracer.finalize(query="user query", final_answer="the answer is 42")
        assert len(trace.steps) == 2
        assert trace.steps[1].parent_step_id == root_id

    def test_add_step_returns_step_id(self) -> None:
        tracer = TraceLensCapture()
        sid = tracer.add_step(
            agent_name="a", step_type="agent", input_text="in", output_text="out"
        )
        assert sid == "step_000"


class TestContextManager:
    """Tests for the step() context manager."""

    def test_basic_context_manager(self) -> None:
        tracer = TraceLensCapture(trace_id="cm_test", project_name="my_project")

        with tracer.step(agent_name="cm_agent", input_text="hello") as io:
            io.output_text = "world"
            io.tool_name = "hello_tool"

        trace = tracer.finalize(query="hello", final_answer="world")
        assert trace.project_name == "my_project"
        assert len(trace.steps) == 1
        
        step = trace.steps[0]
        assert step.agent_name == "cm_agent"
        assert step.io.input_text == "hello"
        assert step.io.output_text == "world"
        assert step.io.tool_name == "hello_tool"
        assert step.parent_step_id is None

    def test_nested_context_managers(self) -> None:
        tracer = TraceLensCapture(trace_id="cm_nested")

        with tracer.step(agent_name="root", input_text="q") as io_root:
            with tracer.step(agent_name="child", input_text="sub_q") as io_child:
                io_child.output_text = "sub_a"
            io_root.output_text = "a"

        trace = tracer.finalize(query="q", final_answer="a")
        
        assert len(trace.steps) == 2
        
        # Child completes first because its with-block exits before root's
        child_step = trace.steps[0]
        root_step = trace.steps[1]
        
        assert child_step.agent_name == "child"
        assert root_step.agent_name == "root"
        
        assert root_step.parent_step_id is None
        assert child_step.parent_step_id == root_step.step_id

class TestAsyncCapture:
    """Tests for async function tracing."""

    @pytest.mark.asyncio
    async def test_async_decorator(self) -> None:
        tracer = TraceLensCapture(trace_id="async_test")

        @tracer.trace_step(agent_name="async_agent", step_type="agent")
        async def async_fn(query: str) -> str:
            return "async result"

        result = await async_fn("test")
        assert result == "async result"
        
        trace = tracer.finalize("test", "async result")
        assert len(trace.steps) == 1
