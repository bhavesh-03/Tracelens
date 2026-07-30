"""Trace capture SDK — lightweight instrumentation for multi-agent pipelines.

Provides a decorator and context-manager interface that developers wrap around
their agent calls to automatically capture execution traces. The SDK is
non-invasive: it records inputs, outputs, timing, and parent-child
relationships without requiring the user to rewrite their agent code.

Usage:
    from tracelens.capture import TraceLensCapture

    tracer = TraceLensCapture()

    @tracer.trace_step(agent_name="router", step_type="router")
    def route_query(query: str) -> str:
        return selected_agent_name

    @tracer.trace_step(agent_name="order_agent", step_type="agent")
    async def handle_order(query: str) -> str:
        return agent_response

    # After the pipeline completes:
    trace = tracer.finalize(query="...", final_answer="...")
"""

from __future__ import annotations

import asyncio
import functools
import time
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal

from tracelens.schema import StepIO, Trace, TraceStep

# Context variable tracks the current parent step ID during nested execution.
# This is how the SDK knows which step invoked the current step — it reads the
# parent from the context when a new step starts, then sets itself as the
# parent for any steps it invokes.
_current_parent: ContextVar[str | None] = ContextVar("_current_parent", default=None)


@dataclass
class _RawStep:
    """Internal representation of a captured step before finalization."""

    step_id: str
    agent_name: str
    step_type: str
    parent_step_id: str | None
    input_text: str
    output_text: str
    model: str | None
    tool_name: str | None
    tool_args: dict[str, Any]
    tool_output: str | None
    timestamp_ms: float
    duration_ms: float
    metadata: dict[str, Any]


class TraceLensCapture:
    """Captures execution traces from a multi-agent pipeline.

    Create one instance per pipeline execution. Use `trace_step()` as a
    decorator or `step()` as a context manager around each agent/tool call.
    Call `finalize()` at the end to get a validated `Trace` object.
    """

    def __init__(self, trace_id: str | None = None) -> None:
        self.trace_id = trace_id or f"trace_{uuid.uuid4().hex[:12]}"
        self._steps: list[_RawStep] = []
        self._step_counter = 0
        self._finalized = False

    def trace_step(
        self,
        agent_name: str,
        step_type: Literal[
            "router", "agent", "tool", "synthesizer", "llm_call", "custom"
        ] = "agent",
        model: str | None = None,
        tool_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Callable:
        """Decorator that wraps a function to capture its execution as a trace step.

        The decorated function's first positional argument is treated as the
        input text, and the return value as the output text. Both are converted
        to strings.

        Works for both sync and async functions.
        """
        def decorator(fn: Callable) -> Callable:
            if asyncio.iscoroutinefunction(fn):
                @functools.wraps(fn)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    input_text = str(args[0]) if args else str(kwargs.get("query", ""))
                    step_id = f"step_{self._step_counter:03d}"
                    self._step_counter += 1
                    parent_id = _current_parent.get()

                    token = _current_parent.set(step_id)
                    start_ms = time.time() * 1000
                    try:
                        result = await fn(*args, **kwargs)
                        output_text = str(result) if result is not None else ""
                        duration_ms = time.time() * 1000 - start_ms

                        self._steps.append(_RawStep(
                            step_id=step_id,
                            agent_name=agent_name,
                            step_type=step_type,
                            parent_step_id=parent_id,
                            input_text=input_text,
                            output_text=output_text,
                            model=model,
                            tool_name=tool_name,
                            tool_args=dict(kwargs) if kwargs else {},
                            tool_output=None,
                            timestamp_ms=start_ms,
                            duration_ms=duration_ms,
                            metadata=metadata or {},
                        ))
                        return result
                    finally:
                        _current_parent.reset(token)

                return async_wrapper
            else:
                @functools.wraps(fn)
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    input_text = str(args[0]) if args else str(kwargs.get("query", ""))
                    step_id = f"step_{self._step_counter:03d}"
                    self._step_counter += 1
                    parent_id = _current_parent.get()

                    token = _current_parent.set(step_id)
                    start_ms = time.time() * 1000
                    try:
                        result = fn(*args, **kwargs)
                        output_text = str(result) if result is not None else ""
                        duration_ms = time.time() * 1000 - start_ms

                        self._steps.append(_RawStep(
                            step_id=step_id,
                            agent_name=agent_name,
                            step_type=step_type,
                            parent_step_id=parent_id,
                            input_text=input_text,
                            output_text=output_text,
                            model=model,
                            tool_name=tool_name,
                            tool_args=dict(kwargs) if kwargs else {},
                            tool_output=None,
                            timestamp_ms=start_ms,
                            duration_ms=duration_ms,
                            metadata=metadata or {},
                        ))
                        return result
                    finally:
                        _current_parent.reset(token)

                return sync_wrapper
        return decorator

    def add_step(
        self,
        agent_name: str,
        step_type: Literal["router", "agent", "tool", "synthesizer", "llm_call", "custom"],
        input_text: str,
        output_text: str,
        parent_step_id: str | None = None,
        model: str | None = None,
        tool_name: str | None = None,
        tool_args: dict[str, Any] | None = None,
        tool_output: str | None = None,
        duration_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Manually add a step to the trace.

        Use this when the decorator approach doesn't fit (e.g., when
        instrumenting a framework that doesn't expose simple function calls).

        Returns the step_id of the newly added step.
        """
        step_id = f"step_{self._step_counter:03d}"
        self._step_counter += 1
        self._steps.append(_RawStep(
            step_id=step_id,
            agent_name=agent_name,
            step_type=step_type,
            parent_step_id=parent_step_id,
            input_text=input_text,
            output_text=output_text,
            model=model,
            tool_name=tool_name,
            tool_args=tool_args or {},
            tool_output=tool_output,
            timestamp_ms=time.time() * 1000,
            duration_ms=duration_ms,
            metadata=metadata or {},
        ))
        return step_id

    def finalize(
        self,
        query: str,
        final_answer: str,
        expected_answer: str | None = None,
        tags: list[str] | None = None,
    ) -> Trace:
        """Convert captured steps into a validated Trace object.

        Raises ValueError if finalize() was already called on this instance.
        Raises pydantic.ValidationError if the trace is structurally invalid
        (e.g., broken parent references, no root step).
        """
        if self._finalized:
            raise ValueError("This TraceLensCapture has already been finalized")
        self._finalized = True

        trace_steps = [
            TraceStep(
                step_id=rs.step_id,
                agent_name=rs.agent_name,
                step_type=rs.step_type,
                parent_step_id=rs.parent_step_id,
                io=StepIO(
                    input_text=rs.input_text,
                    output_text=rs.output_text,
                    model=rs.model,
                    tool_name=rs.tool_name,
                    tool_args=rs.tool_args,
                    tool_output=rs.tool_output,
                ),
                timestamp_ms=rs.timestamp_ms,
                duration_ms=rs.duration_ms,
                metadata=rs.metadata,
            )
            for rs in self._steps
        ]

        return Trace(
            trace_id=self.trace_id,
            query=query,
            final_answer=final_answer,
            steps=trace_steps,
            expected_answer=expected_answer,
            tags=tags or [],
        )

    @property
    def step_count(self) -> int:
        """Number of steps captured so far."""
        return len(self._steps)
