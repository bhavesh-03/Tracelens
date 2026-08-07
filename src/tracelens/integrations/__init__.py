"""Integrations with third-party multi-agent frameworks."""

from .autogen import instrument_autogen_agent
from .crewai import instrument_crew
from .langchain import TraceLensCallbackHandler

__all__ = [
    "instrument_autogen_agent",
    "instrument_crew",
    "TraceLensCallbackHandler",
]
