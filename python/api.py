from typing import Callable
from rust_agent_engine import Agent, LLMConfig

def _tool_decorator(self, func: Callable) -> Callable:
    """
    This decorator allows you to register a function as a tool for the Agent.
    Usage:
        @agent.register_tool
        def my_tool(...):
            ...
    """
    self.register_tool(func)
    return func

Agent.tool = _tool_decorator

__all__ = ["Agent", "LLMConfig"]