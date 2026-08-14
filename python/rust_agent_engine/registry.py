import importlib.util
import sys
from pathlib import Path
from typing import Callable, Dict


_TOOL_REGISTRY: Dict[str, Callable] = {}

def tool(name: str):
    """
    Decorator developers use to register their Python functions with the agent engine.
    """
    def decorator(func: Callable):
        _TOOL_REGISTRY[name] = func
        return func
    return decorator

def get_tool(name: str) -> Callable:
    """Returns the tool (function) with the given name from the registry."""
    if name not in _TOOL_REGISTRY:
        raise ValueError(f"Error: no tool named '{name}' was found. Did you register it with @tool?")
    return _TOOL_REGISTRY[name]

def load_custom_tools(tools_file_path: str):
    """
    Dynamically loads the 'custom_tools.py' file provided by the user as a terminal argument.
    """
    path = Path(tools_file_path)
    if not path.is_file():
        raise FileNotFoundError(f"The file '{tools_file_path}' was not found!")
    
    
    spec = importlib.util.spec_from_file_location("user_custom_tools", path)
    user_module = importlib.util.module_from_spec(spec)
    sys.modules["user_custom_tools"] = user_module
    spec.loader.exec_module(user_module)