# 🦀 Rust Agent Engine

Rust Agent Engine is a lightweight AI agent library powered by a fast Rust core with a simple Python API.

It allows you to easily create AI agents using LLM providers such as OpenAI, Anthropic, Gemini, and local models like Ollama.

---

## 🚀 Installation

### Using uv (recommended)

```bash
uv pip install rust-agent-engine
```

### Using pip

```bash
pip install rust-agent-engine
```

---

## ⚡ Quick Start

Create a simple AI agent in seconds:

```python
from rust_agent_engine import Agent, LLMConfig

config = LLMConfig(
    model="gpt-4o",
    api_key="YOUR_API_KEY",
    provider="openai"
)

agent = Agent(
    name="Assistant",
    system_prompt="You are a helpful AI assistant.",
    config=config
)

response = agent.run(
    user_input="What is Rust?"
)

print(response)
```

---

## 🖥️ Local Model Usage (Ollama)

You can also use local models such as Ollama.

```python
from rust_agent_engine import Agent, LLMConfig

config = LLMConfig(
    model="llama3",
    provider="ollama",
    api_key=""
)

agent = Agent(
    name="LocalAssistant",
    system_prompt="You give short and precise answers.",
    config=config
)

response = agent.run("What is Python?")
print(response)
```

---

## 🌊 Streaming Usage

Receive tokens in real-time as they are generated:

```python
from rust_agent_engine import Agent, LLMConfig

config = LLMConfig(
    model="gpt-4o",
    api_key="YOUR_API_KEY",
    provider="openai"
)

agent = Agent(
    name="Streamer",
    system_prompt="You are a helpful assistant.",
    config=config
)

def stream(token: str):
    print(token, end="", flush=True)

agent.run(
    user_input="What is artificial intelligence?",
    stream_callback=stream
)
```
## Multi Agent
In Rust Agent Engine, an agent can be used as a **tool inside another agent**.

This allows you to build hierarchical systems where:
- One agent delegates work
- Another agent executes it
- The result is returned as a tool output
```python
import sys
from rust_agent_engine import Agent,LLMConfig
import os
from dotenv import load_dotenv
load_dotenv()

config = LLMConfig(
    model="gemini-3.5-flash",  # Local or cloud model
    provider="gemini",
    api_key=os.getenv("API_KEY")
)

def add_numbers(a: float, b: float) -> str:
    """Adds two numbers and returns the result."""
    print(f"\n[MATH TOOL] Calculating: {a} + {b}")
    return str(a + b)


def multiply_numbers(a: float, b: float) -> str:
    """Multiplies two numbers and returns the result."""
    print(f"\n[MATH TOOL] Calculating: {a} * {b}")
    return str(a * b)


math_agent = Agent(
    name="MathExpert",
    system_prompt=(
        "You are a math genius. Focus only on numerical operations, "
        "use tools for calculations, and return precise results."
    ),
    config=config
)

math_agent.register_tool(add_numbers)
math_agent.register_tool(multiply_numbers)


def write_file(file_name: str, content: str) -> str:
    """Writes the given content to a file."""
    print(f"\n[FILE TOOL] Writing to '{file_name}' on disk...")
    try:
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(content)
        return f"{file_name} has been successfully created and saved."
    except Exception as e:
        return f"File write error: {e}"


file_agent = Agent(
    name="FileExpert",
    system_prompt="You are an expert in system file operations. Use tools to handle file writing tasks.",
    config=config
)

file_agent.register_tool(write_file)


def delegate_to_math(task_description: str) -> str:
    """Delegates all math-related tasks to the math expert agent."""
    print(f"\n[CHEF ROUTING] Math Agent activated -> Task: {task_description}")
    return math_agent.run(user_input=task_description)


def delegate_to_file(task_description: str) -> str:
    """Delegates all file-related tasks to the file expert agent."""
    print(f"\n[CHEF ROUTING] File Agent activated -> Task: {task_description}")
    return file_agent.run(user_input=task_description)


chef_agent = Agent(
    name="ChefOrchestrator",
    system_prompt=(
        "You are the system orchestrator agent. "
        "IMPORTANT: NEVER perform math or file operations yourself. "
        "You MUST use the provided tools instead.\n\n"
        "1. Use 'delegate_to_math' for math tasks.\n"
        "2. Use 'delegate_to_file' for file operations.\n\n"
        "Always return only the final result after tool execution."
    ),
    config=config
)

chef_agent.register_tool(delegate_to_math)
chef_agent.register_tool(delegate_to_file)


request = (
    "Multiply 145 by 28. Then save the result "
    "into a file named 'hesap_raporu.txt'."
)

print(f"User Request: {request}\n")
print("Chef Agent is planning and executing...\n" + "=" * 60)
print("Final Output Stream: ", end="")


def callback(token: str):
    print(token, end="", flush=True)


try:
    chef_agent.run(user_input=request, stream_callback=callback)

    print("\n\n" + "=" * 60)
    print("[SUCCESS] Agent system completed the task. No deadlock occurred.")

except Exception as e:
    print(f"\n[ERROR] System failed: {e}")

```
---
---

## 📦 Features

- ⚡ Fast Rust-based core
- 🐍 Simple Python API
- 🤖 Supports OpenAI / Anthropic / Gemini / Ollama
- 🌊 Streaming responses
- 🧠 Lightweight agent system
- 🔌 Works with any OpenAI-compatible API

---

## 💡 Purpose

This library is designed to:

- Build simple AI agents easily
- Integrate LLMs quickly
- Run both local and cloud models
- Provide a minimal and fast agent framework

---

## 📄 License

MIT License