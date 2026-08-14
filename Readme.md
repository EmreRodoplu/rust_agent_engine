# Rust Agent Engine

Rust Agent Engine is a Python-first agent framework powered by a high-performance Rust core. It is designed for building autonomous AI agents that can use tools, maintain memory, and retrieve knowledge with RAG while keeping the developer experience simple and fast.

## Why this project exists

This library gives you the best of both worlds:

- Python for easy application development and orchestration
- Rust for speed, concurrency, and efficient agent execution

It is useful when you want to build systems such as:

- LLM-powered assistants
- internal knowledge bots
- document-aware agents with retrieval
- tool-using workflows and automation
- multi-session agent applications

## Features

- High-performance Rust execution engine
- Python API for building agents quickly
- Custom Python functions can be registered as agent tools
- Session memory support with in-memory and Redis backends
- Retrieval-augmented generation (RAG) for document-aware answers
- CLI for creating projects, validating config, and running chat sessions
- Built-in API server support with FastAPI

## Installation

If you are using a virtual environment, activate it first and then install the package.

## Python usage

The main pattern is simple: create an `Agent`, configure an `LLMConfig`, optionally attach memory, and register Python functions as tools.

### Basic agent example

```python
from rust_agent_engine import Agent, LLMConfig
from rust_agent_engine.memory import AgentMemory

config = LLMConfig(
    model="gpt-4o-mini",
    base_url="https://api.openai.com/v1/chat/completions",
    api_key="YOUR_API_KEY",
)

agent = Agent(
    name="Assistant",
    system_prompt="You are a helpful assistant.",
    config=config,
    memory=AgentMemory.in_memory(),
)

@agent.register_tool
def add_numbers(a: float, b: float) -> str:
    return str(a + b)

response = agent.run(user_input="What is 12 + 37?")
print(response)
```

This is the most common usage pattern in Python: the agent reasons with the model and can call your registered Python functions when needed.

### Real tool-based workflow

```python
from rust_agent_engine import Agent, LLMConfig

config = LLMConfig(
    model="gpt-4o-mini",
    base_url="https://api.openai.com/v1/chat/completions",
    api_key="YOUR_API_KEY",
)

agent = Agent(
    name="ToolAgent",
    system_prompt="Use tools when needed and answer precisely.",
    config=config,
)

@agent.register_tool
def multiply(a: float, b: float) -> float:
    return a * b

@agent.register_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny and 24C."

result = agent.run(user_input="Multiply 8 by 7 and tell me the weather in Berlin.")
print(result)
```

This is one of the main strengths of the library: your Python code can become callable by the agent during runtime.

### RAG example

RAG is useful when the agent needs to answer based on documents or a knowledge base instead of only the model context.

```python
from rust_agent_engine import Agent, LLMConfig
from rust_agent_engine.memory import AgentMemory
from rust_agent_engine.rag import RagEngine

rag = RagEngine(
    model="text-embedding-3-small",
    base_url="https://api.openai.com/v1/embeddings",
    api_key="YOUR_API_KEY",
)

rag.load_document(
    collection="company_rules",
    text="Remote work is required two days per week.",
    source_name="rules.txt",
)

config = LLMConfig(
    model="gpt-4o-mini",
    base_url="https://api.openai.com/v1/chat/completions",
    api_key="YOUR_API_KEY",
)

agent = Agent(
    name="HR_Assistant",
    system_prompt="Answer using the available documents and memory.",
    config=config,
    memory=AgentMemory.in_memory(),
)

agent.add_rag_tool(rag, collection="company_rules", limit=3)
print(agent.run(user_input="How many days of remote work are required?"))
```

This pattern is commonly used for knowledge-base assistants, company policies, documentation search, and FAQ bots.

## CLI usage

The project also provides a command-line interface for practical project workflows.

### 1) Initialize a new project

```bash
cd python
rust-agent init
```

This creates a basic project with:

- `agent_config.yml`
- `custom_tools.py`

### 2) Validate your setup

```bash
rust-agent validate --config agent_config.yml --tools-file custom_tools.py
```

This checks that the YAML config and Python tool file are valid.

### 3) Start an interactive chat session

```bash
rust-agent chat --agent System_Assistant --config agent_config.yml --tools-file custom_tools.py
```

This is the main CLI usage for testing and interacting with the agent in real time.

### 4) Start a FastAPI server for your agent

```bash
rust-agent serve --agent System_Assistant --config agent_config.yml --tools-file custom_tools.py --host 0.0.0.0 --port 8000
```

This exposes the agent as an HTTP API and is useful for integrating it into a web app or backend service.

### 5) Run the agent as a daemon

```bash
rust-agent daemon --agent System_Assistant --config agent_config.yml --tools-file custom_tools.py
```

A daemon is useful when you want the agent to keep running in the background and process scheduled or background tasks without requiring an interactive terminal session. Instead of only responding to one manual chat, the agent stays alive in memory and can be used for long-running automation, periodic jobs, or background orchestration.

Typical daemon use cases include:

- scheduled task execution
- background assistant workers
- recurring workflows triggered by cron or orchestration systems
- long-lived agent services that monitor events or perform automated actions

This mode is especially useful when the agent acts as a worker rather than a direct interactive chatbot.

### 6) RAG management commands

```bash
rust-agent rag ingest --collection company_rules --dir ./docs --type txt --config agent_config.yml
rust-agent rag status --collection company_rules --config agent_config.yml
rust-agent rag drop --collection company_rules --config agent_config.yml
```

These commands let you ingest documents, inspect a collection, and remove it when needed.

### 6) Memory commands

```bash
rust-agent memory list --agent System_Assistant --config agent_config.yml
rust-agent memory prune --agent System_Assistant --session cli_session_001 --max-tokens 2048 --config agent_config.yml
```

Use these commands to inspect active sessions and prune memory when needed.

### 7) List registered tools

```bash
rust-agent tools list --tools-file custom_tools.py
```

This shows the tools loaded from your custom Python file.

## Typical workflow

A common real-world setup looks like this:

1. Run `rust-agent init` to create the starter files.
2. Define your custom Python tools in `custom_tools.py`.
3. Add your LLM settings in `agent_config.yml`.
4. Run `rust-agent validate` to confirm the project is valid.
5. Start a chat with `rust-agent chat`, expose an API with `rust-agent serve`, or keep it alive in the background with `rust-agent daemon`.
6. Add a RAG data store when the agent needs document-aware answers.

## Summary

Rust Agent Engine is designed for developers who want to build agent applications in Python without giving up performance. It is especially useful when you need:

- fast agent execution
- Python-based tool orchestration
- memory-aware conversations
- retrieval over documents
- easy CLI-driven development and testing

This combination makes it a practical framework for both experimentation and real production workflows.
