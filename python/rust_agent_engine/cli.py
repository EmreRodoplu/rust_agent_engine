import typer
import yaml
import uvicorn
from typing import Optional
from pathlib import Path
from rich import print
from rich.table import Table
from rich.console import Console
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import datetime
from .registry import load_custom_tools, _TOOL_REGISTRY
from .parser import create_agent_from_yaml
from .rag import RagEngine
from .orchestrator import AgentOrchestrator

console = Console()


app = typer.Typer(
    name="rust-agent",
    help="Rust-based autonomous AI agent engine CLI interface.",
    add_completion=False,
)

# ---------------------------------------------------------
# SUBCOMMAND GROUPS (Sub-apps)
# ---------------------------------------------------------
rag_app = typer.Typer(help="RAG (vector database and document) management operations.")
memory_app = typer.Typer(help="Manage agents' chat history and token limits.")
tools_app = typer.Typer(help="Show the status of tools registered in the system.")

app.add_typer(rag_app, name="rag")
app.add_typer(memory_app, name="memory")
app.add_typer(tools_app, name="tools")


# ---------------------------------------------------------
# ROOT-LEVEL COMMANDS
# ---------------------------------------------------------

@app.command(name="init")
def init_project():
    """Creates YAML and tool templates for a new project."""
    print("[bold blue]⚙️ Creating project templates...[/bold blue]")
    
    config_path = Path("agent_config.yml")
    tools_path = Path("custom_tools.py")
    
    if config_path.exists() or tools_path.exists():
        print("[bold red]Warning:[/bold red] 'agent_config.yml' or 'custom_tools.py' already exists in this directory. Overwrite was cancelled.")
        raise typer.Exit(code=1)

    yaml_content = """version: "1.0"

resources:
  llm_configs:
    - id: "config"
      model: ""
      base_url: ""
      api_key: "****"  # Replace with your actual API key

  memories:
    - id: "redis"
      type: "redis_memory"
      redis_url: "redis://localhost:6379"
      ttl: 3600  

tools:
  - id: "example_tool_rag"
    type: "rag"
    collection: ""
    limit: 3
    model: ""
    base_url: ""
    redis_vectorstore_url: "redis://localhost:6379" 
  
  - id: "example_tool_custom"
    type: "custom"
    name: "hello_world" # This should match the function name in custom_tools.py You can add more tools here as needed.

agents:
  - name: "System_Assistant"
    system_prompt: "
        You are a Rust-based autonomous AI agent. 
        You can use the tools defined in the configuration to assist users. 
        Always respond in a helpful and concise manner."
    llm_id: "config"
    memory_id: "redis"
    session_id: "system_assistants_session"
    tools:
        - "example_tool_rag"
        - "example_tool_custom"
"""

    tools_content = """from rust_agent_engine.registry import tool

@tool(name="hello_world")
def hello_world(name: str) -> str:
    \"\"\"Greets the user by name.\"\"\"
    return f"Hello {name}, welcome to the Rust-based AI engine!"
"""

    config_path.write_text(yaml_content, encoding="utf-8")
    tools_path.write_text(tools_content, encoding="utf-8")

    print("[green]✓ 'agent_config.yml' and 'custom_tools.py' were created successfully.[/green]")
    print("[bold yellow]💡 Tip:[/bold yellow] To test it, run the following command in the terminal:")
    print("   👉 [bold cyan]rust-agent chat --agent System_Assistant --tools-file custom_tools.py[/bold cyan]")


@app.command(name="validate")
def validate_config(
    config: str = typer.Option("agent_config.yml", "--config", "-c", help="YAML file to validate"),
    tools_file: Optional[str] = typer.Option("custom_tools.py", "--tools-file", "-t", help="Custom tools file")
):
    """Validates the YAML file and tools for CI/CD without running them."""
    print(f"[bold yellow]🔍 Validating: {config}[/bold yellow]")
    
    try:
        if tools_file:
            load_custom_tools(tools_file)
            print(f"[green]✓ Tool file ({tools_file}) is valid.[/green]")
            
        path = Path(config)
        if not path.exists():
            raise FileNotFoundError(f"The file '{config}' was not found.")
            
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            
        if "agents" not in data or not data["agents"]:
            raise ValueError("The YAML file must define at least one agent under 'agents'.")
            
        print("[bold green]✓ All configurations are valid (Pass).[/bold green]")
        
    except Exception as e:
        print(f"[bold red]❌ Validation error:[/bold red] {str(e)}")
        raise typer.Exit(code=1)


@app.command(name="chat")
def chat_mode(
    agent_name: str = typer.Option(..., "--agent", "-a", help="Name of the agent to chat with"),
    config: str = typer.Option("agent_config.yml", "--config", "-c", help="Configuration file"),
    tools_file: Optional[str] = typer.Option("custom_tools.py", "--tools-file", "-t", help="Custom tools file")
):
    """Starts an interactive terminal chat with the specified agent."""
    if not Path(config).exists():
        console.print(f"[bold red]Error:[/bold red] '{config}' was not found. Please run 'rust-agent init'.")
        raise typer.Exit(code=1)

    if tools_file:
        load_custom_tools(tools_file)

    try:
        console.print("[bold yellow]⏳ Configuring the agent (Rust Core)...[/bold yellow]")
        ajan = create_agent_from_yaml(config, agent_name) 
        motor = AgentOrchestrator(ajan)
        for tool_name, tool_func in _TOOL_REGISTRY.items():
            motor.register_background_tool(tool_name, tool_func)

        
        with open(config, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
            
        session_id = "cli_session_001" 
        for agent_data in yaml_data.get("agents", []):
            if agent_data.get("name") == agent_name:
                
                session_id = agent_data.get("session_id", "cli_session_001")
                break

    except ValueError as ve:
         console.print(f"[bold red]Error:[/bold red] {ve}")
         raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Critical parser error:[/bold red] The agent could not be started.\nDetails: {str(e)}")
        raise typer.Exit(code=1)

    console.clear()
    console.print(f"[bold green]🤖 You started chatting with [{agent_name}].[/bold green]")
    console.print(f"[dim]Active Session ID: {session_id}[/dim]")
    console.print("[dim]Type 'exit' or 'quit' to end the conversation.[/dim]\n")

    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]👨‍💻 You[/bold cyan]")
            if user_input.lower() in ["exit", "quit", "q"]:
                console.print(f"[bold green]👋 Goodbye![/bold green]")
                break
            if not user_input.strip():
                continue

            console.print("[dim]The agent is thinking...[/dim]")
            
            current_time_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            prompt = f"[SYSTEM_NOTE: Current UTC Time is {current_time_utc}]\n\n{user_input}"
            
            
            response = ajan.run(user_input=prompt, session_id=session_id)
            
            console.print(f"[bold magenta]🤖 [{agent_name}]:[/bold magenta] {response}")

        except KeyboardInterrupt:
             console.print(f"\n[bold green]👋 Chat ended (Ctrl+C).[/bold green]")
             break
        except Exception as e:
             console.print(f"[bold red]⚠️ Engine error (Rust Core):[/bold red] {str(e)}")


@app.command(name="daemon")
def start_daemon(
    agent_name: str = typer.Option(..., "--agent", "-a", help="Agent to run in the background"),
    config: str = typer.Option("agent_config.yml", "--config", "-c", help="Configuration file"),
    tools_file: Optional[str] = typer.Option("custom_tools.py", "--tools-file", "-t", help="Custom tools file")
):
    """Runs the agent in the background for scheduled (cron) tasks."""
    print(f"[bold blue]🚀 Starting '{agent_name}' in daemon mode...[/bold blue]")
    
    if tools_file:
        load_custom_tools(tools_file)
        
    try:
        ajan = create_agent_from_yaml(config, agent_name)
        motor = AgentOrchestrator(ajan)
        for tool_name, tool_func in _TOOL_REGISTRY.items():
            motor.register_background_tool(tool_name, tool_func)
            
        motor.start_background_engine()
        

        print("[bold green]✓ Agent loaded into memory. Waiting for background tasks (Cron).[/bold green]")
        print("[dim]Press Ctrl+C to stop.[/dim]")
        
        import time
        while True:
            time.sleep(1) 
            
    except KeyboardInterrupt:
        print("\n[bold yellow]🛑 Daemon stopped.[/bold yellow]")
    except Exception as e:
        print(f"[bold red]Error:[/bold red] Daemon could not be started. {e}")
        raise typer.Exit(code=1)


@app.command(name="serve")
def serve_api(
    agent_name: str = typer.Option(..., "--agent", "-a", help="Name of the agent to expose"),
    config: str = typer.Option("agent_config.yml", "--config", "-c", help="Configuration file"),
    tools_file: str = typer.Option("custom_tools.py", "--tools-file", "-t", help="Custom tools file"),
    port: int = typer.Option(8000, "--port", "-p", help="FastAPI server port"),
    host: str = typer.Option("0.0.0.0", "--host", help="Server host address")
):
    """Reads the YAML configuration and exposes the agent as a REST API via FastAPI."""
    if tools_file:
        load_custom_tools(tools_file)

    try:
        print(f"[bold yellow]⏳ Preparing '{agent_name}' for FastAPI...[/bold yellow]")
        ajan = create_agent_from_yaml(config, agent_name)
        motor = AgentOrchestrator(ajan)
        for tool_name, tool_func in _TOOL_REGISTRY.items():
            motor.register_background_tool(tool_name, tool_func)
    except Exception as e:
        print(f"[bold red]Critical error:[/bold red] The agent could not be started. Details: {e}")
        raise typer.Exit(code=1)

    api_app = FastAPI(
        title=f"{agent_name} API",
        description="REST API automatically generated by Rust-Agent Engine.",
        version="1.0.0"
    )

    class ChatRequest(BaseModel):
        query: str
        session_id: str

    class ChatResponse(BaseModel):
        response: str
        agent_name: str

    @api_app.post("/api/chat", response_model=ChatResponse)
    def chat_endpoint(request: ChatRequest):
        try:
            su_an_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            zengin_prompt = f"[SYSTEM_NOTE: Current UTC Time is {su_an_utc}]\n\n{request.query}"
            cevap = ajan.run(user_input=zengin_prompt, session_id=request.session_id)
            return ChatResponse(response=cevap, agent_name=agent_name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Rust Core error: {str(e)}")

    @api_app.get("/api/health")
    async def health_check():
        return {"status": "ok", "engine": "rust_agent_engine", "agent": agent_name}

    print(f"[bold magenta]🌐 Starting FastAPI server at http://{host}:{port}...[/bold magenta]")
    print(f"👉 [dim]API documentation (Swagger UI): http://{host}:{port}/docs[/dim]")
    uvicorn.run(api_app, host=host, port=port, log_level="warning")


# ---------------------------------------------------------
# RAG SUBCOMMANDS (rust-agent rag ...)
# ---------------------------------------------------------

@rag_app.command(name="ingest")
def rag_ingest(
    collection: str = typer.Option(..., "--collection", "-c", help="Target collection (for example: omega_db)"),
    directory: str = typer.Option(..., "--dir", "-d", help="Directory containing the documents to read"),
    doc_type: str = typer.Option(..., "--type", "-t", help="File type (md, txt)"),
    config: str = typer.Option("agent_config.yml", "--config", help="Configuration file")
):
    """Ingests the documents from the specified directory into the RAG engine (Rust Core)."""
    import yaml
    from pathlib import Path
    
    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"[bold red]Error:[/bold red] No directory named '{directory}' was found!")
        raise typer.Exit(code=1)

    dosyalar = list(dir_path.glob(f"**/*.{doc_type}"))
    if not dosyalar:
        print(f"[bold yellow]Warning:[/bold yellow] No files with the '.{doc_type}' extension were found in '{directory}'.")
        raise typer.Exit(code=0)

    print(f"[bold cyan]🔍 Found {len(dosyalar)} '{doc_type}' files. Starting the process...[/bold cyan]")
    
    rag_model = None
    rag_base_url = None
    rag_api_key = None
    redis_vectorstore_url = None

    if Path(config).exists():
        with open(config, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
            for tool_data in yaml_data.get("tools", []):
                if tool_data.get("type") == "rag" and tool_data.get("collection") == collection:
                    rag_model = tool_data.get("model")
                    rag_base_url = tool_data.get("base_url")
                    rag_api_key = tool_data.get("api_key")
                    redis_vectorstore_url = tool_data.get("redis_vectorstore_url")
                    break

    try:
        rag_engine = RagEngine(api_key=rag_api_key, model=rag_model, base_url=rag_base_url, redis_vectorstore_url=redis_vectorstore_url)
    except Exception as e:
        print(f"[bold red]Startup error (Rust Core):[/bold red] {e}")
        raise typer.Exit(code=1)
        
    successful_records = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task(description="Vectorizing documents...", total=len(dosyalar))
        for document_path in dosyalar:
            try:
                content = document_path.read_text(encoding="utf-8")
                rag_engine.load_document(collection=collection, text=content, source_name=document_path.name)
                successful_records += 1
            except Exception as e:
                progress.console.print(f"[red]⚠️ {document_path.name} could not be read: {str(e)}[/red]")
            progress.advance(task)

    print(f"[bold green]✅ Success![/bold green] A total of [bold]{successful_records}[/bold] files were added to the '{collection}' collection.")


@rag_app.command(name="status")
def rag_status(
    collection: str = typer.Option(..., "--collection", "-c", help="Collection whose status will be checked"),
    config: str = typer.Option("agent_config.yml", "--config", help="Configuration file")
):
    """Shows the current status and metrics of the vector collection in the system."""
    print(f"[bold yellow]📊 Checking collection status for '{collection}'...[/bold yellow]")
    import yaml
    from pathlib import Path
    
    rag_model = None
    rag_base_url = None
    rag_api_key = None
    redis_vectorstore_url = None
    
    if Path(config).exists():
        with open(config, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
            for t in yaml_data.get("tools", []):
                if t.get("type") == "rag" and t.get("collection") == collection:
                    rag_model = t.get("model")
                    rag_base_url = t.get("base_url")
                    rag_api_key = t.get("api_key")
                    redis_vectorstore_url = t.get("redis_vectorstore_url")
                    break

    try:
        rag_motoru = RagEngine(
            api_key=rag_api_key, 
            model=rag_model, 
            base_url=rag_base_url, 
            redis_vectorstore_url=redis_vectorstore_url
        )
        
        status = rag_motoru.get_collection_status(collection)
        print(f"[bold green]✓ RAG engine is active.[/bold green]")
        print(f"👉 [bold cyan]{status}[/bold cyan]")
    except Exception as e:
        print(f"[bold red]Error:[/bold red] Could not retrieve RAG status. {e}")
        raise typer.Exit(code=1)


@rag_app.command(name="drop")
def rag_drop(
    collection: str = typer.Option(..., "--collection", "-c", help="Name of the collection to delete"),
    config: str = typer.Option("agent_config.yml", "--config", help="Configuration file")
):
    """Deletes the specified collection and resets the vector memory."""
    print(f"[bold red]🗑️ Deleting collection: {collection}...[/bold red]")
    import yaml
    from pathlib import Path
    
    rag_model = None
    rag_base_url = None
    rag_api_key = None
    redis_vectorstore_url = None
    
    if Path(config).exists():
        with open(config, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
            for t in yaml_data.get("tools", []):
                if t.get("type") == "rag" and t.get("collection") == collection:
                    rag_model = t.get("model")
                    rag_base_url = t.get("base_url")
                    rag_api_key = t.get("api_key")
                    redis_vectorstore_url = t.get("redis_vectorstore_url")
                    break

    try:
        rag_motoru = RagEngine(
            api_key=rag_api_key, 
            model=rag_model, 
            base_url=rag_base_url, 
            redis_vectorstore_url=redis_vectorstore_url
        )
        
        rag_motoru.drop_collection(collection)
    except Exception as e:
        print(f"[bold red]Error:[/bold red] The collection could not be deleted. {e}")
        raise typer.Exit(code=1)


# ---------------------------------------------------------
# MEMORY SUBCOMMANDS (rust-agent memory ...)
# ---------------------------------------------------------

@memory_app.command(name="list")
def memory_list(
    agent_name: str = typer.Option(..., "--agent", "-a", help="Agent whose memory will be inspected"),
    config: str = typer.Option("agent_config.yml", "--config", "-c", help="Configuration file"),
    tools_file: Optional[str] = typer.Option("custom_tools.py", "--tools-file", "-t", help="Custom tools file")
):
    """Shows the active sessions (session_id) for the specified agent."""
    if tools_file:
        load_custom_tools(tools_file)
    try:
        ajan = create_agent_from_yaml(config, agent_name)
        sessions = ajan.get_active_sessions() 
        
        if not sessions:
            print(f"[bold yellow]ℹ️ No active memory session found for '{agent_name}'.[/bold yellow]")
            return

        table = Table(title=f"🧠 {agent_name} - Active Memory Sessions")
        table.add_column("#", justify="center", style="cyan", no_wrap=True)
        table.add_column("Session ID", style="magenta")

        for index, session_id in enumerate(sessions, start=1):
            table.add_row(str(index), session_id)

        print(table)

    except Exception as e:
        print(f"[bold red]Error:[/bold red] Memory information could not be read. Details: {e}")
        raise typer.Exit(code=1)


@memory_app.command(name="prune")
def memory_prune(
    agent_name: str = typer.Option(..., "--agent", "-a", help="Agent whose memory will be pruned"),
    session: str = typer.Option(..., "--session", "-s", help="Session to prune (session_id)"),
    max_tokens: int = typer.Option(2048, "--max-tokens", "-m", help="Maximum token limit to leave"),
    config: str = typer.Option("agent_config.yml", "--config", "-c", help="Configuration file"),
    tools_file: Optional[str] = typer.Option("custom_tools.py", "--tools-file", "-t", help="Custom tools file")
):
    """Prunes old messages from a specific session without disturbing the system prompt/context."""
    print(f"[bold yellow]⏳ Analyzing session '{session}'...[/bold yellow]")
    if tools_file:
        load_custom_tools(tools_file)
    try:
        ajan = create_agent_from_yaml(config, agent_name)
        old_token_count, new_token_count = ajan.prune_memory(session_id=session, max_tokens=max_tokens)
        removed_tokens = old_token_count - new_token_count
        
        print(f"[bold green]✅ Success![/bold green] Memory was pruned successfully.")
        print(f"📉 [bold cyan]Previous Tokens:[/bold cyan] {old_token_count}")
        print(f"📈 [bold cyan]New Tokens:[/bold cyan] {new_token_count}")
        print(f"🗑️ [bold magenta]Removed Tokens:[/bold magenta] {removed_tokens}")
        
    except ValueError as ve:
        print(f"[bold red]Error:[/bold red] {ve}")
        raise typer.Exit(code=1)
    except Exception as e:
        print(f"[bold red]Critical error (Rust Core):[/bold red] A problem occurred while pruning memory. Details: {e}")
        raise typer.Exit(code=1)


# ---------------------------------------------------------
# TOOLS SUBCOMMANDS (rust-agent tools ...)
# ---------------------------------------------------------

@tools_app.command(name="list")
def tools_list(
    tools_file: str = typer.Option("custom_tools.py", "--tools-file", "-t", help="Custom tools file")
):
    """Lists the custom functions and schemas registered in the system."""
    print(f"[bold yellow]🛠️ Loading tools from '{tools_file}'...[/bold yellow]")
    
    try:
        load_custom_tools(tools_file)
        
        if not _TOOL_REGISTRY:
            print("[bold red]Warning:[/bold red] The file was loaded, but no functions marked with @tool were found.")
            raise typer.Exit(code=0)

        table = Table(title="Registered Tools")
        table.add_column("Tool Name", style="cyan", no_wrap=True)
        table.add_column("Description (Docstring)", style="green")
        
        for tool_name, function in _TOOL_REGISTRY.items():
            description = function.__doc__ or "No description provided."
            table.add_row(tool_name, description.strip())
            
        print(table)
        
    except Exception as e:
        print(f"[bold red]Error:[/bold red] A problem occurred while listing the tools: {str(e)}")
        raise typer.Exit(code=1)


def main():
    app()

if __name__ == "__main__":
    main()