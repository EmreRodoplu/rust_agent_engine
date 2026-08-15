from typing import Optional, Callable, Any, List

class AgentMemory:

    @staticmethod
    def in_memory() -> "AgentMemory":
        ...


    @staticmethod
    def redis(
        redis_url: str,
        ttl_seconds: Optional[int] = None
    ) -> "AgentMemory":
        ...

class RagEngine:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        redis_vectorstore_url: Optional[str] = None,
    ) -> None: ...

    def load_document(self, collection: str, text: str, source_name: Optional[str] = None) -> None: ...
    def get_collection_status(self, collection: str) -> dict: ...
    def drop_collection(self, collection: str) -> None: ...

class TaskManager:
    def __init__(self) -> None: ...
    def start_daemon(self, callback: Callable[[str, str, str, str], None]) -> None: ...
    def add_autonomous_task(self, prompt: str, execute_at_iso: str) -> None: ...
    def add_tool_task(self, tool_name: str, args_json: str, execute_at_iso: str) -> None: ...

class LLMConfig:
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> None: ...

class Agent:
    def __init__(
        self,
        name: str,
        system_prompt: str,
        config: LLMConfig,
        memory: Optional[AgentMemory] = None  
    ) -> None: ...

    def run(
        self,
        user_input: str,
        session_id: str,
        stream_callback: Optional[Callable[[str], None]] = None,
        max_tokens: Optional[int] = None,
        max_steps: Optional[int] = None
    ) -> str: ...

    def register_tool(
        self,
        tool: Callable[..., Any]
    ) -> None: ...

    def add_rag_tool(
        self,
        rag_engine: RagEngine,  
        collection: str,
        limit: int = 3
    ) -> None: ...

    def get_active_sessions(self) -> List[str]: ...