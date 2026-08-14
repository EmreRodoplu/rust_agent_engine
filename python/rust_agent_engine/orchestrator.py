import json
import threading
from typing import Dict, Callable, Optional
from ._rust_agent_engine import TaskManager, Agent

__all__ = ["AgentOrchestrator"]

class AgentOrchestrator:
    def __init__(self, agent: Agent, redis_url: Optional[str] = "redis://localhost:6379") -> None:
        """Automatically sets up the system and initializes the TaskManager."""
        self.agent = agent
        self.task_manager = TaskManager(redis_url)
        self.tool_registry: Dict[str, Callable] = {}
        self._inject_scheduler_tool()

    def register_background_tool(self, name: str, func: Callable) -> None:
        """Registers tools that will run in the background (like a cron job)."""
        self.tool_registry[name] = func

    def _inject_scheduler_tool(self) -> None:
        def schedule_task(
            action_type: str, 
            delay_in_seconds: Optional[int] = None, 
            execute_at: Optional[str] = None, 
            prompt: str = "", 
            tool_name: str = "", 
            tool_args: dict = {}
        ) -> str:
            """
            Schedules tasks requested by the user to be executed in the future.
            MANDATORY RULE: If the user asks you to do something in the future, do not just reply "I will do it". YOU MUST CALL THIS TOOL.
            
            Args:
                action_type: Can only take the value 'autonomous_goal' or 'execute_tool'.
                delay_in_seconds: PREFERRED for relative times. Delay in seconds from now. (e.g., 'in 2 hours' -> 7200, 'tomorrow' -> 86400)
                execute_at: OPTIONAL. The exact date/time in UTC ISO 8601. Only use if a specific calendar date is provided.
                prompt: If action_type is 'autonomous_goal', the detail of the task you need to do when the time comes.
                tool_name: If action_type is 'execute_tool', the name of the function to be executed.
                tool_args: JSON arguments for the tool.
            """
            try:
                if action_type == "autonomous_goal":
                    self.task_manager.add_autonomous_task(prompt, execute_at, delay_in_seconds)
                    return "Autonomous task successfully scheduled in the background."
                
                elif action_type == "execute_tool":
                    args_str = json.dumps(tool_args or {})
                    self.task_manager.add_tool_task(tool_name, args_str, execute_at, delay_in_seconds)
                    return f"Tool execution ({tool_name}) successfully scheduled in the background."
                
                return "Error: Invalid action_type."
            except Exception as e:
                return f"Failed to schedule task: {str(e)}"
        
        self.agent.register_tool(schedule_task)

    def start_background_engine(self) -> None:
        """Starts the Rust scheduler and sets up the callback automation."""
        
        def _router_callback(task_id: str, action_type: str, payload: str, args_str: str) -> None:
            if action_type == "autonomous_goal":
                def _run_agent_thread():
                    print(f"\n[Autonomous Wake] Task ID: {task_id}")
                    response = self.agent.run(payload, session_id=f"auto_{task_id}")
                    print(f"[Autonomous Report]: {response}\n")
                
                threading.Thread(target=_run_agent_thread, daemon=True).start()
                
            elif action_type == "execute_tool":
                payload = payload.replace("functions.", "").replace("tools.", "")
                print(f"\n[Tool Triggered] Tool: {payload}")
                args = json.loads(args_str) if args_str else {}
                
                func = self.tool_registry.get(payload)
                if func:
                    try:
                        func(**args)
                    except Exception as e:
                        print(f"[Tool Error] {payload} crashed during execution: {e}")
                else:
                    print(f"[Missing Tool] A tool named '{payload}' could not be found!")

        self.task_manager.start_daemon(_router_callback)
        print("AgentOrchestrator: Background Rust engine activated!")