import json
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
            tool_args: Optional[dict] = None  
        ) -> str:
            """
            Schedules tasks requested by the user to be executed in the future.
            MANDATORY RULE: If the user asks you to do something in the future, do not just reply "I will do it". YOU MUST CALL THIS TOOL.
            
            Args:
                action_type: Can only take the value 'autonomous_goal' or 'execute_tool'.
                delay_in_seconds: PREFERRED for relative times. Delay in seconds from now.
                execute_at: OPTIONAL. The exact date/time in UTC ISO 8601.
                prompt: If action_type is 'autonomous_goal', the detail of the task.
                tool_name: If action_type is 'execute_tool', the name of the function.
                tool_args: JSON arguments for the tool.
            """
            if tool_args is None:
                tool_args = {}
                
            try:
                if action_type == "autonomous_goal":
                    self.task_manager.add_autonomous_task(prompt, execute_at, delay_in_seconds)
                    return "Autonomous task successfully scheduled in the background."
                
                elif action_type == "execute_tool":
                    args_str = json.dumps(tool_args)
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
                print(f"\n[Autonomous Wake] Task ID: {task_id}")
                try:
                    response = self.agent.run(payload, session_id=f"auto_{task_id}")
                    print(f"[Autonomous Report]: {response}\n")
                except Exception as e:
                    print(f"[Autonomous Error]: {e}\n")
                    raise e 
                
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
                        raise e 
                else:
                    msg = f"[Missing Tool] A tool named '{payload}' could not be found!"
                    print(msg)
                    raise ValueError(msg) 

        self.task_manager.start_daemon(_router_callback)
        print("AgentOrchestrator: Background Rust engine activated!")