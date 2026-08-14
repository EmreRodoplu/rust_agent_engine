import yaml
from pathlib import Path
from .registry import get_tool
from . import Agent, LLMConfig
from .memory import AgentMemory
from .rag import RagEngine 

def create_agent_from_yaml(yaml_path: str, target_agent_name: str) -> Agent:
    """
    Reads a YAML file, finds the requested agent, and returns a Rust Core Agent object.
    """
    path = Path(yaml_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

    
    with open(path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
        
    
    agent_data = None
    for a in config.get("agents", []):
        if a.get("name") == target_agent_name:
            agent_data = a
            break
            
    if not agent_data:
        raise ValueError(f"Agent '{target_agent_name}' was not found in the YAML file!")

    
    llm_id = agent_data.get("llm_id")
    llm_data = None
    for l in config.get("resources", {}).get("llm_configs", []):
        if l.get("id") == llm_id:
            llm_data = l
            break
            
    if not llm_data:
        raise ValueError(f"LLM configuration '{llm_id}' was not found under resources!")

    
    memory_id = agent_data.get("memory_id")
    agent_memory = None
    
    if memory_id:
        memory_data = None
        for m in config.get("resources", {}).get("memories", []):
            if m.get("id") == memory_id:
                memory_data = m
                break
                
        if not memory_data:
            raise ValueError(f"Memory configuration '{memory_id}' was not found under resources!")
            
        mem_type = memory_data.get("type")
        if mem_type == "redis_memory":
            redis_url = memory_data.get("redis_url", "redis://localhost:6379")
            ttl = memory_data.get("ttl")
            agent_memory = AgentMemory.redis(redis_url=redis_url, ttl_seconds=ttl)
        elif mem_type == "in_memory":
            agent_memory = AgentMemory.in_memory()
        else:
            raise ValueError(f"Unknown memory type: '{mem_type}'")
    

    
    llm_config = LLMConfig(
        model=llm_data.get("model"),
        api_key=llm_data.get("api_key"),
        provider=llm_data.get("provider"),
        base_url=llm_data.get("base_url")
    )
    
    ajan = Agent(
        name=agent_data.get("name"),
        system_prompt=agent_data.get("system_prompt", ""),
        config=llm_config,
        memory=agent_memory
    )
    
    
    yaml_tools = {t.get("id"): t for t in config.get("tools", [])}

    for tool_id in agent_data.get("tools", []):
        if tool_id not in yaml_tools:
            raise ValueError(f"Tool referenced by '{tool_id}' was not found in the YAML 'tools' section!")
            
        tool_data = yaml_tools[tool_id]
        
        
        if tool_data.get("type") == "custom":
            
            actual_function_name = tool_data.get("name")
            
            tool_func = get_tool(actual_function_name)
            
            ajan.register_tool(tool_func)
            
        
        elif tool_data.get("type") == "rag":
            collection_name = tool_data.get("collection")
            rag_model = tool_data.get("model")
            rag_base_url = tool_data.get("base_url")
            rag_api_key = tool_data.get("api_key")
            redis_vectorstore_url = tool_data.get("redis_vectorstore_url")
            limit_value = tool_data.get("limit", 3)
            
            
            rag_engine = RagEngine(model=rag_model, base_url=rag_base_url, api_key=rag_api_key, redis_vectorstore_url=redis_vectorstore_url)
            
            ajan.add_rag_tool(rag_engine, collection_name, limit_value)
    
    return ajan