use chrono::{DateTime, Duration as ChronoDuration, Utc};
use redis::AsyncCommands;
use redis::aio::MultiplexedConnection;
use tokio::sync::{OnceCell, RwLock};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use uuid::Uuid;
use std::future::Future;
use std::pin::Pin;

pub type TaskFuture = Pin<Box<dyn Future<Output = Result<(), String>> + Send>>;
const PROCESSING_TIMEOUT_SECS: i64 = 300; 

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum TaskStatus {
    Pending,
    InProgress,
    Completed,
    Failed(String),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")] 
pub enum TaskAction {
    AutonomousGoal { prompt: String },
    ExecuteTool { tool_name: String, args: Value },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ScheduledTask {
    pub id: String,
    pub action: TaskAction,
    pub execute_at: DateTime<Utc>,
    pub status: TaskStatus,
    pub created_at: DateTime<Utc>,
}

impl ScheduledTask {
    fn calculate_time(execute_at_iso: Option<String>, delay_in_seconds: Option<i64>) -> Result<DateTime<Utc>, String> {
        if let Some(secs) = delay_in_seconds {
            Ok(Utc::now() + ChronoDuration::seconds(secs))
        } else if let Some(iso) = execute_at_iso {
            DateTime::<Utc>::from_str(&iso).map_err(|e| format!("Invalid date format: {}", e))
        } else {
            Err("Either 'execute_at' or 'delay_in_seconds' must be provided.".to_string())
        }
    }

    pub fn new_autonomous(prompt: String, execute_at_iso: Option<String>, delay_in_seconds: Option<i64>) -> Result<Self, String> {
        let execute_at = Self::calculate_time(execute_at_iso, delay_in_seconds)?;
        Ok(Self {
            id: Uuid::new_v4().to_string(),
            action: TaskAction::AutonomousGoal { prompt },
            execute_at,
            status: TaskStatus::Pending,
            created_at: Utc::now(),
        })
    }

    pub fn new_tool_execution(tool_name: String, args: Value, execute_at_iso: Option<String>, delay_in_seconds: Option<i64>) -> Result<Self, String> {
        let execute_at = Self::calculate_time(execute_at_iso, delay_in_seconds)?;
        Ok(Self {
            id: Uuid::new_v4().to_string(),
            action: TaskAction::ExecuteTool { tool_name, args },
            execute_at,
            status: TaskStatus::Pending,
            created_at: Utc::now(),
        })
    }
}

pub fn get_schedule_tool_schema() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "schedule_task",
            "description": "Schedules a task for the future. You MUST use 'delay_in_seconds' for relative times (like 'in 2 hours', 'tomorrow', 'next week') and ONLY use 'execute_at' if the user gives a specific calendar date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "delay_in_seconds": {
                        "type": "integer",
                        "description": "PREFERRED. Delay in seconds from now. Example: 'in 2 hours' -> 7200, 'tomorrow' -> 86400."
                    },
                    "execute_at": {
                        "type": "string",
                        "description": "OPTIONAL. The exact date/time in UTC ISO 8601. Only use if a specific calendar date is provided."
                    },
                    "action_type": {
                        "type": "string",
                        "enum": ["autonomous_goal", "execute_tool"],
                        "description": "Select 'autonomous_goal' for a prompt/command, or 'execute_tool' to run a tool."
                    },
                    "prompt": {
                        "type": "string",
                        "description": "If action_type is 'autonomous_goal', the command to execute."
                    },
                    "tool_name": {
                        "type": "string"
                    },
                    "tool_args": {
                        "type": "object"
                    }
                },
                "required": ["action_type"] 
            }
        }
    })
}

#[derive(Clone)]
enum TaskBackend {
    InMemory(Arc<RwLock<HashMap<String, ScheduledTask>>>),
    Redis {
        client: redis::Client,
        conn: Arc<OnceCell<MultiplexedConnection>>,
        queue_key: String,
        processing_key: String, 
    },
}

#[derive(Clone)]
pub struct TaskManager {
    backend: TaskBackend,
}

impl TaskManager {
    pub fn new(redis_url: Option<&str>) -> Self {
        if let Some(url) = redis_url {
            if let Ok(client) = redis::Client::open(url) {
                println!("[TaskManager] 🟢 Connected to Redis backend.");
                return Self {
                    backend: TaskBackend::Redis {
                        client,
                        conn: Arc::new(OnceCell::new()),
                        queue_key: "agent:task_queue".to_string(),
                        processing_key: "agent:task_processing".to_string(), 
                    },
                };
            }
        }
        
        println!("[TaskManager] 🟡 Warning: Redis not provided or failed. Using In-Memory backend (Local/Dev Mode).");
        Self {
            backend: TaskBackend::InMemory(Arc::new(RwLock::new(HashMap::new()))),
        }
    }
    async fn get_redis_conn(&self) -> Result<MultiplexedConnection, String> {
        match &self.backend {
            TaskBackend::Redis { client, conn, .. } => {
                let c = conn.get_or_try_init(|| async {
                    client.get_multiplexed_async_connection().await
                }).await.map_err(|e| format!("Failed to connect to Redis: {}", e))?;
                Ok(c.clone())
            }
            _ => Err("Not a Redis backend".to_string()),
        }
    }

    pub async fn add_task(&self, task: ScheduledTask) -> Result<(), String> {
        match &self.backend {
            TaskBackend::InMemory(map) => {
                let mut lock = map.write().await;
                lock.insert(task.id.clone(), task.clone());
                println!("[TaskManager] [InMemory] Task queued: {} (Time: {})", task.id, task.execute_at);
                Ok(())
            }
            TaskBackend::Redis { queue_key, .. } => {
                let mut con = self.get_redis_conn().await?;
                let score = task.execute_at.timestamp();
                let task_json = serde_json::to_string(&task)
                    .map_err(|e| format!("Serialization error: {}", e))?;

                let _: () = con.zadd(queue_key, task_json, score).await
                    .map_err(|e| format!("Redis ZADD error: {}", e))?;
                
                println!("[TaskManager] [Redis] Task queued: {} (Time: {})", task.id, task.execute_at);
                Ok(())
            }
        }
    }

    pub async fn ack_task(&self, task_json: &str) {
        if let TaskBackend::Redis { processing_key, .. } = &self.backend {
            if let Ok(mut con) = self.get_redis_conn().await {
                let _: redis::RedisResult<()> = con.zrem(processing_key, task_json).await;
            }
        }
    }

    pub async fn get_due_tasks(&self) -> Vec<(ScheduledTask, String)> {
        let now = Utc::now();
        let now_ts = now.timestamp();
        let mut due_tasks = Vec::new();

        match &self.backend {
            TaskBackend::InMemory(map) => {
                let mut lock = map.write().await;
                for (_, task) in lock.iter_mut() {
                    if task.status == TaskStatus::Pending && task.execute_at <= now {
                        task.status = TaskStatus::InProgress; 
                        due_tasks.push((task.clone(), String::new()));
                    }
                }
            }
            TaskBackend::Redis { queue_key, processing_key, .. } => {
                if let Ok(mut con) = self.get_redis_conn().await {
                    let timeout_ts = now_ts - PROCESSING_TIMEOUT_SECS;
                    let zombi_tasks_result: redis::RedisResult<Vec<String>> = 
                        con.zrangebyscore(processing_key, 0, timeout_ts).await;
                        
                    if let Ok(zombi_tasks) = zombi_tasks_result {
                        for zt in zombi_tasks {
                            if let Ok(removed) = con.zrem::<&str, &String, i32>(processing_key, &zt).await {
                                if removed > 0 {
                                    println!("[TaskManager] 🧟 Zombie task detected! Re-queueing for retry.");
                                    let _: redis::RedisResult<()> = con.zadd(queue_key, &zt, now_ts).await;
                                }
                            }
                        }
                    }

                    let script = redis::Script::new(r#"
                        local task = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', ARGV[1], 'LIMIT', 0, 1)[1]
                        if task then
                            redis.call('ZREM', KEYS[1], task)
                            redis.call('ZADD', KEYS[2], ARGV[2], task)
                            return task
                        end
                        return nil
                    "#);

                    loop {
                        let script_result: redis::RedisResult<Option<String>> = script
                            .key(queue_key)
                            .key(processing_key)
                            .arg(now_ts)
                            .arg(now_ts) 
                            .invoke_async(&mut con).await;

                        match script_result {
                            Ok(Some(task_json)) => {
                                if let Ok(task) = serde_json::from_str::<ScheduledTask>(&task_json) {
                                    due_tasks.push((task, task_json));
                                } else {
                                    self.ack_task(&task_json).await;
                                }
                            }
                            _ => break, 
                        }
                    }
                }
            }
        }
        
        due_tasks.sort_by_key(|(t, _)| t.execute_at);
        due_tasks
    }

    pub async fn update_status(&self, task: &ScheduledTask, new_status: TaskStatus) {
        match &self.backend {
            TaskBackend::InMemory(map) => {
                let mut lock = map.write().await;
                if let Some(t) = lock.get_mut(&task.id) {
                    t.status = new_status.clone();
                }
            }
            TaskBackend::Redis { .. } => {
                if let TaskStatus::Failed(ref err) = new_status {
                    if let Ok(mut con) = self.get_redis_conn().await {
                        let mut failed_task = task.clone();
                        failed_task.status = TaskStatus::Failed(err.clone());
                        
                        if let Ok(task_json) = serde_json::to_string(&failed_task) {
                            let _: redis::RedisResult<()> = con.rpush("agent:failed_tasks", task_json).await;
                        }
                    }
                }
            }
        }

        match new_status {
            TaskStatus::Completed => println!("[TaskManager] Task {} completed.", task.id),
            TaskStatus::Failed(ref err) => println!("[TaskManager] Task {} failed: {}", task.id, err),
            _ => {}
        }
    }

    pub fn start_daemon<F>(self, executor: F)
    where
    F: Fn(ScheduledTask) -> TaskFuture + Send + Sync + Clone + 'static
    {
        let manager = self.clone();
        tokio::spawn(async move {
            println!("[Daemon] Background task manager started.");
            loop {
                sleep(Duration::from_secs(1)).await;
                let due_tasks = manager.get_due_tasks().await;
                
                for (task, task_json) in due_tasks {
                    println!("[Daemon] Task triggered: {}", task.id);
                    
                    let manager_clone = manager.clone();
                    let executor_result_future = executor(task.clone());
                    
                    tokio::spawn(async move {
                        let result = executor_result_future.await; 
                        manager_clone.ack_task(&task_json).await;

                        match result {
                            Ok(_) => manager_clone.update_status(&task, TaskStatus::Completed).await,
                            Err(e) => manager_clone.update_status(&task, TaskStatus::Failed(e)).await,
                        }
                    });
                }
            }
        });
    }
}