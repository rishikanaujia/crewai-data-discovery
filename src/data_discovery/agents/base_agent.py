# src/data_discovery/agents/base_agent.py

"""
Base Agent Framework for the Data Discovery system.

Provides common functionality, error handling, state management, and integration
with core systems that all specialized agents will inherit from.
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from abc import ABC, abstractmethod
from enum import Enum

# Core system imports
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core.config import get_config
from core.logging_config import get_logger
from core.state_manager import get_state_manager, StateType
from core.database_connector import get_connector
from core.exceptions import DataDiscoveryException


class AgentStatus(Enum):
    """Status of agent execution."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class TaskPriority(Enum):
    """Priority levels for agent tasks."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AgentTask:
    """Represents a task for an agent to execute."""
    task_id: str
    task_type: str
    description: str
    priority: TaskPriority = TaskPriority.MEDIUM
    parameters: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: AgentStatus = AgentStatus.IDLE
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress_percent: float = 0.0

    def get_duration_seconds(self) -> Optional[float]:
        """Get task execution duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            return (datetime.now() - self.started_at).total_seconds()
        return None


@dataclass
class AgentMetrics:
    """Metrics for agent performance tracking."""
    agent_id: str
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    total_execution_time_seconds: float = 0.0
    average_task_time_seconds: float = 0.0
    success_rate: float = 0.0
    last_activity: Optional[datetime] = None

    def update_completion(self, task: AgentTask):
        """Update metrics when a task completes successfully."""
        self.total_tasks_completed += 1
        if task.get_duration_seconds():
            self.total_execution_time_seconds += task.get_duration_seconds()
        self.last_activity = datetime.now()
        self._recalculate_derived_metrics()

    def update_failure(self, task: AgentTask):
        """Update metrics when a task fails."""
        self.total_tasks_failed += 1
        if task.get_duration_seconds():
            self.total_execution_time_seconds += task.get_duration_seconds()
        self.last_activity = datetime.now()
        self._recalculate_derived_metrics()

    def _recalculate_derived_metrics(self):
        """Recalculate derived metrics."""
        total_tasks = self.total_tasks_completed + self.total_tasks_failed
        if total_tasks > 0:
            self.success_rate = self.total_tasks_completed / total_tasks
            self.average_task_time_seconds = self.total_execution_time_seconds / total_tasks


class BaseAgent(ABC):
    """
    Base class for all Data Discovery agents.

    Provides common functionality including logging, error handling,
    state management, and integration with core systems.
    """

    def __init__(self, agent_id: str = None, agent_name: str = None):
        # Agent identification
        self.agent_id = agent_id or f"{self.__class__.__name__}_{uuid.uuid4().hex[:8]}"
        self.agent_name = agent_name or self.__class__.__name__

        # Core system integration
        self.config = get_config()
        self.logger = get_logger(f"agent_{self.agent_id}")
        self.state_manager = get_state_manager()
        self.connector = get_connector()

        # Agent state
        self.status = AgentStatus.IDLE
        self.current_task: Optional[AgentTask] = None
        self.task_queue: List[AgentTask] = []
        self.metrics = AgentMetrics(agent_id=self.agent_id)

        # Configuration
        self.max_retries = 3
        self.timeout_seconds = 300  # 5 minutes default

        self.logger.info("Agent initialized",
                         agent_id=self.agent_id,
                         agent_name=self.agent_name,
                         agent_type=self.__class__.__name__)

    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return list of capabilities this agent provides."""
        pass

    @abstractmethod
    def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute a specific task. Must be implemented by subclasses."""
        pass

    def add_task(self, task_type: str, description: str, parameters: Dict[str, Any] = None,
                 priority: TaskPriority = TaskPriority.MEDIUM) -> str:
        """Add a task to the agent's queue."""
        task = AgentTask(
            task_id=f"{self.agent_id}_{uuid.uuid4().hex[:8]}",
            task_type=task_type,
            description=description,
            priority=priority,
            parameters=parameters or {}
        )

        # Insert task based on priority
        inserted = False
        for i, existing_task in enumerate(self.task_queue):
            if task.priority.value > existing_task.priority.value:
                self.task_queue.insert(i, task)
                inserted = True
                break

        if not inserted:
            self.task_queue.append(task)

        self.logger.info("Task added to queue",
                         task_id=task.task_id,
                         task_type=task_type,
                         priority=priority.name,
                         queue_size=len(self.task_queue))

        return task.task_id

    def run_next_task(self) -> Optional[Dict[str, Any]]:
        """Execute the next task in the queue."""
        if not self.task_queue:
            self.logger.debug("No tasks in queue")
            return None

        task = self.task_queue.pop(0)
        return self.run_task(task)

    def run_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute a specific task with error handling and logging."""
        self.current_task = task
        self.status = AgentStatus.RUNNING
        task.status = AgentStatus.RUNNING
        task.started_at = datetime.now()

        # Save progress checkpoint
        checkpoint_key = f"task_{task.task_id}"

        try:
            self.logger.log_agent_start(self.agent_id, task.description)

            with self.state_manager.checkpoint(checkpoint_key) as checkpoint_id:
                # Save task progress
                self.state_manager.save_state(
                    checkpoint_key,
                    {
                        "task_id": task.task_id,
                        "agent_id": self.agent_id,
                        "status": "running",
                        "started_at": task.started_at.isoformat(),
                        "progress": 0
                    },
                    StateType.AGENT_PROGRESS,
                    ttl_hours=1
                )

                # Execute the task
                result = self._execute_task_with_retry(task)

                # Mark as completed
                task.completed_at = datetime.now()
                task.status = AgentStatus.COMPLETED
                task.result = result
                task.progress_percent = 100.0

                # Update metrics
                self.metrics.update_completion(task)

                # Log completion
                duration = task.get_duration_seconds()
                self.logger.log_agent_complete(self.agent_id, task.description, duration)

                # Save final progress
                self.state_manager.save_state(
                    checkpoint_key,
                    {
                        "task_id": task.task_id,
                        "agent_id": self.agent_id,
                        "status": "completed",
                        "completed_at": task.completed_at.isoformat(),
                        "progress": 100,
                        "result": result
                    },
                    StateType.AGENT_PROGRESS,
                    ttl_hours=24
                )

                self.status = AgentStatus.IDLE
                self.current_task = None

                return result

        except Exception as e:
            # Handle failure
            task.completed_at = datetime.now()
            task.status = AgentStatus.FAILED
            task.error = str(e)

            # Update metrics
            self.metrics.update_failure(task)

            # Log error
            duration = task.get_duration_seconds()
            self.logger.log_agent_error(self.agent_id, task.description, e)

            self.status = AgentStatus.FAILED
            self.current_task = None

            # Re-raise for handling by caller
            raise DataDiscoveryException(
                f"Agent {self.agent_id} failed task {task.task_id}: {str(e)}"
            )

    def _execute_task_with_retry(self, task: AgentTask) -> Dict[str, Any]:
        """Execute task with retry logic."""
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                self.logger.debug("Executing task",
                                  task_id=task.task_id,
                                  attempt=attempt + 1,
                                  max_attempts=self.max_retries)

                # Update progress
                task.progress_percent = (attempt / self.max_retries) * 50  # Up to 50% for attempts

                result = self.execute_task(task)
                return result

            except Exception as e:
                last_exception = e
                self.logger.warning("Task execution attempt failed",
                                    task_id=task.task_id,
                                    attempt=attempt + 1,
                                    error=str(e))

                if attempt < self.max_retries - 1:
                    # Wait before retry (exponential backoff)
                    wait_time = 2 ** attempt
                    self.logger.info("Retrying task",
                                     task_id=task.task_id,
                                     wait_seconds=wait_time)
                    time.sleep(wait_time)

        # All retries failed
        raise last_exception

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "agent_type": self.__class__.__name__,
            "status": self.status.value,
            "capabilities": self.get_capabilities(),
            "current_task": asdict(self.current_task) if self.current_task else None,
            "queue_size": len(self.task_queue),
            "metrics": asdict(self.metrics)
        }

    def pause(self):
        """Pause the agent (if currently running)."""
        if self.status == AgentStatus.RUNNING:
            self.status = AgentStatus.PAUSED
            self.logger.info("Agent paused", agent_id=self.agent_id)

    def resume(self):
        """Resume the agent (if paused)."""
        if self.status == AgentStatus.PAUSED:
            self.status = AgentStatus.RUNNING
            self.logger.info("Agent resumed", agent_id=self.agent_id)

    def clear_queue(self):
        """Clear all pending tasks."""
        cleared_count = len(self.task_queue)
        self.task_queue.clear()
        self.logger.info("Task queue cleared",
                         agent_id=self.agent_id,
                         cleared_tasks=cleared_count)

    def get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result from state manager."""
        return self.state_manager.load_state(cache_key, StateType.AGENT_PROGRESS)

    def cache_result(self, cache_key: str, result: Any, ttl_hours: int = 24):
        """Cache a result in state manager."""
        self.state_manager.save_state(
            cache_key, result, StateType.AGENT_PROGRESS, ttl_hours=ttl_hours
        )


class TestAgent(BaseAgent):
    """Test agent implementation for demonstration."""

    def get_capabilities(self) -> List[str]:
        """Return test agent capabilities."""
        return [
            "database_connection_test",
            "simple_query_execution",
            "state_management_test",
            "error_handling_demo"
        ]

    def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute test tasks."""
        task_type = task.task_type
        parameters = task.parameters

        if task_type == "database_connection_test":
            return self._test_database_connection()

        elif task_type == "simple_query_execution":
            query = parameters.get("query", "SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()")
            return self._execute_simple_query(query)

        elif task_type == "state_management_test":
            return self._test_state_management()

        elif task_type == "error_handling_demo":
            error_type = parameters.get("error_type", "simple")
            return self._demo_error_handling(error_type)

        else:
            raise DataDiscoveryException(f"Unknown task type: {task_type}")

    def _test_database_connection(self) -> Dict[str, Any]:
        """Test database connectivity."""
        self.logger.info("Testing database connection")

        result = self.connector.execute_query("SELECT CURRENT_USER(), CURRENT_WAREHOUSE()")

        return {
            "status": "success",
            "connection_test": "passed",
            "current_user": result.data[0]["CURRENT_USER()"],
            "current_warehouse": result.data[0]["CURRENT_WAREHOUSE()"],
            "execution_time": result.execution_time_seconds
        }

    def _execute_simple_query(self, query: str) -> Dict[str, Any]:
        """Execute a simple SQL query."""
        self.logger.info("Executing query", query=query[:100])

        result = self.connector.execute_query(query)

        return {
            "status": "success",
            "query": query,
            "row_count": result.row_count,
            "columns": result.columns,
            "execution_time": result.execution_time_seconds,
            "sample_data": result.data[:3]  # First 3 rows
        }

    def _test_state_management(self) -> Dict[str, Any]:
        """Test state management functionality."""
        self.logger.info("Testing state management")

        # Save some test data
        test_data = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": self.agent_id,
            "test_value": "state_management_test"
        }

        cache_key = f"test_state_{self.agent_id}"
        self.cache_result(cache_key, test_data, ttl_hours=1)

        # Load it back
        loaded_data = self.get_cached_result(cache_key)

        return {
            "status": "success",
            "cached_data": test_data,
            "loaded_data": loaded_data,
            "cache_working": loaded_data is not None
        }

    def _demo_error_handling(self, error_type: str) -> Dict[str, Any]:
        """Demonstrate error handling."""
        self.logger.info("Demonstrating error handling", error_type=error_type)

        if error_type == "database_error":
            # This will cause a SQL error
            self.connector.execute_query("SELECT * FROM non_existent_table")

        elif error_type == "timeout_error":
            # Simulate a long operation
            time.sleep(10)

        elif error_type == "simple":
            raise ValueError("This is a test error for demonstration")

        return {"status": "success", "message": "No error occurred"}


# Testing and demonstration
if __name__ == "__main__":
    print("Testing Base Agent Framework")
    print("=" * 50)

    try:
        # Create test agent
        print("🤖 Creating test agent...")
        agent = TestAgent(agent_name="TestAgent_Demo")

        print(f"✅ Agent created: {agent.agent_id}")
        print(f"   Name: {agent.agent_name}")
        print(f"   Capabilities: {', '.join(agent.get_capabilities())}")

        # Test 1: Database connection test
        print(f"\n🔍 Test 1: Database Connection")
        task_id = agent.add_task(
            task_type="database_connection_test",
            description="Test database connectivity",
            priority=TaskPriority.HIGH
        )

        result = agent.run_next_task()
        print(f"✅ Database test completed")
        print(f"   User: {result['current_user']}")
        print(f"   Warehouse: {result['current_warehouse']}")
        print(f"   Execution time: {result['execution_time']:.3f}s")

        # Test 2: Simple query execution
        print(f"\n🔍 Test 2: Query Execution")
        agent.add_task(
            task_type="simple_query_execution",
            description="Execute simple query",
            parameters={
                "query": "SELECT COUNT(*) as table_count FROM information_schema.tables WHERE table_schema = 'TPCDS_SF10TCL'"},
            priority=TaskPriority.MEDIUM
        )

        result = agent.run_next_task()
        print(f"✅ Query executed successfully")
        print(f"   Rows returned: {result['row_count']}")
        print(f"   Columns: {result['columns']}")
        print(f"   Sample data: {result['sample_data']}")

        # Test 3: State management
        print(f"\n🔍 Test 3: State Management")
        agent.add_task(
            task_type="state_management_test",
            description="Test state caching",
            priority=TaskPriority.LOW
        )

        result = agent.run_next_task()
        print(f"✅ State management test completed")
        print(f"   Cache working: {result['cache_working']}")
        print(f"   Data round-trip: {'Success' if result['cached_data'] == result['loaded_data'] else 'Failed'}")

        # Test 4: Error handling (safe error)
        print(f"\n🔍 Test 4: Error Handling")
        try:
            agent.add_task(
                task_type="error_handling_demo",
                description="Demonstrate error handling",
                parameters={"error_type": "simple"}
            )
            agent.run_next_task()
        except Exception as e:
            print(f"✅ Error handling working correctly")
            print(f"   Error caught: {type(e).__name__}")
            print(f"   Error message: {str(e)[:100]}")

        # Show agent status and metrics
        print(f"\n📊 Agent Status and Metrics:")
        status = agent.get_status()
        metrics = status['metrics']

        print(f"   Status: {status['status']}")
        print(f"   Tasks completed: {metrics['total_tasks_completed']}")
        print(f"   Tasks failed: {metrics['total_tasks_failed']}")
        print(f"   Success rate: {metrics['success_rate']:.1%}")
        print(f"   Average task time: {metrics['average_task_time_seconds']:.2f}s")

        print(f"\n✅ Base agent framework tested successfully!")
        print(f"   Core functionality: Task execution, error handling, state management")
        print(f"   Integration: Database connector, logging, state manager")
        print(f"   Patterns: Retry logic, progress tracking, metrics")
        print(f"\n🚀 Ready to build specialized agents!")

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        print(f"   Check that core systems are working properly")