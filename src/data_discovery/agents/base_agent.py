# Production enhancements for the Base Agent Framework

import asyncio
from typing import Callable, Awaitable
from concurrent.futures import ThreadPoolExecutor
import threading
from dataclasses import replace


class AsyncAgentMixin:
    """Mixin to add async capabilities to agents."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        self.async_tasks: Dict[str, asyncio.Task] = {}

    async def execute_task_async(self, task: AgentTask) -> Dict[str, Any]:
        """Execute task asynchronously."""
        loop = asyncio.get_event_loop()

        # Run the synchronous execute_task in thread pool
        return await loop.run_in_executor(
            self.thread_pool,
            self.execute_task,
            task
        )

    async def run_task_async(self, task: AgentTask) -> Dict[str, Any]:
        """Async version of run_task with proper error handling."""
        self.current_task = task
        self.status = AgentStatus.RUNNING
        task.status = AgentStatus.RUNNING
        task.started_at = datetime.now()

        try:
            # Create async task
            async_task = asyncio.create_task(self.execute_task_async(task))
            self.async_tasks[task.task_id] = async_task

            # Execute with timeout
            result = await asyncio.wait_for(
                async_task,
                timeout=self.timeout_seconds
            )

            # Success handling
            task.completed_at = datetime.now()
            task.status = AgentStatus.COMPLETED
            task.result = result
            self.metrics.update_completion(task)

            return result

        except asyncio.TimeoutError:
            task.error = f"Task timed out after {self.timeout_seconds} seconds"
            task.status = AgentStatus.FAILED
            self.metrics.update_failure(task)
            raise

        except Exception as e:
            task.error = str(e)
            task.status = AgentStatus.FAILED
            self.metrics.update_failure(task)
            raise

        finally:
            if task.task_id in self.async_tasks:
                del self.async_tasks[task.task_id]
            self.status = AgentStatus.IDLE
            self.current_task = None


class ProgressTracker:
    """Enhanced progress tracking with detailed steps."""

    def __init__(self, task: AgentTask, total_steps: int):
        self.task = task
        self.total_steps = total_steps
        self.current_step = 0
        self.step_descriptions = {}
        self.step_start_times = {}

    def start_step(self, step_name: str, description: str = ""):
        """Start a new step in the task."""
        self.current_step += 1
        self.step_descriptions[self.current_step] = description or step_name
        self.step_start_times[self.current_step] = time.time()

        # Update task progress
        self.task.progress_percent = (self.current_step / self.total_steps) * 100

        return self.current_step

    def complete_step(self, step_number: int = None):
        """Complete a step and log duration."""
        step_num = step_number or self.current_step
        if step_num in self.step_start_times:
            duration = time.time() - self.step_start_times[step_num]
            return duration
        return 0.0

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get detailed progress summary."""
        return {
            "total_steps": self.total_steps,
            "current_step": self.current_step,
            "progress_percent": self.task.progress_percent,
            "completed_steps": list(self.step_descriptions.keys()),
            "current_description": self.step_descriptions.get(self.current_step, "")
        }


class EnhancedBaseAgent(BaseAgent):
    """Enhanced base agent with additional production features."""

    def __init__(self, agent_id: str = None, agent_name: str = None):
        super().__init__(agent_id, agent_name)

        # Enhanced features
        self.task_dependencies: Dict[str, List[str]] = {}
        self.task_callbacks: Dict[str, List[Callable]] = {}
        self.health_check_interval = 30  # seconds
        self.last_health_check = time.time()

        # Circuit breaker for database connections
        self.db_failure_count = 0
        self.db_circuit_open = False
        self.db_circuit_reset_time = None

        # Performance optimization
        self.result_cache_size = 100
        self.recent_results: Dict[str, Any] = {}

    def add_task_with_dependencies(
            self,
            task_type: str,
            description: str,
            dependencies: List[str] = None,
            callback: Callable = None,
            **kwargs
    ) -> str:
        """Add task with dependency management."""
        task_id = self.add_task(task_type, description, **kwargs)

        if dependencies:
            self.task_dependencies[task_id] = dependencies

        if callback:
            if task_id not in self.task_callbacks:
                self.task_callbacks[task_id] = []
            self.task_callbacks[task_id].append(callback)

        return task_id

    def can_execute_task(self, task_id: str) -> bool:
        """Check if task dependencies are satisfied."""
        if task_id not in self.task_dependencies:
            return True

        dependencies = self.task_dependencies[task_id]

        # Check if all dependencies are completed
        for dep_id in dependencies:
            # Look for completed task in metrics or cache
            if not self._is_dependency_satisfied(dep_id):
                return False

        return True

    def _is_dependency_satisfied(self, dependency_id: str) -> bool:
        """Check if a dependency is satisfied."""
        # Check recent results cache
        if dependency_id in self.recent_results:
            return True

        # Check state manager for completed task
        cached_result = self.get_cached_result(f"task_{dependency_id}")
        return cached_result is not None

    def run_next_task_with_dependencies(self) -> Optional[Dict[str, Any]]:
        """Execute next task that has all dependencies satisfied."""
        for i, task in enumerate(self.task_queue):
            if self.can_execute_task(task.task_id):
                # Remove from queue and execute
                executable_task = self.task_queue.pop(i)
                return self.run_task(executable_task)

        self.logger.debug("No executable tasks (dependencies not satisfied)")
        return None

    def run_task_with_progress(self, task: AgentTask, total_steps: int = 1) -> Dict[str, Any]:
        """Execute task with detailed progress tracking."""
        progress_tracker = ProgressTracker(task, total_steps)

        # Store progress tracker for access by execute_task implementation
        task.parameters['_progress_tracker'] = progress_tracker

        try:
            result = self.run_task(task)

            # Execute callbacks
            if task.task_id in self.task_callbacks:
                for callback in self.task_callbacks[task.task_id]:
                    try:
                        callback(task, result)
                    except Exception as e:
                        self.logger.warning("Task callback failed", error=str(e))

            # Cache result
            self.recent_results[task.task_id] = result

            # Trim cache if too large
            if len(self.recent_results) > self.result_cache_size:
                oldest_key = next(iter(self.recent_results))
                del self.recent_results[oldest_key]

            return result

        finally:
            # Clean up
            if task.task_id in self.task_dependencies:
                del self.task_dependencies[task.task_id]
            if task.task_id in self.task_callbacks:
                del self.task_callbacks[task.task_id]

    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check."""
        current_time = time.time()

        # Skip if too recent
        if current_time - self.last_health_check < self.health_check_interval:
            return {"status": "skipped", "reason": "too_recent"}

        self.last_health_check = current_time

        health_status = {
            "agent_id": self.agent_id,
            "status": self.status.value,
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }

        # Database connectivity check
        try:
            if not self.db_circuit_open:
                test_result = self.connector.execute_query("SELECT 1")
                health_status["checks"]["database"] = "healthy"
                self.db_failure_count = 0
            else:
                health_status["checks"]["database"] = "circuit_open"
        except Exception as e:
            self.db_failure_count += 1
            health_status["checks"]["database"] = f"failed: {str(e)}"

            # Open circuit breaker if too many failures
            if self.db_failure_count >= 3:
                self.db_circuit_open = True
                self.db_circuit_reset_time = current_time + 300  # 5 minutes

        # Check if circuit should be reset
        if self.db_circuit_open and current_time > (self.db_circuit_reset_time or 0):
            self.db_circuit_open = False
            self.db_failure_count = 0

        # Memory usage check
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            health_status["checks"]["memory_mb"] = memory_mb
            health_status["checks"]["memory_status"] = "healthy" if memory_mb < 1000 else "high"
        except ImportError:
            health_status["checks"]["memory"] = "psutil_not_available"

        # Task queue health
        health_status["checks"]["task_queue_size"] = len(self.task_queue)
        health_status["checks"]["task_queue_status"] = "healthy" if len(self.task_queue) < 100 else "backlogged"

        # Metrics health
        health_status["checks"]["success_rate"] = self.metrics.success_rate
        health_status["checks"]["metrics_status"] = "healthy" if self.metrics.success_rate > 0.8 else "degraded"

        return health_status

    def get_enhanced_status(self) -> Dict[str, Any]:
        """Get enhanced status with additional details."""
        base_status = self.get_status()

        enhanced_status = {
            **base_status,
            "health_check": self.health_check(),
            "task_dependencies": len(self.task_dependencies),
            "cached_results": len(self.recent_results),
            "database_circuit_open": self.db_circuit_open,
            "executable_tasks": sum(1 for task in self.task_queue if self.can_execute_task(task.task_id))
        }

        return enhanced_status


class MultiStepAgent(EnhancedBaseAgent):
    """Example agent that demonstrates multi-step task execution."""

    def get_capabilities(self) -> List[str]:
        return ["multi_step_analysis", "data_validation", "report_generation"]

    def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute multi-step task with progress tracking."""
        task_type = task.task_type

        if task_type == "multi_step_analysis":
            return self._execute_multi_step_analysis(task)
        elif task_type == "data_validation":
            return self._execute_data_validation(task)
        else:
            raise DataDiscoveryException(f"Unknown task type: {task_type}")

    def _execute_multi_step_analysis(self, task: AgentTask) -> Dict[str, Any]:
        """Example multi-step analysis with progress tracking."""
        progress = task.parameters.get('_progress_tracker')
        results = {}

        # Step 1: Database connection
        if progress:
            progress.start_step("database_connection", "Connecting to database")

        connection_result = self.connector.execute_query("SELECT CURRENT_DATABASE()")
        results["database"] = connection_result.data[0]["CURRENT_DATABASE()"]

        if progress:
            duration = progress.complete_step()
            self.logger.debug("Database connection completed", duration_seconds=duration)

        # Step 2: Schema discovery
        if progress:
            progress.start_step("schema_discovery", "Discovering schema structure")

        time.sleep(1)  # Simulate work
        results["schema_discovered"] = True

        if progress:
            duration = progress.complete_step()
            self.logger.debug("Schema discovery completed", duration_seconds=duration)

        # Step 3: Data analysis
        if progress:
            progress.start_step("data_analysis", "Analyzing data patterns")

        time.sleep(1)  # Simulate work
        results["patterns_found"] = 5

        if progress:
            duration = progress.complete_step()
            self.logger.debug("Data analysis completed", duration_seconds=duration)

        return results

    def _execute_data_validation(self, task: AgentTask) -> Dict[str, Any]:
        """Example data validation task."""
        # Simple validation example
        return {
            "validation_status": "passed",
            "records_validated": 1000,
            "errors_found": 0
        }


# Example usage of enhanced framework
def example_enhanced_agent_usage():
    """Demonstrate enhanced agent capabilities."""

    # Create enhanced agent
    agent = MultiStepAgent(agent_name="EnhancedDemo")

    # Add task with dependencies and callback
    def task_completion_callback(task: AgentTask, result: Dict[str, Any]):
        print(f"Task {task.task_id} completed with result: {result}")

    # Add validation task (no dependencies)
    validation_task_id = agent.add_task_with_dependencies(
        task_type="data_validation",
        description="Validate data quality",
        priority=TaskPriority.HIGH,
        callback=task_completion_callback
    )

    # Add analysis task (depends on validation)
    analysis_task_id = agent.add_task_with_dependencies(
        task_type="multi_step_analysis",
        description="Perform comprehensive analysis",
        dependencies=[validation_task_id],
        priority=TaskPriority.MEDIUM,
        callback=task_completion_callback
    )

    # Execute tasks with dependency resolution
    while agent.task_queue:
        result = agent.run_next_task_with_dependencies()
        if result is None:
            print("No executable tasks available")
            break
        print(f"Task completed: {result}")

    # Show enhanced status
    status = agent.get_enhanced_status()
    print(f"Agent status: {status}")


if __name__ == "__main__":
    example_enhanced_agent_usage()