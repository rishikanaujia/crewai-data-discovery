# Production enhancements for the database connector

import asyncio
from typing import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor, Future
import weakref


class QueryCache:
    """Simple query result cache with TTL."""

    def __init__(self, max_size: int = 1000, default_ttl_seconds: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl_seconds
        self.cache = {}
        self.timestamps = {}
        self.lock = threading.Lock()

    def _generate_key(self, query: str, parameters: Optional[Dict] = None) -> str:
        """Generate cache key from query and parameters."""
        param_str = json.dumps(parameters, sort_keys=True) if parameters else ""
        return hashlib.md5(f"{query}_{param_str}".encode()).hexdigest()

    def get(self, query: str, parameters: Optional[Dict] = None) -> Optional[QueryResult]:
        """Get cached result if available and not expired."""
        key = self._generate_key(query, parameters)

        with self.lock:
            if key in self.cache:
                timestamp = self.timestamps[key]
                if time.time() - timestamp < self.default_ttl:
                    return self.cache[key]
                else:
                    # Expired, remove from cache
                    del self.cache[key]
                    del self.timestamps[key]

        return None

    def put(self, query: str, result: QueryResult, parameters: Optional[Dict] = None):
        """Cache query result."""
        key = self._generate_key(query, parameters)

        with self.lock:
            # Evict oldest if at capacity
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]

            self.cache[key] = result
            self.timestamps[key] = time.time()


class CircuitBreaker:
    """Circuit breaker pattern for database connections."""

    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        self.lock = threading.Lock()

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == "open":
                if time.time() - self.last_failure_time > self.timeout_seconds:
                    self.state = "half-open"
                else:
                    raise CircuitBreakerError("database", self.failure_count)

        try:
            result = func(*args, **kwargs)

            # Success - reset circuit breaker
            with self.lock:
                if self.state == "half-open":
                    self.state = "closed"
                self.failure_count = 0

            return result

        except Exception as e:
            with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time()

                if self.failure_count >= self.failure_threshold:
                    self.state = "open"

            raise e


class EnhancedSnowflakeConnector(SnowflakeConnector):
    """Enhanced connector with caching, circuit breaker, and async support."""

    def __init__(self, config: SnowflakeConfig = None):
        super().__init__(config)

        # Enhanced features
        self.query_cache = QueryCache(
            max_size=self.config.max_sample_rows,
            default_ttl_seconds=300  # 5 minutes
        )
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            timeout_seconds=60
        )
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

        # Performance tracking
        self.slow_query_threshold = 10.0  # seconds
        self.slow_queries = []
        self.query_stats = {
            "total_queries": 0,
            "cached_queries": 0,
            "slow_queries": 0,
            "failed_queries": 0
        }

    def execute_query_cached(
            self,
            query: str,
            parameters: Optional[Dict[str, Any]] = None,
            use_cache: bool = True,
            timeout_seconds: Optional[int] = None
    ) -> QueryResult:
        """Execute query with caching support."""

        # Check cache first
        if use_cache:
            cached_result = self.query_cache.get(query, parameters)
            if cached_result:
                self.query_stats["cached_queries"] += 1
                self.logger.debug("Query result served from cache",
                                  query_hash=hashlib.md5(query.encode()).hexdigest()[:8])
                return cached_result

        # Execute with circuit breaker
        def _execute():
            return self.execute_query(query, parameters, timeout_seconds)

        try:
            result = self.circuit_breaker.call(_execute)

            # Cache successful results for SELECT queries
            if use_cache and query.strip().upper().startswith("SELECT"):
                self.query_cache.put(query, result, parameters)

            # Track performance
            self._track_query_performance(query, result)

            return result

        except Exception as e:
            self.query_stats["failed_queries"] += 1
            raise e

    def _track_query_performance(self, query: str, result: QueryResult):
        """Track query performance metrics."""
        self.query_stats["total_queries"] += 1

        if result.execution_time_seconds > self.slow_query_threshold:
            self.query_stats["slow_queries"] += 1

            # Store slow query details (keep last 50)
            slow_query_info = {
                "query": query[:200] + "..." if len(query) > 200 else query,
                "execution_time": result.execution_time_seconds,
                "row_count": result.row_count,
                "timestamp": datetime.now().isoformat()
            }

            self.slow_queries.append(slow_query_info)
            if len(self.slow_queries) > 50:
                self.slow_queries.pop(0)

            self.logger.warning("Slow query detected",
                                execution_time=result.execution_time_seconds,
                                query_preview=query[:100])

    async def execute_query_async(
            self,
            query: str,
            parameters: Optional[Dict[str, Any]] = None,
            timeout_seconds: Optional[int] = None
    ) -> QueryResult:
        """Execute query asynchronously."""
        loop = asyncio.get_event_loop()

        # Run in thread pool to avoid blocking
        future = loop.run_in_executor(
            self.thread_pool,
            self.execute_query_cached,
            query,
            parameters,
            True,  # use_cache
            timeout_seconds
        )

        return await future

    def batch_execute(
            self,
            queries: List[Tuple[str, Optional[Dict[str, Any]]]],
            max_concurrent: int = 3
    ) -> List[QueryResult]:
        """Execute multiple queries with limited concurrency."""
        results = []

        # Process in batches
        for i in range(0, len(queries), max_concurrent):
            batch = queries[i:i + max_concurrent]

            # Submit batch to thread pool
            futures = []
            for query, params in batch:
                future = self.thread_pool.submit(
                    self.execute_query_cached, query, params
                )
                futures.append(future)

            # Collect results
            for future in futures:
                try:
                    result = future.result(timeout=60)
                    results.append(result)
                except Exception as e:
                    # Create error result
                    error_result = QueryResult(
                        data=[],
                        columns=[],
                        row_count=0,
                        execution_time_seconds=0.0
                    )
                    results.append(error_result)
                    self.logger.error("Batch query failed", error=str(e))

        return results

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        base_metrics = self.get_connection_metrics()

        cache_hit_rate = 0.0
        if self.query_stats["total_queries"] > 0:
            cache_hit_rate = self.query_stats["cached_queries"] / self.query_stats["total_queries"]

        return {
            **base_metrics,
            "query_stats": self.query_stats,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.query_cache.cache),
            "slow_query_count": len(self.slow_queries),
            "circuit_breaker_state": self.circuit_breaker.state,
            "circuit_breaker_failures": self.circuit_breaker.failure_count
        }

    def get_slow_queries(self) -> List[Dict[str, Any]]:
        """Get list of slow queries for optimization."""
        return self.slow_queries.copy()

    def clear_cache(self):
        """Clear query cache."""
        with self.query_cache.lock:
            self.query_cache.cache.clear()
            self.query_cache.timestamps.clear()

        self.logger.info("Query cache cleared")

    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check."""
        health_status = {
            "connection_healthy": False,
            "circuit_breaker_state": self.circuit_breaker.state,
            "active_connections": 0,
            "cache_size": len(self.query_cache.cache),
            "total_queries": self.query_stats["total_queries"],
            "error_rate": 0.0,
            "avg_query_time": 0.0
        }

        try:
            # Test basic connectivity
            result = self.execute_query("SELECT 1")
            health_status["connection_healthy"] = result.row_count > 0

            # Get connection metrics
            metrics = self.get_connection_metrics()
            health_status["active_connections"] = metrics["active_connections"]
            health_status["error_rate"] = metrics["error_rate"]
            health_status["avg_query_time"] = metrics["average_execution_time_seconds"]

        except Exception as e:
            self.logger.error("Health check failed", error=str(e))

        return health_status

    def __del__(self):
        """Cleanup resources on deletion."""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)


# Example usage patterns
async def example_async_usage():
    """Example of using async database operations."""
    connector = EnhancedSnowflakeConnector()

    # Single async query
    result = await connector.execute_query_async("SELECT COUNT(*) FROM customers")
    print(f"Customer count: {result.data[0]}")

    # Batch execution
    queries = [
        ("SELECT COUNT(*) FROM customers", None),
        ("SELECT COUNT(*) FROM orders", None),
        ("SELECT COUNT(*) FROM products", None)
    ]

    results = connector.batch_execute(queries)
    print(f"Executed {len(results)} queries in batch")

    # Performance monitoring
    stats = connector.get_performance_stats()
    print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
    print(f"Circuit breaker state: {stats['circuit_breaker_state']}")


# Integration with existing patterns
def create_production_connector() -> EnhancedSnowflakeConnector:
    """Factory function for production connector."""
    config = get_config()

    if not config.can_connect_to_snowflake():
        logger = get_logger("connector_factory")
        logger.warning("Creating connector without Snowflake credentials")

    return EnhancedSnowflakeConnector(config.snowflake)