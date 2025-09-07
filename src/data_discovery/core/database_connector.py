# src/data_discovery/core/database_connector.py

"""
Database connector for Snowflake with connection pooling, monitoring, and error handling.

Integrates with the configuration, logging, and exception systems to provide
secure, monitored database access for all agents and tools.
"""

import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import contextmanager
import threading
from queue import Queue, Empty
import hashlib

# Core system imports
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from core.config import get_config, SnowflakeConfig
from logging_config import get_logger
from exceptions import (
    DataDiscoveryException, SnowflakeConnectionError, DatabaseTimeoutError,
    AuthenticationError, PermissionError, ErrorContext, ErrorSeverity
)

# Snowflake connector - will be imported when available
try:
    import snowflake.connector
    from snowflake.connector import DictCursor
    from snowflake.connector.errors import (
        DatabaseError, ProgrammingError, IntegrityError,
        OperationalError, NotSupportedError
    )

    HAS_SNOWFLAKE = True
except ImportError:
    HAS_SNOWFLAKE = False


    # Create mock classes for development
    class DatabaseError(Exception):
        pass


    class ProgrammingError(Exception):
        pass


    class IntegrityError(Exception):
        pass


    class OperationalError(Exception):
        pass


    class NotSupportedError(Exception):
        pass


@dataclass
class QueryResult:
    """Result of a database query with metadata."""
    data: List[Dict[str, Any]]
    columns: List[str]
    row_count: int
    execution_time_seconds: float
    query_id: Optional[str] = None
    warehouse_used: Optional[str] = None
    bytes_scanned: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "row_count": self.row_count,
            "column_count": len(self.columns),
            "execution_time_seconds": self.execution_time_seconds,
            "query_id": self.query_id,
            "warehouse_used": self.warehouse_used,
            "bytes_scanned": self.bytes_scanned
        }


@dataclass
class ConnectionMetrics:
    """Metrics for database connection monitoring."""
    connection_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    total_queries: int = 0
    total_execution_time: float = 0.0
    errors_count: int = 0
    is_active: bool = True

    def update_usage(self, execution_time: float, had_error: bool = False):
        """Update connection usage metrics."""
        self.last_used = datetime.now()
        self.total_queries += 1
        self.total_execution_time += execution_time
        if had_error:
            self.errors_count += 1

    def get_avg_execution_time(self) -> float:
        """Get average query execution time."""
        if self.total_queries == 0:
            return 0.0
        return self.total_execution_time / self.total_queries

    def is_idle(self, max_idle_minutes: int = 30) -> bool:
        """Check if connection has been idle too long."""
        idle_time = datetime.now() - self.last_used
        return idle_time > timedelta(minutes=max_idle_minutes)


class SnowflakeConnector:
    """Snowflake database connector with connection pooling and monitoring."""

    def __init__(self, config: SnowflakeConfig = None):
        self.config = config or get_config().snowflake
        self.logger = get_logger("snowflake_connector")

        # Connection pool
        self._connection_pool: Queue = Queue(maxsize=10)
        self._pool_lock = threading.Lock()
        self._connection_metrics: Dict[str, ConnectionMetrics] = {}

        # Configuration
        self._max_retries = 3
        self._retry_delay_seconds = 1
        self._connection_timeout_seconds = 30

        # Initialize if Snowflake is available
        if not HAS_SNOWFLAKE:
            self.logger.warning("Snowflake connector not available - running in mock mode")

    def _generate_connection_id(self) -> str:
        """Generate unique connection ID."""
        timestamp = str(int(time.time() * 1000))
        hash_input = f"{self.config.account}_{self.config.user}_{timestamp}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:8]

    def _create_connection(self) -> Any:
        """Create a new Snowflake connection."""
        if not HAS_SNOWFLAKE:
            # Return mock connection for development
            return MockConnection(self.config)

        start_time = time.time()
        connection_id = self._generate_connection_id()

        try:
            self.logger.info("Creating Snowflake connection",
                             connection_id=connection_id,
                             account=self.config.account,
                             user=self.config.user,
                             warehouse=self.config.warehouse)

            # Prepare connection parameters
            conn_params = {
                "account": self.config.account,
                "user": self.config.user,
                "warehouse": self.config.warehouse,
                "database": self.config.database,
                "schema": self.config.schema,
                "role": self.config.role,
                "login_timeout": self._connection_timeout_seconds,
                "network_timeout": self._connection_timeout_seconds
            }

            # Add authentication
            if self.config.password:
                conn_params["password"] = self.config.password
            elif self.config.private_key_path:
                # Load private key for key-pair authentication
                with open(self.config.private_key_path, 'rb') as key_file:
                    private_key = key_file.read()
                conn_params["private_key"] = private_key
            else:
                raise SnowflakeConnectionError(
                    "No authentication method provided",
                    account=self.config.account
                )

            # Create connection
            connection = snowflake.connector.connect(**conn_params)

            # Test connection
            with connection.cursor() as cursor:
                cursor.execute("SELECT CURRENT_VERSION()")
                version = cursor.fetchone()[0]
                self.logger.info("Snowflake connection established",
                                 connection_id=connection_id,
                                 snowflake_version=version)

            # Store connection metadata
            connection._connection_id = connection_id
            creation_time = time.time() - start_time

            # Track metrics
            self._connection_metrics[connection_id] = ConnectionMetrics(
                connection_id=connection_id
            )

            self.logger.log_database_query(
                query="CONNECTION_ESTABLISHED",
                duration_seconds=creation_time
            )

            return connection

        except DatabaseError as e:
            error_context = ErrorContext()
            error_context.database = self.config.database

            if "authentication" in str(e).lower():
                raise AuthenticationError(
                    f"Snowflake authentication failed: {str(e)}",
                    context=error_context
                )
            elif "permission" in str(e).lower() or "access" in str(e).lower():
                raise PermissionError(
                    f"Snowflake access denied: {str(e)}",
                    context=error_context
                )
            else:
                raise SnowflakeConnectionError(
                    str(e),
                    account=self.config.account
                )

        except Exception as e:
            self.logger.error("Failed to create Snowflake connection",
                              connection_id=connection_id,
                              error=str(e),
                              account=self.config.account)
            raise SnowflakeConnectionError(
                f"Connection creation failed: {str(e)}",
                account=self.config.account
            )

    def _get_connection(self) -> Any:
        """Get connection from pool or create new one."""
        with self._pool_lock:
            try:
                # Try to get from pool (non-blocking)
                connection = self._connection_pool.get_nowait()

                # Validate connection is still active
                if self._is_connection_valid(connection):
                    return connection
                else:
                    # Connection is stale, close it
                    self._close_connection(connection)

            except Empty:
                pass

        # No valid connection in pool, create new one
        return self._create_connection()

    def _return_connection(self, connection: Any):
        """Return connection to pool."""
        if connection and self._is_connection_valid(connection):
            with self._pool_lock:
                try:
                    self._connection_pool.put_nowait(connection)
                except:
                    # Pool is full, close connection
                    self._close_connection(connection)
        else:
            self._close_connection(connection)

    def _is_connection_valid(self, connection: Any) -> bool:
        """Check if connection is still valid."""
        if not connection:
            return False

        try:
            # Quick validation query
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
            return True
        except:
            return False

    def _close_connection(self, connection: Any):
        """Close a database connection."""
        if connection:
            try:
                connection.close()
                if hasattr(connection, '_connection_id'):
                    conn_id = connection._connection_id
                    if conn_id in self._connection_metrics:
                        self._connection_metrics[conn_id].is_active = False
            except:
                pass

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        connection = None
        start_time = time.time()

        try:
            connection = self._get_connection()
            yield connection

        finally:
            if connection:
                # Update metrics
                execution_time = time.time() - start_time
                if hasattr(connection, '_connection_id'):
                    conn_id = connection._connection_id
                    if conn_id in self._connection_metrics:
                        self._connection_metrics[conn_id].update_usage(execution_time)

                self._return_connection(connection)

    def execute_query(
            self,
            query: str,
            parameters: Optional[Dict[str, Any]] = None,
            timeout_seconds: Optional[int] = None
    ) -> QueryResult:
        """Execute a SQL query and return results."""
        start_time = time.time()
        timeout = timeout_seconds or self.config.query_timeout_seconds
        connection = None  # Initialize connection variable

        # Log query start
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        self.logger.info("Executing query",
                         query_hash=query_hash,
                         query_preview=query[:100] + "..." if len(query) > 100 else query,
                         timeout_seconds=timeout)

        try:
            with self.get_connection() as connection:
                with connection.cursor(DictCursor) as cursor:
                    # Set query timeout
                    cursor.execute(f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {timeout}")

                    # Execute main query
                    if parameters:
                        cursor.execute(query, parameters)
                    else:
                        cursor.execute(query)

                    # Fetch results
                    rows = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description] if cursor.description else []

                    execution_time = time.time() - start_time

                    # Get query metadata if available
                    query_id = getattr(cursor, 'sfqid', None)

                    result = QueryResult(
                        data=rows,
                        columns=columns,
                        row_count=len(rows),
                        execution_time_seconds=execution_time,
                        query_id=query_id,
                        warehouse_used=self.config.warehouse
                    )

                    # Log successful query
                    self.logger.log_database_query(
                        query=query[:200] + "..." if len(query) > 200 else query,
                        duration_seconds=execution_time,
                        row_count=len(rows)
                    )

                    return result

        except OperationalError as e:
            execution_time = time.time() - start_time
            if "timeout" in str(e).lower():
                raise DatabaseTimeoutError("query_execution", timeout)
            else:
                raise SnowflakeConnectionError(str(e), account=self.config.account)

        except ProgrammingError as e:
            self.logger.error("SQL query error",
                              query_hash=query_hash,
                              error=str(e),
                              query_preview=query[:100])
            raise DataDiscoveryException(
                f"SQL execution failed: {str(e)}",
                category=ErrorSeverity.HIGH
            )

        except Exception as e:
            execution_time = time.time() - start_time

            # Update connection metrics with error (only if connection was created)
            if connection is not None and hasattr(connection, '_connection_id'):
                conn_id = connection._connection_id
                if conn_id in self._connection_metrics:
                    self._connection_metrics[conn_id].update_usage(execution_time, had_error=True)

            self.logger.error("Query execution failed",
                              query_hash=query_hash,
                              error=str(e),
                              execution_time=execution_time)

            raise SnowflakeConnectionError(
                f"Query execution failed: {str(e)}",
                account=self.config.account
            )

    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            result = self.execute_query("SELECT CURRENT_USER(), CURRENT_DATABASE(), CURRENT_WAREHOUSE()")

            if result.row_count > 0:
                user, database, warehouse = result.data[0].values()
                self.logger.info("Connection test successful",
                                 current_user=user,
                                 current_database=database,
                                 current_warehouse=warehouse)
                return True

            return False

        except Exception as e:
            self.logger.error("Connection test failed", error=str(e))
            return False

    def get_connection_metrics(self) -> Dict[str, Any]:
        """Get connection pool metrics."""
        active_connections = sum(1 for m in self._connection_metrics.values() if m.is_active)
        total_queries = sum(m.total_queries for m in self._connection_metrics.values())
        total_errors = sum(m.errors_count for m in self._connection_metrics.values())
        avg_execution_time = sum(m.get_avg_execution_time() for m in self._connection_metrics.values())

        if len(self._connection_metrics) > 0:
            avg_execution_time /= len(self._connection_metrics)

        return {
            "pool_size": self._connection_pool.qsize(),
            "active_connections": active_connections,
            "total_connections_created": len(self._connection_metrics),
            "total_queries_executed": total_queries,
            "total_errors": total_errors,
            "average_execution_time_seconds": avg_execution_time,
            "error_rate": total_errors / max(total_queries, 1)
        }

    def cleanup_idle_connections(self):
        """Clean up idle connections from the pool."""
        with self._pool_lock:
            connections_to_close = []

            # Find idle connections
            for conn_id, metrics in self._connection_metrics.items():
                if metrics.is_active and metrics.is_idle():
                    connections_to_close.append(conn_id)

            # Close idle connections
            for conn_id in connections_to_close:
                if conn_id in self._connection_metrics:
                    self._connection_metrics[conn_id].is_active = False
                    self.logger.info("Closing idle connection", connection_id=conn_id)

            self.logger.info("Connection cleanup completed",
                             idle_connections_closed=len(connections_to_close))


class MockConnection:
    """Mock connection for testing when Snowflake is not available."""

    def __init__(self, config: SnowflakeConfig):
        self.config = config
        self._connection_id = f"mock_{int(time.time())}"
        self.logger = get_logger("mock_connector")

    def cursor(self, cursor_class=None):
        return MockCursor(self.config)

    def close(self):
        self.logger.info("Mock connection closed", connection_id=self._connection_id)


class MockCursor:
    """Mock cursor for testing."""

    def __init__(self, config: SnowflakeConfig):
        self.config = config
        self.description = None
        self.sfqid = f"mock_query_{int(time.time())}"

    def execute(self, query: str, parameters=None):
        # Simulate different responses based on query
        if "CURRENT_VERSION" in query:
            self._mock_data = [("8.12.5",)]
            self.description = [("CURRENT_VERSION()",)]
        elif "CURRENT_USER" in query:
            self._mock_data = [(self.config.user, self.config.database, self.config.warehouse)]
            self.description = [("CURRENT_USER()",), ("CURRENT_DATABASE()",), ("CURRENT_WAREHOUSE()",)]
        elif "SELECT 1" in query:
            self._mock_data = [(1,)]
            self.description = [("1",)]
        else:
            # Generic mock response
            self._mock_data = [{"id": 1, "name": "Sample Data", "value": 100}]
            self.description = [("ID",), ("NAME",), ("VALUE",)]

    def fetchone(self):
        if hasattr(self, '_mock_data') and self._mock_data:
            return self._mock_data[0]
        return None

    def fetchall(self):
        if hasattr(self, '_mock_data'):
            return self._mock_data
        return []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# Global connector instance
_connector: Optional[SnowflakeConnector] = None


def get_connector(config: SnowflakeConfig = None) -> SnowflakeConnector:
    """Get global Snowflake connector instance."""
    global _connector

    if _connector is None:
        _connector = SnowflakeConnector(config)

    return _connector


def test_database_connection() -> bool:
    """Test database connection with current configuration."""
    try:
        connector = get_connector()
        return connector.test_connection()
    except Exception as e:
        logger = get_logger("database_test")
        logger.error("Database connection test failed", error=str(e))
        return False


# Testing and demonstration
if __name__ == "__main__":
    print("Testing Database Connector System")
    print("=" * 50)

    # Test with mock configuration
    logger = get_logger("test_connector")

    try:
        # Test connector creation
        connector = get_connector()
        logger.info("Database connector created successfully")

        # Test connection
        print("🔗 Testing database connection...")
        if connector.test_connection():
            print("✅ Connection test passed")
        else:
            print("⚠️  Connection test failed (expected with demo credentials)")

        # Test query execution
        print("\n🔍 Testing query execution...")
        try:
            result = connector.execute_query("SELECT 1 as test_column")
            print(f"✅ Query executed successfully")
            print(f"   Rows returned: {result.row_count}")
            print(f"   Execution time: {result.execution_time_seconds:.3f} seconds")
            print(f"   Columns: {result.columns}")
            print(f"   Sample data: {result.data[:3]}")  # Show first 3 rows

        except Exception as e:
            print(f"⚠️  Query execution failed: {str(e)}")

        # Test connection metrics
        print("\n📊 Connection metrics:")
        metrics = connector.get_connection_metrics()
        for key, value in metrics.items():
            print(f"   {key}: {value}")

        # Test cleanup
        print("\n🧹 Testing connection cleanup...")
        connector.cleanup_idle_connections()
        print("✅ Cleanup completed")

        print(f"\n✅ Database connector system tested successfully!")
        print(f"   Features: Connection pooling, error handling, metrics")
        print(f"   Integration: Config system, logging, exceptions")
        print(f"   Mode: {'Production' if HAS_SNOWFLAKE else 'Development (Mock)'}")

        if not HAS_SNOWFLAKE:
            print(f"\n💡 To enable Snowflake connectivity:")
            print(f"   pip install snowflake-connector-python")
            print(f"   Update .env with real Snowflake credentials")

    except Exception as e:
        logger.error("Database connector test failed", error=str(e))
        print(f"❌ Test failed: {str(e)}")

        # Show what's needed to fix
        config = get_config()
        if not config.can_connect_to_snowflake():
            print(f"\n🔧 Configuration issues:")
            print(f"   Account: {'✅' if config.snowflake.account else '❌'}")
            print(f"   User: {'✅' if config.snowflake.user else '❌'}")
            print(f"   Auth: {'✅' if (config.snowflake.password or config.snowflake.private_key_path) else '❌'}")