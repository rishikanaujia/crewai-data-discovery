# src/data_discovery/core/exceptions.py

"""
Custom exception classes for the Data Discovery system.

This module provides structured error handling with categorized exceptions,
detailed error context, and recovery suggestions for different failure scenarios.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import traceback


class ErrorSeverity(Enum):
    """Error severity levels for categorizing exceptions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors for better error handling."""
    CONFIGURATION = "configuration"
    CONNECTION = "connection"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    DATA_QUALITY = "data_quality"
    AGENT_FAILURE = "agent_failure"
    TOOL_FAILURE = "tool_failure"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Detailed context information for an error."""
    timestamp: datetime = field(default_factory=datetime.now)
    agent_id: Optional[str] = None
    tool_name: Optional[str] = None
    database: Optional[str] = None
    table: Optional[str] = None
    query: Optional[str] = None
    user_action: Optional[str] = None
    environment: Optional[str] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for logging."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "tool_name": self.tool_name,
            "database": self.database,
            "table": self.table,
            "query": self.query,
            "user_action": self.user_action,
            "environment": self.environment,
            "additional_info": self.additional_info
        }


class DataDiscoveryException(Exception):
    """
    Base exception class for all Data Discovery system errors.

    Provides structured error handling with context, severity, and recovery suggestions.
    """

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        recovery_suggestions: Optional[List[str]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext()
        self.recovery_suggestions = recovery_suggestions or []
        self.original_exception = original_exception
        self.error_id = self._generate_error_id()
        self.stack_trace = traceback.format_exc()

    def _generate_error_id(self) -> str:
        """Generate a unique error ID for tracking."""
        import hashlib

        error_string = (
            f"{self.category.value}_{self.severity.value}_{self.message}_{datetime.now().isoformat()}"
        )
        error_hash = hashlib.md5(error_string.encode()).hexdigest()[:8]
        return f"{self.category.value.upper()}_{error_hash}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging and serialization."""
        return {
            "error_id": self.error_id,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "context": self.context.to_dict(),
            "recovery_suggestions": self.recovery_suggestions,
            "original_exception": str(self.original_exception)
            if self.original_exception
            else None,
            "stack_trace": self.stack_trace,
        }

    def get_user_friendly_message(self) -> str:
        """Get a user-friendly version of the error message."""
        if self.recovery_suggestions:
            suggestions = "\n".join(
                f"• {suggestion}" for suggestion in self.recovery_suggestions
            )
            return f"{self.message}\n\nSuggestions:\n{suggestions}"
        return self.message


# ---------------- Configuration Exceptions ----------------
class ConfigurationError(DataDiscoveryException):
    """Errors related to system configuration."""

    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        context = kwargs.pop("context", ErrorContext())
        context.additional_info["config_key"] = config_key
        super().__init__(
            message=message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            context=context,
            **kwargs,
        )


class InvalidConfigurationError(ConfigurationError):
    """Configuration values are invalid or missing."""

    def __init__(self, config_key: str, expected_value: str = None, **kwargs):
        message = f"Invalid configuration for '{config_key}'"
        if expected_value:
            message += f". Expected: {expected_value}"
        recovery_suggestions = [
            f"Check the '{config_key}' setting in your configuration",
            "Verify environment variables are set correctly",
            "Review the configuration documentation",
        ]
        super().__init__(
            message=message,
            config_key=config_key,
            recovery_suggestions=recovery_suggestions,
            **kwargs,
        )


# ---------------- Connection Exceptions ----------------
class ConnectionError(DataDiscoveryException):
    """Errors related to database connections."""

    def __init__(self, message: str, database: Optional[str] = None, **kwargs):
        context = kwargs.pop("context", ErrorContext())
        context.database = database
        super().__init__(
            message=message,
            category=ErrorCategory.CONNECTION,
            severity=ErrorSeverity.HIGH,
            context=context,
            **kwargs,
        )


class SnowflakeConnectionError(ConnectionError):
    """Specific errors for Snowflake database connections."""

    def __init__(self, message: str, account: Optional[str] = None, **kwargs):
        context = kwargs.pop("context", ErrorContext())
        context.additional_info["snowflake_account"] = account
        recovery_suggestions = [
            "Verify Snowflake account, username, and password",
            "Check network connectivity to Snowflake",
            "Ensure the Snowflake warehouse is running",
            "Verify the database and schema exist",
            "Check if your IP is whitelisted",
        ]
        super().__init__(
            message=f"Snowflake connection failed: {message}",
            recovery_suggestions=recovery_suggestions,
            context=context,
            **kwargs,
        )


class DatabaseTimeoutError(ConnectionError):
    """Database operations that exceed timeout limits."""

    def __init__(self, operation: str, timeout_seconds: int, **kwargs):
        message = f"Database operation '{operation}' timed out after {timeout_seconds} seconds"
        recovery_suggestions = [
            "Increase the query timeout setting",
            "Optimize the query for better performance",
            "Check database load and performance",
            "Consider breaking large operations into smaller chunks",
        ]
        super().__init__(
            message=message,
            category=ErrorCategory.TIMEOUT,
            recovery_suggestions=recovery_suggestions,
            **kwargs,
        )


# ---------------- Authentication & Permission Exceptions ----------------
class AuthenticationError(DataDiscoveryException):
    """Authentication-related errors."""

    def __init__(self, message: str, **kwargs):
        recovery_suggestions = [
            "Verify your username and password",
            "Check if your account is locked or expired",
            "Ensure you have access to the specified database",
            "Contact your database administrator",
        ]
        super().__init__(
            message=message,
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            recovery_suggestions=recovery_suggestions,
            **kwargs,
        )


class PermissionError(DataDiscoveryException):
    """Permission and authorization errors."""

    def __init__(self, message: str, required_permission: Optional[str] = None, **kwargs):
        context = kwargs.pop("context", ErrorContext())
        context.additional_info["required_permission"] = required_permission
        recovery_suggestions = [
            "Check if you have the required permissions",
            "Contact your database administrator to grant access",
            "Verify your role has the necessary privileges",
            "Ensure you're connected to the correct database/schema",
        ]
        super().__init__(
            message=message,
            category=ErrorCategory.PERMISSION,
            severity=ErrorSeverity.HIGH,
            context=context,
            recovery_suggestions=recovery_suggestions,
            **kwargs,
        )


# ---------------- Agent & Tool Exceptions ----------------
class AgentError(DataDiscoveryException):
    """Errors from CrewAI agents."""

    def __init__(self, message: str, agent_id: str, **kwargs):
        context = kwargs.pop("context", ErrorContext())
        context.agent_id = agent_id
        super().__init__(
            message=message,
            category=ErrorCategory.AGENT_FAILURE,
            context=context,
            **kwargs,
        )


class AgentTimeoutError(DataDiscoveryException):
    """Agent operations that exceed timeout limits."""

    def __init__(self, agent_id: str, operation: str, timeout_seconds: int, **kwargs):
        message = f"Agent '{agent_id}' timed out during '{operation}' after {timeout_seconds} seconds"
        context = kwargs.pop("context", ErrorContext())
        context.agent_id = agent_id
        recovery_suggestions = [
            "Increase the agent timeout setting",
            "Check if the agent is stuck in an infinite loop",
            "Verify the agent's dependencies are available",
            "Consider simplifying the agent's task",
        ]
        super().__init__(
            message=message,
            category=ErrorCategory.TIMEOUT,
            context=context,
            recovery_suggestions=recovery_suggestions,
            **kwargs,
        )


class ToolError(DataDiscoveryException):
    """Errors from agent tools."""

    def __init__(self, message: str, tool_name: str, **kwargs):
        context = kwargs.pop("context", ErrorContext())
        context.tool_name = tool_name
        super().__init__(
            message=message,
            category=ErrorCategory.TOOL_FAILURE,
            context=context,
            **kwargs,
        )


class SchemaInspectionError(ToolError):
    """Errors during database schema inspection."""

    def __init__(self, message: str, database: Optional[str] = None, table: Optional[str] = None, **kwargs):
        context = kwargs.pop("context", ErrorContext())
        context.database = database
        context.table = table
        recovery_suggestions = [
            "Verify the database and table exist",
            "Check if you have SELECT permissions",
            "Ensure the schema is not locked",
            "Try inspecting a smaller subset of tables",
        ]
        super().__init__(
            message=f"Schema inspection failed: {message}",
            tool_name="SchemaInspector",
            context=context,
            recovery_suggestions=recovery_suggestions,
            **kwargs,
        )


# ---------------- Data Quality Exceptions ----------------
class DataQualityError(DataDiscoveryException):
    """Errors related to data quality issues."""

    def __init__(self, message: str, table: Optional[str] = None, column: Optional[str] = None, **kwargs):
        context = kwargs.pop("context", ErrorContext())
        context.table = table
        context.additional_info["column"] = column
        super().__init__(
            message=message,
            category=ErrorCategory.DATA_QUALITY,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            **kwargs,
        )


class InsufficientDataError(DataDiscoveryException):
    """Not enough data to perform analysis."""

    def __init__(self, table: str, required_rows: int, actual_rows: int):
        message = f"Table '{table}' has insufficient data: {actual_rows} rows (minimum: {required_rows})"
        context = ErrorContext()
        context.table = table
        recovery_suggestions = [
            "Choose a table with more data",
            "Lower the minimum data requirements",
            "Check if the table is still being populated",
            "Verify data loading processes",
        ]
        super().__init__(
            message=message,
            category=ErrorCategory.DATA_QUALITY,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            recovery_suggestions=recovery_suggestions,
        )


# ---------------- Validation Exceptions ----------------
class ValidationError(DataDiscoveryException):
    """Data or input validation errors."""

    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None, **kwargs):
        context = kwargs.pop("context", ErrorContext())
        context.additional_info.update({
            "field": field,
            "value": str(value) if value is not None else None,
        })
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            context=context,
            **kwargs,
        )


class SQLValidationError(ValidationError):
    """SQL query validation errors."""

    def __init__(self, message: str, sql_query: Optional[str] = None, **kwargs):
        context = kwargs.pop("context", ErrorContext())
        context.query = sql_query
        recovery_suggestions = [
            "Check SQL syntax for errors",
            "Verify table and column names exist",
            "Ensure proper quoting and escaping",
            "Test the query in a SQL editor first",
        ]
        super().__init__(
            message=f"SQL validation failed: {message}",
            context=context,
            recovery_suggestions=recovery_suggestions,
            **kwargs,
        )


# ---------------- Resource Exceptions ----------------
class ResourceError(DataDiscoveryException):
    """Resource-related errors (memory, disk, etc.)."""

    def __init__(self, message: str, resource_type: str, **kwargs):
        context = kwargs.pop("context", ErrorContext())
        context.additional_info["resource_type"] = resource_type
        super().__init__(
            message=message,
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.HIGH,
            context=context,
            **kwargs,
        )


class CacheError(ResourceError):
    """Cache-related errors."""

    def __init__(self, message: str, cache_key: Optional[str] = None, **kwargs):
        context = kwargs.pop("context", ErrorContext())
        context.additional_info["cache_key"] = cache_key
        recovery_suggestions = [
            "Clear the cache and retry",
            "Check available disk space",
            "Verify cache directory permissions",
            "Reduce cache size limits",
        ]
        super().__init__(
            message=f"Cache error: {message}",
            resource_type="cache",
            context=context,
            recovery_suggestions=recovery_suggestions,
            **kwargs,
        )


# ---------------- Error Handler ----------------
class ErrorHandler:
    """Centralized error handling and logging."""

    def __init__(self, logger=None):
        self.logger = logger
        self.error_counts: Dict[str, int] = {}

    def handle_error(self, error: Union[DataDiscoveryException, Exception], context: Optional[ErrorContext] = None) -> DataDiscoveryException:
        """Handle an error with proper logging and context."""
        if not isinstance(error, DataDiscoveryException):
            error = DataDiscoveryException(
                message=str(error),
                category=ErrorCategory.UNKNOWN,
                context=context,
                original_exception=error,
            )
        error_key = f"{error.category.value}_{error.severity.value}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        if self.logger:
            self._log_error(error)
        return error

    def _log_error(self, error: DataDiscoveryException):
        """Log the error with appropriate level."""
        error_dict = error.to_dict()
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"CRITICAL ERROR [{error.error_id}]: {error.message}", extra=error_dict)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(f"ERROR [{error.error_id}]: {error.message}", extra=error_dict)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"WARNING [{error.error_id}]: {error.message}", extra=error_dict)
        else:
            self.logger.info(f"INFO [{error.error_id}]: {error.message}", extra=error_dict)

    def get_error_summary(self) -> Dict[str, int]:
        """Get a summary of error counts by category and severity."""
        return self.error_counts.copy()

    def reset_error_counts(self):
        """Reset error counters."""
        self.error_counts.clear()



class QueryGenerationError(ToolError):
    """Errors during automated SQL or query generation."""

    def __init__(
        self,
        message: str,
        input_prompt: Optional[str] = None,
        tool_name: str = "QueryGenerator",
        **kwargs
    ):
        context = kwargs.pop("context", ErrorContext())
        context.additional_info["input_prompt"] = input_prompt
        recovery_suggestions = [
            "Verify the input prompt or template is well-formed",
            "Check if the database schema matches the expected structure",
            "Ensure reserved keywords are not used incorrectly",
            "Try simplifying the natural language request",
            "Validate query parts before full generation",
        ]
        super().__init__(
            message=f"Query generation failed: {message}",
            tool_name=tool_name,
            context=context,
            recovery_suggestions=recovery_suggestions,
            severity=ErrorSeverity.HIGH,  # 🔥 Explicit severity level
            **kwargs,
        )


# ---------------- Test & Demonstration ----------------
if __name__ == "__main__":
    print("Testing Data Discovery Exception System")
    print("=" * 50)

    try:
        raise DataDiscoveryException(
            message="Test exception for demonstration",
            category=ErrorCategory.AGENT_FAILURE,
            severity=ErrorSeverity.MEDIUM,
            recovery_suggestions=["This is a test", "Try again later"],
        )
    except DataDiscoveryException as e:
        print("✅ Basic exception created:")
        print(f"   Error ID: {e.error_id}")
        print(f"   Category: {e.category.value}")
        print(f"   Severity: {e.severity.value}")
        print(f"   Message: {e.message}")
        print(f"   Suggestions: {len(e.recovery_suggestions)} items")

    exceptions_to_test = [
        InvalidConfigurationError("SNOWFLAKE_ACCOUNT", "valid account name"),
        SnowflakeConnectionError("Invalid credentials", account="demo-account"),
        AgentTimeoutError("technical_analyst", "schema_discovery", 30),
        SQLValidationError("Syntax error near 'SELCT'", sql_query="SELCT * FROM users"),
        CacheError("Disk full", cache_key="schema_metadata"),
        QueryGenerationError(
            "Could not translate natural language into SQL",
            input_prompt="Show me all customers in 2024"
        ),
    ]

    error_handler = ErrorHandler()
    for i, exc in enumerate(exceptions_to_test, 1):
        handled_error = error_handler.handle_error(exc)
        print(f"✅ Exception {i}: {exc.__class__.__name__}")
        print(f"   ID: {handled_error.error_id}")
        print(f"   Category: {handled_error.category.value}")
        print(f"   Severity: {handled_error.severity.value}")
        print(f"   Recovery suggestions: {len(handled_error.recovery_suggestions)}")
        if "input_prompt" in handled_error.context.additional_info:
            print(f"   Input Prompt: {handled_error.context.additional_info['input_prompt']}")
        print()

    print("📊 Error Summary:")
    summary = error_handler.get_error_summary()
    for error_type, count in summary.items():
        print(f"   {error_type}: {count}")

    print("\n✅ Exception system tested successfully!")
    print(f"   Total exception types: {len(exceptions_to_test)}")
    print(f"   Error categories: {len(ErrorCategory)}")
    print(f"   Severity levels: {len(ErrorSeverity)}")
    print("\n🚀 Exception system is ready for use!")
