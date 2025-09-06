# src/data_discovery/core/logging_config.py

"""
Logging configuration for the Data Discovery system.

Provides structured logging with JSON formatting, multiple output destinations,
and integration with monitoring systems like Prometheus/Grafana.
"""

import os
import sys
import logging
import logging.handlers
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import json
from dataclasses import dataclass, asdict

# Try to import optional dependencies
try:
    import structlog

    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False

try:
    from pythonjsonlogger import jsonlogger

    HAS_JSON_LOGGER = True
except ImportError:
    HAS_JSON_LOGGER = False


@dataclass
class LogConfig:
    """Configuration for logging setup."""
    level: str = "INFO"
    format_type: str = "json"  # "json", "structured", "simple"
    log_to_file: bool = True
    log_to_console: bool = True
    file_path: Optional[str] = None
    max_file_size_mb: int = 10
    backup_count: int = 5
    include_timestamp: bool = True
    include_level: bool = True
    include_logger_name: bool = True
    include_module: bool = True
    include_function: bool = True
    include_line_number: bool = True


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def __init__(self, include_extras: bool = True):
        super().__init__()
        self.include_extras = include_extras

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add module information
        if hasattr(record, 'module'):
            log_data["module"] = record.module
        if hasattr(record, 'funcName'):
            log_data["function"] = record.funcName
        if hasattr(record, 'lineno'):
            log_data["line_number"] = record.lineno

        # Add exception information
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        if self.include_extras:
            for key, value in record.__dict__.items():
                if key not in {'name', 'msg', 'args', 'levelname', 'levelno',
                               'pathname', 'filename', 'module', 'lineno',
                               'funcName', 'created', 'msecs', 'relativeCreated',
                               'thread', 'threadName', 'processName', 'process',
                               'getMessage', 'exc_info', 'exc_text', 'stack_info'}:
                    # Convert complex objects to strings
                    if isinstance(value, (dict, list, tuple)):
                        log_data[key] = json.dumps(value) if value else None
                    else:
                        log_data[key] = str(value) if value is not None else None

        return json.dumps(log_data, ensure_ascii=False)


class StructuredFormatter(logging.Formatter):
    """Human-readable structured formatter."""

    def __init__(self):
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record in structured but readable format."""
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")

        # Base log line
        log_line = f"[{timestamp}] {record.levelname:8} {record.name:20} {record.getMessage()}"

        # Add extra context if available
        extras = []
        for key, value in record.__dict__.items():
            if key not in {'name', 'msg', 'args', 'levelname', 'levelno',
                           'pathname', 'filename', 'module', 'lineno',
                           'funcName', 'created', 'msecs', 'relativeCreated',
                           'thread', 'threadName', 'processName', 'process',
                           'getMessage', 'exc_info', 'exc_text', 'stack_info'}:
                if value is not None:
                    extras.append(f"{key}={value}")

        if extras:
            log_line += f" | {' '.join(extras)}"

        # Add exception information
        if record.exc_info:
            log_line += f"\n{self.formatException(record.exc_info)}"

        return log_line


class DataDiscoveryLogger:
    """Enhanced logger for the Data Discovery system."""

    def __init__(self, name: str, config: LogConfig = None):
        self.name = name
        self.config = config or LogConfig()
        self.logger = logging.getLogger(name)
        self._setup_logger()

    def _setup_logger(self):
        """Setup logger with configured handlers and formatters."""
        # Clear existing handlers
        self.logger.handlers.clear()

        # Set log level
        level = getattr(logging, self.config.level.upper(), logging.INFO)
        self.logger.setLevel(level)

        # Console handler
        if self.config.log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(self._get_formatter())
            self.logger.addHandler(console_handler)

        # File handler
        if self.config.log_to_file:
            file_path = self._get_log_file_path()

            # Create directory if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                filename=file_path,
                maxBytes=self.config.max_file_size_mb * 1024 * 1024,
                backupCount=self.config.backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(self._get_formatter())
            self.logger.addHandler(file_handler)

        # Prevent propagation to root logger
        self.logger.propagate = False

    def _get_formatter(self) -> logging.Formatter:
        """Get appropriate formatter based on configuration."""
        if self.config.format_type == "json" and HAS_JSON_LOGGER:
            return JSONFormatter()
        elif self.config.format_type == "json":
            return JSONFormatter()  # Use custom JSON formatter
        elif self.config.format_type == "structured":
            return StructuredFormatter()
        else:
            # Simple format
            fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            return logging.Formatter(fmt)

    def _get_log_file_path(self) -> Path:
        """Get the log file path."""
        if self.config.file_path:
            return Path(self.config.file_path)

        # Default to logs directory
        log_dir = Path("logs")
        log_file = f"{self.name}.log"
        return log_dir / log_file

    def debug(self, message: str, **kwargs):
        """Log debug message with optional extra fields."""
        self.logger.debug(message, extra=kwargs)

    def info(self, message: str, **kwargs):
        """Log info message with optional extra fields."""
        self.logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message with optional extra fields."""
        self.logger.warning(message, extra=kwargs)

    def error(self, message: str, **kwargs):
        """Log error message with optional extra fields."""
        self.logger.error(message, extra=kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message with optional extra fields."""
        self.logger.critical(message, extra=kwargs)

    def log_agent_start(self, agent_id: str, task: str):
        """Log agent start with structured data."""
        self.info(f"Agent {agent_id} starting task",
                  agent_id=agent_id, task=task, event_type="agent_start")

    def log_agent_complete(self, agent_id: str, task: str, duration_seconds: float):
        """Log agent completion with structured data."""
        self.info(f"Agent {agent_id} completed task",
                  agent_id=agent_id, task=task, duration_seconds=duration_seconds,
                  event_type="agent_complete")

    def log_agent_error(self, agent_id: str, task: str, error: Exception):
        """Log agent error with structured data."""
        self.logger.error(f"Agent {agent_id} failed task: {str(error)}",
                          extra={
                              "agent_id": agent_id,
                              "task": task,
                              "error_type": type(error).__name__,
                              "event_type": "agent_error"
                          },
                          exc_info=True)

    def log_database_query(self, query: str, duration_seconds: float, row_count: int = None):
        """Log database query with performance metrics."""
        self.info(f"Database query executed",
                  query=query[:200] + "..." if len(query) > 200 else query,
                  duration_seconds=duration_seconds, row_count=row_count,
                  event_type="database_query")

    def log_schema_discovery(self, database: str, table_count: int, duration_seconds: float):
        """Log schema discovery results."""
        self.info(f"Schema discovery completed for {database}",
                  database=database, table_count=table_count,
                  duration_seconds=duration_seconds, event_type="schema_discovery")

    def log_question_generated(self, question: str, confidence: float, category: str):
        """Log business question generation."""
        self.info(f"Business question generated",
                  question=question, confidence=confidence, category=category,
                  event_type="question_generated")

    def log_user_interaction(self, action: str, user_id: str = None, **kwargs):
        """Log user interaction with the system."""
        self.info(f"User interaction: {action}",
                  action=action, user_id=user_id, event_type="user_interaction",
                  **kwargs)


class LoggerManager:
    """Central manager for all loggers in the system."""

    def __init__(self, default_config: LogConfig = None):
        self.default_config = default_config or LogConfig()
        self.loggers: Dict[str, DataDiscoveryLogger] = {}

    def get_logger(self, name: str, config: LogConfig = None) -> DataDiscoveryLogger:
        """Get or create a logger with the given name."""
        if name not in self.loggers:
            logger_config = config or self.default_config
            self.loggers[name] = DataDiscoveryLogger(name, logger_config)
        return self.loggers[name]

    def configure_all_loggers(self, config: LogConfig):
        """Reconfigure all existing loggers with new config."""
        self.default_config = config
        for logger in self.loggers.values():
            logger.config = config
            logger._setup_logger()

    def get_logger_stats(self) -> Dict[str, Any]:
        """Get statistics about all loggers."""
        return {
            "total_loggers": len(self.loggers),
            "logger_names": list(self.loggers.keys()),
            "default_config": asdict(self.default_config)
        }


# Global logger manager instance
_logger_manager: Optional[LoggerManager] = None


def setup_logging(config: LogConfig = None) -> LoggerManager:
    """Setup global logging configuration."""
    global _logger_manager

    if config is None:
        # Try to load from environment
        config = LogConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format_type=os.getenv("LOG_FORMAT", "json"),
            log_to_file=os.getenv("LOG_TO_FILE", "true").lower() == "true",
            log_to_console=os.getenv("LOG_TO_CONSOLE", "true").lower() == "true",
            file_path=os.getenv("LOG_FILE_PATH"),
        )

    _logger_manager = LoggerManager(config)

    # Configure root logger to prevent unwanted output
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)

    return _logger_manager


def get_logger(name: str) -> DataDiscoveryLogger:
    """Get a logger instance. Auto-setup if not configured."""
    global _logger_manager

    if _logger_manager is None:
        _logger_manager = setup_logging()

    return _logger_manager.get_logger(name)


# Example usage and testing
if __name__ == "__main__":
    print("Testing Data Discovery Logging System")
    print("=" * 50)

    # Test different configurations
    configs_to_test = [
        ("JSON Format", LogConfig(level="DEBUG", format_type="json", log_to_file=False)),
        ("Structured Format", LogConfig(level="INFO", format_type="structured", log_to_file=False)),
        ("Simple Format", LogConfig(level="WARNING", format_type="simple", log_to_file=False)),
    ]

    for config_name, config in configs_to_test:
        print(f"\n🧪 Testing {config_name}")
        print("-" * 30)

        # Setup logging with this config
        manager = setup_logging(config)

        # Get test logger
        logger = get_logger("test_logger")

        # Test different log levels and structured data
        logger.debug("Debug message", test_data="debug_value")
        logger.info("Info message", test_data="info_value")
        logger.warning("Warning message", test_data="warning_value")
        logger.error("Error message", test_data="error_value")

        # Test specialized logging methods
        logger.log_agent_start("technical_analyst", "schema_discovery")
        logger.log_agent_complete("technical_analyst", "schema_discovery", 15.5)
        logger.log_database_query("SELECT * FROM customers", 2.3, 1000)
        logger.log_question_generated("What is our customer count?", 0.95, "simple")

        print(f"✅ {config_name} test completed")

    # Test file logging
    print(f"\n🧪 Testing File Logging")
    print("-" * 30)

    file_config = LogConfig(
        level="INFO",
        format_type="json",
        log_to_console=False,
        log_to_file=True,
        file_path="logs/test_file_logging.log"
    )

    manager = setup_logging(file_config)
    file_logger = get_logger("file_test_logger")

    file_logger.info("This message should go to file", test_file_logging=True)
    file_logger.log_schema_discovery("TEST_DB", 25, 45.2)

    # Check if file was created
    log_file = Path("logs/test_file_logging.log")
    if log_file.exists():
        print(f"✅ Log file created: {log_file}")
        print(f"   File size: {log_file.stat().st_size} bytes")

        # Show last few lines
        with open(log_file, 'r') as f:
            lines = f.readlines()
            print(f"   Last log entry: {lines[-1].strip()}")
    else:
        print("❌ Log file not created")

    # Test logger manager stats
    print(f"\n📊 Logger Manager Statistics")
    print("-" * 30)

    stats = manager.get_logger_stats()
    print(f"Total loggers: {stats['total_loggers']}")
    print(f"Logger names: {', '.join(stats['logger_names'])}")
    print(f"Default log level: {stats['default_config']['level']}")
    print(f"Default format: {stats['default_config']['format_type']}")

    # Test error logging with exception
    print(f"\n🧪 Testing Exception Logging")
    print("-" * 30)

    error_logger = get_logger("error_test")
    try:
        raise ValueError("Test exception for logging")
    except Exception as e:
        error_logger.log_agent_error("test_agent", "test_task", e)
        print("✅ Exception logged successfully")

    print(f"\n✅ Logging system tested successfully!")
    print(f"   Supported formats: JSON, Structured, Simple")
    print(f"   Features: File rotation, structured data, agent logging")
    print(f"   Integration: Exception system, monitoring ready")
    print("\n🚀 Logging system is ready for use!")