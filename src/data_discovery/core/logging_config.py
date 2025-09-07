# Enhanced logging features for production deployment
import logging
import uuid
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Set, Dict, Any, Optional
import re


class CorrelationFilter(logging.Filter):
    """Add correlation IDs to log records for request tracing."""

    def __init__(self):
        super().__init__()
        self.local = threading.local()

    def filter(self, record):
        """Add correlation ID to log record."""
        correlation_id = getattr(self.local, 'correlation_id', None)
        if correlation_id:
            record.correlation_id = correlation_id
        else:
            # Generate new correlation ID if none exists
            correlation_id = str(uuid.uuid4())[:8]
            self.local.correlation_id = correlation_id
            record.correlation_id = correlation_id
        return True

    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID for current thread."""
        self.local.correlation_id = correlation_id

    def clear_correlation_id(self):
        """Clear correlation ID for current thread."""
        if hasattr(self.local, 'correlation_id'):
            delattr(self.local, 'correlation_id')


class SensitiveDataFilter(logging.Filter):
    """Filter out sensitive data from log messages."""

    def __init__(self):
        super().__init__()
        # Common patterns for sensitive data
        self.patterns = [
            (re.compile(r'password["\s]*[:=]["\s]*[^"\s,}]+', re.IGNORECASE), 'password=***'),
            (re.compile(r'token["\s]*[:=]["\s]*[^"\s,}]+', re.IGNORECASE), 'token=***'),
            (re.compile(r'key["\s]*[:=]["\s]*[^"\s,}]+', re.IGNORECASE), 'key=***'),
            (re.compile(r'\b\d{4}-\d{4}-\d{4}-\d{4}\b'), '****-****-****-****'),  # Credit cards
            (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), '***-**-****'),  # SSN
        ]

    def filter(self, record):
        """Sanitize sensitive data from log message."""
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            message = record.msg
            for pattern, replacement in self.patterns:
                message = pattern.sub(replacement, message)
            record.msg = message

        # Also sanitize extra fields
        for key, value in record.__dict__.items():
            if isinstance(value, str):
                for pattern, replacement in self.patterns:
                    sanitized = pattern.sub(replacement, value)
                    if sanitized != value:
                        setattr(record, key, sanitized)

        return True


class SamplingFilter(logging.Filter):
    """Sample log messages to reduce volume for noisy operations."""

    def __init__(self, sample_rate: float = 0.1, max_per_minute: int = 60):
        super().__init__()
        self.sample_rate = sample_rate
        self.max_per_minute = max_per_minute
        self.message_counts = {}
        self.last_reset = time.time()

    def filter(self, record):
        """Apply sampling to reduce log volume."""
        current_time = time.time()

        # Reset counts every minute
        if current_time - self.last_reset > 60:
            self.message_counts.clear()
            self.last_reset = current_time

        # Check if this message type should be sampled
        message_key = f"{record.name}:{record.levelno}"
        count = self.message_counts.get(message_key, 0)

        # Always log errors and warnings
        if record.levelno >= logging.WARNING:
            self.message_counts[message_key] = count + 1
            return True

        # Sample other messages
        if count >= self.max_per_minute:
            return False

        if random.random() <= self.sample_rate:
            self.message_counts[message_key] = count + 1
            return True

        return False


class EnhancedDataDiscoveryLogger(DataDiscoveryLogger):
    """Enhanced logger with production features."""

    def __init__(self, name: str, config: LogConfig = None):
        super().__init__(name, config)
        self.correlation_filter = CorrelationFilter()
        self.sensitive_filter = SensitiveDataFilter()
        self.sampling_filter = SamplingFilter() if config and hasattr(config,
                                                                      'enable_sampling') and config.enable_sampling else None
        self._setup_filters()

    def _setup_filters(self):
        """Add production filters to logger."""
        # Add correlation ID filter to all handlers
        for handler in self.logger.handlers:
            handler.addFilter(self.correlation_filter)
            handler.addFilter(self.sensitive_filter)
            if self.sampling_filter:
                handler.addFilter(self.sampling_filter)

    @contextmanager
    def correlation_context(self, correlation_id: str = None):
        """Context manager for correlation ID."""
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())[:8]

        self.correlation_filter.set_correlation_id(correlation_id)
        try:
            yield correlation_id
        finally:
            self.correlation_filter.clear_correlation_id()

    def log_performance_metric(self, metric_name: str, value: float, unit: str = "seconds", **kwargs):
        """Log performance metrics in a structured format."""
        self.info(f"Performance metric: {metric_name}",
                  metric_name=metric_name, metric_value=value, metric_unit=unit,
                  event_type="performance_metric", **kwargs)

    def log_business_event(self, event_name: str, **kwargs):
        """Log business events for analytics."""
        self.info(f"Business event: {event_name}",
                  event_name=event_name, event_type="business_event", **kwargs)

    def log_security_event(self, event_type: str, severity: str = "info", **kwargs):
        """Log security-related events."""
        log_method = getattr(self.logger, severity.lower(), self.logger.info)
        log_method(f"Security event: {event_type}",
                   extra={"event_type": "security_event", "security_event_type": event_type, **kwargs})


class MetricsCollector:
    """Collect metrics from log events for monitoring."""

    def __init__(self):
        self.metrics = {
            "agent_starts": 0,
            "agent_completions": 0,
            "agent_errors": 0,
            "database_queries": 0,
            "questions_generated": 0,
            "user_interactions": 0,
        }
        self.performance_metrics = {}
        self.lock = threading.Lock()

    def handle_log_record(self, record):
        """Process log record and extract metrics."""
        if not hasattr(record, 'event_type'):
            return

        with self.lock:
            event_type = record.event_type

            if event_type == "agent_start":
                self.metrics["agent_starts"] += 1
            elif event_type == "agent_complete":
                self.metrics["agent_completions"] += 1
                if hasattr(record, 'duration_seconds'):
                    agent_id = getattr(record, 'agent_id', 'unknown')
                    self.performance_metrics[f"agent_duration_{agent_id}"] = record.duration_seconds
            elif event_type == "agent_error":
                self.metrics["agent_errors"] += 1
            elif event_type == "database_query":
                self.metrics["database_queries"] += 1
                if hasattr(record, 'duration_seconds'):
                    self.performance_metrics["query_duration"] = record.duration_seconds
            elif event_type == "question_generated":
                self.metrics["questions_generated"] += 1
            elif event_type == "user_interaction":
                self.metrics["user_interactions"] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        with self.lock:
            return {
                "counters": self.metrics.copy(),
                "performance": self.performance_metrics.copy(),
                "timestamp": datetime.now().isoformat()
            }

    def reset_metrics(self):
        """Reset all metrics."""
        with self.lock:
            for key in self.metrics:
                self.metrics[key] = 0
            self.performance_metrics.clear()


class MetricsHandler(logging.Handler):
    """Custom handler to extract metrics from log records."""

    def __init__(self, metrics_collector: MetricsCollector):
        super().__init__()
        self.metrics_collector = metrics_collector

    def emit(self, record):
        """Extract metrics from log record."""
        try:
            self.metrics_collector.handle_log_record(record)
        except Exception:
            # Don't let metrics collection interfere with logging
            pass


# Enhanced configuration with production features
@dataclass
class EnhancedLogConfig(LogConfig):
    """Enhanced logging configuration for production."""
    enable_correlation_ids: bool = True
    enable_sensitive_data_filtering: bool = True
    enable_sampling: bool = False
    sample_rate: float = 0.1
    enable_metrics_collection: bool = True
    external_log_endpoint: Optional[str] = None  # For log shipping
    log_retention_days: int = 30


class ProductionLoggerManager(LoggerManager):
    """Production-ready logger manager with enhanced features."""

    def __init__(self, default_config: EnhancedLogConfig = None):
        self.default_config = default_config or EnhancedLogConfig()
        self.loggers: Dict[str, EnhancedDataDiscoveryLogger] = {}
        self.metrics_collector = MetricsCollector() if self.default_config.enable_metrics_collection else None

    def get_logger(self, name: str, config: EnhancedLogConfig = None) -> EnhancedDataDiscoveryLogger:
        """Get or create an enhanced logger."""
        if name not in self.loggers:
            logger_config = config or self.default_config
            logger = EnhancedDataDiscoveryLogger(name, logger_config)

            # Add metrics collection if enabled
            if self.metrics_collector:
                metrics_handler = MetricsHandler(self.metrics_collector)
                logger.logger.addHandler(metrics_handler)

            self.loggers[name] = logger

        return self.loggers[name]

    def get_metrics(self) -> Dict[str, Any]:
        """Get system-wide metrics."""
        if self.metrics_collector:
            return self.metrics_collector.get_metrics()
        return {}

    def reset_metrics(self):
        """Reset all metrics."""
        if self.metrics_collector:
            self.metrics_collector.reset_metrics()


# Usage example for production logging
def example_production_usage():
    """Example of using enhanced logging in production."""

    # Setup production logging
    config = EnhancedLogConfig(
        level="INFO",
        format_type="json",
        enable_correlation_ids=True,
        enable_sensitive_data_filtering=True,
        enable_metrics_collection=True
    )

    manager = ProductionLoggerManager(config)
    logger = manager.get_logger("production_example")

    # Use correlation context for request tracing
    with logger.correlation_context() as correlation_id:
        logger.info("Starting data discovery pipeline", pipeline_id="12345")

        # Log with sensitive data (will be filtered)
        logger.info("Database connection",
                    username="admin",
                    password="secret123",  # Will be sanitized
                    host="db.example.com")

        # Log performance metrics
        logger.log_performance_metric("schema_discovery_time", 45.2, "seconds",
                                      table_count=150, database="analytics")

        # Log business events
        logger.log_business_event("catalog_generated",
                                  question_count=89,
                                  high_confidence_count=67)

        # Log security event
        logger.log_security_event("unauthorized_access_attempt",
                                  severity="warning",
                                  user_ip="192.168.1.100",
                                  attempted_resource="/admin")

    # Get metrics
    metrics = manager.get_metrics()
    print(f"Collected metrics: {metrics}")


if __name__ == "__main__":
    # Test enhanced logging features
    example_production_usage()