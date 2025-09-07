# src/data_discovery/core/config.py - Enhanced Version

import os
import sys
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import logging


# Enhanced environment loading with better error handling
def _load_environment_variables() -> bool:
    """Load environment variables with robust path detection."""
    logger = logging.getLogger(__name__)

    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.warning("python-dotenv not installed. Using system environment only.")
        return False

    # Multiple .env file locations to check
    possible_env_paths = [
        Path.cwd() / '.env',
        Path(__file__).parent.parent.parent.parent / '.env',
        Path.home() / '.env.data_discovery',  # User-specific config
        Path('/etc/data_discovery/.env'),  # System-wide config
    ]

    for env_path in possible_env_paths:
        if env_path.exists() and env_path.is_file():
            try:
                success = load_dotenv(env_path, override=True)
                if success:
                    logger.info(f"Loaded environment from: {env_path}")
                    return True
            except Exception as e:
                logger.warning(f"Failed to load {env_path}: {e}")
                continue

    logger.info("No .env file found, using system environment variables")
    return False


# Load environment on module import
_ENV_LOADED = _load_environment_variables()


class Environment(Enum):
    """Environment types for configuration management."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

    @classmethod
    def from_string(cls, env_str: str) -> 'Environment':
        """Create Environment from string with fallback."""
        try:
            return cls(env_str.lower())
        except ValueError:
            logging.warning(f"Unknown environment '{env_str}', defaulting to development")
            return cls.DEVELOPMENT


@dataclass
class SnowflakeConfig:
    """Snowflake database configuration with enhanced validation."""
    account: str = ""
    user: str = ""
    password: Optional[str] = None
    private_key_path: Optional[str] = None
    private_key_passphrase: Optional[str] = None
    warehouse: str = "COMPUTE_WH"
    database: str = "ANALYTICS_DB"
    schema: str = "PUBLIC"
    role: str = "DATA_DISCOVERY_ROLE"

    # Connection pool settings
    max_tables_per_batch: int = 50
    query_timeout_seconds: int = 30
    max_sample_rows: int = 1000
    connection_timeout_seconds: int = 60
    max_retry_attempts: int = 3

    # Performance tuning
    client_session_keep_alive: bool = True
    autocommit: bool = True

    def __post_init__(self):
        """Validate Snowflake configuration."""
        if self.account and not self.account.endswith('.snowflakecomputing.com'):
            if '.' not in self.account:
                self.account = f"{self.account}.snowflakecomputing.com"

    def has_credentials(self) -> bool:
        """Check if we have sufficient credentials to connect."""
        return bool(
            self.account and
            self.user and
            (self.password or self._has_valid_private_key())
        )

    def _has_valid_private_key(self) -> bool:
        """Check if private key configuration is valid."""
        if not self.private_key_path:
            return False

        key_path = Path(self.private_key_path)
        return key_path.exists() and key_path.is_file()

    def get_connection_params(self) -> Dict[str, Any]:
        """Get connection parameters for Snowflake connector."""
        params = {
            'account': self.account,
            'user': self.user,
            'warehouse': self.warehouse,
            'database': self.database,
            'schema': self.schema,
            'role': self.role,
            'client_session_keep_alive': self.client_session_keep_alive,
            'autocommit': self.autocommit,
            'login_timeout': self.connection_timeout_seconds,
            'network_timeout': self.query_timeout_seconds
        }

        if self.password:
            params['password'] = self.password
        elif self._has_valid_private_key():
            params['private_key_path'] = self.private_key_path
            if self.private_key_passphrase:
                params['private_key_passphrase'] = self.private_key_passphrase

        return params


@dataclass
class AnalysisConfig:
    """Analysis and query generation configuration with validation."""
    min_confidence_threshold: float = 0.7
    max_questions_per_category: int = 25
    enable_cross_table_joins: bool = True
    max_join_tables: int = 3
    sample_data_rows: int = 100

    # Advanced analysis settings
    enable_statistical_profiling: bool = True
    enable_pattern_detection: bool = True
    max_column_cardinality: int = 1000
    null_threshold_percent: float = 95.0

    def __post_init__(self):
        """Validate analysis configuration."""
        if not 0.0 <= self.min_confidence_threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")

        if self.max_questions_per_category < 1:
            raise ValueError("Max questions per category must be positive")

        if not 0.0 <= self.null_threshold_percent <= 100.0:
            raise ValueError("Null threshold must be between 0.0 and 100.0")


@dataclass
class SecurityConfig:
    """Security and governance configuration."""
    pii_detection_enabled: bool = True
    mask_sensitive_preview: bool = True
    governance_strict_mode: bool = False
    max_pii_scan_rows: int = 10000
    sensitive_patterns_file: Optional[str] = None

    # Additional security settings
    enable_query_sanitization: bool = True
    max_query_complexity: int = 10  # Limit complex queries
    allowed_functions: Optional[list] = None  # Whitelist SQL functions
    forbidden_keywords: list = field(default_factory=lambda: ['DELETE', 'DROP', 'TRUNCATE', 'ALTER'])

    def get_pii_patterns_path(self) -> Path:
        """Get path to PII patterns file with fallback."""
        if self.sensitive_patterns_file:
            custom_path = Path(self.sensitive_patterns_file)
            if custom_path.exists():
                return custom_path

        # Default path
        default_path = Path(__file__).parent.parent.parent.parent / "data" / "reference" / "pii_patterns.json"
        return default_path

    def is_query_allowed(self, query: str) -> tuple[bool, str]:
        """Check if a query is allowed based on security rules."""
        query_upper = query.upper()

        for keyword in self.forbidden_keywords:
            if keyword in query_upper:
                return False, f"Forbidden keyword detected: {keyword}"

        return True, "Query allowed"


@dataclass
class UIConfig:
    """User interface configuration with validation."""
    enable_sql_execution: bool = False  # Security: preview only by default
    max_results_display: int = 100
    cache_results_minutes: int = 15
    gradio_port: int = 7860
    gradio_host: str = "0.0.0.0"
    share_publicly: bool = False

    # UI enhancement settings
    enable_dark_mode: bool = True
    max_export_rows: int = 10000
    enable_real_time_updates: bool = False
    session_timeout_minutes: int = 480  # 8 hours

    def __post_init__(self):
        """Validate UI configuration."""
        if not 1024 <= self.gradio_port <= 65535:
            raise ValueError("Gradio port must be between 1024 and 65535")

        if self.max_results_display < 1:
            raise ValueError("Max results display must be positive")


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    enable_metrics: bool = True
    metrics_port: int = 8080
    log_level: str = "INFO"
    log_file_path: Optional[str] = None
    enable_performance_tracking: bool = True
    health_check_interval_seconds: int = 30

    # Enhanced monitoring
    enable_structured_logging: bool = True
    log_rotation_size_mb: int = 100
    max_log_files: int = 5
    enable_error_alerting: bool = False

    def __post_init__(self):
        """Validate monitoring configuration."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.log_level.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")

    def get_log_file_path(self) -> Optional[Path]:
        """Get resolved log file path."""
        if self.log_file_path:
            return Path(self.log_file_path)
        return None

    def setup_logging(self) -> None:
        """Configure Python logging based on settings."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        if self.enable_structured_logging:
            log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'

        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format=log_format,
            handlers=[
                logging.StreamHandler(sys.stdout),
                *([logging.FileHandler(self.get_log_file_path())] if self.log_file_path else [])
            ]
        )


@dataclass
class CacheConfig:
    """Caching configuration with enhanced options."""
    enable_caching: bool = True
    cache_directory: str = "cache"
    schema_cache_ttl_hours: int = 24
    query_cache_ttl_hours: int = 6
    profile_cache_ttl_hours: int = 12
    max_cache_size_mb: int = 1024

    # Cache cleanup settings
    enable_auto_cleanup: bool = True
    cleanup_interval_hours: int = 6
    cache_compression: bool = True

    def get_cache_directory(self) -> Path:
        """Get resolved cache directory path."""
        cache_path = Path(self.cache_directory).resolve()
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path

    def get_cache_subdirectory(self, subdir: str) -> Path:
        """Get a specific cache subdirectory."""
        sub_path = self.get_cache_directory() / subdir
        sub_path.mkdir(parents=True, exist_ok=True)
        return sub_path


@dataclass
class DataDiscoveryConfig:
    """Main configuration class for the Data Discovery system."""
    environment: Environment
    snowflake: SnowflakeConfig
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    @classmethod
    def from_env(cls, environment: Union[Environment, str, None] = None) -> "DataDiscoveryConfig":
        """Create configuration from environment variables."""
        if environment is None:
            env_name = os.getenv("ENVIRONMENT", "development")
            environment = Environment.from_string(env_name)
        elif isinstance(environment, str):
            environment = Environment.from_string(environment)

        # Helper function for environment variable parsing
        def get_bool(key: str, default: bool = False) -> bool:
            return os.getenv(key, str(default)).lower() in ('true', '1', 'yes', 'on')

        def get_int(key: str, default: int) -> int:
            try:
                return int(os.getenv(key, str(default)))
            except ValueError:
                return default

        def get_float(key: str, default: float) -> float:
            try:
                return float(os.getenv(key, str(default)))
            except ValueError:
                return default

        # Create configurations
        snowflake_config = SnowflakeConfig(
            account=os.getenv("SNOWFLAKE_ACCOUNT", ""),
            user=os.getenv("SNOWFLAKE_USER", ""),
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            private_key_path=os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH"),
            private_key_passphrase=os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE"),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
            database=os.getenv("SNOWFLAKE_DATABASE", "ANALYTICS_DB"),
            schema=os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC"),
            role=os.getenv("SNOWFLAKE_ROLE", "DATA_DISCOVERY_ROLE"),
            max_tables_per_batch=get_int("SNOWFLAKE_MAX_TABLES_BATCH", 50),
            query_timeout_seconds=get_int("SNOWFLAKE_QUERY_TIMEOUT", 30),
            max_sample_rows=get_int("SNOWFLAKE_MAX_SAMPLE_ROWS", 1000),
            connection_timeout_seconds=get_int("SNOWFLAKE_CONNECTION_TIMEOUT", 60),
            client_session_keep_alive=get_bool("SNOWFLAKE_KEEP_ALIVE", True)
        )

        analysis_config = AnalysisConfig(
            min_confidence_threshold=get_float("ANALYSIS_MIN_CONFIDENCE", 0.7),
            max_questions_per_category=get_int("ANALYSIS_MAX_QUESTIONS", 25),
            enable_cross_table_joins=get_bool("ANALYSIS_ENABLE_JOINS", True),
            max_join_tables=get_int("ANALYSIS_MAX_JOIN_TABLES", 3),
            sample_data_rows=get_int("ANALYSIS_SAMPLE_ROWS", 100)
        )

        security_config = SecurityConfig(
            pii_detection_enabled=get_bool("SECURITY_PII_DETECTION", True),
            mask_sensitive_preview=get_bool("SECURITY_MASK_PREVIEW", True),
            governance_strict_mode=get_bool("SECURITY_STRICT_MODE", False),
            max_pii_scan_rows=get_int("SECURITY_PII_SCAN_ROWS", 10000),
            sensitive_patterns_file=os.getenv("SECURITY_PII_PATTERNS_FILE")
        )

        ui_config = UIConfig(
            enable_sql_execution=get_bool("UI_ENABLE_SQL_EXEC", False),
            max_results_display=get_int("UI_MAX_RESULTS", 100),
            cache_results_minutes=get_int("UI_CACHE_MINUTES", 15),
            gradio_port=get_int("GRADIO_PORT", 7860),
            gradio_host=os.getenv("GRADIO_HOST", "0.0.0.0"),
            share_publicly=get_bool("GRADIO_SHARE", False)
        )

        monitoring_config = MonitoringConfig(
            enable_metrics=get_bool("MONITORING_ENABLE", True),
            metrics_port=get_int("MONITORING_PORT", 8080),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file_path=os.getenv("LOG_FILE_PATH"),
            enable_performance_tracking=get_bool("MONITORING_PERFORMANCE", True),
            health_check_interval_seconds=get_int("MONITORING_HEALTH_INTERVAL", 30)
        )

        cache_config = CacheConfig(
            enable_caching=get_bool("CACHE_ENABLE", True),
            cache_directory=os.getenv("CACHE_DIRECTORY", "cache"),
            schema_cache_ttl_hours=get_int("CACHE_SCHEMA_TTL_HOURS", 24),
            query_cache_ttl_hours=get_int("CACHE_QUERY_TTL_HOURS", 6),
            max_cache_size_mb=get_int("CACHE_MAX_SIZE_MB", 1024)
        )

        config = cls(
            environment=environment,
            snowflake=snowflake_config,
            analysis=analysis_config,
            security=security_config,
            ui=ui_config,
            monitoring=monitoring_config,
            cache=cache_config
        )

        # Setup logging immediately
        config.monitoring.setup_logging()

        return config

    def validate(self) -> tuple[bool, list[str]]:
        """Validate the entire configuration and return issues."""
        issues = []

        # Environment-specific validation
        if self.environment == Environment.PRODUCTION:
            if not self.snowflake.has_credentials():
                issues.append("Snowflake credentials are required for production")

            if self.ui.enable_sql_execution:
                issues.append("SQL execution should be disabled in production")

            if self.ui.share_publicly:
                issues.append("Public sharing should be disabled in production")

        # Cache directory validation
        try:
            self.cache.get_cache_directory()
        except Exception as e:
            issues.append(f"Cache directory validation failed: {e}")

        # PII patterns validation
        if self.security.pii_detection_enabled:
            pii_path = self.security.get_pii_patterns_path()
            if not pii_path.exists():
                issues.append(f"PII patterns file not found: {pii_path}")

        # Port conflicts
        if self.ui.gradio_port == self.monitoring.metrics_port:
            issues.append("Gradio port conflicts with metrics port")

        return len(issues) == 0, issues

    def can_connect_to_snowflake(self) -> bool:
        """Check if we have enough information to connect to Snowflake."""
        return self.snowflake.has_credentials()

    def summary(self) -> str:
        """Get a human-readable configuration summary."""
        lines = [
            f"Data Discovery Configuration Summary",
            f"{'=' * 40}",
            f"Environment: {self.environment.value}",
            f"Snowflake Account: {self.snowflake.account or 'NOT SET'}",
            f"Snowflake Ready: {'✓' if self.can_connect_to_snowflake() else '✗'}",
            f"Cache Directory: {self.cache.get_cache_directory()}",
            f"UI Port: {self.ui.gradio_port}",
            f"Log Level: {self.monitoring.log_level}",
            f"PII Detection: {'✓' if self.security.pii_detection_enabled else '✗'}",
            f"SQL Execution: {'✓' if self.ui.enable_sql_execution else '✗'}",
        ]
        return '\n'.join(lines)


# Enhanced global configuration management
class ConfigManager:
    """Thread-safe configuration manager."""

    _instance: Optional[DataDiscoveryConfig] = None
    _lock = None

    @classmethod
    def get_config(cls, reload: bool = False) -> DataDiscoveryConfig:
        """Get configuration instance with thread safety."""
        if cls._lock is None:
            import threading
            cls._lock = threading.Lock()

        with cls._lock:
            if cls._instance is None or reload:
                cls._instance = DataDiscoveryConfig.from_env()

                # Validate configuration
                is_valid, issues = cls._instance.validate()
                if not is_valid:
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Configuration validation issues: {issues}")

        return cls._instance

    @classmethod
    def set_config(cls, config: DataDiscoveryConfig) -> None:
        """Set configuration instance."""
        if cls._lock is None:
            import threading
            cls._lock = threading.Lock()

        with cls._lock:
            cls._instance = config


# Public API
def get_config(reload: bool = False) -> DataDiscoveryConfig:
    """Get the global configuration instance."""
    return ConfigManager.get_config(reload=reload)


def set_config(config: DataDiscoveryConfig) -> None:
    """Set the global configuration instance."""
    ConfigManager.set_config(config)


if __name__ == "__main__":
    # Enhanced testing and validation
    print("Data Discovery Configuration System Test")
    print("=" * 50)

    try:
        config = get_config()
        print(config.summary())

        is_valid, issues = config.validate()
        print(f"\nValidation: {'PASSED' if is_valid else 'FAILED'}")

        if issues:
            print("Issues found:")
            for issue in issues:
                print(f"  - {issue}")

        # Test Snowflake connection readiness
        if config.can_connect_to_snowflake():
            print("\n✓ Ready for Snowflake connection!")
        else:
            print("\n⚠ Snowflake credentials needed for connection")

    except Exception as e:
        logging.error(f"Configuration test failed: {e}")
        sys.exit(1)