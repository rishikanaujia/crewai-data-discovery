# src/data_discovery/core/config.py

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

# Load environment variables FIRST - before any other code
print("🔧 Loading environment variables...")
try:
    from dotenv import load_dotenv

    # Find and load .env file
    env_paths = [
        Path.cwd() / '.env',  # Current directory
        Path(__file__).parent.parent.parent.parent / '.env'  # Project root
    ]

    env_loaded = False
    for env_path in env_paths:
        if env_path.exists():
            print(f"🔍 Found .env at: {env_path}")
            result = load_dotenv(env_path, override=True)
            print(f"🔍 Load result: {result}")
            if result:
                env_loaded = True
                print(f"✅ Successfully loaded .env from: {env_path}")
                # Test immediate loading
                test_account = os.getenv("SNOWFLAKE_ACCOUNT")
                print(f"✅ Test variable loaded: {test_account or 'NOT FOUND'}")
                break

    if not env_loaded:
        print("⚠️  No .env file found or loaded successfully")

except ImportError:
    print("⚠️  python-dotenv not found. Install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠️  Error loading .env file: {e}")


class Environment(Enum):
    """Environment types for configuration management."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


@dataclass
class SnowflakeConfig:
    """Snowflake database configuration."""
    account: str = ""
    user: str = ""
    password: Optional[str] = None
    private_key_path: Optional[str] = None
    warehouse: str = "COMPUTE_WH"
    database: str = "ANALYTICS_DB"
    schema: str = "PUBLIC"
    role: str = "DATA_DISCOVERY_ROLE"
    max_tables_per_batch: int = 50
    query_timeout_seconds: int = 30
    max_sample_rows: int = 1000

    def has_credentials(self) -> bool:
        """Check if we have credentials to connect."""
        return bool(self.account and self.user and (self.password or self.private_key_path))


@dataclass
class AnalysisConfig:
    """Analysis and query generation configuration."""
    min_confidence_threshold: float = 0.7
    max_questions_per_category: int = 25
    enable_cross_table_joins: bool = True
    max_join_tables: int = 3
    sample_data_rows: int = 100

    def __post_init__(self):
        """Validate analysis configuration."""
        if not 0.0 <= self.min_confidence_threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")


@dataclass
class SecurityConfig:
    """Security and governance configuration."""
    pii_detection_enabled: bool = True
    mask_sensitive_preview: bool = True
    governance_strict_mode: bool = False
    max_pii_scan_rows: int = 10000
    sensitive_patterns_file: Optional[str] = None

    def get_pii_patterns_path(self) -> Path:
        """Get path to PII patterns file."""
        if self.sensitive_patterns_file:
            return Path(self.sensitive_patterns_file)
        return Path(__file__).parent.parent.parent.parent / "data" / "reference" / "pii_patterns.json"


@dataclass
class UIConfig:
    """User interface configuration."""
    enable_sql_execution: bool = False  # Security: preview only
    max_results_display: int = 100
    cache_results_minutes: int = 15
    gradio_port: int = 7860
    gradio_host: str = "0.0.0.0"
    share_publicly: bool = False

    def __post_init__(self):
        """Validate UI configuration."""
        if self.gradio_port < 1024 or self.gradio_port > 65535:
            raise ValueError("Gradio port must be between 1024 and 65535")


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    enable_metrics: bool = True
    metrics_port: int = 8080
    log_level: str = "INFO"
    log_file_path: Optional[str] = None
    enable_performance_tracking: bool = True
    health_check_interval_seconds: int = 30

    def get_log_file_path(self) -> Optional[Path]:
        """Get resolved log file path."""
        if self.log_file_path:
            return Path(self.log_file_path)
        return None


@dataclass
class CacheConfig:
    """Caching configuration."""
    enable_caching: bool = True
    cache_directory: str = "cache"
    schema_cache_ttl_hours: int = 24
    query_cache_ttl_hours: int = 6
    max_cache_size_mb: int = 1024

    def get_cache_directory(self) -> Path:
        """Get resolved cache directory path."""
        return Path(self.cache_directory).resolve()


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
    def from_env(cls, environment: Environment = None) -> "DataDiscoveryConfig":
        """Create configuration from environment variables."""
        if environment is None:
            env_name = os.getenv("ENVIRONMENT", "development").lower()
            environment = Environment(env_name)

        # Snowflake configuration from environment
        snowflake_config = SnowflakeConfig(
            account=os.getenv("SNOWFLAKE_ACCOUNT", ""),
            user=os.getenv("SNOWFLAKE_USER", ""),
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            private_key_path=os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH"),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
            database=os.getenv("SNOWFLAKE_DATABASE", "ANALYTICS_DB"),
            schema=os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC"),
            role=os.getenv("SNOWFLAKE_ROLE", "DATA_DISCOVERY_ROLE"),
            max_tables_per_batch=int(os.getenv("SNOWFLAKE_MAX_TABLES_BATCH", "50")),
            query_timeout_seconds=int(os.getenv("SNOWFLAKE_QUERY_TIMEOUT", "30")),
            max_sample_rows=int(os.getenv("SNOWFLAKE_MAX_SAMPLE_ROWS", "1000"))
        )

        # Analysis configuration
        analysis_config = AnalysisConfig(
            min_confidence_threshold=float(os.getenv("ANALYSIS_MIN_CONFIDENCE", "0.7")),
            max_questions_per_category=int(os.getenv("ANALYSIS_MAX_QUESTIONS", "25")),
            enable_cross_table_joins=os.getenv("ANALYSIS_ENABLE_JOINS", "true").lower() == "true"
        )

        # Security configuration
        security_config = SecurityConfig(
            pii_detection_enabled=os.getenv("SECURITY_PII_DETECTION", "true").lower() == "true",
            mask_sensitive_preview=os.getenv("SECURITY_MASK_PREVIEW", "true").lower() == "true",
            governance_strict_mode=os.getenv("SECURITY_STRICT_MODE", "false").lower() == "true"
        )

        # UI configuration
        ui_config = UIConfig(
            enable_sql_execution=os.getenv("UI_ENABLE_SQL_EXEC", "false").lower() == "true",
            max_results_display=int(os.getenv("UI_MAX_RESULTS", "100")),
            cache_results_minutes=int(os.getenv("UI_CACHE_MINUTES", "15")),
            gradio_port=int(os.getenv("GRADIO_PORT", "7860")),
            gradio_host=os.getenv("GRADIO_HOST", "0.0.0.0"),
            share_publicly=os.getenv("GRADIO_SHARE", "false").lower() == "true"
        )

        # Monitoring configuration
        monitoring_config = MonitoringConfig(
            enable_metrics=os.getenv("MONITORING_ENABLE", "true").lower() == "true",
            metrics_port=int(os.getenv("MONITORING_PORT", "8080")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file_path=os.getenv("LOG_FILE_PATH")
        )

        # Cache configuration
        cache_config = CacheConfig(
            enable_caching=os.getenv("CACHE_ENABLE", "true").lower() == "true",
            cache_directory=os.getenv("CACHE_DIRECTORY", "cache"),
            schema_cache_ttl_hours=int(os.getenv("CACHE_SCHEMA_TTL_HOURS", "24")),
            query_cache_ttl_hours=int(os.getenv("CACHE_QUERY_TTL_HOURS", "6"))
        )

        return cls(
            environment=environment,
            snowflake=snowflake_config,
            analysis=analysis_config,
            security=security_config,
            ui=ui_config,
            monitoring=monitoring_config,
            cache=cache_config
        )

    def can_connect_to_snowflake(self) -> bool:
        """Check if we have enough information to connect to Snowflake."""
        return self.snowflake.has_credentials()

    def validate(self) -> bool:
        """Validate the entire configuration based on environment."""
        try:
            # Only validate Snowflake connection for production
            if self.environment == Environment.PRODUCTION:
                if not self.snowflake.has_credentials():
                    raise ValueError("Snowflake credentials are required for production")
            else:
                # For development/testing, just warn about missing credentials
                if not self.snowflake.has_credentials():
                    missing_creds = []
                    if not self.snowflake.account:
                        missing_creds.append("account")
                    if not self.snowflake.user:
                        missing_creds.append("user")
                    if not self.snowflake.password and not self.snowflake.private_key_path:
                        missing_creds.append("password/private_key")

                    if missing_creds:
                        print(f"⚠️  Development mode: Missing Snowflake credentials: {', '.join(missing_creds)}")
                        print("   Set these in .env file when ready to connect to Snowflake")

            # Validate cache directory is writable
            cache_dir = self.cache.get_cache_directory()
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Validate PII patterns file exists if specified
            if self.security.sensitive_patterns_file:
                pii_path = self.security.get_pii_patterns_path()
                if not pii_path.exists():
                    print(f"⚠️  PII patterns file not found: {pii_path} (will use defaults)")

            return True

        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "environment": self.environment.value,
            "snowflake": {
                "account": self.snowflake.account,
                "user": self.snowflake.user,
                "warehouse": self.snowflake.warehouse,
                "database": self.snowflake.database,
                "schema": self.snowflake.schema,
                "role": self.snowflake.role,
                "max_tables_per_batch": self.snowflake.max_tables_per_batch,
                "query_timeout_seconds": self.snowflake.query_timeout_seconds,
                "max_sample_rows": self.snowflake.max_sample_rows
            },
            "analysis": {
                "min_confidence_threshold": self.analysis.min_confidence_threshold,
                "max_questions_per_category": self.analysis.max_questions_per_category,
                "enable_cross_table_joins": self.analysis.enable_cross_table_joins
            },
            "security": {
                "pii_detection_enabled": self.security.pii_detection_enabled,
                "mask_sensitive_preview": self.security.mask_sensitive_preview,
                "governance_strict_mode": self.security.governance_strict_mode
            },
            "ui": {
                "enable_sql_execution": self.ui.enable_sql_execution,
                "max_results_display": self.ui.max_results_display,
                "cache_results_minutes": self.ui.cache_results_minutes,
                "gradio_port": self.ui.gradio_port,
                "gradio_host": self.ui.gradio_host
            },
            "monitoring": {
                "enable_metrics": self.monitoring.enable_metrics,
                "metrics_port": self.monitoring.metrics_port,
                "log_level": self.monitoring.log_level
            },
            "cache": {
                "enable_caching": self.cache.enable_caching,
                "cache_directory": self.cache.cache_directory,
                "schema_cache_ttl_hours": self.cache.schema_cache_ttl_hours,
                "query_cache_ttl_hours": self.cache.query_cache_ttl_hours
            }
        }

    def save_to_file(self, config_path: str) -> None:
        """Save configuration to JSON file."""
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# Global configuration instance
_config: Optional[DataDiscoveryConfig] = None


def get_config() -> DataDiscoveryConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = DataDiscoveryConfig.from_env()
    return _config


def set_config(config: DataDiscoveryConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def reload_config() -> DataDiscoveryConfig:
    """Reload configuration from environment."""
    global _config
    _config = DataDiscoveryConfig.from_env()
    return _config


if __name__ == "__main__":
    # Example usage and testing
    print("\nTesting Data Discovery Configuration System")
    print("=" * 50)

    # Debug: Show what environment variables are loaded
    print("🔍 Debug: Environment Variables Check")
    env_vars = [
        "ENVIRONMENT", "SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER",
        "SNOWFLAKE_PASSWORD", "SNOWFLAKE_DATABASE"
    ]
    for var in env_vars:
        value = os.getenv(var, "NOT SET")
        # Mask password for security
        if "PASSWORD" in var and value != "NOT SET":
            value = "*" * len(value)
        print(f"   {var}: {value}")

    # Check if any environment variables are set
    env_vars_set = [var for var in env_vars if os.getenv(var)]
    print(f"   📊 Variables set: {len(env_vars_set)}/{len(env_vars)}")
    print()

    try:
        # Test creating config from environment
        config = DataDiscoveryConfig.from_env()
        print("✅ Configuration created from environment")

        # Test validation
        is_valid = config.validate()
        print(f"✅ Configuration validation: {'PASSED' if is_valid else 'FAILED'}")

        # Check Snowflake connectivity readiness
        can_connect = config.can_connect_to_snowflake()
        print(f"🔗 Snowflake connection ready: {'YES' if can_connect else 'NO'}")

        # Print configuration summary
        print(f"\nConfiguration Summary:")
        print(f"Environment: {config.environment.value}")
        print(f"Snowflake Account: {config.snowflake.account or 'NOT SET'}")
        print(f"Snowflake User: {config.snowflake.user or 'NOT SET'}")
        print(f"Snowflake Database: {config.snowflake.database}")
        print(f"Analysis Confidence Threshold: {config.analysis.min_confidence_threshold}")
        print(f"Security PII Detection: {config.security.pii_detection_enabled}")
        print(f"UI Port: {config.ui.gradio_port}")
        print(f"Cache Enabled: {config.cache.enable_caching}")
        print(f"Cache Directory: {config.cache.get_cache_directory()}")

        # Test serialization
        config_dict = config.to_dict()
        print(f"✅ Configuration serialization successful")

        # Create sample config file for reference
        sample_config_path = "sample_config.json"
        config.save_to_file(sample_config_path)
        print(f"✅ Sample configuration saved to {sample_config_path}")

        print(f"\n{'=' * 50}")
        if can_connect:
            print("🚀 Configuration system is ready for Snowflake connection!")
        else:
            print("🔧 Configuration system is ready for development!")
            print("   Add Snowflake credentials to .env file when ready to connect.")
            print("\n   Example .env entries:")
            print("   SNOWFLAKE_ACCOUNT=your-account.snowflakecomputing.com")
            print("   SNOWFLAKE_USER=your-username")
            print("   SNOWFLAKE_PASSWORD=your-password")

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        print("Please check your environment variables or configuration.")

        # Additional debug info on failure
        print(f"\n🔍 Debug Info:")
        print(f"   Current working directory: {os.getcwd()}")
        print(f"   .env file exists: {Path('.env').exists()}")
        print(f"   .env file path: {Path('.env').absolute()}")

        # Check if dotenv is properly installed
        try:
            import dotenv

            print(f"   ✅ python-dotenv installed successfully")
        except ImportError:
            print(f"   ❌ python-dotenv not installed!")

        # Try to load .env manually for debugging
        env_file = Path('.env')
        if env_file.exists():
            print(f"   .env file content (first 10 lines):")
            with open(env_file, 'r') as f:
                for i, line in enumerate(f):
                    if i >= 10:
                        break
                    print(f"      {line.strip()}")

            # Try manual loading as a test
            print(f"\n   🔧 Attempting manual .env loading...")
            try:
                from dotenv import load_dotenv

                result = load_dotenv(env_file, override=True)
                print(f"   Manual load result: {result}")

                # Check if variables are now available
                test_var = os.getenv("SNOWFLAKE_ACCOUNT")
                print(f"   Test variable after manual load: {test_var or 'STILL NOT SET'}")
            except Exception as load_error:
                print(f"   Manual load failed: {load_error}")
        else:
            print(f"   ❌ .env file not found!")
            print(f"   Expected path: {env_file.absolute()}")