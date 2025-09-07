# src/data_discovery/tools/analysis/data_profiler.py

"""
Data Profiler Tool for comprehensive statistical analysis and data quality assessment.

Performs data profiling, quality scoring, distribution analysis, and anomaly detection
on Snowflake tables with intelligent sampling for large datasets.
"""

import time
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import statistics

# Core system imports
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from core.database_connector import get_connector
from core.state_manager import get_state_manager, StateType
from core.logging_config import get_logger
from core.config import get_config
from core.exceptions import (
    DataDiscoveryException, DataQualityError, DatabaseTimeoutError, ErrorContext
)


class DataType(Enum):
    """Data type categories for profiling."""
    NUMERIC = "numeric"
    STRING = "string"
    DATE = "date"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"


class QualityDimension(Enum):
    """Data quality dimensions."""
    COMPLETENESS = "completeness"  # Non-null percentage
    UNIQUENESS = "uniqueness"  # Unique value percentage
    VALIDITY = "validity"  # Valid format percentage
    CONSISTENCY = "consistency"  # Consistent format percentage
    ACCURACY = "accuracy"  # Accurate value percentage


@dataclass
class ColumnProfile:
    """Statistical profile of a database column."""
    column_name: str
    data_type: str
    inferred_type: DataType
    total_rows: int
    null_count: int = 0
    unique_count: int = 0
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    mean_value: Optional[float] = None
    median_value: Optional[float] = None
    std_dev: Optional[float] = None

    # String-specific metrics
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    avg_length: Optional[float] = None

    # Quality scores
    completeness_score: float = 0.0
    uniqueness_score: float = 0.0
    validity_score: float = 0.0
    overall_quality_score: float = 0.0

    # Patterns and anomalies
    common_patterns: List[str] = field(default_factory=list)
    anomalies: List[str] = field(default_factory=list)
    sample_values: List[Any] = field(default_factory=list)

    def get_null_percentage(self) -> float:
        """Get percentage of null values."""
        if self.total_rows == 0:
            return 0.0
        return (self.null_count / self.total_rows) * 100

    def get_unique_percentage(self) -> float:
        """Get percentage of unique values."""
        if self.total_rows == 0:
            return 0.0
        return (self.unique_count / self.total_rows) * 100

    def is_likely_primary_key(self) -> bool:
        """Check if column appears to be a primary key."""
        return (
                self.null_count == 0 and
                self.unique_count == self.total_rows and
                self.total_rows > 0
        )

    def is_likely_foreign_key(self) -> bool:
        """Check if column appears to be a foreign key."""
        name_lower = self.column_name.lower()
        return (
                ('_id' in name_lower or '_key' in name_lower or name_lower.endswith('_sk')) and
                self.unique_count < self.total_rows * 0.9 and
                self.null_count < self.total_rows * 0.1
        )


@dataclass
class TableProfile:
    """Complete statistical profile of a database table."""
    table_name: str
    schema_name: str
    database_name: str
    total_rows: int
    total_columns: int
    columns: List[ColumnProfile] = field(default_factory=list)
    profiling_duration_seconds: float = 0.0
    sample_size: int = 0
    profiled_at: datetime = field(default_factory=datetime.now)

    # Table-level quality metrics
    overall_completeness: float = 0.0
    overall_uniqueness: float = 0.0
    overall_quality_score: float = 0.0

    # Recommendations
    quality_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def get_column_by_name(self, column_name: str) -> Optional[ColumnProfile]:
        """Get column profile by name."""
        return next((col for col in self.columns if col.column_name == column_name), None)

    def get_primary_key_candidates(self) -> List[ColumnProfile]:
        """Get columns that could be primary keys."""
        return [col for col in self.columns if col.is_likely_primary_key()]

    def get_foreign_key_candidates(self) -> List[ColumnProfile]:
        """Get columns that could be foreign keys."""
        return [col for col in self.columns if col.is_likely_foreign_key()]

    def get_low_quality_columns(self, threshold: float = 0.5) -> List[ColumnProfile]:
        """Get columns with quality scores below threshold."""
        return [col for col in self.columns if col.overall_quality_score < threshold]


class DataProfiler:
    """Tool for comprehensive data profiling and quality assessment."""

    def __init__(self):
        self.config = get_config()
        self.connector = get_connector()
        self.state_manager = get_state_manager()
        self.logger = get_logger("data_profiler")

        # Configuration
        self.max_sample_size = self.config.snowflake.max_sample_rows
        self.large_table_threshold = 1000000  # 1M rows
        self.max_profiling_time_seconds = 300  # 5 minutes

        self.logger.info("Data Profiler initialized",
                         max_sample_size=self.max_sample_size,
                         large_table_threshold=self.large_table_threshold)

    def profile_table(self, database: str, schema: str, table_name: str,
                      force_refresh: bool = False,
                      detailed_analysis: bool = True) -> TableProfile:
        """
        Profile a complete table with statistical analysis.

        Args:
            database: Database name
            schema: Schema name
            table_name: Table name
            force_refresh: Skip cache and force fresh profiling
            detailed_analysis: Include detailed statistical analysis

        Returns:
            TableProfile with complete statistical analysis
        """
        cache_key = f"table_profile_{database}_{schema}_{table_name}"

        # Check cache first
        if not force_refresh:
            cached_profile = self.state_manager.load_state(cache_key, StateType.DATA_QUALITY)
            if cached_profile:
                self.logger.info("Using cached table profile",
                                 table=table_name,
                                 profiled_at=cached_profile.get('profiled_at'))
                return self._dict_to_table_profile(cached_profile)

        start_time = time.time()
        self.logger.info("Starting table profiling",
                         database=database,
                         schema=schema,
                         table=table_name,
                         detailed_analysis=detailed_analysis)

        try:
            # Get table metadata
            full_table_name = f'"{database}"."{schema}"."{table_name}"'

            # Get row count
            total_rows = self._get_table_row_count(full_table_name)

            # Get column information
            columns_info = self._get_table_columns(database, schema, table_name)

            # Determine sampling strategy
            sample_size = self._calculate_sample_size(total_rows)

            # Profile each column
            column_profiles = []
            for col_info in columns_info:
                column_profile = self._profile_column(
                    full_table_name,
                    col_info,
                    total_rows,
                    sample_size,
                    detailed_analysis
                )
                column_profiles.append(column_profile)

            # Calculate table-level metrics
            table_profile = TableProfile(
                table_name=table_name,
                schema_name=schema,
                database_name=database,
                total_rows=total_rows,
                total_columns=len(columns_info),
                columns=column_profiles,
                sample_size=sample_size,
                profiling_duration_seconds=time.time() - start_time
            )

            # Calculate overall quality metrics
            self._calculate_table_quality_metrics(table_profile)

            # Generate recommendations
            self._generate_quality_recommendations(table_profile)

            # Cache the results
            self.state_manager.save_state(
                cache_key,
                self._table_profile_to_dict(table_profile),
                StateType.DATA_QUALITY,
                ttl_hours=24
            )

            self.logger.info("Table profiling completed",
                             table=table_name,
                             duration=table_profile.profiling_duration_seconds,
                             sample_size=sample_size,
                             quality_score=table_profile.overall_quality_score)

            return table_profile

        except Exception as e:
            self.logger.error("Table profiling failed",
                              table=table_name,
                              error=str(e))
            raise DataQualityError(
                f"Failed to profile table {table_name}: {str(e)}",
                table=table_name
            )

    def _get_table_row_count(self, full_table_name: str) -> int:
        """Get exact row count for the table."""
        try:
            query = f"SELECT COUNT(*) as row_count FROM {full_table_name}"
            result = self.connector.execute_query(query, timeout_seconds=60)

            if result.data:
                return result.data[0]['ROW_COUNT']
            return 0

        except Exception as e:
            self.logger.warning("Failed to get exact row count",
                                table=full_table_name,
                                error=str(e))
            # Try approximate count
            try:
                query = f"SELECT APPROXIMATE_COUNT_DISTINCT(*) as approx_count FROM {full_table_name}"
                result = self.connector.execute_query(query, timeout_seconds=30)
                if result.data:
                    return result.data[0]['APPROX_COUNT']
            except:
                pass
            return 0

    def _get_table_columns(self, database: str, schema: str, table_name: str) -> List[Dict[str, Any]]:
        """Get column information for the table."""
        query = """
        SELECT 
            column_name,
            data_type,
            is_nullable,
            column_default,
            character_maximum_length,
            numeric_precision,
            numeric_scale
        FROM information_schema.columns 
        WHERE table_catalog = %(database)s
          AND table_schema = %(schema)s 
          AND table_name = %(table_name)s
        ORDER BY ordinal_position
        """

        result = self.connector.execute_query(query, {
            'database': database,
            'schema': schema,
            'table_name': table_name
        })

        return result.data

    def _calculate_sample_size(self, total_rows: int) -> int:
        """Calculate appropriate sample size for profiling."""
        if total_rows <= self.max_sample_size:
            return total_rows

        # For large tables, use statistical sampling
        if total_rows > self.large_table_threshold:
            # Use margin of error formula for 95% confidence
            margin_of_error = 0.05  # 5%
            confidence_level = 1.96  # 95%

            sample_size = (confidence_level ** 2 * 0.25) / (margin_of_error ** 2)
            sample_size = int(sample_size / (1 + (sample_size - 1) / total_rows))

            return min(sample_size, self.max_sample_size)

        return min(total_rows, self.max_sample_size)

    def _profile_column(self, full_table_name: str, col_info: Dict[str, Any],
                        total_rows: int, sample_size: int, detailed_analysis: bool) -> ColumnProfile:
        """Profile a single column with statistical analysis."""
        column_name = col_info['COLUMN_NAME']
        data_type = col_info['DATA_TYPE']

        # Create base profile
        profile = ColumnProfile(
            column_name=column_name,
            data_type=data_type,
            inferred_type=self._infer_data_type(data_type),
            total_rows=total_rows
        )

        try:
            # Build sampling query
            sample_clause = ""
            if sample_size < total_rows:
                sample_percent = (sample_size / total_rows) * 100
                sample_clause = f"SAMPLE ({sample_percent:.2f})"

            # Basic statistics query
            basic_query = f"""
            SELECT 
                COUNT(*) as total_count,
                COUNT("{column_name}") as non_null_count,
                COUNT(DISTINCT "{column_name}") as unique_count
            FROM {full_table_name} {sample_clause}
            """

            basic_result = self.connector.execute_query(basic_query, timeout_seconds=30)

            if basic_result.data:
                data = basic_result.data[0]
                profile.null_count = data['TOTAL_COUNT'] - data['NON_NULL_COUNT']
                profile.unique_count = data['UNIQUE_COUNT']

            # Type-specific analysis
            if detailed_analysis:
                self._detailed_column_analysis(full_table_name, profile, sample_clause)

            # Calculate quality scores
            self._calculate_column_quality_scores(profile)

            # Get sample values
            profile.sample_values = self._get_sample_values(full_table_name, column_name, sample_clause)

        except Exception as e:
            self.logger.warning("Column profiling failed",
                                column=column_name,
                                error=str(e))
            profile.anomalies.append(f"Profiling error: {str(e)}")

        return profile

    def _detailed_column_analysis(self, full_table_name: str, profile: ColumnProfile, sample_clause: str):
        """Perform detailed type-specific analysis."""
        column_name = profile.column_name

        if profile.inferred_type == DataType.NUMERIC:
            self._analyze_numeric_column(full_table_name, profile, sample_clause)
        elif profile.inferred_type == DataType.STRING:
            self._analyze_string_column(full_table_name, profile, sample_clause)
        elif profile.inferred_type == DataType.DATE:
            self._analyze_date_column(full_table_name, profile, sample_clause)

    def _analyze_numeric_column(self, full_table_name: str, profile: ColumnProfile, sample_clause: str):
        """Analyze numeric column statistics."""
        column_name = profile.column_name

        try:
            query = f"""
            SELECT 
                MIN("{column_name}") as min_val,
                MAX("{column_name}") as max_val,
                AVG("{column_name}") as mean_val,
                MEDIAN("{column_name}") as median_val,
                STDDEV("{column_name}") as std_dev
            FROM {full_table_name} {sample_clause}
            WHERE "{column_name}" IS NOT NULL
            """

            result = self.connector.execute_query(query, timeout_seconds=30)

            if result.data and result.data[0]['MIN_VAL'] is not None:
                data = result.data[0]
                profile.min_value = data['MIN_VAL']
                profile.max_value = data['MAX_VAL']
                profile.mean_value = float(data['MEAN_VAL']) if data['MEAN_VAL'] is not None else None
                profile.median_value = float(data['MEDIAN_VAL']) if data['MEDIAN_VAL'] is not None else None
                profile.std_dev = float(data['STD_DEV']) if data['STD_DEV'] is not None else None

                # Detect anomalies
                if profile.mean_value and profile.std_dev:
                    # Check for potential outliers (values beyond 3 standard deviations)
                    outlier_query = f"""
                    SELECT COUNT(*) as outlier_count
                    FROM {full_table_name} {sample_clause}
                    WHERE "{column_name}" IS NOT NULL 
                      AND (
                        "{column_name}" < {profile.mean_value - 3 * profile.std_dev} OR
                        "{column_name}" > {profile.mean_value + 3 * profile.std_dev}
                      )
                    """

                    outlier_result = self.connector.execute_query(outlier_query, timeout_seconds=30)
                    if outlier_result.data and outlier_result.data[0]['OUTLIER_COUNT'] > 0:
                        outlier_count = outlier_result.data[0]['OUTLIER_COUNT']
                        outlier_percent = (outlier_count / profile.total_rows) * 100
                        if outlier_percent > 1:  # More than 1% outliers
                            profile.anomalies.append(f"High outlier count: {outlier_percent:.1f}% of values")

        except Exception as e:
            self.logger.warning("Numeric analysis failed",
                                column=column_name,
                                error=str(e))

    def _analyze_string_column(self, full_table_name: str, profile: ColumnProfile, sample_clause: str):
        """Analyze string column patterns and lengths."""
        column_name = profile.column_name

        try:
            # Length analysis
            length_query = f"""
            SELECT 
                MIN(LENGTH("{column_name}")) as min_length,
                MAX(LENGTH("{column_name}")) as max_length,
                AVG(LENGTH("{column_name}")) as avg_length
            FROM {full_table_name} {sample_clause}
            WHERE "{column_name}" IS NOT NULL
            """

            result = self.connector.execute_query(length_query, timeout_seconds=30)

            if result.data and result.data[0]['MIN_LENGTH'] is not None:
                data = result.data[0]
                profile.min_length = data['MIN_LENGTH']
                profile.max_length = data['MAX_LENGTH']
                profile.avg_length = float(data['AVG_LENGTH']) if data['AVG_LENGTH'] is not None else None

            # Pattern analysis for common formats
            self._analyze_string_patterns(full_table_name, profile, sample_clause)

        except Exception as e:
            self.logger.warning("String analysis failed",
                                column=column_name,
                                error=str(e))

    def _analyze_string_patterns(self, full_table_name: str, profile: ColumnProfile, sample_clause: str):
        """Analyze string patterns for format validation."""
        column_name = profile.column_name

        # Common patterns to check
        patterns = {
            'email': r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            'phone': r"^\+?1?-?\.?\s?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})$",
            'url': r"^https?://[^\s/$.?#].[^\s]*$",
            'uuid': r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        }

        for pattern_name, pattern in patterns.items():
            try:
                pattern_query = f"""
                SELECT COUNT(*) as pattern_matches
                FROM {full_table_name} {sample_clause}
                WHERE "{column_name}" IS NOT NULL 
                  AND REGEXP_LIKE("{column_name}", '{pattern}', 'i')
                """

                result = self.connector.execute_query(pattern_query, timeout_seconds=30)

                if result.data and result.data[0]['PATTERN_MATCHES'] > 0:
                    matches = result.data[0]['PATTERN_MATCHES']
                    non_null_count = profile.total_rows - profile.null_count
                    match_percent = (matches / non_null_count) * 100 if non_null_count > 0 else 0

                    if match_percent > 80:  # More than 80% match the pattern
                        profile.common_patterns.append(f"{pattern_name}: {match_percent:.1f}%")

            except Exception:
                # Pattern check failed, continue with others
                continue

    def _analyze_date_column(self, full_table_name: str, profile: ColumnProfile, sample_clause: str):
        """Analyze date column ranges and patterns."""
        column_name = profile.column_name

        try:
            date_query = f"""
            SELECT 
                MIN("{column_name}") as min_date,
                MAX("{column_name}") as max_date,
                COUNT(DISTINCT DATE_TRUNC('year', "{column_name}")) as year_count,
                COUNT(DISTINCT DATE_TRUNC('month', "{column_name}")) as month_count
            FROM {full_table_name} {sample_clause}
            WHERE "{column_name}" IS NOT NULL
            """

            result = self.connector.execute_query(date_query, timeout_seconds=30)

            if result.data and result.data[0]['MIN_DATE'] is not None:
                data = result.data[0]
                profile.min_value = data['MIN_DATE']
                profile.max_value = data['MAX_DATE']

                # Check for date range anomalies
                year_count = data['YEAR_COUNT']
                if year_count > 50:  # More than 50 years of data
                    profile.anomalies.append(f"Wide date range: {year_count} years")

        except Exception as e:
            self.logger.warning("Date analysis failed",
                                column=column_name,
                                error=str(e))

    def _calculate_column_quality_scores(self, profile: ColumnProfile):
        """Calculate quality scores for a column."""
        # Completeness (non-null percentage)
        if profile.total_rows > 0:
            profile.completeness_score = 1.0 - (profile.null_count / profile.total_rows)
        else:
            profile.completeness_score = 0.0

        # Uniqueness score
        if profile.total_rows > 0:
            profile.uniqueness_score = profile.unique_count / profile.total_rows
        else:
            profile.uniqueness_score = 0.0

        # Validity score (based on patterns and anomalies)
        profile.validity_score = 1.0
        if profile.anomalies:
            profile.validity_score -= len(profile.anomalies) * 0.1

        profile.validity_score = max(0.0, profile.validity_score)

        # Overall quality score (weighted average)
        profile.overall_quality_score = (
                profile.completeness_score * 0.4 +
                profile.uniqueness_score * 0.3 +
                profile.validity_score * 0.3
        )

    def _calculate_table_quality_metrics(self, table_profile: TableProfile):
        """Calculate table-level quality metrics."""
        if not table_profile.columns:
            return

        # Overall completeness
        completeness_scores = [col.completeness_score for col in table_profile.columns]
        table_profile.overall_completeness = statistics.mean(completeness_scores)

        # Overall uniqueness
        uniqueness_scores = [col.uniqueness_score for col in table_profile.columns]
        table_profile.overall_uniqueness = statistics.mean(uniqueness_scores)

        # Overall quality score
        quality_scores = [col.overall_quality_score for col in table_profile.columns]
        table_profile.overall_quality_score = statistics.mean(quality_scores)

    def _generate_quality_recommendations(self, table_profile: TableProfile):
        """Generate data quality recommendations."""
        recommendations = []
        quality_issues = []

        # Check for low completeness
        low_completeness_cols = [col for col in table_profile.columns if col.completeness_score < 0.8]
        if low_completeness_cols:
            quality_issues.append(f"{len(low_completeness_cols)} columns have high null rates")
            recommendations.append("Investigate data collection processes for incomplete columns")

        # Check for potential duplicates
        potential_pk_cols = table_profile.get_primary_key_candidates()
        if not potential_pk_cols:
            quality_issues.append("No clear primary key identified")
            recommendations.append("Define and enforce primary key constraints")

        # Check for anomalies
        anomaly_cols = [col for col in table_profile.columns if col.anomalies]
        if anomaly_cols:
            quality_issues.append(f"{len(anomaly_cols)} columns have data anomalies")
            recommendations.append("Review and clean anomalous data patterns")

        # Performance recommendations
        if table_profile.total_rows > 10000000:  # 10M+ rows
            recommendations.append("Consider partitioning strategy for large table")

        table_profile.quality_issues = quality_issues
        table_profile.recommendations = recommendations

    def _get_sample_values(self, full_table_name: str, column_name: str, sample_clause: str) -> List[Any]:
        """Get sample values from the column."""
        try:
            query = f"""
            SELECT DISTINCT "{column_name}" as sample_value
            FROM {full_table_name} {sample_clause}
            WHERE "{column_name}" IS NOT NULL
            LIMIT 10
            """

            result = self.connector.execute_query(query, timeout_seconds=30)

            if result.data:
                return [row['SAMPLE_VALUE'] for row in result.data]

        except Exception:
            pass

        return []

    def _infer_data_type(self, snowflake_type: str) -> DataType:
        """Infer logical data type from Snowflake type."""
        type_lower = snowflake_type.lower()

        if any(t in type_lower for t in ['number', 'decimal', 'numeric', 'int', 'float', 'double']):
            return DataType.NUMERIC
        elif any(t in type_lower for t in ['varchar', 'char', 'text', 'string']):
            return DataType.STRING
        elif any(t in type_lower for t in ['date', 'timestamp', 'time']):
            return DataType.DATE
        elif 'boolean' in type_lower:
            return DataType.BOOLEAN
        else:
            return DataType.UNKNOWN

    def _table_profile_to_dict(self, profile: TableProfile) -> Dict[str, Any]:
        """Convert TableProfile to dictionary for caching."""
        return {
            "table_name": profile.table_name,
            "schema_name": profile.schema_name,
            "database_name": profile.database_name,
            "total_rows": profile.total_rows,
            "total_columns": profile.total_columns,
            "sample_size": profile.sample_size,
            "profiling_duration_seconds": profile.profiling_duration_seconds,
            "profiled_at": profile.profiled_at.isoformat(),
            "overall_completeness": profile.overall_completeness,
            "overall_uniqueness": profile.overall_uniqueness,
            "overall_quality_score": profile.overall_quality_score,
            "quality_issues": profile.quality_issues,
            "recommendations": profile.recommendations,
            "columns": [
                {
                    "column_name": col.column_name,
                    "data_type": col.data_type,
                    "inferred_type": col.inferred_type.value,
                    "total_rows": col.total_rows,
                    "null_count": col.null_count,
                    "unique_count": col.unique_count,
                    "min_value": str(col.min_value) if col.min_value is not None else None,
                    "max_value": str(col.max_value) if col.max_value is not None else None,
                    "mean_value": col.mean_value,
                    "median_value": col.median_value,
                    "std_dev": col.std_dev,
                    "min_length": col.min_length,
                    "max_length": col.max_length,
                    "avg_length": col.avg_length,
                    "completeness_score": col.completeness_score,
                    "uniqueness_score": col.uniqueness_score,
                    "validity_score": col.validity_score,
                    "overall_quality_score": col.overall_quality_score,
                    "common_patterns": col.common_patterns,
                    "anomalies": col.anomalies,
                    "sample_values": [str(v) for v in col.sample_values]
                }
                for col in profile.columns
            ]
        }

    def _dict_to_table_profile(self, data: Dict[str, Any]) -> TableProfile:
        """Convert dictionary back to TableProfile."""
        columns = []

        for col_data in data.get("columns", []):
            column = ColumnProfile(
                column_name=col_data["column_name"],
                data_type=col_data["data_type"],
                inferred_type=DataType(col_data["inferred_type"]),
                total_rows=col_data["total_rows"],
                null_count=col_data["null_count"],
                unique_count=col_data["unique_count"],
                min_value=col_data.get("min_value"),
                max_value=col_data.get("max_value"),
                mean_value=col_data.get("mean_value"),
                median_value=col_data.get("median_value"),
                std_dev=col_data.get("std_dev"),
                min_length=col_data.get("min_length"),
                max_length=col_data.get("max_length"),
                avg_length=col_data.get("avg_length"),
                completeness_score=col_data["completeness_score"],
                uniqueness_score=col_data["uniqueness_score"],
                validity_score=col_data["validity_score"],
                overall_quality_score=col_data["overall_quality_score"],
                common_patterns=col_data["common_patterns"],
                anomalies=col_data["anomalies"],
                sample_values=col_data["sample_values"]
            )
            columns.append(column)

        return TableProfile(
            table_name=data["table_name"],
            schema_name=data["schema_name"],
            database_name=data["database_name"],
            total_rows=data["total_rows"],
            total_columns=data["total_columns"],
            columns=columns,
            sample_size=data["sample_size"],
            profiling_duration_seconds=data["profiling_duration_seconds"],
            profiled_at=datetime.fromisoformat(data["profiled_at"]),
            overall_completeness=data["overall_completeness"],
            overall_uniqueness=data["overall_uniqueness"],
            overall_quality_score=data["overall_quality_score"],
            quality_issues=data["quality_issues"],
            recommendations=data["recommendations"]
        )


# Testing and demonstration
if __name__ == "__main__":
    print("Testing Data Profiler Tool")
    print("=" * 50)

    logger = get_logger("test_data_profiler")

    try:
        # Create Data Profiler
        print("Creating Data Profiler...")
        profiler = DataProfiler()

        print(f"Data Profiler created")
        print(f"   Max sample size: {profiler.max_sample_size:,}")
        print(f"   Large table threshold: {profiler.large_table_threshold:,}")

        # Test with different table sizes
        test_tables = [
            ("SNOWFLAKE_SAMPLE_DATA", "TPCDS_SF10TCL", "CUSTOMER"),  # Medium table
            ("SNOWFLAKE_SAMPLE_DATA", "TPCDS_SF10TCL", "ITEM"),  # Smaller table
            ("SNOWFLAKE_SAMPLE_DATA", "TPCDS_SF10TCL", "DATE_DIM"),  # Small reference table
        ]

        for database, schema, table_name in test_tables:
            print(f"\nTest: Profiling {table_name}")
            try:
                profile = profiler.profile_table(
                    database=database,
                    schema=schema,
                    table_name=table_name,
                    detailed_analysis=True
                )

                print(f"Table Profile Completed")
                print(f"   Table: {profile.table_name}")
                print(f"   Total rows: {profile.total_rows:,}")
                print(f"   Total columns: {profile.total_columns}")
                print(f"   Sample size: {profile.sample_size:,}")
                print(f"   Profiling duration: {profile.profiling_duration_seconds:.2f}s")
                print(f"   Overall quality score: {profile.overall_quality_score:.2f}")

                # Show column insights
                print(f"\n   Column Quality Summary:")
                for col in profile.columns[:5]:  # Show first 5 columns
                    null_pct = col.get_null_percentage()
                    unique_pct = col.get_unique_percentage()
                    print(f"      {col.column_name}: Quality {col.overall_quality_score:.2f}, "
                          f"Nulls {null_pct:.1f}%, Unique {unique_pct:.1f}%")

                # Show data insights
                pk_candidates = profile.get_primary_key_candidates()
                if pk_candidates:
                    print(f"   Primary key candidates: {[col.column_name for col in pk_candidates]}")

                fk_candidates = profile.get_foreign_key_candidates()
                if fk_candidates:
                    print(f"   Foreign key candidates: {[col.column_name for col in fk_candidates]}")

                # Show quality issues and recommendations
                if profile.quality_issues:
                    print(f"   Quality issues: {len(profile.quality_issues)}")
                    for issue in profile.quality_issues[:3]:
                        print(f"      - {issue}")

                if profile.recommendations:
                    print(f"   Recommendations: {len(profile.recommendations)}")
                    for rec in profile.recommendations[:2]:
                        print(f"      - {rec}")

            except Exception as e:
                print(f"Failed to profile {table_name}: {str(e)}")

        # Test caching
        print(f"\nTesting cache functionality...")
        start_time = time.time()
        cached_profile = profiler.profile_table(
            database="SNOWFLAKE_SAMPLE_DATA",
            schema="TPCDS_SF10TCL",
            table_name="CUSTOMER"
        )
        cache_time = time.time() - start_time

        print(f"Cache test completed in {cache_time:.3f} seconds")
        print(f"   Cache hit: {'Yes' if cache_time < 1.0 else 'No'}")

        print(f"\nData Profiler tool tested successfully!")
        print(f"   Features: Statistical profiling, quality scoring, anomaly detection")
        print(f"   Integration: Database connector, state manager, intelligent sampling")
        print(f"   Performance: Efficient large table handling with sampling")
        print(f"\nReady to support Data Scientist Agent!")

    except Exception as e:
        logger.error("Data Profiler test failed", error=str(e))
        print(f"Test failed: {str(e)}")
        print(f"Check that database connector and schema access are working correctly")