# src/data_discovery/agents/data_scientist_agent.py

"""
Data Scientist Agent for comprehensive statistical analysis and data quality assessment.

Orchestrates data profiling, applies statistical methods, detects patterns and anomalies,
and generates data-driven insights and recommendations across multiple tables.
"""

import time
import statistics
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import math

# Core system imports
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent, AgentTask, TaskPriority, AgentStatus
from tools.analysis.data_profiler import DataProfiler, TableProfile, ColumnProfile, DataType, QualityDimension
from core.state_manager import StateType
from core.exceptions import DataDiscoveryException, DataQualityError, ErrorContext


class AnalysisType(Enum):
    """Types of statistical analysis."""
    DESCRIPTIVE = "descriptive"
    DIAGNOSTIC = "diagnostic"
    PREDICTIVE = "predictive"
    PRESCRIPTIVE = "prescriptive"


class DataPattern(Enum):
    """Types of data patterns that can be detected."""
    CORRELATION = "correlation"
    SEASONALITY = "seasonality"
    OUTLIER = "outlier"
    DISTRIBUTION = "distribution"
    DEPENDENCY = "dependency"


@dataclass
class StatisticalInsight:
    """Statistical insight with supporting data."""
    insight_id: str
    insight_type: AnalysisType
    pattern_type: DataPattern
    title: str
    description: str
    confidence: float  # 0.0 - 1.0
    significance: str  # low, medium, high, critical
    tables_involved: List[str] = field(default_factory=list)
    columns_involved: List[str] = field(default_factory=list)
    statistical_measures: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "insight_id": self.insight_id,
            "insight_type": self.insight_type.value,
            "pattern_type": self.pattern_type.value,
            "title": self.title,
            "description": self.description,
            "confidence": self.confidence,
            "significance": self.significance,
            "tables_involved": self.tables_involved,
            "columns_involved": self.columns_involved,
            "statistical_measures": self.statistical_measures,
            "recommendations": self.recommendations
        }


@dataclass
class DataQualityScorecard:
    """Comprehensive data quality assessment."""
    database: str
    schema_name: str
    overall_score: float
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    table_scores: Dict[str, float] = field(default_factory=dict)
    critical_issues: List[str] = field(default_factory=list)
    improvement_opportunities: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)

    def get_grade(self) -> str:
        """Get letter grade based on overall score."""
        if self.overall_score >= 0.9:
            return "A"
        elif self.overall_score >= 0.8:
            return "B"
        elif self.overall_score >= 0.7:
            return "C"
        elif self.overall_score >= 0.6:
            return "D"
        else:
            return "F"


@dataclass
class DataScienceReport:
    """Comprehensive data science analysis report."""
    report_id: str
    database: str
    schema_name: str
    tables_analyzed: int
    analysis_duration_seconds: float
    quality_scorecard: DataQualityScorecard
    statistical_insights: List[StatisticalInsight] = field(default_factory=list)
    table_profiles: List[TableProfile] = field(default_factory=list)
    cross_table_analysis: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)

    def get_high_confidence_insights(self) -> List[StatisticalInsight]:
        """Get insights with high confidence scores."""
        return [insight for insight in self.statistical_insights if insight.confidence > 0.8]

    def get_critical_insights(self) -> List[StatisticalInsight]:
        """Get critical significance insights."""
        return [insight for insight in self.statistical_insights if insight.significance == "critical"]


class DataScientistAgent(BaseAgent):
    """
    Data Scientist Agent for statistical analysis and data quality assessment.

    Responsibilities:
    - Orchestrate data profiling across multiple tables
    - Apply statistical methods for pattern detection
    - Assess data quality comprehensively
    - Generate scientific insights and recommendations
    - Create data quality scorecards
    """

    def __init__(self, agent_id: str = None):
        super().__init__(
            agent_id=agent_id,
            agent_name="DataScientist"
        )

        # Initialize tools
        self.data_profiler = DataProfiler()

        # Configuration
        self.max_tables_per_batch = 10
        self.min_insight_confidence = 0.6
        self.correlation_threshold = 0.7

        self.logger.info("Data Scientist Agent initialized",
                         max_tables_per_batch=self.max_tables_per_batch,
                         min_insight_confidence=self.min_insight_confidence)

    def get_capabilities(self) -> List[str]:
        """Return capabilities of the Data Scientist Agent."""
        return [
            "data_profiling",
            "statistical_analysis",
            "quality_assessment",
            "pattern_detection",
            "anomaly_detection",
            "scientific_reporting"
        ]

    def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute data science tasks."""
        task_type = task.task_type
        parameters = task.parameters

        if task_type == "profile_schema":
            database = parameters.get("database")
            schema_name = parameters.get("schema_name")
            table_list = parameters.get("table_list")
            return self._profile_schema_tables(database, schema_name, table_list)

        elif task_type == "comprehensive_analysis":
            technical_analysis = parameters.get("technical_analysis")
            return self._perform_comprehensive_analysis(technical_analysis)

        elif task_type == "quality_assessment":
            table_profiles = parameters.get("table_profiles")
            return self._assess_data_quality(table_profiles)

        elif task_type == "detect_patterns":
            table_profiles = parameters.get("table_profiles")
            return self._detect_data_patterns(table_profiles)

        elif task_type == "generate_insights":
            table_profiles = parameters.get("table_profiles")
            return self._generate_statistical_insights(table_profiles)

        else:
            raise DataDiscoveryException(f"Unknown task type: {task_type}")

    def _profile_schema_tables(self, database: str, schema_name: str,
                               table_list: List[str] = None) -> Dict[str, Any]:
        """Profile multiple tables in a schema."""
        self.logger.info("Starting schema profiling",
                         database=database,
                         schema=schema_name,
                         table_count=len(table_list) if table_list else "all")

        try:
            if self.current_task:
                self.current_task.progress_percent = 10.0

            profiles = []
            total_tables = len(table_list) if table_list else 0

            for i, table_name in enumerate(table_list or []):
                try:
                    self.logger.debug("Profiling table",
                                      table=table_name,
                                      progress=f"{i + 1}/{total_tables}")

                    profile = self.data_profiler.profile_table(
                        database=database,
                        schema=schema_name,
                        table_name=table_name,
                        detailed_analysis=True
                    )
                    profiles.append(profile)

                    # Update progress
                    if self.current_task:
                        progress = 10.0 + (i + 1) / total_tables * 80.0
                        self.current_task.progress_percent = progress

                except Exception as e:
                    self.logger.warning("Failed to profile table",
                                        table=table_name,
                                        error=str(e))
                    continue

            if self.current_task:
                self.current_task.progress_percent = 100.0

            # Cache profiles
            cache_key = f"schema_profiles_{database}_{schema_name}"
            self.cache_result(cache_key, profiles, ttl_hours=12)

            self.logger.info("Schema profiling completed",
                             database=database,
                             schema=schema_name,
                             profiles_created=len(profiles),
                             failed_tables=total_tables - len(profiles))

            return {
                "status": "success",
                "database": database,
                "schema_name": schema_name,
                "profiles_created": len(profiles),
                "failed_tables": total_tables - len(profiles),
                "cache_key": cache_key,
                "profile_summary": [
                    {
                        "table_name": profile.table_name,
                        "total_rows": profile.total_rows,
                        "total_columns": profile.total_columns,
                        "quality_score": profile.overall_quality_score,
                        "profiling_duration": profile.profiling_duration_seconds
                    }
                    for profile in profiles
                ]
            }

        except Exception as e:
            self.logger.error("Schema profiling failed",
                              database=database,
                              schema=schema_name,
                              error=str(e))
            raise DataQualityError(f"Schema profiling failed: {str(e)}")

    def _perform_comprehensive_analysis(self, technical_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive data science analysis."""
        start_time = time.time()

        self.logger.info("Starting comprehensive data science analysis")

        try:
            analysis_info = technical_analysis.get('analysis', {})
            table_summary = technical_analysis.get('table_summary', [])

            database = analysis_info.get('database')
            schema_name = analysis_info.get('schema_name')

            if self.current_task:
                self.current_task.progress_percent = 5.0

            # Extract table names for profiling
            table_names = [table['name'] for table in table_summary[:self.max_tables_per_batch]]

            # Profile tables
            profiling_result = self._profile_schema_tables(database, schema_name, table_names)
            profiles = self.get_cached_result(profiling_result['cache_key'])

            if self.current_task:
                self.current_task.progress_percent = 40.0

            # Assess data quality
            quality_scorecard = self._create_quality_scorecard(profiles, database, schema_name)

            if self.current_task:
                self.current_task.progress_percent = 60.0

            # Generate statistical insights
            insights = self._generate_statistical_insights(profiles)

            if self.current_task:
                self.current_task.progress_percent = 80.0

            # Perform cross-table analysis
            cross_table_analysis = self._perform_cross_table_analysis(profiles)

            # Generate recommendations
            recommendations = self._generate_data_science_recommendations(
                profiles, quality_scorecard, insights
            )

            if self.current_task:
                self.current_task.progress_percent = 100.0

            # Create comprehensive report
            analysis_duration = time.time() - start_time

            report = DataScienceReport(
                report_id=f"ds_analysis_{database}_{int(time.time())}",
                database=database,
                schema_name=schema_name,
                tables_analyzed=len(profiles),
                analysis_duration_seconds=analysis_duration,
                quality_scorecard=quality_scorecard,
                statistical_insights=insights,
                table_profiles=profiles,
                cross_table_analysis=cross_table_analysis,
                recommendations=recommendations
            )

            # Cache the report
            cache_key = f"data_science_report_{database}_{schema_name}"
            self.cache_result(cache_key, report, ttl_hours=24)

            self.logger.info("Comprehensive analysis completed",
                             database=database,
                             schema=schema_name,
                             tables_analyzed=len(profiles),
                             insights_generated=len(insights),
                             quality_grade=quality_scorecard.get_grade(),
                             duration_seconds=analysis_duration)

            return {
                "status": "success",
                "report": {
                    "report_id": report.report_id,
                    "database": database,
                    "schema_name": schema_name,
                    "tables_analyzed": len(profiles),
                    "analysis_duration_seconds": analysis_duration,
                    "quality_grade": quality_scorecard.get_grade(),
                    "overall_quality_score": quality_scorecard.overall_score
                },
                "insights_summary": {
                    "total_insights": len(insights),
                    "high_confidence": len(report.get_high_confidence_insights()),
                    "critical_insights": len(report.get_critical_insights()),
                    "patterns_detected": len(set(insight.pattern_type.value for insight in insights))
                },
                "quality_summary": {
                    "overall_score": quality_scorecard.overall_score,
                    "grade": quality_scorecard.get_grade(),
                    "critical_issues": len(quality_scorecard.critical_issues),
                    "improvement_opportunities": len(quality_scorecard.improvement_opportunities)
                },
                "recommendations": len(recommendations),
                "cache_key": cache_key
            }

        except Exception as e:
            self.logger.error("Comprehensive analysis failed", error=str(e))
            raise DataDiscoveryException(f"Data science analysis failed: {str(e)}")

    def _create_quality_scorecard(self, profiles: List[TableProfile],
                                  database: str, schema_name: str) -> DataQualityScorecard:
        """Create comprehensive data quality scorecard."""
        if not profiles:
            return DataQualityScorecard(
                database=database,
                schema_name=schema_name,
                overall_score=0.0
            )

        # Calculate dimension scores
        dimension_scores = {}

        # Completeness score
        completeness_scores = [profile.overall_completeness for profile in profiles]
        dimension_scores[QualityDimension.COMPLETENESS.value] = statistics.mean(completeness_scores)

        # Uniqueness score
        uniqueness_scores = [profile.overall_uniqueness for profile in profiles]
        dimension_scores[QualityDimension.UNIQUENESS.value] = statistics.mean(uniqueness_scores)

        # Overall quality scores by table
        table_scores = {
            profile.table_name: profile.overall_quality_score
            for profile in profiles
        }

        # Overall score
        overall_score = statistics.mean(table_scores.values())

        # Identify critical issues
        critical_issues = []
        improvement_opportunities = []

        for profile in profiles:
            if profile.overall_quality_score < 0.5:
                critical_issues.append(f"Table {profile.table_name} has very low quality score")

            if profile.overall_completeness < 0.7:
                critical_issues.append(f"Table {profile.table_name} has high null rates")

            low_quality_cols = profile.get_low_quality_columns()
            if len(low_quality_cols) > profile.total_columns * 0.3:
                improvement_opportunities.append(
                    f"Table {profile.table_name}: {len(low_quality_cols)} columns need quality improvement"
                )

        return DataQualityScorecard(
            database=database,
            schema_name=schema_name,
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            table_scores=table_scores,
            critical_issues=critical_issues,
            improvement_opportunities=improvement_opportunities
        )

    def _generate_statistical_insights(self, profiles: List[TableProfile]) -> List[StatisticalInsight]:
        """Generate statistical insights from table profiles."""
        insights = []

        # Data distribution insights
        insights.extend(self._analyze_data_distributions(profiles))

        # Quality pattern insights
        insights.extend(self._analyze_quality_patterns(profiles))

        # Anomaly detection insights
        insights.extend(self._detect_statistical_anomalies(profiles))

        # Relationship insights
        insights.extend(self._analyze_table_relationships(profiles))

        # Filter by confidence threshold
        insights = [insight for insight in insights if insight.confidence >= self.min_insight_confidence]

        # Sort by confidence and significance
        significance_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        insights.sort(
            key=lambda x: (significance_order.get(x.significance, 0), x.confidence),
            reverse=True
        )

        return insights

    def _analyze_data_distributions(self, profiles: List[TableProfile]) -> List[StatisticalInsight]:
        """Analyze data distributions across tables."""
        insights = []

        for profile in profiles:
            numeric_columns = [col for col in profile.columns if col.inferred_type == DataType.NUMERIC]

            for col in numeric_columns:
                if col.mean_value is not None and col.std_dev is not None and col.std_dev > 0:
                    # Check for normal distribution (coefficient of variation)
                    cv = col.std_dev / abs(col.mean_value) if col.mean_value != 0 else float('inf')

                    if cv > 2.0:  # High variability
                        insights.append(StatisticalInsight(
                            insight_id=f"high_variability_{profile.table_name}_{col.column_name}",
                            insight_type=AnalysisType.DESCRIPTIVE,
                            pattern_type=DataPattern.DISTRIBUTION,
                            title=f"High Variability in {col.column_name}",
                            description=f"Column {col.column_name} in {profile.table_name} shows high variability "
                                        f"(CV={cv:.2f}), indicating diverse value ranges.",
                            confidence=0.8,
                            significance="medium",
                            tables_involved=[profile.table_name],
                            columns_involved=[col.column_name],
                            statistical_measures={"coefficient_variation": cv, "std_dev": col.std_dev},
                            recommendations=[
                                "Consider data transformation for analysis",
                                "Investigate outliers and extreme values",
                                "Review data collection consistency"
                            ]
                        ))

        return insights

    def _analyze_quality_patterns(self, profiles: List[TableProfile]) -> List[StatisticalInsight]:
        """Analyze quality patterns across tables."""
        insights = []

        # Overall quality assessment
        quality_scores = [profile.overall_quality_score for profile in profiles]
        avg_quality = statistics.mean(quality_scores)

        if avg_quality < 0.7:
            insights.append(StatisticalInsight(
                insight_id="overall_quality_concern",
                insight_type=AnalysisType.DIAGNOSTIC,
                pattern_type=DataPattern.DISTRIBUTION,
                title="Schema-wide Quality Concerns",
                description=f"Average data quality score ({avg_quality:.2f}) indicates systematic quality issues "
                            f"across {len(profiles)} tables.",
                confidence=0.9,
                significance="high" if avg_quality < 0.5 else "medium",
                tables_involved=[p.table_name for p in profiles],
                statistical_measures={"average_quality": avg_quality},
                recommendations=[
                    "Implement data quality monitoring",
                    "Establish data governance processes",
                    "Review data collection procedures"
                ]
            ))

        # Completeness patterns
        completeness_scores = [profile.overall_completeness for profile in profiles]
        if statistics.stdev(completeness_scores) > 0.3:  # High variation in completeness
            insights.append(StatisticalInsight(
                insight_id="completeness_inconsistency",
                insight_type=AnalysisType.DIAGNOSTIC,
                pattern_type=DataPattern.DEPENDENCY,
                title="Inconsistent Data Completeness",
                description="Large variation in data completeness across tables suggests "
                            "inconsistent data collection processes.",
                confidence=0.8,
                significance="medium",
                statistical_measures={"completeness_std": statistics.stdev(completeness_scores)},
                recommendations=[
                    "Standardize data collection processes",
                    "Implement mandatory field validation",
                    "Review ETL pipeline completeness"
                ]
            ))

        return insights

    def _detect_statistical_anomalies(self, profiles: List[TableProfile]) -> List[StatisticalInsight]:
        """Detect statistical anomalies in the data."""
        insights = []

        for profile in profiles:
            # Check for tables with unusual row counts
            if profile.total_rows == 0:
                insights.append(StatisticalInsight(
                    insight_id=f"empty_table_{profile.table_name}",
                    insight_type=AnalysisType.DIAGNOSTIC,
                    pattern_type=DataPattern.OUTLIER,
                    title=f"Empty Table Detected",
                    description=f"Table {profile.table_name} contains no data, which may indicate "
                                f"process failures or timing issues.",
                    confidence=1.0,
                    significance="high",
                    tables_involved=[profile.table_name],
                    recommendations=[
                        "Investigate data loading processes",
                        "Check for timing or scheduling issues",
                        "Verify source system connectivity"
                    ]
                ))

            # Check for columns with all null values
            all_null_columns = [col for col in profile.columns if col.null_count == col.total_rows]
            if all_null_columns:
                insights.append(StatisticalInsight(
                    insight_id=f"all_null_columns_{profile.table_name}",
                    insight_type=AnalysisType.DIAGNOSTIC,
                    pattern_type=DataPattern.OUTLIER,
                    title=f"All-Null Columns in {profile.table_name}",
                    description=f"Found {len(all_null_columns)} columns with 100% null values, "
                                f"indicating potential data source issues.",
                    confidence=0.9,
                    significance="medium",
                    tables_involved=[profile.table_name],
                    columns_involved=[col.column_name for col in all_null_columns],
                    recommendations=[
                        "Review source system data availability",
                        "Check ETL transformation logic",
                        "Consider removing unused columns"
                    ]
                ))

        return insights

    def _analyze_table_relationships(self, profiles: List[TableProfile]) -> List[StatisticalInsight]:
        """Analyze relationships between tables."""
        insights = []

        # Find potential foreign key relationships based on column names and patterns
        fk_candidates = {}
        for profile in profiles:
            for col in profile.columns:
                if col.is_likely_foreign_key():
                    base_name = col.column_name.replace('_SK', '').replace('_ID', '').replace('_KEY', '')
                    if base_name not in fk_candidates:
                        fk_candidates[base_name] = []
                    fk_candidates[base_name].append((profile.table_name, col.column_name))

        # Identify many-to-many relationships
        many_to_many = [base for base, refs in fk_candidates.items() if len(refs) > 2]
        if many_to_many:
            insights.append(StatisticalInsight(
                insight_id="many_to_many_relationships",
                insight_type=AnalysisType.DESCRIPTIVE,
                pattern_type=DataPattern.DEPENDENCY,
                title="Complex Table Relationships Detected",
                description=f"Found {len(many_to_many)} entities with many-to-many relationships, "
                            f"indicating complex business domain modeling.",
                confidence=0.7,
                significance="medium",
                statistical_measures={"relationship_count": len(many_to_many)},
                recommendations=[
                    "Document table relationship diagrams",
                    "Consider relationship optimization for queries",
                    "Implement referential integrity constraints"
                ]
            ))

        return insights

    def _perform_cross_table_analysis(self, profiles: List[TableProfile]) -> Dict[str, Any]:
        """Perform analysis across multiple tables."""
        analysis = {
            "table_count": len(profiles),
            "total_rows": sum(profile.total_rows for profile in profiles),
            "total_columns": sum(profile.total_columns for profile in profiles),
            "size_distribution": self._analyze_table_sizes(profiles),
            "quality_distribution": self._analyze_quality_distribution(profiles),
            "data_type_distribution": self._analyze_data_types(profiles)
        }

        return analysis

    def _analyze_table_sizes(self, profiles: List[TableProfile]) -> Dict[str, Any]:
        """Analyze table size distribution."""
        sizes = [profile.total_rows for profile in profiles if profile.total_rows > 0]

        if not sizes:
            return {"status": "no_data"}

        return {
            "min_rows": min(sizes),
            "max_rows": max(sizes),
            "avg_rows": statistics.mean(sizes),
            "median_rows": statistics.median(sizes),
            "large_tables": len([s for s in sizes if s > 1000000]),
            "small_tables": len([s for s in sizes if s < 10000])
        }

    def _analyze_quality_distribution(self, profiles: List[TableProfile]) -> Dict[str, Any]:
        """Analyze quality score distribution."""
        scores = [profile.overall_quality_score for profile in profiles]

        return {
            "avg_quality": statistics.mean(scores),
            "min_quality": min(scores),
            "max_quality": max(scores),
            "high_quality_tables": len([s for s in scores if s > 0.8]),
            "low_quality_tables": len([s for s in scores if s < 0.5])
        }

    def _analyze_data_types(self, profiles: List[TableProfile]) -> Dict[str, Any]:
        """Analyze data type distribution across all columns."""
        type_counts = {}

        for profile in profiles:
            for col in profile.columns:
                data_type = col.inferred_type.value
                type_counts[data_type] = type_counts.get(data_type, 0) + 1

        total_columns = sum(type_counts.values())

        return {
            "type_counts": type_counts,
            "type_percentages": {
                dtype: (count / total_columns) * 100
                for dtype, count in type_counts.items()
            } if total_columns > 0 else {}
        }

    def _generate_data_science_recommendations(self, profiles: List[TableProfile],
                                               scorecard: DataQualityScorecard,
                                               insights: List[StatisticalInsight]) -> List[str]:
        """Generate data science recommendations."""
        recommendations = []

        # Quality-based recommendations
        if scorecard.overall_score < 0.7:
            recommendations.append(
                f"Implement comprehensive data quality improvement program - "
                f"current score ({scorecard.overall_score:.2f}) indicates systematic issues"
            )

        # Statistical recommendations
        high_confidence_insights = [i for i in insights if i.confidence > 0.8]
        if len(high_confidence_insights) > 3:
            recommendations.append(
                f"Prioritize addressing {len(high_confidence_insights)} high-confidence statistical findings"
            )

        # Scale recommendations
        total_rows = sum(profile.total_rows for profile in profiles)
        if total_rows > 100000000:  # 100M+ rows
            recommendations.append(
                "Consider implementing data sampling strategies for statistical analysis on large datasets"
            )

        # Pattern-based recommendations
        outlier_insights = [i for i in insights if i.pattern_type == DataPattern.OUTLIER]
        if outlier_insights:
            recommendations.append(
                "Implement automated outlier detection and handling procedures"
            )

        return recommendations

    def _assess_data_quality(self, table_profiles: List[TableProfile]) -> Dict[str, Any]:
        """Assess data quality from table profiles."""
        scorecard = self._create_quality_scorecard(table_profiles, "unknown", "unknown")

        return {
            "status": "success",
            "quality_scorecard": {
                "overall_score": scorecard.overall_score,
                "grade": scorecard.get_grade(),
                "dimension_scores": scorecard.dimension_scores,
                "critical_issues": scorecard.critical_issues,
                "improvement_opportunities": scorecard.improvement_opportunities
            }
        }

    def _detect_data_patterns(self, table_profiles: List[TableProfile]) -> Dict[str, Any]:
        """Detect patterns in the data."""
        insights = self._generate_statistical_insights(table_profiles)

        pattern_summary = {}
        for pattern_type in DataPattern:
            pattern_insights = [i for i in insights if i.pattern_type == pattern_type]
            pattern_summary[pattern_type.value] = len(pattern_insights)

        return {
            "status": "success",
            "patterns_detected": pattern_summary,
            "total_insights": len(insights),
            "high_confidence_patterns": len([i for i in insights if i.confidence > 0.8])
        }


# Testing and demonstration
if __name__ == "__main__":
    print("Testing Data Scientist Agent")
    print("=" * 50)

    try:
        # Create Data Scientist Agent
        print("Creating Data Scientist Agent...")
        agent = DataScientistAgent()

        print(f"Agent created: {agent.agent_id}")
        print(f"   Name: {agent.agent_name}")
        print(f"   Capabilities: {', '.join(agent.get_capabilities())}")

        # Sample technical analysis (simulating Technical Analyst output)
        print(f"\nCreating sample technical analysis...")
        sample_technical_analysis = {
            "analysis": {
                "database": "SNOWFLAKE_SAMPLE_DATA",
                "schema_name": "TPCDS_SF10TCL",
                "tables_analyzed": 4
            },
            "table_summary": [
                {"name": "CUSTOMER", "business_domain": "dimension", "row_count": 65000000},
                {"name": "ITEM", "business_domain": "dimension", "row_count": 462000},
                {"name": "DATE_DIM", "business_domain": "dimension", "row_count": 73049},
                {"name": "CALL_CENTER", "business_domain": "dimension", "row_count": 54}
            ]
        }

        print(f"Sample analysis created with {len(sample_technical_analysis['table_summary'])} tables")

        # Test 1: Profile Schema Tables
        print(f"\nTest 1: Profile Schema Tables")
        table_names = [table["name"] for table in sample_technical_analysis["table_summary"]]

        profiling_task_id = agent.add_task(
            task_type="profile_schema",
            description="Profile multiple tables in schema",
            parameters={
                "database": "SNOWFLAKE_SAMPLE_DATA",
                "schema_name": "TPCDS_SF10TCL",
                "table_list": table_names[:3]  # Limit to 3 tables for testing
            },
            priority=TaskPriority.HIGH
        )

        profiling_result = agent.run_next_task()
        print(f"Schema profiling completed")
        print(f"   Profiles created: {profiling_result['profiles_created']}")
        print(f"   Failed tables: {profiling_result['failed_tables']}")

        # Show profile summary
        print(f"\n   Profile summary:")
        for profile_summary in profiling_result['profile_summary']:
            print(f"      {profile_summary['table_name']}: "
                  f"{profile_summary['total_rows']:,} rows, "
                  f"Quality {profile_summary['quality_score']:.2f}, "
                  f"{profile_summary['profiling_duration']:.1f}s")

        # Test 2: Comprehensive Analysis
        print(f"\nTest 2: Comprehensive Data Science Analysis")
        analysis_task_id = agent.add_task(
            task_type="comprehensive_analysis",
            description="Perform comprehensive data science analysis",
            parameters={
                "technical_analysis": sample_technical_analysis
            },
            priority=TaskPriority.HIGH
        )

        analysis_result = agent.run_next_task()
        print(f"Comprehensive analysis completed")
        print(f"   Tables analyzed: {analysis_result['report']['tables_analyzed']}")
        print(f"   Analysis duration: {analysis_result['report']['analysis_duration_seconds']:.2f}s")
        print(f"   Quality grade: {analysis_result['report']['quality_grade']}")
        print(f"   Overall quality score: {analysis_result['report']['overall_quality_score']:.2f}")

        # Show insights summary
        print(f"\n   Statistical insights:")
        print(f"      Total insights: {analysis_result['insights_summary']['total_insights']}")
        print(f"      High confidence: {analysis_result['insights_summary']['high_confidence']}")
        print(f"      Critical insights: {analysis_result['insights_summary']['critical_insights']}")
        print(f"      Patterns detected: {analysis_result['insights_summary']['patterns_detected']}")

        # Show quality summary
        print(f"\n   Quality assessment:")
        print(f"      Overall score: {analysis_result['quality_summary']['overall_score']:.2f}")
        print(f"      Grade: {analysis_result['quality_summary']['grade']}")
        print(f"      Critical issues: {analysis_result['quality_summary']['critical_issues']}")
        print(f"      Improvement opportunities: {analysis_result['quality_summary']['improvement_opportunities']}")

        print(f"      Recommendations: {analysis_result['recommendations']}")

        # Test 3: Quality Assessment
        print(f"\nTest 3: Data Quality Assessment")

        # Get cached profiles from previous test
        cached_profiles = agent.get_cached_result(profiling_result['cache_key'])

        agent.add_task(
            task_type="quality_assessment",
            description="Assess data quality from profiles",
            parameters={
                "table_profiles": cached_profiles
            },
            priority=TaskPriority.MEDIUM
        )

        quality_result = agent.run_next_task()
        print(f"Quality assessment completed")

        scorecard = quality_result['quality_scorecard']
        print(f"   Overall score: {scorecard['overall_score']:.2f}")
        print(f"   Grade: {scorecard['grade']}")

        if scorecard['dimension_scores']:
            print(f"   Dimension scores:")
            for dimension, score in scorecard['dimension_scores'].items():
                print(f"      {dimension}: {score:.2f}")

        # Show agent status and metrics
        print(f"\nAgent Status and Metrics:")
        status = agent.get_status()
        metrics = status['metrics']

        print(f"   Status: {status['status']}")
        print(f"   Tasks completed: {metrics['total_tasks_completed']}")
        print(f"   Tasks failed: {metrics['total_tasks_failed']}")
        print(f"   Success rate: {metrics['success_rate']:.1%}")
        print(f"   Average task time: {metrics['average_task_time_seconds']:.2f}s")

        print(f"\nData Scientist Agent tested successfully!")
        print(f"   Core capabilities: Data profiling, statistical analysis, quality assessment")
        print(f"   Scientific rigor: Pattern detection, anomaly identification, insights generation")
        print(f"   Integration: Data Profiler tool, Technical Analyst, comprehensive reporting")
        print(f"\nReady to complete the data discovery pipeline!")

    except Exception as e:
        print(f"Test failed: {str(e)}")
        agent.logger.error("Data Scientist Agent test failed", error=str(e))
        print(f"   Check that Data Profiler tool is working correctly")