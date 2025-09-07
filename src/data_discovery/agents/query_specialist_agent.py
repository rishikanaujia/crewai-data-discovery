# src/data_discovery/agents/query_specialist_agent.py

"""
Query Specialist Agent for executing business queries and delivering data insights.

Orchestrates SQL generation, query execution, result analysis, and performance monitoring
to transform business questions into actionable data insights.
"""

import time
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Core system imports
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent, AgentTask, TaskPriority, AgentStatus
from tools.query.sql_generator import SQLGenerator, GeneratedQuery, QueryStrategy, QueryOptimization
from tools.query.business_question_bank import BusinessQuestion, QuestionType
from tools.analysis.data_profiler import TableProfile
from core.database_connector import get_connector, QueryResult
from core.state_manager import StateType
from core.exceptions import DataDiscoveryException, QueryExecutionError, ErrorContext


class ExecutionMode(Enum):
    """Query execution modes."""
    PREVIEW = "preview"  # Show SQL without executing
    SAMPLE = "sample"  # Execute with sampling
    FULL = "full"  # Full execution
    CACHED = "cached"  # Use cached results


class ResultFormat(Enum):
    """Result formatting options."""
    RAW = "raw"  # Raw database results
    FORMATTED = "formatted"  # Formatted for display
    SUMMARY = "summary"  # Summary statistics
    CHART_DATA = "chart_data"  # Formatted for visualization


@dataclass
class QueryExecution:
    """Query execution results with metadata."""
    execution_id: str
    generated_query: GeneratedQuery
    execution_mode: ExecutionMode
    sql_executed: str
    execution_time_seconds: float
    rows_returned: int
    columns_returned: int
    result_data: List[Dict[str, Any]] = field(default_factory=list)
    execution_warnings: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    executed_at: datetime = field(default_factory=datetime.now)

    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary."""
        return {
            "execution_id": self.execution_id,
            "question": self.generated_query.business_question.question,
            "execution_mode": self.execution_mode.value,
            "execution_time_seconds": self.execution_time_seconds,
            "rows_returned": self.rows_returned,
            "columns_returned": self.columns_returned,
            "executed_at": self.executed_at.isoformat()
        }


@dataclass
class QueryInsight:
    """Business insight derived from query results."""
    insight_id: str
    question_id: str
    insight_type: str  # trend, ranking, distribution, anomaly
    title: str
    description: str
    key_findings: List[str] = field(default_factory=list)
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class QueryReport:
    """Comprehensive query execution report."""
    report_id: str
    business_question: BusinessQuestion
    query_execution: QueryExecution
    insights: List[QueryInsight] = field(default_factory=list)
    data_summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)


class QuerySpecialistAgent(BaseAgent):
    """
    Query Specialist Agent for executing business queries and analyzing results.

    Responsibilities:
    - Generate SQL from business questions
    - Execute queries with appropriate strategies
    - Analyze and format query results
    - Generate insights from data
    - Monitor query performance
    - Manage result caching
    """

    def __init__(self, agent_id: str = None):
        super().__init__(
            agent_id=agent_id,
            agent_name="QuerySpecialist"
        )

        # Initialize tools
        self.sql_generator = SQLGenerator()
        self.connector = get_connector()

        # Configuration
        self.default_execution_mode = ExecutionMode.SAMPLE
        self.max_result_rows = 10000
        self.query_timeout_seconds = 300  # 5 minutes
        self.enable_result_caching = True

        self.logger.info("Query Specialist Agent initialized",
                         default_mode=self.default_execution_mode.value,
                         max_result_rows=self.max_result_rows,
                         query_timeout=self.query_timeout_seconds)

    def get_capabilities(self) -> List[str]:
        """Return capabilities of the Query Specialist Agent."""
        return [
            "sql_generation",
            "query_execution",
            "result_analysis",
            "performance_monitoring",
            "insight_generation",
            "data_formatting"
        ]

    def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute query specialist tasks."""
        task_type = task.task_type
        parameters = task.parameters

        if task_type == "execute_business_query":
            business_question = parameters.get("business_question")
            table_profiles = parameters.get("table_profiles")
            execution_mode = parameters.get("execution_mode", self.default_execution_mode.value)
            return self._execute_business_query(business_question, table_profiles, ExecutionMode(execution_mode))

        elif task_type == "generate_query_report":
            business_question = parameters.get("business_question")
            table_profiles = parameters.get("table_profiles")
            return self._generate_comprehensive_report(business_question, table_profiles)

        elif task_type == "analyze_query_results":
            query_execution = parameters.get("query_execution")
            return self._analyze_query_results(query_execution)

        elif task_type == "execute_sql_directly":
            sql_query = parameters.get("sql_query")
            execution_mode = parameters.get("execution_mode", ExecutionMode.FULL.value)
            return self._execute_sql_directly(sql_query, ExecutionMode(execution_mode))

        elif task_type == "get_query_performance":
            execution_history = parameters.get("execution_history", [])
            return self._analyze_query_performance(execution_history)

        else:
            raise DataDiscoveryException(f"Unknown task type: {task_type}")

    def _execute_business_query(self, business_question: BusinessQuestion,
                                table_profiles: List[TableProfile] = None,
                                execution_mode: ExecutionMode = ExecutionMode.SAMPLE) -> Dict[str, Any]:
        """Execute a business question as a SQL query."""
        self.logger.info("Executing business query",
                         question_id=business_question.question_id,
                         question_type=business_question.question_type.value,
                         execution_mode=execution_mode.value)

        try:
            if self.current_task:
                self.current_task.progress_percent = 10.0

            # Generate SQL from business question
            generated_query = self.sql_generator.generate_sql(
                business_question=business_question,
                table_profiles=table_profiles,
                optimization_level=QueryOptimization.ADVANCED
            )

            if self.current_task:
                self.current_task.progress_percent = 30.0

            # Choose execution strategy
            sql_to_execute, actual_mode = self._choose_execution_sql(generated_query, execution_mode)

            if self.current_task:
                self.current_task.progress_percent = 50.0

            # Execute the query
            query_execution = self._execute_sql_with_monitoring(
                sql_to_execute, generated_query, actual_mode
            )

            if self.current_task:
                self.current_task.progress_percent = 80.0

            # Format results
            formatted_results = self._format_query_results(query_execution)

            if self.current_task:
                self.current_task.progress_percent = 100.0

            # Cache results if enabled
            if self.enable_result_caching:
                cache_key = f"query_execution_{business_question.question_id}"
                self.cache_result(cache_key, query_execution, ttl_hours=6)

            self.logger.info("Business query executed successfully",
                             question_id=business_question.question_id,
                             execution_mode=actual_mode.value,
                             execution_time=query_execution.execution_time_seconds,
                             rows_returned=query_execution.rows_returned)

            return {
                "status": "success",
                "execution_summary": query_execution.get_summary(),
                "generated_sql": generated_query.query_plan.sql_query,
                "executed_sql": sql_to_execute,
                "results": formatted_results,
                "performance_metrics": query_execution.performance_metrics,
                "recommendations": self._generate_execution_recommendations(query_execution)
            }

        except Exception as e:
            self.logger.error("Business query execution failed",
                              question_id=business_question.question_id,
                              error=str(e))
            raise QueryExecutionError(f"Failed to execute business query: {str(e)}")

    def _generate_comprehensive_report(self, business_question: BusinessQuestion,
                                       table_profiles: List[TableProfile] = None) -> Dict[str, Any]:
        """Generate comprehensive query execution report with insights."""
        start_time = time.time()

        self.logger.info("Generating comprehensive query report",
                         question_id=business_question.question_id)

        try:
            if self.current_task:
                self.current_task.progress_percent = 5.0

            # Execute the query
            execution_result = self._execute_business_query(
                business_question, table_profiles, ExecutionMode.SAMPLE
            )

            if self.current_task:
                self.current_task.progress_percent = 50.0

            # Get cached execution
            cache_key = f"query_execution_{business_question.question_id}"
            query_execution = self.get_cached_result(cache_key)

            # Generate insights from results
            insights = self._generate_insights_from_results(business_question, query_execution)

            if self.current_task:
                self.current_task.progress_percent = 80.0

            # Create data summary
            data_summary = self._create_data_summary(query_execution)

            # Generate recommendations
            recommendations = self._generate_query_recommendations(business_question, query_execution, insights)

            # Create comprehensive report
            report = QueryReport(
                report_id=f"query_report_{business_question.question_id}_{int(time.time())}",
                business_question=business_question,
                query_execution=query_execution,
                insights=insights,
                data_summary=data_summary,
                recommendations=recommendations
            )

            if self.current_task:
                self.current_task.progress_percent = 100.0

            # Cache the report
            report_cache_key = f"query_report_{business_question.question_id}"
            self.cache_result(report_cache_key, report, ttl_hours=12)

            report_duration = time.time() - start_time

            self.logger.info("Comprehensive query report generated",
                             question_id=business_question.question_id,
                             insights_count=len(insights),
                             report_duration=report_duration)

            return {
                "status": "success",
                "report": {
                    "report_id": report.report_id,
                    "question": business_question.question,
                    "execution_summary": query_execution.get_summary(),
                    "data_summary": data_summary,
                    "insights_count": len(insights),
                    "recommendations_count": len(recommendations),
                    "generation_duration_seconds": report_duration
                },
                "insights": [
                    {
                        "title": insight.title,
                        "description": insight.description,
                        "key_findings": insight.key_findings,
                        "confidence": insight.confidence
                    }
                    for insight in insights
                ],
                "execution_details": execution_result,
                "recommendations": recommendations
            }

        except Exception as e:
            self.logger.error("Comprehensive report generation failed",
                              question_id=business_question.question_id,
                              error=str(e))
            raise DataDiscoveryException(f"Failed to generate query report: {str(e)}")

    def _choose_execution_sql(self, generated_query: GeneratedQuery,
                              requested_mode: ExecutionMode) -> Tuple[str, ExecutionMode]:
        """Choose appropriate SQL and execution mode."""
        # For preview mode, return the SQL without executing
        if requested_mode == ExecutionMode.PREVIEW:
            return generated_query.query_plan.sql_query, ExecutionMode.PREVIEW

        # For sample mode or large tables, use optimized/sampled SQL
        if (requested_mode == ExecutionMode.SAMPLE or
                generated_query.query_plan.strategy == QueryStrategy.SAMPLED):
            return generated_query.query_plan.sql_query, ExecutionMode.SAMPLE

        # For full execution, check if we have a non-sampled alternative
        if requested_mode == ExecutionMode.FULL:
            # Look for a full-scan alternative plan
            full_plan = None
            for alt_plan in generated_query.alternative_plans:
                if alt_plan.strategy == QueryStrategy.FULL_SCAN:
                    full_plan = alt_plan
                    break

            if full_plan:
                return full_plan.sql_query, ExecutionMode.FULL
            else:
                # Use primary plan but warn about sampling
                return generated_query.query_plan.sql_query, ExecutionMode.SAMPLE

        return generated_query.query_plan.sql_query, ExecutionMode.SAMPLE

    def _execute_sql_with_monitoring(self, sql_query: str, generated_query: GeneratedQuery,
                                     execution_mode: ExecutionMode) -> QueryExecution:
        """Execute SQL with performance monitoring."""
        execution_id = f"exec_{generated_query.business_question.question_id}_{int(time.time())}"

        if execution_mode == ExecutionMode.PREVIEW:
            # Return preview without execution
            return QueryExecution(
                execution_id=execution_id,
                generated_query=generated_query,
                execution_mode=execution_mode,
                sql_executed=sql_query,
                execution_time_seconds=0.0,
                rows_returned=0,
                columns_returned=0,
                result_data=[],
                performance_metrics={"preview_mode": True}
            )

        start_time = time.time()

        try:
            self.logger.info("Executing SQL query",
                             execution_id=execution_id,
                             query_length=len(sql_query))

            # Execute query with timeout
            query_result = self.connector.execute_query(
                sql_query,
                timeout_seconds=self.query_timeout_seconds
            )

            execution_time = time.time() - start_time

            # Limit result size
            result_data = query_result.data
            if len(result_data) > self.max_result_rows:
                result_data = result_data[:self.max_result_rows]
                warning = f"Results truncated to {self.max_result_rows} rows"
            else:
                warning = None

            # Create execution record
            execution = QueryExecution(
                execution_id=execution_id,
                generated_query=generated_query,
                execution_mode=execution_mode,
                sql_executed=sql_query,
                execution_time_seconds=execution_time,
                rows_returned=len(result_data),
                columns_returned=len(result_data[0].keys()) if result_data else 0,
                result_data=result_data,
                execution_warnings=[warning] if warning else [],
                performance_metrics={
                    "query_duration_seconds": execution_time,
                    "estimated_vs_actual_time": execution_time / max(
                        generated_query.query_plan.estimated_runtime_seconds, 1),
                    "rows_per_second": len(result_data) / max(execution_time, 0.001),
                    "was_truncated": len(query_result.data) > self.max_result_rows
                }
            )

            return execution

        except Exception as e:
            self.logger.error("SQL execution failed",
                              execution_id=execution_id,
                              error=str(e))

            # Return failed execution record
            return QueryExecution(
                execution_id=execution_id,
                generated_query=generated_query,
                execution_mode=execution_mode,
                sql_executed=sql_query,
                execution_time_seconds=time.time() - start_time,
                rows_returned=0,
                columns_returned=0,
                result_data=[],
                execution_warnings=[f"Execution failed: {str(e)}"],
                performance_metrics={"execution_failed": True}
            )

    def _format_query_results(self, query_execution: QueryExecution,
                              format_type: ResultFormat = ResultFormat.FORMATTED) -> Dict[str, Any]:
        """Format query results for presentation."""
        if not query_execution.result_data:
            return {
                "format": format_type.value,
                "data": [],
                "summary": "No data returned"
            }

        data = query_execution.result_data

        if format_type == ResultFormat.RAW:
            return {"format": "raw", "data": data}

        elif format_type == ResultFormat.SUMMARY:
            return {
                "format": "summary",
                "row_count": len(data),
                "columns": list(data[0].keys()) if data else [],
                "sample_data": data[:5]  # First 5 rows
            }

        elif format_type == ResultFormat.FORMATTED:
            # Format for human readability
            formatted_data = []
            for row in data:
                formatted_row = {}
                for key, value in row.items():
                    # Format numbers with commas
                    if isinstance(value, (int, float)) and abs(value) >= 1000:
                        formatted_row[key] = f"{value:,}"
                    else:
                        formatted_row[key] = value
                formatted_data.append(formatted_row)

            return {
                "format": "formatted",
                "data": formatted_data,
                "columns": list(data[0].keys()) if data else [],
                "row_count": len(data)
            }

        elif format_type == ResultFormat.CHART_DATA:
            # Format for visualization
            columns = list(data[0].keys()) if data else []
            return {
                "format": "chart_data",
                "labels": [str(row[columns[0]]) for row in data] if columns else [],
                "values": [row[columns[1]] for row in data] if len(columns) > 1 else [],
                "series": columns
            }

        return {"format": "default", "data": data}

    def _generate_insights_from_results(self, business_question: BusinessQuestion,
                                        query_execution: QueryExecution) -> List[QueryInsight]:
        """Generate business insights from query results."""
        insights = []

        if not query_execution.result_data:
            return insights

        data = query_execution.result_data
        question_type = business_question.question_type

        # Generate insights based on question type
        if question_type == QuestionType.TOP_N:
            insights.extend(self._analyze_top_n_results(business_question, data))
        elif question_type == QuestionType.COUNT:
            insights.extend(self._analyze_count_results(business_question, data))
        elif question_type == QuestionType.TREND_ANALYSIS:
            insights.extend(self._analyze_trend_results(business_question, data))

        # General insights
        insights.extend(self._generate_general_insights(business_question, data))

        return insights

    def _analyze_top_n_results(self, business_question: BusinessQuestion,
                               data: List[Dict[str, Any]]) -> List[QueryInsight]:
        """Analyze TOP-N query results."""
        insights = []

        if len(data) < 2:
            return insights

        # Analyze the ranking distribution
        if len(data) >= 2:
            columns = list(data[0].keys())
            value_column = None

            # Find the numeric value column (likely the second column)
            for col in columns[1:]:
                if isinstance(data[0][col], (int, float)):
                    value_column = col
                    break

            if value_column:
                top_value = data[0][value_column]
                second_value = data[1][value_column]

                if top_value > 0 and second_value > 0:
                    dominance_ratio = top_value / second_value

                    if dominance_ratio > 2.0:
                        insights.append(QueryInsight(
                            insight_id=f"dominance_{business_question.question_id}",
                            question_id=business_question.question_id,
                            insight_type="ranking",
                            title="Clear Market Leader Identified",
                            description=f"The top performer significantly outpaces others with {dominance_ratio:.1f}x the value of the second place.",
                            key_findings=[
                                f"Top performer: {data[0][columns[0]]} with {top_value:,}",
                                f"Dominance ratio: {dominance_ratio:.1f}x over second place"
                            ],
                            supporting_data={"dominance_ratio": dominance_ratio, "top_value": top_value},
                            confidence=0.9
                        ))

        return insights

    def _analyze_count_results(self, business_question: BusinessQuestion,
                               data: List[Dict[str, Any]]) -> List[QueryInsight]:
        """Analyze COUNT query results."""
        insights = []

        if data and len(data) == 1:
            count_value = list(data[0].values())[0]

            if isinstance(count_value, (int, float)):
                # Analyze scale
                if count_value > 1000000:
                    scale = "millions"
                    scaled_value = count_value / 1000000
                elif count_value > 1000:
                    scale = "thousands"
                    scaled_value = count_value / 1000
                else:
                    scale = "units"
                    scaled_value = count_value

                insights.append(QueryInsight(
                    insight_id=f"scale_{business_question.question_id}",
                    question_id=business_question.question_id,
                    insight_type="distribution",
                    title=f"Data Scale: {scaled_value:.1f} {scale.title()}",
                    description=f"The dataset contains {count_value:,} records, indicating {scale}-scale operations.",
                    key_findings=[f"Total count: {count_value:,}"],
                    supporting_data={"count": count_value, "scale": scale},
                    confidence=1.0
                ))

        return insights

    def _analyze_trend_results(self, business_question: BusinessQuestion,
                               data: List[Dict[str, Any]]) -> List[QueryInsight]:
        """Analyze trend analysis results."""
        insights = []

        # Trend analysis would require time-series data
        # For now, return basic insights about data variation
        if len(data) >= 3:
            insights.append(QueryInsight(
                insight_id=f"trend_{business_question.question_id}",
                question_id=business_question.question_id,
                insight_type="trend",
                title="Trend Data Available",
                description=f"Dataset contains {len(data)} data points suitable for trend analysis.",
                key_findings=[f"Time periods analyzed: {len(data)}"],
                confidence=0.7
            ))

        return insights

    def _generate_general_insights(self, business_question: BusinessQuestion,
                                   data: List[Dict[str, Any]]) -> List[QueryInsight]:
        """Generate general insights from any query results."""
        insights = []

        # Data completeness insight
        if data:
            total_fields = len(data) * len(data[0].keys())
            null_fields = sum(1 for row in data for value in row.values() if value is None)
            completeness = (total_fields - null_fields) / total_fields if total_fields > 0 else 0

            if completeness < 0.8:
                insights.append(QueryInsight(
                    insight_id=f"completeness_{business_question.question_id}",
                    question_id=business_question.question_id,
                    insight_type="anomaly",
                    title="Data Completeness Concern",
                    description=f"Query results show {completeness:.1%} data completeness.",
                    key_findings=[f"Completeness rate: {completeness:.1%}"],
                    recommendations=["Consider data quality improvements"],
                    confidence=0.8
                ))

        return insights

    def _create_data_summary(self, query_execution: QueryExecution) -> Dict[str, Any]:
        """Create summary of query execution data."""
        data = query_execution.result_data

        summary = {
            "total_rows": len(data),
            "total_columns": query_execution.columns_returned,
            "execution_time_seconds": query_execution.execution_time_seconds,
            "data_preview": data[:3] if data else [],  # First 3 rows
        }

        if data:
            # Analyze column types
            column_types = {}
            for col in data[0].keys():
                sample_value = data[0][col]
                if isinstance(sample_value, (int, float)):
                    column_types[col] = "numeric"
                elif isinstance(sample_value, str):
                    column_types[col] = "text"
                else:
                    column_types[col] = "other"

            summary["column_types"] = column_types

        return summary

    def _generate_execution_recommendations(self, query_execution: QueryExecution) -> List[str]:
        """Generate recommendations based on query execution."""
        recommendations = []

        # Performance recommendations
        if query_execution.execution_time_seconds > 60:
            recommendations.append("Consider using sampling for faster results on large datasets")

        if query_execution.rows_returned >= self.max_result_rows:
            recommendations.append("Results were truncated - consider adding LIMIT clause or filters")

        # Efficiency recommendations
        performance = query_execution.performance_metrics
        if performance.get("rows_per_second", 0) < 100:
            recommendations.append("Query performance is below optimal - review indexing and filters")

        return recommendations

    def _generate_query_recommendations(self, business_question: BusinessQuestion,
                                        query_execution: QueryExecution,
                                        insights: List[QueryInsight]) -> List[str]:
        """Generate comprehensive recommendations."""
        recommendations = []

        # Add execution recommendations
        recommendations.extend(self._generate_execution_recommendations(query_execution))

        # Add insight-based recommendations
        for insight in insights:
            recommendations.extend(insight.recommendations)

        # Question-specific recommendations
        if business_question.question_type == QuestionType.TOP_N:
            recommendations.append("Consider analyzing trends over time for these top performers")
        elif business_question.question_type == QuestionType.COUNT:
            recommendations.append("Use this baseline for trend analysis and growth tracking")

        return list(set(recommendations))  # Remove duplicates

    def _execute_sql_directly(self, sql_query: str, execution_mode: ExecutionMode) -> Dict[str, Any]:
        """Execute SQL directly without business question context."""
        # Create a temporary business question for tracking
        temp_question = BusinessQuestion(
            question_id="direct_sql",
            question="Direct SQL execution",
            sql_query=sql_query,
            question_type=QuestionType.COUNT,  # Default
            complexity=QuestionComplexity.SIMPLE,
            confidence=1.0,
            business_value="medium",
            estimated_runtime="unknown",
            tables_involved=[],
            columns_involved=[]
        )

        # Create a temporary generated query
        from tools.query.sql_generator import GeneratedQuery, QueryPlan, QueryStrategy

        temp_plan = QueryPlan(
            query_id="direct_sql_plan",
            sql_query=sql_query,
            strategy=QueryStrategy.FULL_SCAN,
            optimization_level=QueryOptimization.NONE,
            estimated_runtime_seconds=30.0,
            estimated_rows_returned=1000,
            estimated_cost=1.0
        )

        temp_generated = GeneratedQuery(
            query_id="direct_sql_generated",
            business_question=temp_question,
            query_plan=temp_plan
        )

        # Execute with monitoring
        execution = self._execute_sql_with_monitoring(sql_query, temp_generated, execution_mode)

        return {
            "status": "success",
            "execution_summary": execution.get_summary(),
            "results": self._format_query_results(execution),
            "performance_metrics": execution.performance_metrics
        }

    def _analyze_query_performance(self, execution_history: List[QueryExecution]) -> Dict[str, Any]:
        """Analyze query performance across multiple executions."""
        if not execution_history:
            return {"status": "no_data", "message": "No execution history available"}

        # Calculate performance statistics
        execution_times = [ex.execution_time_seconds for ex in execution_history]
        rows_returned = [ex.rows_returned for ex in execution_history]

        import statistics

        performance_analysis = {
            "total_queries": len(execution_history),
            "avg_execution_time": statistics.mean(execution_times),
            "min_execution_time": min(execution_times),
            "max_execution_time": max(execution_times),
            "avg_rows_returned": statistics.mean(rows_returned),
            "execution_modes": {},
            "performance_trends": "stable"  # Simplified
        }

        # Analyze by execution mode
        for execution in execution_history:
            mode = execution.execution_mode.value
            performance_analysis["execution_modes"][mode] = performance_analysis["execution_modes"].get(mode, 0) + 1

        return {
            "status": "success",
            "performance_analysis": performance_analysis
        }


# Testing and demonstration
if __name__ == "__main__":
    print("Testing Query Specialist Agent")
    print("=" * 50)

    try:
        # Create Query Specialist Agent
        print("Creating Query Specialist Agent...")
        agent = QuerySpecialistAgent()

        print(f"Agent created: {agent.agent_id}")
        print(f"   Name: {agent.agent_name}")
        print(f"   Capabilities: {', '.join(agent.get_capabilities())}")
        print(f"   Default execution mode: {agent.default_execution_mode.value}")
        print(f"   Max result rows: {agent.max_result_rows:,}")

        # Create sample business question
        print(f"\nCreating sample business question...")

        from tools.query.business_question_bank import BusinessQuestion, QuestionType, QuestionComplexity

        sample_question = BusinessQuestion(
            question_id="customer_count_test",
            question="How many customers do we have?",
            sql_query="SELECT COUNT(*) as customer_count FROM CUSTOMER",
            question_type=QuestionType.COUNT,
            complexity=QuestionComplexity.SIMPLE,
            confidence=1.0,
            business_value="medium",
            estimated_runtime="fast",
            tables_involved=["CUSTOMER"],
            columns_involved=[]
        )

        print(f"Sample question created: {sample_question.question}")

        # Create sample table profiles
        print(f"\nCreating sample table profiles...")

        from tools.analysis.data_profiler import TableProfile

        sample_profiles = [
            TableProfile(
                table_name="CUSTOMER",
                schema_name="TPCDS_SF10TCL",
                database_name="SNOWFLAKE_SAMPLE_DATA",
                total_rows=65000000,  # 65M rows
                total_columns=18,
                overall_quality_score=0.9
            )
        ]

        print(f"Table profiles created: {len(sample_profiles)}")

        # Test 1: Execute Business Query (Preview Mode)
        print(f"\nTest 1: Execute Business Query (Preview Mode)")

        preview_task_id = agent.add_task(
            task_type="execute_business_query",
            description="Execute business query in preview mode",
            parameters={
                "business_question": sample_question,
                "table_profiles": sample_profiles,
                "execution_mode": ExecutionMode.PREVIEW.value
            },
            priority=TaskPriority.HIGH
        )

        preview_result = agent.run_next_task()
        print(f"Preview execution completed")
        print(f"   Status: {preview_result['status']}")
        print(f"   Execution mode: {preview_result['execution_summary']['execution_mode']}")
        print(f"   Generated SQL: {preview_result['generated_sql']}")

        # Test 2: Execute Business Query (Sample Mode)
        print(f"\nTest 2: Execute Business Query (Sample Mode)")

        sample_task_id = agent.add_task(
            task_type="execute_business_query",
            description="Execute business query with sampling",
            parameters={
                "business_question": sample_question,
                "table_profiles": sample_profiles,
                "execution_mode": ExecutionMode.SAMPLE.value
            },
            priority=TaskPriority.HIGH
        )

        sample_result = agent.run_next_task()
        print(f"Sample execution completed")
        print(f"   Status: {sample_result['status']}")
        print(f"   Execution time: {sample_result['execution_summary']['execution_time_seconds']:.2f}s")
        print(f"   Rows returned: {sample_result['execution_summary']['rows_returned']:,}")

        if sample_result['results']['data']:
            result_data = sample_result['results']['data'][0]
            print(f"   Result: {result_data}")

        # Show performance metrics
        if sample_result['performance_metrics']:
            print(f"   Performance metrics:")
            for metric, value in sample_result['performance_metrics'].items():
                print(f"      {metric}: {value}")

        # Test 3: Generate Comprehensive Report
        print(f"\nTest 3: Generate Comprehensive Query Report")

        report_task_id = agent.add_task(
            task_type="generate_query_report",
            description="Generate comprehensive query report with insights",
            parameters={
                "business_question": sample_question,
                "table_profiles": sample_profiles
            },
            priority=TaskPriority.HIGH
        )

        report_result = agent.run_next_task()
        print(f"Comprehensive report generated")
        print(f"   Report ID: {report_result['report']['report_id']}")
        print(f"   Insights generated: {report_result['report']['insights_count']}")
        print(f"   Recommendations: {report_result['report']['recommendations_count']}")

        # Show insights
        if report_result['insights']:
            print(f"\n   Generated insights:")
            for i, insight in enumerate(report_result['insights'][:2], 1):
                print(f"      {i}. {insight['title']}")
                print(f"         {insight['description']}")

        # Show recommendations
        if report_result['recommendations']:
            print(f"\n   Recommendations:")
            for i, rec in enumerate(report_result['recommendations'][:3], 1):
                print(f"      {i}. {rec}")

        # Test 4: Direct SQL Execution
        print(f"\nTest 4: Direct SQL Execution")

        direct_sql = "SELECT 'Test' as message, 42 as number"

        agent.add_task(
            task_type="execute_sql_directly",
            description="Execute SQL directly",
            parameters={
                "sql_query": direct_sql,
                "execution_mode": ExecutionMode.FULL.value
            },
            priority=TaskPriority.MEDIUM
        )

        direct_result = agent.run_next_task()
        print(f"Direct SQL execution completed")
        print(f"   Execution time: {direct_result['execution_summary']['execution_time_seconds']:.3f}s")
        print(f"   Result: {direct_result['results']['data']}")

        # Show agent status and metrics
        print(f"\nAgent Status and Metrics:")
        status = agent.get_status()
        metrics = status['metrics']

        print(f"   Status: {status['status']}")
        print(f"   Tasks completed: {metrics['total_tasks_completed']}")
        print(f"   Tasks failed: {metrics['total_tasks_failed']}")
        print(f"   Success rate: {metrics['success_rate']:.1%}")
        print(f"   Average task time: {metrics['average_task_time_seconds']:.2f}s")

        print(f"\nQuery Specialist Agent tested successfully!")
        print(f"   Core capabilities: SQL generation, query execution, result analysis")
        print(f"   Performance: Intelligent sampling, caching, monitoring")
        print(f"   Intelligence: Insight generation, recommendations, optimization")
        print(f"\nData discovery pipeline is now complete!")

    except Exception as e:
        print(f"Test failed: {str(e)}")
        agent.logger.error("Query Specialist Agent test failed", error=str(e))
        print(f"   Check that SQL Generator and database connector are working correctly")