# src/data_discovery/tools/query/sql_generator.py

"""
SQL Generator Tool for converting business questions into optimized SQL queries.

Transforms business question templates into executable SQL with performance optimizations,
parameter validation, and intelligent query planning based on statistical profiles.
"""

import time
import re
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Core system imports
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from tools.query.business_question_bank import BusinessQuestion, QuestionType, QuestionComplexity
from tools.analysis.data_profiler import TableProfile, ColumnProfile, DataType
from core.database_connector import get_connector
from core.state_manager import get_state_manager, StateType
from core.logging_config import get_logger
from core.config import get_config
from core.exceptions import DataDiscoveryException, QueryGenerationError, ErrorContext


class QueryStrategy(Enum):
    """Query execution strategies."""
    FULL_SCAN = "full_scan"  # Full table scan
    SAMPLED = "sampled"  # Statistical sampling
    INDEXED = "indexed"  # Use indexes when available
    PARTITIONED = "partitioned"  # Partition-aware queries
    CACHED = "cached"  # Use cached results


class QueryOptimization(Enum):
    """Query optimization techniques."""
    NONE = "none"
    BASIC = "basic"  # Basic optimizations
    ADVANCED = "advanced"  # Advanced optimizations
    AGGRESSIVE = "aggressive"  # All optimizations


@dataclass
class QueryParameter:
    """Query parameter with validation."""
    name: str
    value: Any
    data_type: str
    is_required: bool = True
    default_value: Any = None
    validation_pattern: Optional[str] = None

    def validate(self) -> bool:
        """Validate parameter value."""
        if self.is_required and self.value is None:
            return False

        if self.validation_pattern and self.value:
            return bool(re.match(self.validation_pattern, str(self.value)))

        return True


@dataclass
class QueryPlan:
    """Execution plan for a generated query."""
    query_id: str
    sql_query: str
    strategy: QueryStrategy
    optimization_level: QueryOptimization
    estimated_runtime_seconds: float
    estimated_rows_returned: int
    estimated_cost: float
    parameters: List[QueryParameter] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # Required tables
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def get_query_hash(self) -> str:
        """Get unique hash for the query."""
        query_content = f"{self.sql_query}_{self.strategy.value}_{self.optimization_level.value}"
        return hashlib.md5(query_content.encode()).hexdigest()[:8]


@dataclass
class GeneratedQuery:
    """Complete generated query with metadata."""
    query_id: str
    business_question: BusinessQuestion
    query_plan: QueryPlan
    alternative_plans: List[QueryPlan] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)
    generation_duration_seconds: float = 0.0

    def get_best_plan(self) -> QueryPlan:
        """Get the best execution plan."""
        return self.query_plan

    def get_fastest_plan(self) -> QueryPlan:
        """Get the fastest execution plan."""
        all_plans = [self.query_plan] + self.alternative_plans
        return min(all_plans, key=lambda p: p.estimated_runtime_seconds)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query_id": self.query_id,
            "business_question": self.business_question.to_dict(),
            "query_plan": {
                "query_id": self.query_plan.query_id,
                "sql_query": self.query_plan.sql_query,
                "strategy": self.query_plan.strategy.value,
                "optimization_level": self.query_plan.optimization_level.value,
                "estimated_runtime_seconds": self.query_plan.estimated_runtime_seconds,
                "estimated_rows_returned": self.query_plan.estimated_rows_returned,
                "estimated_cost": self.query_plan.estimated_cost,
                "warnings": self.query_plan.warnings,
                "recommendations": self.query_plan.recommendations
            },
            "alternative_plans_count": len(self.alternative_plans),
            "generated_at": self.generated_at.isoformat(),
            "generation_duration_seconds": self.generation_duration_seconds
        }


class SQLGenerator:
    """Tool for generating optimized SQL queries from business questions."""

    def __init__(self):
        self.config = get_config()
        self.connector = get_connector()
        self.state_manager = get_state_manager()
        self.logger = get_logger("sql_generator")

        # Configuration
        self.large_table_threshold = 1000000  # 1M rows
        self.default_sample_percent = 5.0
        self.max_query_timeout = 300  # 5 minutes
        self.enable_query_optimization = True

        # Query templates and patterns
        self.optimization_patterns = self._load_optimization_patterns()

        self.logger.info("SQL Generator initialized",
                         large_table_threshold=self.large_table_threshold,
                         enable_optimization=self.enable_query_optimization)

    def generate_sql(self, business_question: BusinessQuestion,
                     table_profiles: List[TableProfile] = None,
                     optimization_level: QueryOptimization = QueryOptimization.BASIC,
                     force_refresh: bool = False) -> GeneratedQuery:
        """
        Generate optimized SQL from a business question.

        Args:
            business_question: Business question to convert to SQL
            table_profiles: Statistical profiles for optimization
            optimization_level: Level of optimization to apply
            force_refresh: Skip cache and force fresh generation

        Returns:
            GeneratedQuery with optimized SQL and execution plans
        """
        start_time = time.time()

        # Check cache first
        cache_key = f"generated_query_{business_question.question_id}_{optimization_level.value}"
        if not force_refresh:
            cached_query = self.state_manager.load_state(cache_key, StateType.GENERATED_QUERIES)
            if cached_query:
                self.logger.info("Using cached generated query",
                                 question_id=business_question.question_id)
                return self._dict_to_generated_query(cached_query)

        self.logger.info("Generating SQL from business question",
                         question_id=business_question.question_id,
                         question_type=business_question.question_type.value,
                         complexity=business_question.complexity.value)

        try:
            # Generate base SQL query
            base_sql = self._generate_base_sql(business_question)

            # Create query parameters
            parameters = self._extract_query_parameters(business_question, base_sql)

            # Choose optimal strategy
            strategy = self._choose_query_strategy(business_question, table_profiles)

            # Apply optimizations
            optimized_sql = self._apply_optimizations(
                base_sql, business_question, table_profiles, optimization_level
            )

            # Estimate performance
            performance_estimate = self._estimate_query_performance(
                optimized_sql, business_question, table_profiles
            )

            # Create primary query plan
            primary_plan = QueryPlan(
                query_id=f"{business_question.question_id}_primary",
                sql_query=optimized_sql,
                strategy=strategy,
                optimization_level=optimization_level,
                estimated_runtime_seconds=performance_estimate['runtime_seconds'],
                estimated_rows_returned=performance_estimate['rows_returned'],
                estimated_cost=performance_estimate['cost'],
                parameters=parameters,
                dependencies=business_question.tables_involved,
                warnings=performance_estimate.get('warnings', []),
                recommendations=performance_estimate.get('recommendations', [])
            )

            # Generate alternative plans
            alternative_plans = self._generate_alternative_plans(
                business_question, table_profiles, base_sql
            )

            # Create generated query
            generated_query = GeneratedQuery(
                query_id=f"query_{business_question.question_id}_{int(time.time())}",
                business_question=business_question,
                query_plan=primary_plan,
                alternative_plans=alternative_plans,
                generation_duration_seconds=time.time() - start_time
            )

            # Cache the result
            self.state_manager.save_state(
                cache_key,
                generated_query.to_dict(),
                StateType.GENERATED_QUERIES,
                ttl_hours=6
            )

            self.logger.info("SQL generation completed",
                             question_id=business_question.question_id,
                             strategy=strategy.value,
                             estimated_runtime=primary_plan.estimated_runtime_seconds,
                             alternative_plans=len(alternative_plans),
                             generation_duration=generated_query.generation_duration_seconds)

            return generated_query

        except Exception as e:
            self.logger.error("SQL generation failed",
                              question_id=business_question.question_id,
                              error=str(e))
            raise QueryGenerationError(
                f"Failed to generate SQL for question {business_question.question_id}: {str(e)}"
            )

    def _generate_base_sql(self, business_question: BusinessQuestion) -> str:
        """Generate base SQL from business question template."""
        # Use the SQL template from the business question
        base_sql = business_question.sql_query

        # Clean and format SQL
        base_sql = self._clean_sql(base_sql)

        # Validate SQL syntax
        if not self._validate_sql_syntax(base_sql):
            raise QueryGenerationError(f"Invalid SQL syntax in question template")

        return base_sql

    def _extract_query_parameters(self, business_question: BusinessQuestion,
                                  sql_query: str) -> List[QueryParameter]:
        """Extract and validate query parameters."""
        parameters = []

        # Find parameter placeholders in SQL
        param_pattern = r'\{(\w+)\}'
        param_matches = re.findall(param_pattern, sql_query)

        for param_name in param_matches:
            # Determine parameter type and validation
            param_type = self._infer_parameter_type(param_name)
            validation_pattern = self._get_validation_pattern(param_type)

            parameter = QueryParameter(
                name=param_name,
                value=None,  # Will be set during execution
                data_type=param_type,
                validation_pattern=validation_pattern
            )
            parameters.append(parameter)

        return parameters

    def _choose_query_strategy(self, business_question: BusinessQuestion,
                               table_profiles: List[TableProfile] = None) -> QueryStrategy:
        """Choose optimal query execution strategy."""
        if not table_profiles:
            return QueryStrategy.FULL_SCAN

        # Check table sizes
        max_rows = max((profile.total_rows for profile in table_profiles
                        if profile.table_name in business_question.tables_involved), default=0)

        if max_rows > 100000000:  # 100M+ rows
            return QueryStrategy.SAMPLED
        elif max_rows > self.large_table_threshold:
            return QueryStrategy.INDEXED
        else:
            return QueryStrategy.FULL_SCAN

    def _apply_optimizations(self, sql_query: str, business_question: BusinessQuestion,
                             table_profiles: List[TableProfile] = None,
                             optimization_level: QueryOptimization = QueryOptimization.BASIC) -> str:
        """Apply query optimizations based on statistical profiles."""
        if not self.enable_query_optimization or optimization_level == QueryOptimization.NONE:
            return sql_query

        optimized_sql = sql_query

        # Basic optimizations
        if optimization_level.value in ['basic', 'advanced', 'aggressive']:
            optimized_sql = self._apply_basic_optimizations(optimized_sql, business_question)

        # Advanced optimizations
        if optimization_level.value in ['advanced', 'aggressive']:
            optimized_sql = self._apply_advanced_optimizations(
                optimized_sql, business_question, table_profiles
            )

        # Aggressive optimizations
        if optimization_level == QueryOptimization.AGGRESSIVE:
            optimized_sql = self._apply_aggressive_optimizations(
                optimized_sql, business_question, table_profiles
            )

        return optimized_sql

    def _apply_basic_optimizations(self, sql_query: str,
                                   business_question: BusinessQuestion) -> str:
        """Apply basic SQL optimizations."""
        optimized_sql = sql_query

        # Add query hints for Snowflake
        if 'ORDER BY' in optimized_sql.upper() and 'LIMIT' in optimized_sql.upper():
            # For top-N queries, use TOP instead of ORDER BY + LIMIT when possible
            if business_question.question_type == QuestionType.TOP_N:
                optimized_sql = self._optimize_top_n_query(optimized_sql)

        # Optimize GROUP BY clauses
        optimized_sql = self._optimize_group_by(optimized_sql)

        return optimized_sql

    def _apply_advanced_optimizations(self, sql_query: str,
                                      business_question: BusinessQuestion,
                                      table_profiles: List[TableProfile] = None) -> str:
        """Apply advanced optimizations based on statistics."""
        optimized_sql = sql_query

        if not table_profiles:
            return optimized_sql

        # Add sampling for large tables
        for profile in table_profiles:
            if (profile.table_name in business_question.tables_involved and
                    profile.total_rows > self.large_table_threshold):
                # Add SAMPLE clause for large tables
                table_pattern = rf'FROM\s+(?:\"[^\"]+\"\.){{0,2}}\"?{re.escape(profile.table_name)}\"?'

                sample_percent = self._calculate_sample_percent(profile.total_rows)
                replacement = f'FROM "{profile.database_name}"."{profile.schema_name}"."{profile.table_name}" SAMPLE ({sample_percent}%)'

                optimized_sql = re.sub(table_pattern, replacement, optimized_sql, flags=re.IGNORECASE)

        return optimized_sql

    def _apply_aggressive_optimizations(self, sql_query: str,
                                        business_question: BusinessQuestion,
                                        table_profiles: List[TableProfile] = None) -> str:
        """Apply aggressive optimizations for maximum performance."""
        optimized_sql = sql_query

        # Add result caching hint
        optimized_sql = f"-- CACHE RESULT\n{optimized_sql}"

        # Use approximate functions where possible
        if 'COUNT(*)' in optimized_sql.upper():
            # For very large tables, suggest approximate count
            for profile in table_profiles or []:
                if (profile.table_name in business_question.tables_involved and
                        profile.total_rows > 10000000):  # 10M+ rows

                    # Note: Keep exact count but add comment about approximate option
                    optimized_sql = optimized_sql.replace(
                        'COUNT(*)',
                        'COUNT(*) -- Consider APPROXIMATE_COUNT_DISTINCT for even faster results'
                    )

        return optimized_sql

    def _estimate_query_performance(self, sql_query: str,
                                    business_question: BusinessQuestion,
                                    table_profiles: List[TableProfile] = None) -> Dict[str, Any]:
        """Estimate query performance characteristics."""
        estimate = {
            'runtime_seconds': 30.0,  # Default estimate
            'rows_returned': 1000,
            'cost': 1.0,
            'warnings': [],
            'recommendations': []
        }

        if not table_profiles:
            return estimate

        # Calculate based on table sizes and query complexity
        total_rows = sum(profile.total_rows for profile in table_profiles
                         if profile.table_name in business_question.tables_involved)

        # Base runtime estimate
        if total_rows > 100000000:  # 100M+ rows
            estimate['runtime_seconds'] = 120.0
            estimate['warnings'].append("Query involves very large tables (100M+ rows)")
            estimate['recommendations'].append("Consider using sampling for faster results")
        elif total_rows > 10000000:  # 10M+ rows
            estimate['runtime_seconds'] = 60.0
        elif total_rows > 1000000:  # 1M+ rows
            estimate['runtime_seconds'] = 15.0
        else:
            estimate['runtime_seconds'] = 5.0

        # Adjust for query complexity
        complexity_multiplier = {
            QuestionComplexity.SIMPLE: 1.0,
            QuestionComplexity.INTERMEDIATE: 2.0,
            QuestionComplexity.ADVANCED: 4.0
        }

        multiplier = complexity_multiplier.get(business_question.complexity, 1.0)
        estimate['runtime_seconds'] *= multiplier

        # Estimate rows returned
        if business_question.question_type == QuestionType.COUNT:
            estimate['rows_returned'] = 1
        elif business_question.question_type == QuestionType.TOP_N:
            estimate['rows_returned'] = 10  # Default top-N
        else:
            estimate['rows_returned'] = min(1000, int(total_rows * 0.01))  # 1% sample

        # Cost estimate (relative)
        estimate['cost'] = estimate['runtime_seconds'] / 30.0

        return estimate

    def _generate_alternative_plans(self, business_question: BusinessQuestion,
                                    table_profiles: List[TableProfile],
                                    base_sql: str) -> List[QueryPlan]:
        """Generate alternative execution plans."""
        alternatives = []

        # Sampled version for large tables
        if table_profiles:
            max_rows = max((profile.total_rows for profile in table_profiles
                            if profile.table_name in business_question.tables_involved), default=0)

            if max_rows > self.large_table_threshold:
                sampled_sql = self._apply_advanced_optimizations(
                    base_sql, business_question, table_profiles
                )

                sampled_estimate = self._estimate_query_performance(
                    sampled_sql, business_question, table_profiles
                )

                alternatives.append(QueryPlan(
                    query_id=f"{business_question.question_id}_sampled",
                    sql_query=sampled_sql,
                    strategy=QueryStrategy.SAMPLED,
                    optimization_level=QueryOptimization.ADVANCED,
                    estimated_runtime_seconds=sampled_estimate['runtime_seconds'] * 0.3,  # 30% of full
                    estimated_rows_returned=sampled_estimate['rows_returned'],
                    estimated_cost=sampled_estimate['cost'] * 0.3,
                    dependencies=business_question.tables_involved,
                    recommendations=["Faster execution with statistical sampling"]
                ))

        return alternatives

    def _optimize_top_n_query(self, sql_query: str) -> str:
        """Optimize TOP-N queries."""
        # For Snowflake, ORDER BY + LIMIT is usually optimal
        return sql_query

    def _optimize_group_by(self, sql_query: str) -> str:
        """Optimize GROUP BY clauses."""
        # Add query planning hints
        if 'GROUP BY' in sql_query.upper():
            return f"-- Use columnar storage optimization\n{sql_query}"
        return sql_query

    def _calculate_sample_percent(self, total_rows: int) -> float:
        """Calculate appropriate sample percentage."""
        if total_rows > 1000000000:  # 1B+ rows
            return 1.0
        elif total_rows > 100000000:  # 100M+ rows
            return 2.0
        elif total_rows > 10000000:  # 10M+ rows
            return 5.0
        else:
            return 10.0

    def _clean_sql(self, sql_query: str) -> str:
        """Clean and format SQL query."""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', sql_query.strip())

        # Ensure proper formatting
        cleaned = cleaned.replace(',', ', ')
        cleaned = cleaned.replace('(', ' (')
        cleaned = cleaned.replace(')', ') ')

        # Clean up extra spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)

        return cleaned

    def _validate_sql_syntax(self, sql_query: str) -> bool:
        """Basic SQL syntax validation."""
        # Check for required keywords
        sql_upper = sql_query.upper()

        if not sql_upper.startswith('SELECT'):
            return False

        # Check for balanced parentheses
        if sql_query.count('(') != sql_query.count(')'):
            return False

        # Check for basic SQL structure
        required_patterns = [r'SELECT\s+', r'FROM\s+']
        for pattern in required_patterns:
            if not re.search(pattern, sql_upper):
                return False

        return True

    def _infer_parameter_type(self, param_name: str) -> str:
        """Infer parameter data type from name."""
        name_lower = param_name.lower()

        if any(keyword in name_lower for keyword in ['count', 'limit', 'top', 'n']):
            return 'INTEGER'
        elif any(keyword in name_lower for keyword in ['date', 'time']):
            return 'DATE'
        elif any(keyword in name_lower for keyword in ['amount', 'price', 'percent']):
            return 'DECIMAL'
        else:
            return 'VARCHAR'

    def _get_validation_pattern(self, param_type: str) -> Optional[str]:
        """Get regex validation pattern for parameter type."""
        patterns = {
            'INTEGER': r'^\d+$',
            'DECIMAL': r'^\d+(\.\d+)?$',
            'DATE': r'^\d{4}-\d{2}-\d{2}$',
            'VARCHAR': r'^.+$'
        }
        return patterns.get(param_type)

    def _load_optimization_patterns(self) -> Dict[str, Any]:
        """Load SQL optimization patterns."""
        return {
            'large_table_threshold': self.large_table_threshold,
            'sample_strategies': {
                'random': 'SAMPLE({}%)',
                'systematic': 'SAMPLE SYSTEM({}%)'
            },
            'index_hints': {
                'use_index': '/*+ USE_INDEX */',
                'no_index': '/*+ NO_INDEX_SCAN */'
            }
        }

    def _dict_to_generated_query(self, data: Dict[str, Any]) -> GeneratedQuery:
        """Convert dictionary back to GeneratedQuery object."""
        # This would implement the reverse conversion
        # For now, return a placeholder
        raise NotImplementedError("Cache deserialization not yet implemented")

    def validate_query(self, sql_query: str) -> Dict[str, Any]:
        """Validate SQL query without executing it."""
        validation_result = {
            'is_valid': True,
            'syntax_errors': [],
            'warnings': [],
            'recommendations': []
        }

        try:
            # Basic syntax validation
            if not self._validate_sql_syntax(sql_query):
                validation_result['is_valid'] = False
                validation_result['syntax_errors'].append("Invalid SQL syntax")

            # Check for potentially expensive operations
            sql_upper = sql_query.upper()

            if 'SELECT *' in sql_upper:
                validation_result['warnings'].append("Using SELECT * may be inefficient")
                validation_result['recommendations'].append("Specify only required columns")

            if 'CROSS JOIN' in sql_upper:
                validation_result['warnings'].append("CROSS JOIN detected - ensure this is intentional")

            if sql_upper.count('JOIN') > 5:
                validation_result['warnings'].append("Complex query with many joins")
                validation_result['recommendations'].append("Consider breaking into smaller queries")

        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['syntax_errors'].append(f"Validation error: {str(e)}")

        return validation_result

    def get_query_statistics(self, queries: List[GeneratedQuery]) -> Dict[str, Any]:
        """Get statistics about generated queries."""
        if not queries:
            return {"total_queries": 0}

        stats = {
            "total_queries": len(queries),
            "by_strategy": {},
            "by_complexity": {},
            "avg_generation_time": statistics.mean([q.generation_duration_seconds for q in queries]),
            "avg_estimated_runtime": statistics.mean([q.query_plan.estimated_runtime_seconds for q in queries]),
            "performance_distribution": {
                "fast": len([q for q in queries if q.query_plan.estimated_runtime_seconds < 10]),
                "medium": len([q for q in queries if 10 <= q.query_plan.estimated_runtime_seconds < 60]),
                "slow": len([q for q in queries if q.query_plan.estimated_runtime_seconds >= 60])
            }
        }

        # Strategy distribution
        for query in queries:
            strategy = query.query_plan.strategy.value
            stats["by_strategy"][strategy] = stats["by_strategy"].get(strategy, 0) + 1

        # Complexity distribution
        for query in queries:
            complexity = query.business_question.complexity.value
            stats["by_complexity"][complexity] = stats["by_complexity"].get(complexity, 0) + 1

        return stats


# Testing and demonstration
if __name__ == "__main__":
    print("Testing SQL Generator Tool")
    print("=" * 50)

    logger = get_logger("test_sql_generator")

    try:
        # Create SQL Generator
        print("🔧 Creating SQL Generator...")
        generator = SQLGenerator()

        print(f"✅ SQL Generator created")
        print(f"   Large table threshold: {generator.large_table_threshold:,}")
        print(f"   Default sample percent: {generator.default_sample_percent}%")
        print(f"   Query optimization: {'Enabled' if generator.enable_query_optimization else 'Disabled'}")

        # Create sample business questions
        print(f"\n📋 Creating sample business questions...")

        from tools.query.business_question_bank import BusinessQuestion, QuestionType, QuestionComplexity

        sample_questions = [
            BusinessQuestion(
                question_id="top_customers_revenue",
                question="Who are the top 10 customers by revenue?",
                sql_query="SELECT SS_CUSTOMER_SK, SUM(SS_SALES_PRICE) as total_revenue FROM STORE_SALES GROUP BY SS_CUSTOMER_SK ORDER BY total_revenue DESC LIMIT 10",
                question_type=QuestionType.TOP_N,
                complexity=QuestionComplexity.INTERMEDIATE,
                confidence=0.9,
                business_value="high",
                estimated_runtime="medium",
                tables_involved=["STORE_SALES"],
                columns_involved=["SS_CUSTOMER_SK", "SS_SALES_PRICE"]
            ),
            BusinessQuestion(
                question_id="total_sales_count",
                question="How many sales transactions do we have?",
                sql_query="SELECT COUNT(*) as total_transactions FROM STORE_SALES",
                question_type=QuestionType.COUNT,
                complexity=QuestionComplexity.SIMPLE,
                confidence=1.0,
                business_value="medium",
                estimated_runtime="fast",
                tables_involved=["STORE_SALES"],
                columns_involved=[]
            )
        ]

        print(f"✅ Sample questions created: {len(sample_questions)}")

        # Create sample table profiles for optimization
        print(f"\n📊 Creating sample table profiles...")

        from tools.analysis.data_profiler import TableProfile, ColumnProfile, DataType

        sample_profiles = [
            TableProfile(
                table_name="STORE_SALES",
                schema_name="TPCDS_SF10TCL",
                database_name="SNOWFLAKE_SAMPLE_DATA",
                total_rows=28800239865,  # 28.8B rows - very large
                total_columns=23,
                overall_quality_score=0.8
            ),
            TableProfile(
                table_name="CUSTOMER",
                schema_name="TPCDS_SF10TCL",
                database_name="SNOWFLAKE_SAMPLE_DATA",
                total_rows=65000000,  # 65M rows - large
                total_columns=18,
                overall_quality_score=0.9
            )
        ]

        print(f"✅ Table profiles created: {len(sample_profiles)}")

        # Test 1: Generate SQL for Top-N Query
        print(f"\n🔍 Test 1: Generate SQL for Top Customers Query")

        top_customers_question = sample_questions[0]

        generated_query = generator.generate_sql(
            business_question=top_customers_question,
            table_profiles=sample_profiles,
            optimization_level=QueryOptimization.ADVANCED
        )

        print(f"✅ SQL generation completed")
        print(f"   Query ID: {generated_query.query_id}")
        print(f"   Strategy: {generated_query.query_plan.strategy.value}")
        print(f"   Optimization: {generated_query.query_plan.optimization_level.value}")
        print(f"   Estimated runtime: {generated_query.query_plan.estimated_runtime_seconds:.1f}s")
        print(f"   Estimated rows: {generated_query.query_plan.estimated_rows_returned:,}")
        print(f"   Generation time: {generated_query.generation_duration_seconds:.3f}s")

        print(f"\n   Generated SQL:")
        print(f"   {generated_query.query_plan.sql_query}")

        if generated_query.query_plan.warnings:
            print(f"\n   Warnings:")
            for warning in generated_query.query_plan.warnings:
                print(f"      - {warning}")

        if generated_query.query_plan.recommendations:
            print(f"   Recommendations:")
            for rec in generated_query.query_plan.recommendations:
                print(f"      - {rec}")

        # Test 2: Generate SQL for Count Query
        print(f"\n🔍 Test 2: Generate SQL for Count Query")

        count_question = sample_questions[1]

        count_generated_query = generator.generate_sql(
            business_question=count_question,
            table_profiles=sample_profiles,
            optimization_level=QueryOptimization.AGGRESSIVE
        )

        print(f"✅ Count query generation completed")
        print(f"   Strategy: {count_generated_query.query_plan.strategy.value}")
        print(f"   Estimated runtime: {count_generated_query.query_plan.estimated_runtime_seconds:.1f}s")
        print(f"   Alternative plans: {len(count_generated_query.alternative_plans)}")

        print(f"\n   Generated SQL:")
        print(f"   {count_generated_query.query_plan.sql_query}")

        # Test 3: Query Validation
        print(f"\n🔍 Test 3: Query Validation")

        test_queries = [
            "SELECT COUNT(*) FROM STORE_SALES",  # Valid
            "SELECT * FROM",  # Invalid - incomplete
            "SELECT SS_CUSTOMER_SK, SUM(SS_SALES_PRICE) FROM STORE_SALES GROUP BY SS_CUSTOMER_SK"  # Valid
        ]

        for i, test_sql in enumerate(test_queries, 1):
            validation = generator.validate_query(test_sql)
            status = "✅ Valid" if validation['is_valid'] else "❌ Invalid"
            print(f"   Query {i}: {status}")

            if validation['syntax_errors']:
                print(f"      Errors: {', '.join(validation['syntax_errors'])}")
            if validation['warnings']:
                print(f"      Warnings: {', '.join(validation['warnings'])}")

        # Test 4: Query Statistics
        print(f"\n🔍 Test 4: Query Statistics")

        all_queries = [generated_query, count_generated_query]
        stats = generator.get_query_statistics(all_queries)

        print(f"✅ Query statistics calculated")
        print(f"   Total queries: {stats['total_queries']}")
        print(f"   Average generation time: {stats['avg_generation_time']:.3f}s")
        print(f"   Average estimated runtime: {stats['avg_estimated_runtime']:.1f}s")

        print(f"\n   Performance distribution:")
        perf_dist = stats['performance_distribution']
        print(f"      Fast (<10s): {perf_dist['fast']} queries")
        print(f"      Medium (10-60s): {perf_dist['medium']} queries")
        print(f"      Slow (>60s): {perf_dist['slow']} queries")

        print(f"\n   Strategy distribution:")
        for strategy, count in stats['by_strategy'].items():
            print(f"      {strategy}: {count} queries")

        print(f"\n✅ SQL Generator tool tested successfully!")
        print(f"   Features: SQL generation, optimization, validation, performance estimation")
        print(f"   Integration: Business questions, table profiles, database connector")
        print(f"   Intelligence: Strategy selection, sampling, caching")
        print(f"\n🚀 Ready to support Query Specialist Agent!")

    except Exception as e:
        logger.error("SQL Generator test failed", error=str(e))
        print(f"❌ Test failed: {str(e)}")
        print(f"   Check that business question templates and table profiles are correct")