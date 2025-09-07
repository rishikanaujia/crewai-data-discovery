# src/data_discovery/tools/query/business_question_bank.py

"""
Business Question Bank Tool for generating template-based business questions.

Transforms technical schema metadata into business-relevant questions using
domain-specific templates and intelligent question generation patterns.
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

# Core system imports
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from core.logging_config import get_logger
from core.state_manager import get_state_manager, StateType
from core.config import get_config
from core.exceptions import DataDiscoveryException


class QuestionType(Enum):
    """Types of business questions that can be generated."""
    TREND_ANALYSIS = "trend_analysis"
    TOP_N = "top_n"
    COMPARISON = "comparison"
    DISTRIBUTION = "distribution"
    ANOMALY = "anomaly"
    COUNT = "count"
    AGGREGATION = "aggregation"
    RELATIONSHIP = "relationship"


class QuestionComplexity(Enum):
    """Complexity levels for business questions."""
    SIMPLE = "simple"  # Single table, basic aggregation
    INTERMEDIATE = "intermediate"  # Multiple tables, joins
    ADVANCED = "advanced"  # Complex analytics, window functions


@dataclass
class QuestionTemplate:
    """Template for generating business questions."""
    template_id: str
    question_type: QuestionType
    complexity: QuestionComplexity
    question_template: str
    sql_template: str
    description: str
    required_columns: List[str] = field(default_factory=list)
    required_table_types: List[str] = field(default_factory=list)
    business_domains: List[str] = field(default_factory=list)

    def generate_question(self, **kwargs) -> str:
        """Generate actual question from template."""
        try:
            return self.question_template.format(**kwargs)
        except KeyError as e:
            raise DataDiscoveryException(f"Missing template parameter: {e}")


@dataclass
class BusinessQuestion:
    """Generated business question with metadata."""
    question_id: str
    question: str
    sql_query: str
    question_type: QuestionType
    complexity: QuestionComplexity
    confidence: float  # 0.0 - 1.0
    business_value: str  # low, medium, high
    estimated_runtime: str  # fast, medium, slow
    tables_involved: List[str] = field(default_factory=list)
    columns_involved: List[str] = field(default_factory=list)
    business_context: str = ""
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "question_id": self.question_id,
            "question": self.question,
            "sql_query": self.sql_query,
            "question_type": self.question_type.value,
            "complexity": self.complexity.value,
            "confidence": self.confidence,
            "business_value": self.business_value,
            "estimated_runtime": self.estimated_runtime,
            "tables_involved": self.tables_involved,
            "columns_involved": self.columns_involved,
            "business_context": self.business_context,
            "generated_at": self.generated_at.isoformat()
        }


class BusinessQuestionBank:
    """Tool for generating business questions from schema metadata."""

    def __init__(self):
        self.config = get_config()
        self.logger = get_logger("business_question_bank")
        self.state_manager = get_state_manager()

        # Load question templates
        self.templates = self._load_question_templates()

        # Configuration
        self.max_questions_per_category = self.config.analysis.max_questions_per_category
        self.min_confidence_threshold = self.config.analysis.min_confidence_threshold

        self.logger.info("Business Question Bank initialized",
                         template_count=len(self.templates),
                         max_questions_per_category=self.max_questions_per_category)

    def _load_question_templates(self) -> List[QuestionTemplate]:
        """Load predefined question templates."""
        templates = []

        # Trend Analysis Templates
        templates.extend([
            QuestionTemplate(
                template_id="trend_sales_monthly",
                question_type=QuestionType.TREND_ANALYSIS,
                complexity=QuestionComplexity.SIMPLE,
                question_template="What is the monthly {metric} trend for {entity}?",
                sql_template="SELECT DATE_TRUNC('month', {date_column}) as month, SUM({metric_column}) as total_{metric} FROM {table} GROUP BY 1 ORDER BY 1",
                description="Monthly trend analysis for sales metrics",
                required_columns=["date_column", "metric_column"],
                required_table_types=["fact"],
                business_domains=["fact", "sales"]
            ),
            QuestionTemplate(
                template_id="trend_customer_growth",
                question_type=QuestionType.TREND_ANALYSIS,
                complexity=QuestionComplexity.INTERMEDIATE,
                question_template="How has customer acquisition changed over time?",
                sql_template="SELECT DATE_TRUNC('month', {date_column}) as month, COUNT(DISTINCT {customer_id}) as new_customers FROM {table} GROUP BY 1 ORDER BY 1",
                description="Customer acquisition trend over time",
                required_columns=["date_column", "customer_id"],
                business_domains=["dimension", "customer"]
            )
        ])

        # Top-N Templates
        templates.extend([
            QuestionTemplate(
                template_id="top_customers_revenue",
                question_type=QuestionType.TOP_N,
                complexity=QuestionComplexity.INTERMEDIATE,
                question_template="Who are the top {n} customers by {metric}?",
                sql_template="SELECT {customer_column}, SUM({metric_column}) as total_{metric} FROM {table} GROUP BY 1 ORDER BY 2 DESC LIMIT {n}",
                description="Top customers by revenue or other metrics",
                required_columns=["customer_column", "metric_column"],
                business_domains=["fact", "customer"]
            ),
            QuestionTemplate(
                template_id="top_products_sales",
                question_type=QuestionType.TOP_N,
                complexity=QuestionComplexity.SIMPLE,
                question_template="What are the top {n} best-selling {entity}?",
                sql_template="SELECT {entity_column}, SUM({quantity_column}) as total_quantity FROM {table} GROUP BY 1 ORDER BY 2 DESC LIMIT {n}",
                description="Top selling products or items",
                required_columns=["entity_column", "quantity_column"],
                business_domains=["fact", "product"]
            )
        ])

        # Count Templates
        templates.extend([
            QuestionTemplate(
                template_id="count_entities",
                question_type=QuestionType.COUNT,
                complexity=QuestionComplexity.SIMPLE,
                question_template="How many {entity} do we have?",
                sql_template="SELECT COUNT(*) as total_{entity} FROM {table}",
                description="Simple count of entities",
                required_columns=[],
                business_domains=["dimension", "fact"]
            ),
            QuestionTemplate(
                template_id="count_active_entities",
                question_type=QuestionType.COUNT,
                complexity=QuestionComplexity.SIMPLE,
                question_template="How many active {entity} are there?",
                sql_template="SELECT COUNT(*) as active_{entity} FROM {table} WHERE {status_column} = 'ACTIVE'",
                description="Count of active entities",
                required_columns=["status_column"],
                business_domains=["dimension"]
            )
        ])

        # Distribution Templates
        templates.extend([
            QuestionTemplate(
                template_id="distribution_by_category",
                question_type=QuestionType.DISTRIBUTION,
                complexity=QuestionComplexity.SIMPLE,
                question_template="What is the distribution of {entity} by {category}?",
                sql_template="SELECT {category_column}, COUNT(*) as count FROM {table} GROUP BY 1 ORDER BY 2 DESC",
                description="Distribution analysis by category",
                required_columns=["category_column"],
                business_domains=["dimension", "fact"]
            ),
            QuestionTemplate(
                template_id="distribution_revenue_by_region",
                question_type=QuestionType.DISTRIBUTION,
                complexity=QuestionComplexity.INTERMEDIATE,
                question_template="How is revenue distributed across different regions?",
                sql_template="SELECT {region_column}, SUM({revenue_column}) as total_revenue FROM {table} GROUP BY 1 ORDER BY 2 DESC",
                description="Revenue distribution by geographic regions",
                required_columns=["region_column", "revenue_column"],
                business_domains=["fact", "geographic"]
            )
        ])

        # Comparison Templates
        templates.extend([
            QuestionTemplate(
                template_id="compare_periods",
                question_type=QuestionType.COMPARISON,
                complexity=QuestionComplexity.ADVANCED,
                question_template="How does {metric} compare between {period1} and {period2}?",
                sql_template="""
                SELECT 
                    period,
                    SUM({metric_column}) as total_{metric}
                FROM (
                    SELECT '{period1}' as period, {metric_column} FROM {table} WHERE {date_column} BETWEEN '{start1}' AND '{end1}'
                    UNION ALL
                    SELECT '{period2}' as period, {metric_column} FROM {table} WHERE {date_column} BETWEEN '{start2}' AND '{end2}'
                ) GROUP BY 1
                """,
                description="Period-over-period comparison",
                required_columns=["metric_column", "date_column"],
                business_domains=["fact"]
            )
        ])

        return templates

    def generate_questions_for_schema(self, business_tables: List[Dict[str, Any]],
                                      max_questions: int = None) -> List[BusinessQuestion]:
        """Generate business questions for a schema with classified tables."""
        max_questions = max_questions or self.max_questions_per_category

        self.logger.info("Generating business questions",
                         table_count=len(business_tables),
                         max_questions=max_questions)

        questions = []

        # Group tables by business domain
        tables_by_domain = self._group_tables_by_domain(business_tables)

        # Generate questions for each domain
        for domain, tables in tables_by_domain.items():
            domain_questions = self._generate_questions_for_domain(domain, tables, max_questions)
            questions.extend(domain_questions)

            self.logger.debug("Generated questions for domain",
                              domain=domain,
                              table_count=len(tables),
                              question_count=len(domain_questions))

        # Sort by confidence and business value
        questions = self._rank_questions(questions)

        self.logger.info("Business questions generated",
                         total_questions=len(questions),
                         high_confidence=len([q for q in questions if q.confidence > 0.8]))

        return questions

    def _group_tables_by_domain(self, business_tables: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group tables by business domain."""
        tables_by_domain = {}

        for table in business_tables:
            domain = table.get('business_domain', 'unknown')
            if domain not in tables_by_domain:
                tables_by_domain[domain] = []
            tables_by_domain[domain].append(table)

        return tables_by_domain

    def _generate_questions_for_domain(self, domain: str, tables: List[Dict[str, Any]],
                                       max_questions: int) -> List[BusinessQuestion]:
        """Generate questions for a specific business domain."""
        questions = []

        # Find templates for this domain
        domain_templates = [t for t in self.templates if domain in t.business_domains or not t.business_domains]

        for table in tables:
            table_questions = self._generate_questions_for_table(table, domain_templates)
            questions.extend(table_questions)

            # Limit questions per domain
            if len(questions) >= max_questions:
                questions = questions[:max_questions]
                break

        return questions

    def _generate_questions_for_table(self, table: Dict[str, Any],
                                      templates: List[QuestionTemplate]) -> List[BusinessQuestion]:
        """Generate questions for a specific table."""
        questions = []
        table_name = table.get('name', 'unknown_table')
        columns = table.get('columns', [])

        for template in templates:
            try:
                # Check if table meets template requirements
                if not self._table_meets_requirements(table, template):
                    continue

                # Generate question parameters
                question_params = self._extract_question_parameters(table, template)
                if not question_params:
                    continue

                # Generate question and SQL
                question_text = template.generate_question(**question_params)
                sql_query = self._generate_sql_from_template(template, table, question_params)

                # Calculate confidence
                confidence = self._calculate_question_confidence(table, template, question_params)

                if confidence >= self.min_confidence_threshold:
                    question = BusinessQuestion(
                        question_id=f"{table_name}_{template.template_id}_{len(questions)}",
                        question=question_text,
                        sql_query=sql_query,
                        question_type=template.question_type,
                        complexity=template.complexity,
                        confidence=confidence,
                        business_value=self._assess_business_value(table, template),
                        estimated_runtime=self._estimate_runtime(table, template),
                        tables_involved=[table_name],
                        columns_involved=list(question_params.values()),
                        business_context=template.description
                    )
                    questions.append(question)

            except Exception as e:
                self.logger.warning("Failed to generate question from template",
                                    table=table_name,
                                    template=template.template_id,
                                    error=str(e))

        return questions

    def _table_meets_requirements(self, table: Dict[str, Any], template: QuestionTemplate) -> bool:
        """Check if table meets template requirements."""
        # Check table type requirements
        if template.required_table_types:
            table_type = table.get('table_type', 'TABLE').upper()
            if table_type not in [t.upper() for t in template.required_table_types]:
                return False

        # Check for required column patterns
        columns = [col.get('name', '').lower() for col in table.get('columns', [])]

        for required_col in template.required_columns:
            # Check for exact match or pattern match
            if not self._find_matching_column(columns, required_col):
                return False

        return True

    def _find_matching_column(self, columns: List[str], pattern: str) -> Optional[str]:
        """Find column matching a pattern."""
        pattern_lower = pattern.lower()

        # Column type patterns
        column_patterns = {
            'date_column': ['date', 'time', 'created', 'updated', 'modified'],
            'metric_column': ['amount', 'total', 'price', 'cost', 'revenue', 'quantity', 'sales'],
            'customer_column': ['customer', 'cust_id', 'customer_id'],
            'customer_id': ['customer_id', 'cust_id', 'c_customer_id'],
            'entity_column': ['name', 'title', 'description'],
            'quantity_column': ['quantity', 'qty', 'count', 'number'],
            'category_column': ['category', 'type', 'class', 'segment'],
            'region_column': ['region', 'state', 'country', 'location', 'city'],
            'revenue_column': ['revenue', 'sales', 'amount', 'total'],
            'status_column': ['status', 'state', 'active', 'inactive']
        }

        if pattern_lower in column_patterns:
            for keyword in column_patterns[pattern_lower]:
                matching_cols = [col for col in columns if keyword in col]
                if matching_cols:
                    return matching_cols[0]

        # Direct pattern matching
        matching_cols = [col for col in columns if pattern_lower in col]
        return matching_cols[0] if matching_cols else None

    def _extract_question_parameters(self, table: Dict[str, Any],
                                     template: QuestionTemplate) -> Optional[Dict[str, Any]]:
        """Extract parameters needed for question generation."""
        table_name = table.get('name', '')
        columns = [col.get('name', '') for col in table.get('columns', [])]
        columns_lower = [col.lower() for col in columns]

        params = {
            'table': table_name,
            'entity': self._extract_entity_name(table_name),
            'n': 10  # Default top-N value
        }

        # Map required columns to actual columns
        for required_col in template.required_columns:
            actual_col = self._find_matching_column(columns_lower, required_col)
            if actual_col:
                # Find the original case column name
                original_col = next(col for col in columns if col.lower() == actual_col)
                params[required_col] = original_col
            else:
                return None  # Required column not found

        # Add metric name based on columns found
        if 'metric_column' in params:
            params['metric'] = self._extract_metric_name(params['metric_column'])

        return params

    def _extract_entity_name(self, table_name: str) -> str:
        """Extract entity name from table name."""
        # Remove common prefixes/suffixes
        entity = table_name.lower()
        entity = re.sub(r'^(dim_|fact_|ref_)', '', entity)
        entity = re.sub(r'(_dim|_fact|_ref)$', '', entity)

        # Handle plural forms
        if entity.endswith('s') and len(entity) > 3:
            entity = entity[:-1]

        return entity.replace('_', ' ')

    def _extract_metric_name(self, column_name: str) -> str:
        """Extract metric name from column name."""
        metric = column_name.lower()

        # Common metric mappings
        if 'amount' in metric or 'total' in metric:
            return 'revenue'
        elif 'price' in metric or 'cost' in metric:
            return 'cost'
        elif 'quantity' in metric or 'qty' in metric:
            return 'quantity'
        elif 'sales' in metric:
            return 'sales'
        else:
            return metric.replace('_', ' ')

    def _generate_sql_from_template(self, template: QuestionTemplate, table: Dict[str, Any],
                                    params: Dict[str, Any]) -> str:
        """Generate SQL from template and parameters."""
        try:
            sql = template.sql_template.format(**params)
            # Clean up SQL formatting
            sql = re.sub(r'\s+', ' ', sql.strip())
            return sql
        except KeyError as e:
            self.logger.warning("Missing SQL template parameter",
                                template=template.template_id,
                                missing_param=str(e))
            return f"-- SQL generation failed: missing parameter {e}"

    def _calculate_question_confidence(self, table: Dict[str, Any], template: QuestionTemplate,
                                       params: Dict[str, Any]) -> float:
        """Calculate confidence score for generated question."""
        confidence = 1.0

        # Reduce confidence if table has no row count
        if not table.get('row_count'):
            confidence -= 0.2

        # Reduce confidence for complex templates on simple tables
        if template.complexity == QuestionComplexity.ADVANCED and table.get('complexity') == 'simple':
            confidence -= 0.3

        # Increase confidence for well-named columns
        for param_name, param_value in params.items():
            if param_name.endswith('_column') and param_value:
                if any(keyword in param_value.lower() for keyword in ['id', 'name', 'date', 'amount']):
                    confidence += 0.1

        # Business domain bonus
        domain = table.get('business_domain', 'unknown')
        if domain in ['fact', 'dimension'] and template.business_domains:
            if domain in template.business_domains:
                confidence += 0.2

        return min(1.0, max(0.0, confidence))

    def _assess_business_value(self, table: Dict[str, Any], template: QuestionTemplate) -> str:
        """Assess business value of the question."""
        # High value for revenue/sales questions
        if template.question_type in [QuestionType.TREND_ANALYSIS, QuestionType.TOP_N]:
            if any(keyword in table.get('name', '').lower() for keyword in ['sales', 'revenue', 'customer']):
                return 'high'

        # Medium value for operational questions
        if template.question_type in [QuestionType.COUNT, QuestionType.DISTRIBUTION]:
            return 'medium'

        return 'low'

    def _estimate_runtime(self, table: Dict[str, Any], template: QuestionTemplate) -> str:
        """Estimate query runtime."""
        row_count = table.get('row_count', 0)

        if template.complexity == QuestionComplexity.ADVANCED:
            return 'slow'
        elif row_count and row_count > 1000000:
            return 'medium'
        else:
            return 'fast'

    def _rank_questions(self, questions: List[BusinessQuestion]) -> List[BusinessQuestion]:
        """Rank questions by confidence and business value."""
        value_scores = {'high': 3, 'medium': 2, 'low': 1}

        return sorted(questions,
                      key=lambda q: (q.confidence, value_scores.get(q.business_value, 1)),
                      reverse=True)

    def save_questions(self, questions: List[BusinessQuestion], cache_key: str) -> str:
        """Save generated questions to cache."""
        questions_data = [q.to_dict() for q in questions]

        self.state_manager.save_state(
            cache_key,
            questions_data,
            StateType.BUSINESS_QUESTIONS,
            ttl_hours=12
        )

        self.logger.info("Business questions saved",
                         question_count=len(questions),
                         cache_key=cache_key)

        return cache_key

    def load_questions(self, cache_key: str) -> Optional[List[BusinessQuestion]]:
        """Load questions from cache."""
        questions_data = self.state_manager.load_state(cache_key, StateType.BUSINESS_QUESTIONS)

        if questions_data:
            questions = []
            for q_data in questions_data:
                question = BusinessQuestion(
                    question_id=q_data['question_id'],
                    question=q_data['question'],
                    sql_query=q_data['sql_query'],
                    question_type=QuestionType(q_data['question_type']),
                    complexity=QuestionComplexity(q_data['complexity']),
                    confidence=q_data['confidence'],
                    business_value=q_data['business_value'],
                    estimated_runtime=q_data['estimated_runtime'],
                    tables_involved=q_data['tables_involved'],
                    columns_involved=q_data['columns_involved'],
                    business_context=q_data['business_context'],
                    generated_at=datetime.fromisoformat(q_data['generated_at'])
                )
                questions.append(question)

            self.logger.info("Business questions loaded",
                             question_count=len(questions),
                             cache_key=cache_key)

            return questions

        return None


# Testing and demonstration
if __name__ == "__main__":
    print("Testing Business Question Bank Tool")
    print("=" * 50)

    logger = get_logger("test_business_question_bank")

    try:
        # Create Business Question Bank
        print("🤖 Creating Business Question Bank...")
        question_bank = BusinessQuestionBank()

        print(f"✅ Question Bank created")
        print(f"   Templates loaded: {len(question_bank.templates)}")
        print(f"   Max questions per category: {question_bank.max_questions_per_category}")

        # Sample business tables (simulating Technical Analyst output)
        print(f"\n📋 Creating sample business table metadata...")
        sample_business_tables = [
            {
                "name": "STORE_SALES",
                "business_domain": "fact",
                "classification": "core_business",
                "table_type": "TABLE",
                "row_count": 28800239865,
                "complexity": "complex",
                "columns": [
                    {"name": "SS_SOLD_DATE_SK", "data_type": "NUMBER"},
                    {"name": "SS_CUSTOMER_SK", "data_type": "NUMBER"},
                    {"name": "SS_ITEM_SK", "data_type": "NUMBER"},
                    {"name": "SS_QUANTITY", "data_type": "NUMBER"},
                    {"name": "SS_SALES_PRICE", "data_type": "NUMBER"},
                    {"name": "SS_NET_PAID", "data_type": "NUMBER"}
                ]
            },
            {
                "name": "CUSTOMER",
                "business_domain": "dimension",
                "classification": "core_business",
                "table_type": "TABLE",
                "row_count": 65000000,
                "complexity": "moderate",
                "columns": [
                    {"name": "C_CUSTOMER_SK", "data_type": "NUMBER"},
                    {"name": "C_CUSTOMER_ID", "data_type": "TEXT"},
                    {"name": "C_FIRST_NAME", "data_type": "TEXT"},
                    {"name": "C_LAST_NAME", "data_type": "TEXT"},
                    {"name": "C_EMAIL_ADDRESS", "data_type": "TEXT"}
                ]
            },
            {
                "name": "ITEM",
                "business_domain": "dimension",
                "classification": "reference",
                "table_type": "TABLE",
                "row_count": 462000,
                "complexity": "simple",
                "columns": [
                    {"name": "I_ITEM_SK", "data_type": "NUMBER"},
                    {"name": "I_ITEM_ID", "data_type": "TEXT"},
                    {"name": "I_ITEM_DESC", "data_type": "TEXT"},
                    {"name": "I_CATEGORY", "data_type": "TEXT"},
                    {"name": "I_CLASS", "data_type": "TEXT"}
                ]
            },
            {
                "name": "DATE_DIM",
                "business_domain": "dimension",
                "classification": "reference",
                "table_type": "TABLE",
                "row_count": 73049,
                "complexity": "simple",
                "columns": [
                    {"name": "D_DATE_SK", "data_type": "NUMBER"},
                    {"name": "D_DATE", "data_type": "DATE"},
                    {"name": "D_YEAR", "data_type": "NUMBER"},
                    {"name": "D_MONTH_SEQ", "data_type": "NUMBER"}
                ]
            }
        ]

        print(f"✅ Sample metadata created: {len(sample_business_tables)} tables")

        # Test question generation
        print(f"\n🔍 Test 1: Generate Business Questions")
        questions = question_bank.generate_questions_for_schema(
            sample_business_tables,
            max_questions=15
        )

        print(f"✅ Questions generated: {len(questions)}")

        # Show questions by type
        questions_by_type = {}
        for question in questions:
            q_type = question.question_type.value
            if q_type not in questions_by_type:
                questions_by_type[q_type] = []
            questions_by_type[q_type].append(question)

        print(f"\n   Questions by type:")
        for q_type, type_questions in questions_by_type.items():
            print(f"      {q_type}: {len(type_questions)} questions")

        # Show top questions
        print(f"\n   Top business questions:")
        for i, question in enumerate(questions[:5], 1):
            print(f"      {i}. {question.question}")
            print(f"         Confidence: {question.confidence:.2f}, Value: {question.business_value}")
            print(f"         SQL: {question.sql_query[:80]}...")

        # Test question saving and loading
        print(f"\n💾 Test 2: Save and Load Questions")
        cache_key = "test_business_questions_tpcds"
        question_bank.save_questions(questions, cache_key)

        loaded_questions = question_bank.load_questions(cache_key)
        print(f"✅ Questions cached and loaded")
        print(f"   Original count: {len(questions)}")
        print(f"   Loaded count: {len(loaded_questions)}")
        print(f"   Data integrity: {'✅ OK' if len(questions) == len(loaded_questions) else '❌ ERROR'}")

        # Test high-confidence questions
        print(f"\n🎯 Test 3: High-Confidence Questions")
        high_confidence = [q for q in questions if q.confidence > 0.8]
        print(f"   High-confidence questions: {len(high_confidence)}")

        if high_confidence:
            best_question = high_confidence[0]
            print(f"   Best question: {best_question.question}")
            print(f"   Confidence: {best_question.confidence:.2f}")
            print(f"   Complexity: {best_question.complexity.value}")
            print(f"   Tables: {', '.join(best_question.tables_involved)}")
            print(f"   SQL: {best_question.sql_query}")

        # Test question analysis
        print(f"\n📊 Question Analysis:")
        complexities = [q.complexity.value for q in questions]
        for complexity in set(complexities):
            count = complexities.count(complexity)
            print(f"   {complexity}: {count} questions")

        business_values = [q.business_value for q in questions]
        for value in set(business_values):
            count = business_values.count(value)
            print(f"   {value} value: {count} questions")

        print(f"\n✅ Business Question Bank tested successfully!")
        print(f"   Template system: {len(question_bank.templates)} templates")
        print(f"   Question generation: {len(questions)} questions from {len(sample_business_tables)} tables")
        print(f"   Caching: Save/load functionality working")
        print(f"   Intelligence: Confidence scoring and business value assessment")
        print(f"\n🚀 Ready to support Business Analyst Agent!")

    except Exception as e:
        logger.error("Business Question Bank test failed", error=str(e))
        print(f"❌ Test failed: {str(e)}")