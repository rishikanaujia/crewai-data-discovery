# src/data_discovery/agents/technical_analyst_agent.py

"""
Technical Analyst Agent for comprehensive database schema discovery and analysis.

Orchestrates schema discovery, classifies business domains, identifies relationships,
and provides technical insights that form the foundation for all other agents.
"""

import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Core system imports
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent, AgentTask, TaskPriority, AgentStatus
from tools.analysis.schema_inspector import SchemaInspector, SchemaInfo, TableInfo, ColumnInfo
from core.state_manager import StateType
from core.exceptions import DataDiscoveryException, SchemaInspectionError, ErrorContext


class BusinessDomain(Enum):
    """Business domain classifications for tables."""
    FACT = "fact"  # Large transactional tables
    DIMENSION = "dimension"  # Reference/lookup tables
    BRIDGE = "bridge"  # Many-to-many relationship tables
    STAGING = "staging"  # ETL staging tables
    CONFIGURATION = "configuration"  # System configuration tables
    AUDIT = "audit"  # Audit and logging tables
    UNKNOWN = "unknown"  # Unclassified tables


class TableClassification(Enum):
    """Technical classification of table purposes."""
    CORE_BUSINESS = "core_business"  # Primary business entities
    OPERATIONAL = "operational"  # Day-to-day operations
    ANALYTICAL = "analytical"  # Analytics and reporting
    REFERENCE = "reference"  # Static reference data
    SYSTEM = "system"  # System/technical tables
    DEPRECATED = "deprecated"  # Legacy/unused tables


@dataclass
class BusinessTableInfo:
    """Enhanced table information with business context."""
    table_info: TableInfo
    business_domain: BusinessDomain = BusinessDomain.UNKNOWN
    classification: TableClassification = TableClassification.CORE_BUSINESS
    business_description: str = ""
    key_metrics: List[str] = field(default_factory=list)
    related_tables: List[str] = field(default_factory=list)
    data_quality_score: float = 0.0
    business_criticality: str = "medium"  # low, medium, high, critical
    usage_frequency: str = "unknown"  # daily, weekly, monthly, rarely, unknown

    def get_summary(self) -> Dict[str, Any]:
        """Get business summary of the table."""
        return {
            "name": self.table_info.name,
            "business_domain": self.business_domain.value,
            "classification": self.classification.value,
            "description": self.business_description,
            "row_count": self.table_info.row_count,
            "column_count": self.table_info.get_column_count(),
            "criticality": self.business_criticality,
            "quality_score": self.data_quality_score,
            "key_metrics": self.key_metrics,
            "related_tables": self.related_tables
        }


@dataclass
class TechnicalAnalysis:
    """Complete technical analysis results."""
    schema_info: SchemaInfo
    business_tables: List[BusinessTableInfo] = field(default_factory=list)
    domain_summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    analysis_duration_seconds: float = 0.0

    def get_tables_by_domain(self, domain: BusinessDomain) -> List[BusinessTableInfo]:
        """Get tables by business domain."""
        return [t for t in self.business_tables if t.business_domain == domain]

    def get_high_value_tables(self) -> List[BusinessTableInfo]:
        """Get tables with high business value."""
        return [t for t in self.business_tables
                if t.business_criticality in ["high", "critical"]
                and t.data_quality_score > 0.7]


class TechnicalAnalystAgent(BaseAgent):
    """
    Technical Analyst Agent for comprehensive schema discovery and business classification.

    Responsibilities:
    - Discover database schemas using SchemaInspector
    - Classify tables by business domain and purpose
    - Identify relationships and data flow patterns
    - Assess data quality and business criticality
    - Generate technical recommendations
    """

    def __init__(self, agent_id: str = None):
        super().__init__(
            agent_id=agent_id,
            agent_name="TechnicalAnalyst"
        )

        # Initialize tools
        self.schema_inspector = SchemaInspector()

        # Configuration
        self.enable_deep_analysis = True
        self.include_row_counts = True
        self.max_analysis_time_seconds = 600  # 10 minutes

        self.logger.info("Technical Analyst Agent initialized",
                         enable_deep_analysis=self.enable_deep_analysis,
                         max_analysis_time=self.max_analysis_time_seconds)

    def get_capabilities(self) -> List[str]:
        """Return capabilities of the Technical Analyst Agent."""
        return [
            "schema_discovery",
            "table_classification",
            "relationship_analysis",
            "data_quality_assessment",
            "business_domain_mapping",
            "technical_recommendations"
        ]

    def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute technical analysis tasks."""
        task_type = task.task_type
        parameters = task.parameters

        if task_type == "schema_discovery":
            database = parameters.get("database")
            schema_name = parameters.get("schema_name")
            force_refresh = parameters.get("force_refresh", False)
            return self._discover_schema(database, schema_name, force_refresh)

        elif task_type == "full_technical_analysis":
            database = parameters.get("database")
            schema_name = parameters.get("schema_name")
            return self._perform_full_analysis(database, schema_name)

        elif task_type == "classify_tables":
            schema_cache_key = parameters.get("schema_cache_key")
            return self._classify_tables(schema_cache_key)

        elif task_type == "analyze_relationships":
            schema_cache_key = parameters.get("schema_cache_key")
            return self._analyze_relationships(schema_cache_key)

        elif task_type == "assess_data_quality":
            schema_cache_key = parameters.get("schema_cache_key")
            return self._assess_data_quality(schema_cache_key)

        else:
            raise DataDiscoveryException(f"Unknown task type: {task_type}")

    def _discover_schema(self, database: str = None, schema_name: str = None,
                         force_refresh: bool = False) -> Dict[str, Any]:
        """Discover database schema using SchemaInspector."""
        self.logger.info("Starting schema discovery",
                         database=database or "default",
                         schema=schema_name or "default",
                         force_refresh=force_refresh)

        try:
            # Update progress
            if self.current_task:
                self.current_task.progress_percent = 10.0

            # Discover schema
            schema_info = self.schema_inspector.discover_schema(
                database=database,
                schema_name=schema_name,
                force_refresh=force_refresh,
                include_row_counts=self.include_row_counts
            )

            # Update progress
            if self.current_task:
                self.current_task.progress_percent = 80.0

            # Cache the schema info for other tasks
            cache_key = f"schema_discovery_{schema_info.database}_{schema_info.schema_name}"
            self.cache_result(cache_key, schema_info, ttl_hours=24)

            # Update progress
            if self.current_task:
                self.current_task.progress_percent = 100.0

            self.logger.info("Schema discovery completed",
                             database=schema_info.database,
                             schema=schema_info.schema_name,
                             table_count=schema_info.get_table_count(),
                             total_columns=schema_info.get_total_columns(),
                             discovery_duration=schema_info.discovery_duration_seconds)

            return {
                "status": "success",
                "schema_info": {
                    "database": schema_info.database,
                    "schema_name": schema_info.schema_name,
                    "table_count": schema_info.get_table_count(),
                    "total_columns": schema_info.get_total_columns(),
                    "discovery_duration_seconds": schema_info.discovery_duration_seconds,
                    "discovered_at": schema_info.discovered_at.isoformat()
                },
                "cache_key": cache_key,
                "tables": [
                    {
                        "name": table.name,
                        "type": table.table_type,
                        "columns": table.get_column_count(),
                        "rows": table.row_count,
                        "complexity": table.estimate_complexity()
                    }
                    for table in schema_info.tables
                ]
            }

        except Exception as e:
            self.logger.error("Schema discovery failed",
                              database=database,
                              schema=schema_name,
                              error=str(e))
            raise SchemaInspectionError(
                f"Failed to discover schema: {str(e)}",
                database=database,
                table=schema_name
            )

    def _perform_full_analysis(self, database: str = None, schema_name: str = None) -> Dict[str, Any]:
        """Perform comprehensive technical analysis."""
        start_time = time.time()

        self.logger.info("Starting full technical analysis",
                         database=database or "default",
                         schema=schema_name or "default")

        try:
            # Step 1: Schema Discovery (20%)
            if self.current_task:
                self.current_task.progress_percent = 5.0

            schema_info = self.schema_inspector.discover_schema(
                database=database,
                schema_name=schema_name,
                include_row_counts=self.include_row_counts
            )

            if self.current_task:
                self.current_task.progress_percent = 20.0

            # Step 2: Table Classification (40%)
            self.logger.info("Classifying tables by business domain")
            business_tables = self._classify_all_tables(schema_info)

            if self.current_task:
                self.current_task.progress_percent = 40.0

            # Step 3: Relationship Analysis (60%)
            self.logger.info("Analyzing table relationships")
            relationship_analysis = self.schema_inspector.analyze_table_relationships(schema_info)

            if self.current_task:
                self.current_task.progress_percent = 60.0

            # Step 4: Data Quality Assessment (80%)
            self.logger.info("Assessing data quality")
            self._assess_table_quality(business_tables)

            if self.current_task:
                self.current_task.progress_percent = 80.0

            # Step 5: Generate Recommendations (100%)
            self.logger.info("Generating recommendations")
            recommendations = self._generate_recommendations(business_tables, relationship_analysis)

            if self.current_task:
                self.current_task.progress_percent = 100.0

            # Create complete analysis
            analysis_duration = time.time() - start_time

            analysis = TechnicalAnalysis(
                schema_info=schema_info,
                business_tables=business_tables,
                domain_summary=self._create_domain_summary(business_tables),
                recommendations=recommendations,
                analysis_duration_seconds=analysis_duration
            )

            # Cache complete analysis
            cache_key = f"full_analysis_{schema_info.database}_{schema_info.schema_name}"
            self.cache_result(cache_key, analysis, ttl_hours=12)

            self.logger.info("Full technical analysis completed",
                             database=schema_info.database,
                             schema=schema_info.schema_name,
                             analysis_duration=analysis_duration,
                             tables_analyzed=len(business_tables),
                             recommendations_count=len(recommendations))

            return {
                "status": "success",
                "analysis": {
                    "database": schema_info.database,
                    "schema_name": schema_info.schema_name,
                    "analysis_duration_seconds": analysis_duration,
                    "tables_analyzed": len(business_tables),
                    "domain_summary": analysis.domain_summary,
                    "recommendations": recommendations,
                    "cache_key": cache_key
                },
                "table_summary": [table.get_summary() for table in business_tables],
                "high_value_tables": [t.get_summary() for t in analysis.get_high_value_tables()],
                "relationship_insights": relationship_analysis
            }

        except Exception as e:
            self.logger.error("Full technical analysis failed",
                              database=database,
                              schema=schema_name,
                              error=str(e))
            raise DataDiscoveryException(f"Technical analysis failed: {str(e)}")

    def _classify_all_tables(self, schema_info: SchemaInfo) -> List[BusinessTableInfo]:
        """Classify all tables in the schema."""
        business_tables = []

        for table in schema_info.tables:
            business_table = BusinessTableInfo(table_info=table)

            # Classify by business domain
            business_table.business_domain = self._classify_business_domain(table)

            # Classify by technical purpose
            business_table.classification = self._classify_table_purpose(table)

            # Generate business description
            business_table.business_description = self._generate_business_description(table)

            # Identify key metrics
            business_table.key_metrics = self._identify_key_metrics(table)

            # Assess business criticality
            business_table.business_criticality = self._assess_business_criticality(table)

            business_tables.append(business_table)

        return business_tables

    def _classify_business_domain(self, table: TableInfo) -> BusinessDomain:
        """Classify table by business domain."""
        table_name = table.name.lower()
        column_names = [col.name.lower() for col in table.columns]

        # Fact table indicators
        if (table.row_count and table.row_count > 100000 and
                len(table.get_foreign_key_columns()) >= 2):
            return BusinessDomain.FACT

        # Large tables with sales/transaction indicators
        if any(keyword in table_name for keyword in ['sales', 'orders', 'transactions', 'events']):
            return BusinessDomain.FACT

        # Dimension table indicators
        if (table.get_column_count() <= 20 and
                len(table.get_id_columns()) >= 1 and
                any(keyword in table_name for keyword in ['customer', 'product', 'item', 'catalog', 'store'])):
            return BusinessDomain.DIMENSION

        # Bridge table indicators
        if (table.get_column_count() <= 10 and
                len(table.get_foreign_key_columns()) >= 2 and
                any(keyword in table_name for keyword in ['bridge', 'link', 'assoc'])):
            return BusinessDomain.BRIDGE

        # Staging indicators
        if any(keyword in table_name for keyword in ['staging', 'stage', 'temp', 'tmp']):
            return BusinessDomain.STAGING

        # Configuration indicators
        if any(keyword in table_name for keyword in ['config', 'setting', 'parameter', 'lookup']):
            return BusinessDomain.CONFIGURATION

        # Audit indicators
        if any(keyword in table_name for keyword in ['audit', 'log', 'history', 'track']):
            return BusinessDomain.AUDIT

        return BusinessDomain.UNKNOWN

    def _classify_table_purpose(self, table: TableInfo) -> TableClassification:
        """Classify table by technical purpose."""
        table_name = table.name.lower()

        # Core business entities
        if any(keyword in table_name for keyword in
               ['customer', 'product', 'order', 'sale', 'invoice', 'payment']):
            return TableClassification.CORE_BUSINESS

        # Operational tables
        if any(keyword in table_name for keyword in
               ['inventory', 'shipment', 'warehouse', 'call_center']):
            return TableClassification.OPERATIONAL

        # Analytical tables
        if any(keyword in table_name for keyword in
               ['fact', 'dim', 'summary', 'aggregate', 'report']):
            return TableClassification.ANALYTICAL

        # Reference data
        if any(keyword in table_name for keyword in
               ['ref', 'lookup', 'code', 'type', 'category', 'region']):
            return TableClassification.REFERENCE

        return TableClassification.CORE_BUSINESS

    def _generate_business_description(self, table: TableInfo) -> str:
        """Generate human-readable business description."""
        table_name = table.name.lower()
        domain = self._classify_business_domain(table)

        # Generate description based on domain and naming patterns
        if domain == BusinessDomain.FACT:
            if 'sales' in table_name:
                return f"Transaction records for {table_name.replace('_', ' ')} with detailed metrics and dimensions"
            elif 'returns' in table_name:
                return f"Product return transactions with associated costs and reasons"
            else:
                return f"Fact table containing transactional data for {table_name.replace('_', ' ')}"

        elif domain == BusinessDomain.DIMENSION:
            if 'customer' in table_name:
                return f"Customer master data with demographics and contact information"
            elif 'product' in table_name or 'item' in table_name:
                return f"Product catalog with specifications and categorization"
            elif 'store' in table_name:
                return f"Store location and operational details"
            else:
                return f"Reference data for {table_name.replace('_', ' ')} entities"

        elif domain == BusinessDomain.CONFIGURATION:
            return f"System configuration and lookup values for {table_name.replace('_', ' ')}"

        else:
            return f"Business data table: {table_name.replace('_', ' ')}"

    def _identify_key_metrics(self, table: TableInfo) -> List[str]:
        """Identify key business metrics in the table."""
        metrics = []

        for column in table.columns:
            col_name = column.name.lower()

            # Common metric patterns
            if any(keyword in col_name for keyword in
                   ['amount', 'total', 'price', 'cost', 'revenue', 'quantity', 'count']):
                metrics.append(column.name)
            elif any(keyword in col_name for keyword in
                     ['net', 'gross', 'profit', 'margin', 'discount']):
                metrics.append(column.name)

        return metrics[:5]  # Limit to top 5 metrics

    def _assess_business_criticality(self, table: TableInfo) -> str:
        """Assess business criticality of the table."""
        table_name = table.name.lower()

        # Critical tables
        if any(keyword in table_name for keyword in
               ['sales', 'order', 'customer', 'payment', 'invoice']):
            return "critical"

        # High importance
        if any(keyword in table_name for keyword in
               ['product', 'inventory', 'shipment', 'returns']):
            return "high"

        # Medium importance
        if any(keyword in table_name for keyword in
               ['catalog', 'promotion', 'warehouse', 'store']):
            return "medium"

        return "low"

    def _assess_table_quality(self, business_tables: List[BusinessTableInfo]):
        """Assess data quality for each table."""
        for business_table in business_tables:
            table = business_table.table_info
            quality_score = 1.0  # Start with perfect score

            # Reduce score for missing row counts
            if table.row_count is None:
                quality_score -= 0.2

            # Reduce score for too many nullable columns
            nullable_ratio = len(table.get_nullable_columns()) / max(table.get_column_count(), 1)
            if nullable_ratio > 0.8:
                quality_score -= 0.3

            # Reduce score for no primary keys
            if len(table.get_primary_key_columns()) == 0:
                quality_score -= 0.2

            # Reduce score for no foreign keys in large tables
            if table.row_count and table.row_count > 10000 and len(table.get_foreign_key_columns()) == 0:
                quality_score -= 0.1

            business_table.data_quality_score = max(0.0, quality_score)

    def _generate_recommendations(self, business_tables: List[BusinessTableInfo],
                                  relationship_analysis: Dict[str, Any]) -> List[str]:
        """Generate technical recommendations."""
        recommendations = []

        # Analyze fact tables
        fact_tables = [t for t in business_tables if t.business_domain == BusinessDomain.FACT]
        if len(fact_tables) > 5:
            recommendations.append(
                f"Consider partitioning strategy for {len(fact_tables)} fact tables to improve query performance"
            )

        # Analyze data quality
        low_quality_tables = [t for t in business_tables if t.data_quality_score < 0.5]
        if low_quality_tables:
            recommendations.append(
                f"Review data quality for {len(low_quality_tables)} tables with quality scores below 50%"
            )

        # Analyze missing relationships
        unrelated_tables = [t for t in business_tables if not t.related_tables]
        if len(unrelated_tables) > len(business_tables) * 0.3:
            recommendations.append(
                "Consider documenting table relationships - many tables appear isolated"
            )

        # Performance recommendations
        large_tables = [t for t in business_tables if t.table_info.row_count and t.table_info.row_count > 1000000]
        if large_tables:
            recommendations.append(
                f"Monitor query performance for {len(large_tables)} tables with over 1M rows"
            )

        # Business value recommendations
        high_value_tables = [t for t in business_tables if t.business_criticality == "critical"]
        if high_value_tables:
            recommendations.append(
                f"Prioritize data governance for {len(high_value_tables)} business-critical tables"
            )

        return recommendations

    def _create_domain_summary(self, business_tables: List[BusinessTableInfo]) -> Dict[str, Any]:
        """Create summary by business domain."""
        summary = {}

        for domain in BusinessDomain:
            domain_tables = [t for t in business_tables if t.business_domain == domain]
            if domain_tables:
                summary[domain.value] = {
                    "count": len(domain_tables),
                    "tables": [t.table_info.name for t in domain_tables],
                    "total_rows": sum(t.table_info.row_count or 0 for t in domain_tables),
                    "avg_quality_score": sum(t.data_quality_score for t in domain_tables) / len(domain_tables)
                }

        return summary

    def _classify_tables(self, schema_cache_key: str) -> Dict[str, Any]:
        """Classify tables from cached schema."""
        schema_info = self.get_cached_result(schema_cache_key)
        if not schema_info:
            raise DataDiscoveryException("Schema not found in cache")

        business_tables = self._classify_all_tables(schema_info)

        return {
            "status": "success",
            "classified_tables": [table.get_summary() for table in business_tables],
            "domain_distribution": self._create_domain_summary(business_tables)
        }

    def _analyze_relationships(self, schema_cache_key: str) -> Dict[str, Any]:
        """Analyze table relationships from cached schema."""
        schema_info = self.get_cached_result(schema_cache_key)
        if not schema_info:
            raise DataDiscoveryException("Schema not found in cache")

        relationship_analysis = self.schema_inspector.analyze_table_relationships(schema_info)

        return {
            "status": "success",
            "relationship_analysis": relationship_analysis
        }

    def _assess_data_quality(self, schema_cache_key: str) -> Dict[str, Any]:
        """Assess data quality from cached schema."""
        schema_info = self.get_cached_result(schema_cache_key)
        if not schema_info:
            raise DataDiscoveryException("Schema not found in cache")

        business_tables = self._classify_all_tables(schema_info)
        self._assess_table_quality(business_tables)

        quality_summary = {
            "average_quality_score": sum(t.data_quality_score for t in business_tables) / len(business_tables),
            "high_quality_tables": len([t for t in business_tables if t.data_quality_score > 0.8]),
            "low_quality_tables": len([t for t in business_tables if t.data_quality_score < 0.5]),
            "table_scores": {t.table_info.name: t.data_quality_score for t in business_tables}
        }

        return {
            "status": "success",
            "quality_assessment": quality_summary
        }


# Testing and demonstration
if __name__ == "__main__":
    print("Testing Technical Analyst Agent")
    print("=" * 50)

    try:
        # Create Technical Analyst Agent
        print("🤖 Creating Technical Analyst Agent...")
        agent = TechnicalAnalystAgent()

        print(f"✅ Agent created: {agent.agent_id}")
        print(f"   Name: {agent.agent_name}")
        print(f"   Capabilities: {', '.join(agent.get_capabilities())}")

        # Test 1: Schema Discovery
        print(f"\n🔍 Test 1: Schema Discovery")
        schema_task_id = agent.add_task(
            task_type="schema_discovery",
            description="Discover database schema structure",
            parameters={
                "database": "SNOWFLAKE_SAMPLE_DATA",
                "schema_name": "TPCDS_SF10TCL",
                "force_refresh": False
            },
            priority=TaskPriority.HIGH
        )

        schema_result = agent.run_next_task()
        print(f"✅ Schema discovery completed")
        print(f"   Database: {schema_result['schema_info']['database']}")
        print(f"   Schema: {schema_result['schema_info']['schema_name']}")
        print(f"   Tables: {schema_result['schema_info']['table_count']}")
        print(f"   Columns: {schema_result['schema_info']['total_columns']}")
        print(f"   Discovery time: {schema_result['schema_info']['discovery_duration_seconds']:.2f}s")

        # Show sample tables with complexity
        print(f"\n   Sample tables by complexity:")
        for table in schema_result['tables'][:5]:
            row_info = f", {table['rows']:,} rows" if table['rows'] else ""
            print(f"      {table['name']}: {table['complexity']} ({table['columns']} cols{row_info})")

        # Test 2: Full Technical Analysis
        print(f"\n🔍 Test 2: Full Technical Analysis")
        analysis_task_id = agent.add_task(
            task_type="full_technical_analysis",
            description="Perform comprehensive technical analysis",
            parameters={
                "database": "SNOWFLAKE_SAMPLE_DATA",
                "schema_name": "TPCDS_SF10TCL"
            },
            priority=TaskPriority.HIGH
        )

        analysis_result = agent.run_next_task()
        print(f"✅ Technical analysis completed")
        print(f"   Analysis duration: {analysis_result['analysis']['analysis_duration_seconds']:.2f}s")
        print(f"   Tables analyzed: {analysis_result['analysis']['tables_analyzed']}")
        print(f"   Recommendations: {len(analysis_result['analysis']['recommendations'])}")

        # Show domain distribution
        print(f"\n   Business domain distribution:")
        for domain, info in analysis_result['analysis']['domain_summary'].items():
            print(f"      {domain}: {info['count']} tables, {info['total_rows']:,} total rows")

        # Show high-value tables
        print(f"\n   High-value tables:")
        for table in analysis_result['high_value_tables'][:3]:
            print(f"      {table['name']}: {table['criticality']} criticality, {table['quality_score']:.2f} quality")

        # Show recommendations
        print(f"\n   Key recommendations:")
        for i, recommendation in enumerate(analysis_result['analysis']['recommendations'][:3], 1):
            print(f"      {i}. {recommendation}")

        # Test 3: Table Classification
        print(f"\n🔍 Test 3: Table Classification")
        agent.add_task(
            task_type="classify_tables",
            description="Classify tables by business domain",
            parameters={
                "schema_cache_key": schema_result['cache_key']
            },
            priority=TaskPriority.MEDIUM
        )

        classification_result = agent.run_next_task()
        print(f"✅ Table classification completed")

        # Show classification summary
        print(f"   Domain distribution:")
        for domain, info in classification_result['domain_distribution'].items():
            avg_quality = info['avg_quality_score']
            print(f"      {domain}: {info['count']} tables, avg quality: {avg_quality:.2f}")

        # Show agent status and metrics
        print(f"\n📊 Agent Status and Metrics:")
        status = agent.get_status()
        metrics = status['metrics']

        print(f"   Status: {status['status']}")
        print(f"   Tasks completed: {metrics['total_tasks_completed']}")
        print(f"   Tasks failed: {metrics['total_tasks_failed']}")
        print(f"   Success rate: {metrics['success_rate']:.1%}")
        print(f"   Average task time: {metrics['average_task_time_seconds']:.2f}s")

        print(f"\n✅ Technical Analyst Agent tested successfully!")
        print(f"   Core capabilities: Schema discovery, classification, analysis")
        print(f"   Business insights: Domain mapping, quality assessment, recommendations")
        print(f"   Integration: SchemaInspector, BaseAgent, state management")
        print(f"\n🚀 Ready to support Business Analyst and other agents!")

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        agent.logger.error("Technical Analyst Agent test failed", error=str(e))
        print(f"   Check that SchemaInspector tool is working correctly")