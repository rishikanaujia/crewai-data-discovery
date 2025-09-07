# src/data_discovery/agents/business_analyst_agent.py

"""
Business Analyst Agent for transforming technical findings into business insights.

Orchestrates business domain analysis, generates contextual business questions,
and creates executive-friendly summaries with actionable recommendations.
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
from tools.query.business_question_bank import BusinessQuestionBank, BusinessQuestion, QuestionType
from core.state_manager import StateType
from core.exceptions import DataDiscoveryException, ErrorContext


class BusinessDomain(Enum):
    """Business domain focus areas."""
    SALES_PERFORMANCE = "sales_performance"
    CUSTOMER_ANALYTICS = "customer_analytics"
    PRODUCT_MANAGEMENT = "product_management"
    FINANCIAL_REPORTING = "financial_reporting"
    OPERATIONAL_EFFICIENCY = "operational_efficiency"
    MARKET_ANALYSIS = "market_analysis"
    RISK_MANAGEMENT = "risk_management"


class InsightType(Enum):
    """Types of business insights."""
    OPPORTUNITY = "opportunity"
    RISK = "risk"
    TREND = "trend"
    ANOMALY = "anomaly"
    RECOMMENDATION = "recommendation"
    BENCHMARK = "benchmark"


@dataclass
class BusinessInsight:
    """Business insight with context and impact."""
    insight_id: str
    insight_type: InsightType
    domain: BusinessDomain
    title: str
    description: str
    impact_level: str  # low, medium, high, critical
    confidence: float  # 0.0 - 1.0
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    recommended_actions: List[str] = field(default_factory=list)
    related_questions: List[str] = field(default_factory=list)
    business_value: str = "medium"  # low, medium, high

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "insight_id": self.insight_id,
            "insight_type": self.insight_type.value,
            "domain": self.domain.value,
            "title": self.title,
            "description": self.description,
            "impact_level": self.impact_level,
            "confidence": self.confidence,
            "supporting_data": self.supporting_data,
            "recommended_actions": self.recommended_actions,
            "related_questions": self.related_questions,
            "business_value": self.business_value
        }


@dataclass
class BusinessReport:
    """Comprehensive business analysis report."""
    report_id: str
    database: str
    schema_name: str
    executive_summary: str
    key_insights: List[BusinessInsight] = field(default_factory=list)
    business_questions: List[BusinessQuestion] = field(default_factory=list)
    domain_analysis: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    opportunity_assessment: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)

    def get_high_impact_insights(self) -> List[BusinessInsight]:
        """Get high and critical impact insights."""
        return [insight for insight in self.key_insights
                if insight.impact_level in ["high", "critical"]]

    def get_insights_by_domain(self, domain: BusinessDomain) -> List[BusinessInsight]:
        """Get insights for specific business domain."""
        return [insight for insight in self.key_insights if insight.domain == domain]


class BusinessAnalystAgent(BaseAgent):
    """
    Business Analyst Agent for transforming technical analysis into business insights.

    Responsibilities:
    - Generate business questions from technical metadata
    - Create domain-specific business insights
    - Identify opportunities and risks
    - Produce executive-friendly reports
    - Recommend business actions
    """

    def __init__(self, agent_id: str = None):
        super().__init__(
            agent_id=agent_id,
            agent_name="BusinessAnalyst"
        )

        # Initialize tools
        self.question_bank = BusinessQuestionBank()

        # Business domain expertise
        self.domain_keywords = self._load_domain_keywords()
        self.industry_context = self._load_industry_context()

        # Configuration
        self.max_insights_per_domain = 5
        self.min_insight_confidence = 0.6

        self.logger.info("Business Analyst Agent initialized",
                         max_insights_per_domain=self.max_insights_per_domain,
                         domain_count=len(self.domain_keywords))

    def get_capabilities(self) -> List[str]:
        """Return capabilities of the Business Analyst Agent."""
        return [
            "business_question_generation",
            "domain_analysis",
            "insight_generation",
            "risk_assessment",
            "opportunity_identification",
            "executive_reporting"
        ]

    def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute business analysis tasks."""
        task_type = task.task_type
        parameters = task.parameters

        if task_type == "generate_business_questions":
            technical_analysis = parameters.get("technical_analysis")
            return self._generate_business_questions(technical_analysis)

        elif task_type == "create_business_report":
            technical_analysis = parameters.get("technical_analysis")
            return self._create_comprehensive_report(technical_analysis)

        elif task_type == "analyze_business_domain":
            domain = parameters.get("domain")
            business_tables = parameters.get("business_tables")
            return self._analyze_specific_domain(domain, business_tables)

        elif task_type == "identify_opportunities":
            business_tables = parameters.get("business_tables")
            return self._identify_business_opportunities(business_tables)

        elif task_type == "assess_risks":
            business_tables = parameters.get("business_tables")
            return self._assess_business_risks(business_tables)

        else:
            raise DataDiscoveryException(f"Unknown task type: {task_type}")

    def _generate_business_questions(self, technical_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate business questions from technical analysis."""
        self.logger.info("Generating business questions from technical analysis")

        try:
            # Extract business tables from technical analysis
            business_tables = technical_analysis.get('table_summary', [])

            if self.current_task:
                self.current_task.progress_percent = 20.0

            # Generate questions using Question Bank
            questions = self.question_bank.generate_questions_for_schema(
                business_tables,
                max_questions=self.config.analysis.max_questions_per_category
            )

            if self.current_task:
                self.current_task.progress_percent = 60.0

            # Add business context to questions
            enhanced_questions = self._enhance_questions_with_context(questions, business_tables)

            if self.current_task:
                self.current_task.progress_percent = 80.0

            # Cache the questions
            cache_key = f"business_questions_{technical_analysis.get('analysis', {}).get('database', 'unknown')}"
            self.question_bank.save_questions(enhanced_questions, cache_key)

            if self.current_task:
                self.current_task.progress_percent = 100.0

            self.logger.info("Business questions generated",
                             question_count=len(enhanced_questions),
                             high_confidence=len([q for q in enhanced_questions if q.confidence > 0.8]))

            return {
                "status": "success",
                "questions": [q.to_dict() for q in enhanced_questions],
                "question_summary": {
                    "total_questions": len(enhanced_questions),
                    "by_type": self._group_questions_by_type(enhanced_questions),
                    "by_complexity": self._group_questions_by_complexity(enhanced_questions),
                    "high_confidence": len([q for q in enhanced_questions if q.confidence > 0.8]),
                    "high_value": len([q for q in enhanced_questions if q.business_value == "high"])
                },
                "cache_key": cache_key
            }

        except Exception as e:
            self.logger.error("Failed to generate business questions", error=str(e))
            raise DataDiscoveryException(f"Business question generation failed: {str(e)}")

    def _create_comprehensive_report(self, technical_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive business analysis report."""
        start_time = time.time()

        self.logger.info("Creating comprehensive business report")

        try:
            analysis_info = technical_analysis.get('analysis', {})
            business_tables = technical_analysis.get('table_summary', [])

            if self.current_task:
                self.current_task.progress_percent = 10.0

            # Generate business questions
            questions = self.question_bank.generate_questions_for_schema(business_tables)
            enhanced_questions = self._enhance_questions_with_context(questions, business_tables)

            if self.current_task:
                self.current_task.progress_percent = 30.0

            # Generate business insights
            insights = self._generate_business_insights(business_tables, enhanced_questions)

            if self.current_task:
                self.current_task.progress_percent = 50.0

            # Analyze by business domain
            domain_analysis = self._analyze_all_domains(business_tables)

            if self.current_task:
                self.current_task.progress_percent = 70.0

            # Create executive summary
            executive_summary = self._create_executive_summary(
                analysis_info, insights, domain_analysis
            )

            if self.current_task:
                self.current_task.progress_percent = 85.0

            # Generate recommendations
            recommendations = self._generate_strategic_recommendations(
                business_tables, insights, domain_analysis
            )

            # Assess risks and opportunities
            risk_assessment = self._assess_business_risks(business_tables)
            opportunity_assessment = self._identify_business_opportunities(business_tables)

            if self.current_task:
                self.current_task.progress_percent = 100.0

            # Create business report
            report = BusinessReport(
                report_id=f"business_report_{analysis_info.get('database', 'unknown')}_{int(time.time())}",
                database=analysis_info.get('database', 'Unknown'),
                schema_name=analysis_info.get('schema_name', 'Unknown'),
                executive_summary=executive_summary,
                key_insights=insights,
                business_questions=enhanced_questions,
                domain_analysis=domain_analysis,
                recommendations=recommendations,
                risk_assessment=risk_assessment['risk_analysis'],
                opportunity_assessment=opportunity_assessment['opportunities']
            )

            # Cache the report
            cache_key = f"business_report_{analysis_info.get('database', 'unknown')}"
            self.cache_result(cache_key, report, ttl_hours=24)

            duration = time.time() - start_time

            self.logger.info("Business report created",
                             database=report.database,
                             insights_count=len(insights),
                             questions_count=len(enhanced_questions),
                             recommendations_count=len(recommendations),
                             duration_seconds=duration)

            return {
                "status": "success",
                "report": {
                    "report_id": report.report_id,
                    "database": report.database,
                    "schema_name": report.schema_name,
                    "executive_summary": executive_summary,
                    "generation_duration_seconds": duration
                },
                "insights_summary": {
                    "total_insights": len(insights),
                    "high_impact": len(report.get_high_impact_insights()),
                    "by_domain": {domain.value: len(report.get_insights_by_domain(domain))
                                  for domain in BusinessDomain},
                    "recommendations": len(recommendations)
                },
                "questions_summary": {
                    "total_questions": len(enhanced_questions),
                    "high_confidence": len([q for q in enhanced_questions if q.confidence > 0.8])
                },
                "cache_key": cache_key
            }

        except Exception as e:
            self.logger.error("Failed to create business report", error=str(e))
            raise DataDiscoveryException(f"Business report creation failed: {str(e)}")

    def _enhance_questions_with_context(self, questions: List[BusinessQuestion],
                                        business_tables: List[Dict[str, Any]]) -> List[BusinessQuestion]:
        """Add business context and domain knowledge to questions."""
        enhanced_questions = []

        for question in questions:
            # Add industry-specific context
            enhanced_context = self._add_industry_context(question, business_tables)
            question.business_context = enhanced_context

            # Adjust confidence based on business relevance
            business_relevance = self._assess_business_relevance(question, business_tables)
            question.confidence = min(1.0, question.confidence * business_relevance)

            enhanced_questions.append(question)

        return enhanced_questions

    def _generate_business_insights(self, business_tables: List[Dict[str, Any]],
                                    questions: List[BusinessQuestion]) -> List[BusinessInsight]:
        """Generate business insights from tables and questions."""
        insights = []

        # Analyze table patterns for insights
        insights.extend(self._analyze_table_patterns(business_tables))

        # Generate insights from high-value questions
        insights.extend(self._insights_from_questions(questions))

        # Add domain-specific insights
        insights.extend(self._generate_domain_insights(business_tables))

        # Filter and rank insights
        insights = [insight for insight in insights if insight.confidence >= self.min_insight_confidence]
        insights = sorted(insights, key=lambda x: (x.confidence, x.impact_level), reverse=True)

        return insights[:self.max_insights_per_domain * len(BusinessDomain)]

    def _analyze_table_patterns(self, business_tables: List[Dict[str, Any]]) -> List[BusinessInsight]:
        """Analyze table patterns to generate insights."""
        insights = []

        # Large fact tables indicate high transaction volume
        large_fact_tables = [t for t in business_tables
                             if t.get('business_domain') == 'fact' and
                             t.get('row_count', 0) > 1000000]

        if large_fact_tables:
            insights.append(BusinessInsight(
                insight_id=f"high_volume_transactions_{len(large_fact_tables)}",
                insight_type=InsightType.TREND,
                domain=BusinessDomain.SALES_PERFORMANCE,
                title="High Transaction Volume Detected",
                description=f"Found {len(large_fact_tables)} fact tables with over 1M records, "
                            f"indicating high business transaction volume.",
                impact_level="high",
                confidence=0.9,
                supporting_data={"large_tables": [t['name'] for t in large_fact_tables]},
                recommended_actions=[
                    "Implement data archiving strategy for historical data",
                    "Consider partitioning strategies for query performance",
                    "Monitor query performance on large tables"
                ],
                business_value="high"
            ))

        # Missing relationships indicate data silos
        isolated_tables = [t for t in business_tables if not t.get('related_tables')]
        if len(isolated_tables) > len(business_tables) * 0.3:
            insights.append(BusinessInsight(
                insight_id="data_silos_detected",
                insight_type=InsightType.RISK,
                domain=BusinessDomain.OPERATIONAL_EFFICIENCY,
                title="Data Silos Identified",
                description=f"{len(isolated_tables)} tables appear isolated without clear relationships, "
                            f"potentially limiting analytical capabilities.",
                impact_level="medium",
                confidence=0.7,
                supporting_data={"isolated_tables": [t['name'] for t in isolated_tables]},
                recommended_actions=[
                    "Document table relationships and dependencies",
                    "Implement data lineage tracking",
                    "Review data integration opportunities"
                ]
            ))

        return insights

    def _insights_from_questions(self, questions: List[BusinessQuestion]) -> List[BusinessInsight]:
        """Generate insights from high-value business questions."""
        insights = []

        # High-confidence questions indicate well-structured data
        high_confidence_questions = [q for q in questions if q.confidence > 0.9]
        if len(high_confidence_questions) > len(questions) * 0.5:
            insights.append(BusinessInsight(
                insight_id="well_structured_data",
                insight_type=InsightType.OPPORTUNITY,
                domain=BusinessDomain.OPERATIONAL_EFFICIENCY,
                title="Well-Structured Data Foundation",
                description=f"{len(high_confidence_questions)} high-confidence business questions "
                            f"can be generated, indicating good data structure for analytics.",
                impact_level="high",
                confidence=0.8,
                supporting_data={"high_confidence_count": len(high_confidence_questions)},
                recommended_actions=[
                    "Leverage data structure for advanced analytics",
                    "Implement self-service BI capabilities",
                    "Develop automated reporting dashboards"
                ],
                business_value="high"
            ))

        return insights

    def _generate_domain_insights(self, business_tables: List[Dict[str, Any]]) -> List[BusinessInsight]:
        """Generate domain-specific business insights."""
        insights = []

        # Sales domain analysis
        sales_tables = [t for t in business_tables
                        if any(keyword in t.get('name', '').lower()
                               for keyword in ['sales', 'order', 'revenue'])]

        if sales_tables:
            insights.append(BusinessInsight(
                insight_id="sales_analytics_ready",
                insight_type=InsightType.OPPORTUNITY,
                domain=BusinessDomain.SALES_PERFORMANCE,
                title="Sales Analytics Capabilities Available",
                description=f"Found {len(sales_tables)} sales-related tables enabling "
                            f"comprehensive revenue and performance analysis.",
                impact_level="high",
                confidence=0.85,
                supporting_data={"sales_tables": [t['name'] for t in sales_tables]},
                recommended_actions=[
                    "Implement sales performance dashboards",
                    "Develop customer lifetime value models",
                    "Create sales forecasting capabilities"
                ],
                business_value="high"
            ))

        # Customer domain analysis
        customer_tables = [t for t in business_tables
                           if 'customer' in t.get('name', '').lower()]

        if customer_tables:
            insights.append(BusinessInsight(
                insight_id="customer_analytics_potential",
                insight_type=InsightType.OPPORTUNITY,
                domain=BusinessDomain.CUSTOMER_ANALYTICS,
                title="Customer Analytics Foundation Present",
                description=f"Customer data tables available for segmentation, "
                            f"behavior analysis, and personalization initiatives.",
                impact_level="medium",
                confidence=0.8,
                supporting_data={"customer_tables": [t['name'] for t in customer_tables]},
                recommended_actions=[
                    "Develop customer segmentation models",
                    "Implement customer journey analytics",
                    "Create personalization engines"
                ]
            ))

        return insights

    def _analyze_all_domains(self, business_tables: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze all business domains."""
        domain_analysis = {}

        for domain in BusinessDomain:
            domain_tables = self._get_domain_tables(business_tables, domain)
            if domain_tables:
                domain_analysis[domain.value] = {
                    "table_count": len(domain_tables),
                    "tables": [t['name'] for t in domain_tables],
                    "total_rows": sum(t.get('row_count', 0) for t in domain_tables),
                    "readiness_score": self._calculate_domain_readiness(domain_tables),
                    "key_capabilities": self._identify_domain_capabilities(domain, domain_tables)
                }

        return domain_analysis

    def _get_domain_tables(self, business_tables: List[Dict[str, Any]],
                           domain: BusinessDomain) -> List[Dict[str, Any]]:
        """Get tables relevant to a business domain."""
        domain_keywords = self.domain_keywords.get(domain.value, [])

        relevant_tables = []
        for table in business_tables:
            table_name = table.get('name', '').lower()
            if any(keyword in table_name for keyword in domain_keywords):
                relevant_tables.append(table)

        return relevant_tables

    def _calculate_domain_readiness(self, domain_tables: List[Dict[str, Any]]) -> float:
        """Calculate readiness score for a business domain."""
        if not domain_tables:
            return 0.0

        score = 0.0

        # Data volume score
        total_rows = sum(t.get('row_count', 0) for t in domain_tables)
        if total_rows > 1000000:
            score += 0.3
        elif total_rows > 100000:
            score += 0.2
        elif total_rows > 10000:
            score += 0.1

        # Data quality score
        avg_quality = sum(t.get('quality_score', 0.5) for t in domain_tables) / len(domain_tables)
        score += avg_quality * 0.4

        # Completeness score
        if len(domain_tables) >= 3:
            score += 0.3
        elif len(domain_tables) >= 2:
            score += 0.2
        else:
            score += 0.1

        return min(1.0, score)

    def _identify_domain_capabilities(self, domain: BusinessDomain,
                                      domain_tables: List[Dict[str, Any]]) -> List[str]:
        """Identify capabilities available for a domain."""
        capabilities = []

        if domain == BusinessDomain.SALES_PERFORMANCE:
            if any('sales' in t.get('name', '').lower() for t in domain_tables):
                capabilities.append("Sales trend analysis")
            if any('customer' in t.get('name', '').lower() for t in domain_tables):
                capabilities.append("Customer performance tracking")

        elif domain == BusinessDomain.CUSTOMER_ANALYTICS:
            capabilities.extend([
                "Customer segmentation",
                "Behavior analysis",
                "Lifetime value calculation"
            ])

        return capabilities

    def _create_executive_summary(self, analysis_info: Dict[str, Any],
                                  insights: List[BusinessInsight],
                                  domain_analysis: Dict[str, Any]) -> str:
        """Create executive summary of business analysis."""
        database = analysis_info.get('database', 'Unknown')
        table_count = analysis_info.get('tables_analyzed', 0)

        high_impact_insights = [i for i in insights if i.impact_level in ['high', 'critical']]
        opportunities = [i for i in insights if i.insight_type == InsightType.OPPORTUNITY]
        risks = [i for i in insights if i.insight_type == InsightType.RISK]

        summary = f"""
Business Analysis Summary for {database}

Dataset Overview:
- {table_count} tables analyzed across multiple business domains
- {len(insights)} key business insights identified
- {len(high_impact_insights)} high-impact findings requiring attention

Key Findings:
- {len(opportunities)} business opportunities identified for growth and optimization
- {len(risks)} potential risks detected requiring mitigation
- {len([d for d in domain_analysis.values() if d['readiness_score'] > 0.7])} domains show high analytical readiness

Strategic Recommendations:
Focus on high-value opportunities in sales analytics and customer insights while addressing
data quality and integration challenges to unlock full analytical potential.
        """.strip()

        return summary

    def _generate_strategic_recommendations(self, business_tables: List[Dict[str, Any]],
                                            insights: List[BusinessInsight],
                                            domain_analysis: Dict[str, Any]) -> List[str]:
        """Generate strategic business recommendations."""
        recommendations = []

        # Data-driven recommendations
        high_readiness_domains = [d for d, info in domain_analysis.items()
                                  if info['readiness_score'] > 0.7]

        if high_readiness_domains:
            recommendations.append(
                f"Prioritize analytics initiatives in {', '.join(high_readiness_domains)} "
                f"domains due to high data readiness scores"
            )

        # Opportunity-based recommendations
        opportunities = [i for i in insights if i.insight_type == InsightType.OPPORTUNITY]
        if opportunities:
            recommendations.append(
                f"Implement {len(opportunities)} identified opportunities to drive business value"
            )

        # Risk mitigation recommendations
        risks = [i for i in insights if i.insight_type == InsightType.RISK]
        if risks:
            recommendations.append(
                f"Address {len(risks)} identified risks to prevent potential business impact"
            )

        return recommendations

    def _assess_business_risks(self, business_tables: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess business risks from data analysis."""
        risks = []

        # Data quality risks
        low_quality_tables = [t for t in business_tables if t.get('quality_score', 1.0) < 0.5]
        if low_quality_tables:
            risks.append({
                "risk_type": "data_quality",
                "severity": "medium",
                "description": f"{len(low_quality_tables)} tables have low data quality scores",
                "impact": "Poor decision-making due to unreliable data"
            })

        # Compliance risks
        large_tables = [t for t in business_tables if t.get('row_count', 0) > 10000000]
        if large_tables:
            risks.append({
                "risk_type": "compliance",
                "severity": "high",
                "description": f"{len(large_tables)} large tables may contain sensitive data",
                "impact": "Potential privacy and compliance violations"
            })

        return {
            "risk_analysis": risks,
            "risk_score": len(risks) / max(len(business_tables), 1),
            "mitigation_priority": "high" if len(risks) > 3 else "medium"
        }

    def _identify_business_opportunities(self, business_tables: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify business opportunities from data analysis."""
        opportunities = []

        # Analytics opportunities
        fact_tables = [t for t in business_tables if t.get('business_domain') == 'fact']
        if len(fact_tables) >= 2:
            opportunities.append({
                "opportunity_type": "analytics",
                "value_potential": "high",
                "description": "Cross-domain analytics capabilities available",
                "business_impact": "Enhanced decision-making and insights"
            })

        # Customer analytics opportunities
        if any('customer' in t.get('name', '').lower() for t in business_tables):
            opportunities.append({
                "opportunity_type": "customer_insights",
                "value_potential": "medium",
                "description": "Customer data available for segmentation and personalization",
                "business_impact": "Improved customer experience and retention"
            })

        return {
            "opportunities": opportunities,
            "opportunity_score": len(opportunities) / 5.0,  # Normalize to 0-1
            "implementation_priority": "high" if len(opportunities) > 2 else "medium"
        }

    def _group_questions_by_type(self, questions: List[BusinessQuestion]) -> Dict[str, int]:
        """Group questions by type."""
        type_counts = {}
        for question in questions:
            q_type = question.question_type.value
            type_counts[q_type] = type_counts.get(q_type, 0) + 1
        return type_counts

    def _group_questions_by_complexity(self, questions: List[BusinessQuestion]) -> Dict[str, int]:
        """Group questions by complexity."""
        complexity_counts = {}
        for question in questions:
            complexity = question.complexity.value
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        return complexity_counts

    def _add_industry_context(self, question: BusinessQuestion,
                              business_tables: List[Dict[str, Any]]) -> str:
        """Add industry-specific context to questions."""
        context = question.business_context

        # Add retail/e-commerce context for TPCDS-like schemas
        if any('sales' in t.get('name', '').lower() for t in business_tables):
            context += " This analysis supports retail performance optimization and customer experience enhancement."

        return context

    def _assess_business_relevance(self, question: BusinessQuestion,
                                   business_tables: List[Dict[str, Any]]) -> float:
        """Assess business relevance of a question."""
        relevance = 1.0

        # High relevance for sales and customer questions
        if any(keyword in question.question.lower()
               for keyword in ['sales', 'revenue', 'customer', 'profit']):
            relevance *= 1.2

        return min(1.0, relevance)

    def _load_domain_keywords(self) -> Dict[str, List[str]]:
        """Load domain-specific keywords."""
        return {
            "sales_performance": ["sales", "revenue", "order", "transaction", "payment"],
            "customer_analytics": ["customer", "client", "user", "account"],
            "product_management": ["product", "item", "catalog", "inventory"],
            "financial_reporting": ["financial", "accounting", "budget", "cost"],
            "operational_efficiency": ["operation", "process", "workflow", "efficiency"],
            "market_analysis": ["market", "competitor", "segment", "channel"],
            "risk_management": ["risk", "compliance", "audit", "security"]
        }

    def _load_industry_context(self) -> Dict[str, str]:
        """Load industry-specific context."""
        return {
            "retail": "Retail industry focuses on customer experience, inventory optimization, and sales performance",
            "finance": "Financial services emphasize risk management, compliance, and customer portfolio analysis",
            "healthcare": "Healthcare prioritizes patient outcomes, operational efficiency, and regulatory compliance"
        }


# Testing and demonstration
if __name__ == "__main__":
    print("Testing Business Analyst Agent")
    print("=" * 50)

    try:
        # Create Business Analyst Agent
        print("🤖 Creating Business Analyst Agent...")
        agent = BusinessAnalystAgent()

        print(f"✅ Agent created: {agent.agent_id}")
        print(f"   Name: {agent.agent_name}")
        print(f"   Capabilities: {', '.join(agent.get_capabilities())}")

        # Sample technical analysis (simulating Technical Analyst output)
        print(f"\n📋 Creating sample technical analysis...")
        sample_technical_analysis = {
            "analysis": {
                "database": "SNOWFLAKE_SAMPLE_DATA",
                "schema_name": "TPCDS_SF10TCL",
                "analysis_duration_seconds": 0.5,
                "tables_analyzed": 24
            },
            "table_summary": [
                {
                    "name": "STORE_SALES",
                    "business_domain": "fact",
                    "classification": "core_business",
                    "row_count": 28800239865,
                    "quality_score": 0.8,
                    "business_criticality": "critical",
                    "related_tables": ["CUSTOMER", "ITEM", "DATE_DIM"]
                },
                {
                    "name": "CUSTOMER",
                    "business_domain": "dimension",
                    "classification": "core_business",
                    "row_count": 65000000,
                    "quality_score": 0.9,
                    "business_criticality": "high",
                    "related_tables": ["STORE_SALES"]
                },
                {
                    "name": "ITEM",
                    "business_domain": "dimension",
                    "classification": "reference",
                    "row_count": 462000,
                    "quality_score": 0.85,
                    "business_criticality": "medium",
                    "related_tables": ["STORE_SALES"]
                },
                {
                    "name": "WEB_SALES",
                    "business_domain": "fact",
                    "classification": "core_business",
                    "row_count": 7199963324,
                    "quality_score": 0.75,
                    "business_criticality": "high",
                    "related_tables": ["CUSTOMER", "ITEM"]
                }
            ]
        }

        print(f"✅ Sample analysis created with {len(sample_technical_analysis['table_summary'])} tables")

        # Test 1: Generate Business Questions
        print(f"\n🔍 Test 1: Generate Business Questions")
        questions_task_id = agent.add_task(
            task_type="generate_business_questions",
            description="Generate business questions from technical analysis",
            parameters={
                "technical_analysis": sample_technical_analysis
            },
            priority=TaskPriority.HIGH
        )

        questions_result = agent.run_next_task()
        print(f"✅ Business questions generated")
        print(f"   Total questions: {questions_result['question_summary']['total_questions']}")
        print(f"   High confidence: {questions_result['question_summary']['high_confidence']}")
        print(f"   High value: {questions_result['question_summary']['high_value']}")

        # Show question types
        print(f"\n   Questions by type:")
        for q_type, count in questions_result['question_summary']['by_type'].items():
            print(f"      {q_type}: {count}")

        # Test 2: Create Comprehensive Business Report
        print(f"\n🔍 Test 2: Create Business Report")
        report_task_id = agent.add_task(
            task_type="create_business_report",
            description="Create comprehensive business analysis report",
            parameters={
                "technical_analysis": sample_technical_analysis
            },
            priority=TaskPriority.HIGH
        )

        report_result = agent.run_next_task()
        print(f"✅ Business report created")
        print(f"   Report ID: {report_result['report']['report_id']}")
        print(f"   Database: {report_result['report']['database']}")
        print(f"   Generation time: {report_result['report']['generation_duration_seconds']:.2f}s")

        # Show insights summary
        print(f"\n   Business insights:")
        print(f"      Total insights: {report_result['insights_summary']['total_insights']}")
        print(f"      High impact: {report_result['insights_summary']['high_impact']}")
        print(f"      Recommendations: {report_result['insights_summary']['recommendations']}")

        # Show domain distribution
        print(f"\n   Domain analysis:")
        for domain, count in report_result['insights_summary']['by_domain'].items():
            if count > 0:
                print(f"      {domain}: {count} insights")

        # Test 3: Identify Opportunities
        print(f"\n🔍 Test 3: Identify Business Opportunities")
        agent.add_task(
            task_type="identify_opportunities",
            description="Identify business opportunities",
            parameters={
                "business_tables": sample_technical_analysis['table_summary']
            },
            priority=TaskPriority.MEDIUM
        )

        opportunities_result = agent.run_next_task()
        print(f"✅ Opportunities identified")
        print(f"   Opportunities found: {len(opportunities_result['opportunities'])}")
        print(f"   Opportunity score: {opportunities_result['opportunity_score']:.2f}")

        if opportunities_result['opportunities']:
            print(f"\n   Key opportunities:")
            for i, opp in enumerate(opportunities_result['opportunities'][:3], 1):
                print(f"      {i}. {opp['description']} (Value: {opp['value_potential']})")

        # Show agent status and metrics
        print(f"\n📊 Agent Status and Metrics:")
        status = agent.get_status()
        metrics = status['metrics']

        print(f"   Status: {status['status']}")
        print(f"   Tasks completed: {metrics['total_tasks_completed']}")
        print(f"   Tasks failed: {metrics['total_tasks_failed']}")
        print(f"   Success rate: {metrics['success_rate']:.1%}")
        print(f"   Average task time: {metrics['average_task_time_seconds']:.2f}s")

        print(f"\n✅ Business Analyst Agent tested successfully!")
        print(f"   Core capabilities: Question generation, insight analysis, reporting")
        print(f"   Business intelligence: Domain analysis, opportunity identification")
        print(f"   Integration: Technical Analyst, Business Question Bank")
        print(f"\n🚀 Ready to support Data Scientist and Query Specialist agents!")

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        agent.logger.error("Business Analyst Agent test failed", error=str(e))
        print(f"   Check that Technical Analyst and Question Bank are working correctly")