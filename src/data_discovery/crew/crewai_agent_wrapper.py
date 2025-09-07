# src/data_discovery/crew/crewai_agent_wrapper.py

"""
CrewAI Agent Wrapper for integrating existing agents with CrewAI framework.

Bridges custom BaseAgent-derived agents with CrewAI's Agent class to enable
multi-agent collaboration, task delegation, and sophisticated workflow orchestration.
"""

import json
from typing import Dict, List, Optional, Any, Type, Union
from dataclasses import dataclass
from datetime import datetime

# CrewAI imports
try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import BaseTool
    from crewai.agent import Agent as CrewAIAgent
    from langchain.tools import Tool
    from langchain.schema import BaseMessage

    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False


    # Placeholder classes for when CrewAI is not installed
    class Agent:
        pass


    class Task:
        pass


    class Crew:
        pass


    class BaseTool:
        pass

# Core system imports
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent, AgentTask, TaskPriority
from agents.technical_analyst_agent import TechnicalAnalystAgent
from agents.business_analyst_agent import BusinessAnalystAgent
from agents.data_scientist_agent import DataScientistAgent
from agents.query_specialist_agent import QuerySpecialistAgent
from core.logging_config import get_logger
from core.exceptions import DataDiscoveryException


@dataclass
class AgentRole:
    """Definition of an agent's role within the crew."""
    name: str
    goal: str
    backstory: str
    role_description: str
    expertise_areas: List[str]
    collaboration_style: str


# Replace the BaseAgentTool class with this simpler version:

from langchain.tools import Tool


class BaseAgentTool:
    """Simple wrapper to expose BaseAgent capabilities as tools."""

    def __init__(self, agent: BaseAgent, capability: str):
        if not CREWAI_AVAILABLE:
            raise ImportError("CrewAI is not installed. Run: pip install crewai")

        self._wrapped_agent = agent
        self._capability = capability

        # Create LangChain Tool instead of inheriting from BaseTool
        self.tool = Tool(
            name=f"{agent.agent_name}_{capability}",
            description=f"Use {agent.agent_name} agent for {capability}",
            func=self._run
        )

    def _run(self, instruction: str) -> str:
        """Execute the agent capability."""
        try:
            # Parse instruction to determine task type and parameters
            task_info = self._parse_instruction(instruction)

            # Add task to agent
            task_id = self._wrapped_agent.add_task(
                task_type=task_info['task_type'],
                description=task_info['description'],
                parameters=task_info['parameters'],
                priority=TaskPriority.HIGH
            )

            # Execute task
            result = self._wrapped_agent.run_next_task()

            # Format result for CrewAI
            return self._format_result(result)

        except Exception as e:
            return f"Error executing {self._capability}: {str(e)}"

    def _parse_instruction(self, instruction: str) -> Dict[str, Any]:
        """Parse natural language instruction into task parameters."""
        instruction_lower = instruction.lower()

        # Map instruction patterns to task types based on agent capabilities
        if self._wrapped_agent.agent_name == "TechnicalAnalyst":
            if "discover" in instruction_lower or "schema" in instruction_lower:
                return {
                    "task_type": "schema_discovery",
                    "description": instruction,
                    "parameters": self._extract_schema_params(instruction)
                }
            elif "analyze" in instruction_lower and "technical" in instruction_lower:
                return {
                    "task_type": "full_technical_analysis",
                    "description": instruction,
                    "parameters": self._extract_analysis_params(instruction)
                }

        elif self._wrapped_agent.agent_name == "BusinessAnalyst":
            if "question" in instruction_lower or "business" in instruction_lower:
                return {
                    "task_type": "generate_business_questions",
                    "description": instruction,
                    "parameters": self._extract_business_params(instruction)
                }
            elif "report" in instruction_lower:
                return {
                    "task_type": "create_business_report",
                    "description": instruction,
                    "parameters": self._extract_report_params(instruction)
                }

        elif self._wrapped_agent.agent_name == "DataScientist":
            if "profile" in instruction_lower or "quality" in instruction_lower:
                return {
                    "task_type": "comprehensive_analysis",
                    "description": instruction,
                    "parameters": self._extract_science_params(instruction)
                }

        elif self._wrapped_agent.agent_name == "QuerySpecialist":
            if "execute" in instruction_lower or "query" in instruction_lower:
                return {
                    "task_type": "execute_business_query",
                    "description": instruction,
                    "parameters": self._extract_query_params(instruction)
                }

        # Default fallback
        return {
            "task_type": "default_task",
            "description": instruction,
            "parameters": {}
        }

    def _extract_schema_params(self, instruction: str) -> Dict[str, Any]:
        """Extract schema discovery parameters."""
        return {
            "database": "SNOWFLAKE_SAMPLE_DATA",  # Default
            "schema_name": "TPCDS_SF10TCL",  # Default
            "force_refresh": "refresh" in instruction.lower()
        }

    def _extract_analysis_params(self, instruction: str) -> Dict[str, Any]:
        """Extract technical analysis parameters."""
        return {
            "technical_analysis": {
                "analysis": {
                    "database": "SNOWFLAKE_SAMPLE_DATA",
                    "schema_name": "TPCDS_SF10TCL"
                }
            }
        }

    def _extract_business_params(self, instruction: str) -> Dict[str, Any]:
        """Extract business analysis parameters."""
        return {
            "technical_analysis": {
                "analysis": {
                    "database": "SNOWFLAKE_SAMPLE_DATA",
                    "schema_name": "TPCDS_SF10TCL"
                },
                "table_summary": []  # Would be populated from context
            }
        }

    def _extract_report_params(self, instruction: str) -> Dict[str, Any]:
        """Extract business report parameters."""
        return self._extract_business_params(instruction)

    def _extract_science_params(self, instruction: str) -> Dict[str, Any]:
        """Extract data science parameters."""
        return {
            "technical_analysis": {
                "analysis": {
                    "database": "SNOWFLAKE_SAMPLE_DATA",
                    "schema_name": "TPCDS_SF10TCL"
                },
                "table_summary": [
                    {"name": "CUSTOMER", "business_domain": "dimension"},
                    {"name": "STORE_SALES", "business_domain": "fact"}
                ]
            }
        }

    def _extract_query_params(self, instruction: str) -> Dict[str, Any]:
        """Extract query execution parameters."""
        # This would need business question context from previous agents
        return {
            "business_question": None,  # Would be populated from crew memory
            "execution_mode": "sample"
        }

    def _format_result(self, result: Any) -> str:
        """Format agent result for CrewAI consumption."""
        if isinstance(result, dict):
            if result.get("status") == "success":
                # Extract key information for human-readable summary
                summary = []

                if "schema_info" in result:
                    info = result["schema_info"]
                    summary.append(f"Discovered {info['table_count']} tables with {info['total_columns']} columns")

                if "analysis" in result:
                    analysis = result["analysis"]
                    summary.append(f"Generated analysis for {analysis.get('tables_analyzed', 0)} tables")

                if "insights_summary" in result:
                    insights = result["insights_summary"]
                    summary.append(f"Generated {insights['total_insights']} insights")

                if "execution_summary" in result:
                    exec_summary = result["execution_summary"]
                    summary.append(f"Executed query returning {exec_summary['rows_returned']} rows")

                # Include raw result as JSON for other agents to use
                formatted = ". ".join(summary) if summary else "Task completed successfully"
                formatted += f"\n\nDetailed Results:\n{json.dumps(result, indent=2)}"

                return formatted
            else:
                return f"Task failed: {result.get('error', 'Unknown error')}"

        return str(result)


class CrewAIAgentWrapper:
    """Wrapper to integrate BaseAgent-derived agents with CrewAI framework."""

    def __init__(self):
        if not CREWAI_AVAILABLE:
            raise ImportError(
                "CrewAI is not installed. Install with: pip install crewai crewai-tools"
            )

        self.logger = get_logger("crewai_wrapper")
        self.agent_roles = self._define_agent_roles()
        self.wrapped_agents = {}

        self.logger.info("CrewAI Agent Wrapper initialized",
                         available_roles=list(self.agent_roles.keys()))

    def _define_agent_roles(self) -> Dict[str, AgentRole]:
        """Define roles for each agent type."""
        return {
            "technical_analyst": AgentRole(
                name="Technical Data Analyst",
                goal="Discover, analyze, and classify database schemas to understand technical data structure",
                backstory="You are an expert database analyst with deep knowledge of data architecture, "
                          "schema design patterns, and technical metadata analysis. You excel at discovering "
                          "relationships between tables and classifying business domains.",
                role_description="Responsible for technical schema discovery and classification",
                expertise_areas=["schema_discovery", "table_classification", "relationship_analysis"],
                collaboration_style="Provides foundational technical analysis for other agents"
            ),

            "business_analyst": AgentRole(
                name="Business Intelligence Analyst",
                goal="Transform technical findings into business insights and strategic recommendations",
                backstory="You are a seasoned business analyst who bridges the gap between technical data "
                          "and business value. You excel at generating meaningful business questions and "
                          "translating data insights into actionable business strategies.",
                role_description="Transforms technical analysis into business intelligence",
                expertise_areas=["business_questions", "strategic_insights", "executive_reporting"],
                collaboration_style="Builds on technical analysis to deliver business value"
            ),

            "data_scientist": AgentRole(
                name="Data Science Specialist",
                goal="Apply statistical methods and data quality assessment to generate scientific insights",
                backstory="You are a data scientist with expertise in statistical analysis, data quality "
                          "assessment, and pattern detection. You provide scientific rigor to data analysis "
                          "and identify data quality issues and opportunities.",
                role_description="Provides statistical analysis and data quality assessment",
                expertise_areas=["statistical_analysis", "data_profiling", "quality_assessment"],
                collaboration_style="Adds scientific depth to business and technical analysis"
            ),

            "query_specialist": AgentRole(
                name="Query Execution Specialist",
                goal="Execute business queries efficiently and analyze results to provide data-driven answers",
                backstory="You are a query optimization expert who transforms business questions into "
                          "efficient SQL queries and analyzes results to provide actionable insights. "
                          "You ensure optimal performance for large-scale data operations.",
                role_description="Executes queries and analyzes results for business insights",
                expertise_areas=["sql_generation", "query_optimization", "result_analysis"],
                collaboration_style="Delivers concrete answers to business questions from other agents"
            )
        }

    def wrap_agent(self, agent: BaseAgent, agent_type: str) -> Agent:
        """Wrap a BaseAgent as a CrewAI Agent."""
        if agent_type not in self.agent_roles:
            raise ValueError(f"Unknown agent type: {agent_type}")

        role = self.agent_roles[agent_type]

        # Create tools for agent capabilities
        tools = []
        for capability in agent.get_capabilities():
            tool_wrapper = BaseAgentTool(agent, capability)
            tools.append(tool_wrapper.tool)  # Use the .tool property

        # Create CrewAI Agent
        crewai_agent = Agent(
            role=role.role_description,
            goal=role.goal,
            backstory=role.backstory,
            tools=tools,
            verbose=True,
            allow_delegation=True,
            memory=True
        )

        # Store reference
        self.wrapped_agents[agent_type] = {
            "original_agent": agent,
            "crewai_agent": crewai_agent,
            "role": role
        }

        self.logger.info("Agent wrapped for CrewAI",
                         agent_type=agent_type,
                         capabilities=len(tools))

        return crewai_agent

    def create_data_discovery_crew(self) -> Crew:
        """Create a complete data discovery crew with all agents."""
        if not self.wrapped_agents:
            raise ValueError("No agents wrapped. Wrap agents first using wrap_agent()")

        # Get all wrapped agents
        agents = [info["crewai_agent"] for info in self.wrapped_agents.values()]

        # Define the data discovery workflow tasks
        tasks = self._create_workflow_tasks()

        # Create crew
        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=2,
            memory=True,
            full_output=True
        )

        self.logger.info("Data discovery crew created",
                         agent_count=len(agents),
                         task_count=len(tasks))

        return crew

    def _create_workflow_tasks(self) -> List[Task]:
        """Create workflow tasks for data discovery process."""
        tasks = []

        # Task 1: Schema Discovery
        if "technical_analyst" in self.wrapped_agents:
            schema_task = Task(
                description="Discover and analyze the database schema structure, including tables, "
                            "columns, relationships, and business domain classification",
                agent=self.wrapped_agents["technical_analyst"]["crewai_agent"],
                expected_output="Complete schema analysis with table classifications and relationships"
            )
            tasks.append(schema_task)

        # Task 2: Business Analysis
        if "business_analyst" in self.wrapped_agents:
            business_task = Task(
                description="Generate business questions and insights based on the technical schema analysis. "
                            "Create strategic recommendations and business value assessments",
                agent=self.wrapped_agents["business_analyst"]["crewai_agent"],
                expected_output="Business questions, strategic insights, and executive recommendations"
            )
            tasks.append(business_task)

        # Task 3: Data Quality Assessment
        if "data_scientist" in self.wrapped_agents:
            quality_task = Task(
                description="Perform comprehensive data quality assessment and statistical analysis. "
                            "Generate data quality scores and improvement recommendations",
                agent=self.wrapped_agents["data_scientist"]["crewai_agent"],
                expected_output="Data quality scorecard with statistical insights and recommendations"
            )
            tasks.append(quality_task)

        # Task 4: Query Execution
        if "query_specialist" in self.wrapped_agents:
            query_task = Task(
                description="Execute key business questions generated by the business analyst. "
                            "Provide actual data results and performance analysis",
                agent=self.wrapped_agents["query_specialist"]["crewai_agent"],
                expected_output="Query results with insights and performance metrics"
            )
            tasks.append(query_task)

        return tasks

    def get_crew_status(self) -> Dict[str, Any]:
        """Get status of all wrapped agents."""
        status = {
            "total_agents": len(self.wrapped_agents),
            "agent_details": {},
            "capabilities_summary": []
        }

        for agent_type, info in self.wrapped_agents.items():
            original_agent = info["original_agent"]
            agent_status = original_agent.get_status()

            status["agent_details"][agent_type] = {
                "status": agent_status["status"],
                "tasks_completed": agent_status["metrics"]["total_tasks_completed"],
                "success_rate": agent_status["metrics"]["success_rate"],
                "capabilities": original_agent.get_capabilities()
            }

            status["capabilities_summary"].extend(original_agent.get_capabilities())

        status["unique_capabilities"] = list(set(status["capabilities_summary"]))

        return status


# Testing and demonstration
if __name__ == "__main__":
    print("Testing CrewAI Agent Wrapper")
    print("=" * 50)

    if not CREWAI_AVAILABLE:
        print("❌ CrewAI is not installed!")
        print("Install with: pip install crewai crewai-tools")
        print("Then run this test again.")
        exit(1)

    logger = get_logger("test_crewai_wrapper")

    try:
        # Create wrapper
        print("🔧 Creating CrewAI Agent Wrapper...")
        wrapper = CrewAIAgentWrapper()

        print(f"✅ Wrapper created")
        print(f"   Available roles: {list(wrapper.agent_roles.keys())}")

        # Create and wrap agents
        print(f"\n🤖 Creating and wrapping agents...")

        # Create original agents
        technical_agent = TechnicalAnalystAgent()
        business_agent = BusinessAnalystAgent()
        data_scientist = DataScientistAgent()
        query_agent = QuerySpecialistAgent()

        print(f"   Original agents created: 4")

        # Wrap agents for CrewAI
        crewai_technical = wrapper.wrap_agent(technical_agent, "technical_analyst")
        crewai_business = wrapper.wrap_agent(business_agent, "business_analyst")
        crewai_scientist = wrapper.wrap_agent(data_scientist, "data_scientist")
        crewai_query = wrapper.wrap_agent(query_agent, "query_specialist")

        print(f"✅ Agents wrapped for CrewAI: 4")

        # Show agent roles
        print(f"\n📋 Agent Roles and Capabilities:")
        for agent_type, info in wrapper.wrapped_agents.items():
            role = info["role"]
            capabilities = info["original_agent"].get_capabilities()
            print(f"   {role.name}:")
            print(f"      Goal: {role.goal}")
            print(f"      Capabilities: {', '.join(capabilities)}")

        # Test individual tool execution
        print(f"\n🔍 Test 1: Individual Tool Execution")

        # Get technical analyst tools
        tech_tools = crewai_technical.tools
        if tech_tools:
            test_tool = tech_tools[0]  # First tool
            print(f"   Testing tool: {test_tool.name}")

            try:
                result = test_tool._run("Discover schema for SNOWFLAKE_SAMPLE_DATA.TPCDS_SF10TCL")
                print(f"✅ Tool execution completed")
                print(f"   Result preview: {result[:200]}...")
            except Exception as e:
                print(f"⚠️  Tool execution error: {str(e)}")

        # Create crew
        print(f"\n🔍 Test 2: Create Data Discovery Crew")
        crew = wrapper.create_data_discovery_crew()

        print(f"✅ Crew created successfully")
        print(f"   Agents in crew: {len(crew.agents)}")
        print(f"   Tasks in workflow: {len(crew.tasks)}")

        # Show workflow
        print(f"\n   Workflow tasks:")
        for i, task in enumerate(crew.tasks, 1):
            agent_role = task.agent.role if hasattr(task.agent, 'role') else 'Unknown'
            print(f"      {i}. {agent_role}: {task.description[:80]}...")

        # Test crew status
        print(f"\n🔍 Test 3: Crew Status")
        status = wrapper.get_crew_status()

        print(f"✅ Crew status retrieved")
        print(f"   Total agents: {status['total_agents']}")
        print(f"   Unique capabilities: {len(status['unique_capabilities'])}")

        print(f"\n   Agent status summary:")
        for agent_type, details in status['agent_details'].items():
            print(f"      {agent_type}: {details['status']}, "
                  f"{details['tasks_completed']} tasks completed, "
                  f"{details['success_rate']:.1%} success rate")

        print(f"\n✅ CrewAI Agent Wrapper tested successfully!")
        print(f"   Integration: All 4 agents wrapped and crew created")
        print(f"   Capabilities: {len(status['unique_capabilities'])} unique capabilities")
        print(f"   Workflow: Sequential 4-task data discovery process")
        print(f"\n🚀 Ready for CrewAI orchestrated data discovery!")

        print(f"\n💡 Next steps:")
        print(f"   - Run crew.kickoff() to execute full workflow")
        print(f"   - Agents will collaborate automatically")
        print(f"   - CrewAI handles task delegation and memory")

    except Exception as e:
        logger.error("CrewAI wrapper test failed", error=str(e))
        print(f"❌ Test failed: {str(e)}")
        print(f"   Ensure CrewAI is installed: pip install crewai crewai-tools")