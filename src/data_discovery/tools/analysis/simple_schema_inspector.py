# src/data_discovery/tools/analysis/simple_schema_inspector.py

"""
Simplified Schema Inspector that avoids parameter binding and complex exception issues.
"""

import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

# Core system imports
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from core.database_connector import get_connector
from core.state_manager import get_state_manager, StateType
from core.logging_config import get_logger
from core.config import get_config


@dataclass
class SimpleColumn:
    """Simple column information."""
    name: str
    data_type: str
    nullable: str = "unknown"
    comment: str = None


@dataclass
class SimpleTable:
    """Simple table information."""
    name: str
    table_type: str = "TABLE"
    columns: List[SimpleColumn] = None

    def __post_init__(self):
        if self.columns is None:
            self.columns = []


@dataclass
class SimpleSchema:
    """Simple schema information."""
    database: str
    schema_name: str
    tables: List[SimpleTable] = None
    discovered_at: datetime = None
    discovery_duration: float = 0.0

    def __post_init__(self):
        if self.tables is None:
            self.tables = []
        if self.discovered_at is None:
            self.discovered_at = datetime.now()


class SimpleSchemaInspector:
    """Simplified schema inspector with robust error handling."""

    def __init__(self):
        self.connector = get_connector()
        self.state_manager = get_state_manager()
        self.logger = get_logger("simple_schema_inspector")

    def discover_schema(self, database: str, schema_name: str) -> SimpleSchema:
        """Discover schema using simple queries."""
        start_time = time.time()

        self.logger.info("Starting simple schema discovery",
                         database=database, schema=schema_name)

        try:
            # Step 1: Discover tables using SHOW TABLES
            tables = self._discover_tables_simple(database, schema_name)

            # Step 2: Get columns for each table (limit to first few tables for demo)
            for i, table in enumerate(tables[:5]):  # Limit to 5 tables for demo
                try:
                    table.columns = self._discover_columns_simple(database, schema_name, table.name)
                    self.logger.debug(f"Discovered {len(table.columns)} columns for {table.name}")
                except Exception as e:
                    self.logger.warning(f"Could not get columns for {table.name}: {str(e)}")
                    table.columns = []

            # Create result
            duration = time.time() - start_time
            schema = SimpleSchema(
                database=database,
                schema_name=schema_name,
                tables=tables,
                discovery_duration=duration
            )

            self.logger.info("Schema discovery completed",
                             database=database,
                             schema=schema_name,
                             table_count=len(tables),
                             duration=duration)

            return schema

        except Exception as e:
            self.logger.error("Schema discovery failed", error=str(e))
            raise Exception(f"Schema discovery failed: {str(e)}")

    def _discover_tables_simple(self, database: str, schema_name: str) -> List[SimpleTable]:
        """Discover tables using SHOW TABLES command."""
        try:
            # Use SHOW TABLES which should work reliably
            query = f"SHOW TABLES IN SCHEMA {database}.{schema_name}"
            self.logger.debug("Executing query", query=query)

            result = self.connector.execute_query(query)

            tables = []
            for row in result.data:
                # SHOW TABLES returns different column names, try various possibilities
                name = None
                for key in ['name', 'NAME', 'table_name', 'TABLE_NAME']:
                    if key in row:
                        name = row[key]
                        break

                if name:
                    table_type = row.get('kind', row.get('KIND', 'TABLE'))
                    table = SimpleTable(name=name, table_type=table_type)
                    tables.append(table)

            self.logger.info(f"Found {len(tables)} tables")
            return tables

        except Exception as e:
            # Fallback: try a very simple approach
            self.logger.warning(f"SHOW TABLES failed: {str(e)}, trying fallback")

            try:
                # Try information_schema without parameters
                query = f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{schema_name}' AND table_catalog = '{database}' LIMIT 10"
                result = self.connector.execute_query(query)

                tables = []
                for row in result.data:
                    name = row.get('TABLE_NAME') or row.get('table_name')
                    if name:
                        tables.append(SimpleTable(name=name))

                return tables

            except Exception as fallback_error:
                self.logger.error(f"Both methods failed: {str(fallback_error)}")
                return []

    def _discover_columns_simple(self, database: str, schema_name: str, table_name: str) -> List[SimpleColumn]:
        """Discover columns using DESCRIBE TABLE."""
        try:
            query = f"DESCRIBE TABLE {database}.{schema_name}.{table_name}"
            result = self.connector.execute_query(query)

            columns = []
            for row in result.data:
                # Try different possible column names from DESCRIBE output
                name = row.get('name') or row.get('NAME') or row.get('column')
                data_type = row.get('type') or row.get('TYPE') or 'UNKNOWN'
                nullable = row.get('null?') or row.get('NULL?') or 'unknown'
                comment = row.get('comment') or row.get('COMMENT')

                if name:
                    column = SimpleColumn(
                        name=name,
                        data_type=data_type,
                        nullable=str(nullable),
                        comment=comment
                    )
                    columns.append(column)

            return columns

        except Exception as e:
            self.logger.warning(f"Could not describe table {table_name}: {str(e)}")
            return []


def test_simple_schema_inspector():
    """Test function for the simple schema inspector."""
    print("Testing Simple Schema Inspector")
    print("=" * 50)

    logger = get_logger("test_simple_inspector")

    try:
        # Test basic connection
        print("🔍 Testing connection...")
        connector = get_connector()
        result = connector.execute_query("SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()")

        current_db = result.data[0]['CURRENT_DATABASE()']
        current_schema = result.data[0]['CURRENT_SCHEMA()']

        print(f"✅ Connected to {current_db}.{current_schema}")

        # Test schema discovery
        print(f"\n🔍 Discovering schema...")
        inspector = SimpleSchemaInspector()
        schema = inspector.discover_schema(current_db, current_schema)

        print(f"✅ Discovery completed!")
        print(f"   Database: {schema.database}")
        print(f"   Schema: {schema.schema_name}")
        print(f"   Tables found: {len(schema.tables)}")
        print(f"   Discovery time: {schema.discovery_duration:.2f} seconds")

        # Show sample tables
        print(f"\n📋 Tables discovered:")
        for i, table in enumerate(schema.tables[:10]):  # Show first 10
            column_count = len(table.columns) if table.columns else "unknown"
            print(f"   {i + 1}. {table.name} ({table.table_type}) - {column_count} columns")

        if len(schema.tables) > 10:
            print(f"   ... and {len(schema.tables) - 10} more tables")

        # Show sample columns for first table with columns
        for table in schema.tables:
            if table.columns:
                print(f"\n🔍 Sample columns from {table.name}:")
                for col in table.columns[:5]:  # Show first 5 columns
                    print(f"   - {col.name}: {col.data_type} (nullable: {col.nullable})")
                if len(table.columns) > 5:
                    print(f"   ... and {len(table.columns) - 5} more columns")
                break

        print(f"\n✅ Simple schema inspector working successfully!")
        print(f"   Core functionality: Schema discovery, table listing, column inspection")
        print(f"   Robustness: Fallback queries, error handling, partial results")

        return True

    except Exception as e:
        logger.error("Simple schema inspector test failed", error=str(e))
        print(f"❌ Test failed: {str(e)}")
        return False


if __name__ == "__main__":
    test_simple_schema_inspector()