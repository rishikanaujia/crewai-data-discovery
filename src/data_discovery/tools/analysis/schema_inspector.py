# src/data_discovery/tools/analysis/schema_inspector.py

"""
Schema Inspector Tool for comprehensive database schema discovery and analysis.

Discovers table structures, relationships, constraints, and metadata from Snowflake
with intelligent caching, error handling, and progress tracking.
"""

import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import re

# Core system imports
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from core.database_connector import get_connector
from core.state_manager import get_state_manager, StateType
from core.logging_config import get_logger
from core.config import get_config
from core.exceptions import (
    DataDiscoveryException, SchemaInspectionError,
    DatabaseTimeoutError, ErrorContext
)


@dataclass
class ColumnInfo:
    """Information about a database column."""
    name: str
    data_type: str
    is_nullable: bool
    default_value: Optional[str] = None
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_key_table: Optional[str] = None
    foreign_key_column: Optional[str] = None
    character_maximum_length: Optional[int] = None
    numeric_precision: Optional[int] = None
    numeric_scale: Optional[int] = None
    comment: Optional[str] = None

    def get_type_summary(self) -> str:
        """Get a human-readable type summary."""
        if self.data_type.upper() in ['VARCHAR', 'CHAR', 'TEXT']:
            if self.character_maximum_length:
                return f"{self.data_type}({self.character_maximum_length})"
            return self.data_type
        elif self.data_type.upper() in ['NUMBER', 'DECIMAL', 'NUMERIC']:
            if self.numeric_precision and self.numeric_scale:
                return f"{self.data_type}({self.numeric_precision},{self.numeric_scale})"
            elif self.numeric_precision:
                return f"{self.data_type}({self.numeric_precision})"
            return self.data_type
        return self.data_type

    def is_likely_id_column(self) -> bool:
        """Check if column is likely an ID column."""
        name_lower = self.name.lower()
        return (
                name_lower in ['id', 'key', 'pk'] or
                name_lower.endswith('_id') or
                name_lower.endswith('_key') or
                self.is_primary_key
        )

    def is_likely_date_column(self) -> bool:
        """Check if column is likely a date/time column."""
        name_lower = self.name.lower()
        return (
                'date' in name_lower or
                'time' in name_lower or
                'created' in name_lower or
                'updated' in name_lower or
                'modified' in name_lower or
                self.data_type.upper() in ['DATE', 'TIMESTAMP', 'TIMESTAMP_NTZ', 'TIMESTAMP_LTZ', 'TIMESTAMP_TZ']
        )


@dataclass
class TableInfo:
    """Information about a database table."""
    name: str
    schema: str
    database: str
    table_type: str = "TABLE"  # TABLE, VIEW, MATERIALIZED VIEW
    row_count: Optional[int] = None
    size_bytes: Optional[int] = None
    comment: Optional[str] = None
    created_at: Optional[datetime] = None
    last_altered: Optional[datetime] = None
    columns: List[ColumnInfo] = field(default_factory=list)

    def get_column_count(self) -> int:
        """Get number of columns."""
        return len(self.columns)

    def get_primary_key_columns(self) -> List[ColumnInfo]:
        """Get primary key columns."""
        return [col for col in self.columns if col.is_primary_key]

    def get_foreign_key_columns(self) -> List[ColumnInfo]:
        """Get foreign key columns."""
        return [col for col in self.columns if col.is_foreign_key]

    def get_nullable_columns(self) -> List[ColumnInfo]:
        """Get nullable columns."""
        return [col for col in self.columns if col.is_nullable]

    def get_id_columns(self) -> List[ColumnInfo]:
        """Get columns that appear to be IDs."""
        return [col for col in self.columns if col.is_likely_id_column()]

    def get_date_columns(self) -> List[ColumnInfo]:
        """Get date/time columns."""
        return [col for col in self.columns if col.is_likely_date_column()]

    def estimate_complexity(self) -> str:
        """Estimate table complexity based on structure."""
        column_count = self.get_column_count()
        fk_count = len(self.get_foreign_key_columns())

        if column_count <= 5 and fk_count == 0:
            return "simple"
        elif column_count <= 15 and fk_count <= 2:
            return "moderate"
        else:
            return "complex"


@dataclass
class SchemaInfo:
    """Information about a database schema."""
    database: str
    schema_name: str
    tables: List[TableInfo] = field(default_factory=list)
    discovered_at: datetime = field(default_factory=datetime.now)
    discovery_duration_seconds: float = 0.0

    def get_table_count(self) -> int:
        """Get number of tables."""
        return len(self.tables)

    def get_total_columns(self) -> int:
        """Get total number of columns across all tables."""
        return sum(table.get_column_count() for table in self.tables)

    def get_tables_by_type(self) -> Dict[str, List[TableInfo]]:
        """Group tables by type."""
        by_type = {}
        for table in self.tables:
            table_type = table.table_type
            if table_type not in by_type:
                by_type[table_type] = []
            by_type[table_type].append(table)
        return by_type

    def get_largest_tables(self, limit: int = 5) -> List[TableInfo]:
        """Get largest tables by row count."""
        return sorted(
            [t for t in self.tables if t.row_count is not None],
            key=lambda t: t.row_count,
            reverse=True
        )[:limit]

    def find_related_tables(self, table_name: str) -> List[Tuple[TableInfo, str]]:
        """Find tables related to the given table through foreign keys."""
        related = []
        target_table = next((t for t in self.tables if t.name == table_name), None)

        if target_table:
            # Find tables that reference this table
            for table in self.tables:
                for column in table.get_foreign_key_columns():
                    if column.foreign_key_table == table_name:
                        related.append((table, f"references {table_name}.{column.foreign_key_column}"))

            # Find tables this table references
            for column in target_table.get_foreign_key_columns():
                if column.foreign_key_table:
                    ref_table = next((t for t in self.tables if t.name == column.foreign_key_table), None)
                    if ref_table:
                        related.append((ref_table, f"referenced by {table_name}.{column.name}"))

        return related


class SchemaInspector:
    """Tool for discovering and analyzing database schemas."""

    def __init__(self):
        self.config = get_config()
        self.connector = get_connector()
        self.state_manager = get_state_manager()
        self.logger = get_logger("schema_inspector")

        # Configuration
        self.max_tables_per_batch = self.config.snowflake.max_tables_per_batch
        self.sample_row_limit = self.config.snowflake.max_sample_rows

    def discover_schema(
            self,
            database: str = None,
            schema_name: str = None,
            force_refresh: bool = False,
            include_row_counts: bool = True
    ) -> SchemaInfo:
        """
        Discover complete schema information for a database.

        Args:
            database: Database name (uses config default if None)
            schema_name: Schema name (uses config default if None)
            force_refresh: Skip cache and force fresh discovery
            include_row_counts: Whether to include row count statistics

        Returns:
            SchemaInfo object with complete schema metadata
        """
        # Use defaults from config
        database = database or self.config.snowflake.database
        schema_name = schema_name or self.config.snowflake.schema

        cache_key = f"{database}.{schema_name}"

        # Check cache first (unless force refresh)
        if not force_refresh:
            cached_schema = self.state_manager.load_state(
                cache_key, StateType.SCHEMA_METADATA
            )

            if cached_schema:
                self.logger.info("Using cached schema",
                                 database=database,
                                 schema=schema_name,
                                 table_count=len(cached_schema.get('tables', [])))

                # Convert from dict back to SchemaInfo
                return self._dict_to_schema_info(cached_schema)

        # Perform fresh discovery
        start_time = time.time()

        with self.state_manager.checkpoint(f"schema_discovery_{database}_{schema_name}"):
            self.logger.info("Starting schema discovery",
                             database=database,
                             schema=schema_name,
                             include_row_counts=include_row_counts)

            try:
                # Discover tables
                tables = self._discover_tables(database, schema_name)
                self.logger.info("Tables discovered",
                                 database=database,
                                 schema=schema_name,
                                 table_count=len(tables))

                # Discover columns for each table
                for i, table in enumerate(tables):
                    self.logger.debug("Discovering columns",
                                      table=table.name,
                                      progress=f"{i + 1}/{len(tables)}")

                    table.columns = self._discover_columns(database, schema_name, table.name)

                    # Get row count if requested
                    if include_row_counts:
                        table.row_count = self._get_table_row_count(database, schema_name, table.name)

                # Create schema info
                discovery_duration = time.time() - start_time
                schema_info = SchemaInfo(
                    database=database,
                    schema_name=schema_name,
                    tables=tables,
                    discovery_duration_seconds=discovery_duration
                )

                # Cache the results
                self.state_manager.save_state(
                    cache_key,
                    self._schema_info_to_dict(schema_info),
                    StateType.SCHEMA_METADATA,
                    ttl_hours=24,
                    dependencies=[f"database_{database}"]
                )

                self.logger.log_schema_discovery(
                    database=database,
                    table_count=len(tables),
                    duration_seconds=discovery_duration
                )

                return schema_info

            except Exception as e:
                self.logger.error("Schema discovery failed",
                                  database=database,
                                  schema=schema_name,
                                  error=str(e))

                raise SchemaInspectionError(
                    f"Failed to discover schema for {database}.{schema_name}: {str(e)}",
                    database=database,
                    table=schema_name
                )

    def _discover_tables(self, database: str, schema_name: str) -> List[TableInfo]:
        """Discover all tables in the schema."""
        query = """
        SELECT 
        table_name,
        table_type,
        comment,
        created,
        last_altered,
        row_count,
        bytes
    FROM information_schema.tables 
    WHERE table_schema = %(table_schema)s
      AND table_catalog = %(table_catalog)s
    ORDER BY table_name
        """

        try:
            result = self.connector.execute_query(query, {
                'table_schema': schema_name,
                'table_catalog': database
            })

            tables = []
            for row in result.data:
                table = TableInfo(
                    name=row['TABLE_NAME'],
                    schema=schema_name,
                    database=database,
                    table_type=row.get('TABLE_TYPE', 'TABLE'),
                    comment=row.get('COMMENT'),
                    created_at=row.get('CREATED'),
                    last_altered=row.get('LAST_ALTERED'),
                    row_count=row.get('ROW_COUNT'),
                    size_bytes=row.get('BYTES')
                )
                tables.append(table)

            return tables

        except Exception as e:
            raise SchemaInspectionError(
                f"Failed to discover tables: {str(e)}",
                database=database,
                table=schema_name
            )

    def _discover_columns(self, database: str, schema_name: str, table_name: str) -> List[ColumnInfo]:
        """Discover columns for a specific table."""
        query = """
        SELECT 
        column_name,
        data_type,
        is_nullable,
        column_default,
        character_maximum_length,
        numeric_precision,
        numeric_scale,
        comment
    FROM information_schema.columns 
    WHERE table_schema = %(table_schema)s
      AND table_catalog = %(table_catalog)s
      AND table_name = %(table_name)s
    ORDER BY ordinal_position
        """

        try:
            result = self.connector.execute_query(query, {
                'table_schema': schema_name,
                'table_catalog': database,
                'table_name': table_name
            })

            columns = []
            for row in result.data:
                column = ColumnInfo(
                    name=row['COLUMN_NAME'],
                    data_type=row['DATA_TYPE'],
                    is_nullable=row['IS_NULLABLE'] == 'YES',
                    default_value=row.get('COLUMN_DEFAULT'),
                    character_maximum_length=row.get('CHARACTER_MAXIMUM_LENGTH'),
                    numeric_precision=row.get('NUMERIC_PRECISION'),
                    numeric_scale=row.get('NUMERIC_SCALE'),
                    comment=row.get('COMMENT')
                )
                columns.append(column)

            # Try to detect primary keys (Snowflake-specific approach)
            try:
                pk_columns = self._discover_primary_keys(database, schema_name, table_name)
                for column in columns:
                    if column.name in pk_columns:
                        column.is_primary_key = True
            except:
                # Primary key detection failed, continue without it
                pass

            return columns

        except Exception as e:
            raise DataDiscoveryException(
                f"Failed to discover columns for {table_name}: {str(e)}"
            )

    def _discover_primary_keys(self, database: str, schema_name: str, table_name: str) -> List[str]:
        """Discover primary key columns (Snowflake-specific)."""
        query = """
        SELECT 
        kcu.column_name
    FROM information_schema.table_constraints tc
    JOIN information_schema.key_column_usage kcu
      ON tc.constraint_name = kcu.constraint_name
     AND tc.table_schema = kcu.table_schema
     AND tc.table_name = kcu.table_name
    WHERE tc.constraint_type = 'PRIMARY KEY'
      AND tc.table_schema = %(table_schema)s
      AND tc.table_catalog = %(table_catalog)s
      AND tc.table_name = %(table_name)s
    ORDER BY kcu.ordinal_position
        """

        try:
            full_table_name = f'"{database}"."{schema_name}"."{table_name}"'
            result = self.connector.execute_query(query, {'table_name': full_table_name})

            pk_columns = []
            for row in result.data:
                pk_columns.append(row.get('column_name', ''))

            return pk_columns

        except Exception:
            # Primary key discovery failed, return empty list
            return []

    def _get_table_row_count(self, database: str, schema_name: str, table_name: str) -> Optional[int]:
        """Get accurate row count for a table."""
        query = f'SELECT COUNT(*) as row_count FROM "{database}"."{schema_name}"."{table_name}"'

        try:
            result = self.connector.execute_query(query, timeout_seconds=30)
            if result.data:
                return result.data[0]['ROW_COUNT']
        except Exception as e:
            self.logger.warning("Failed to get row count",
                                table=table_name,
                                error=str(e))

        return None

    def analyze_table_relationships(self, schema_info: SchemaInfo) -> Dict[str, Any]:
        """Analyze relationships between tables in the schema."""
        analysis = {
            "total_tables": schema_info.get_table_count(),
            "total_columns": schema_info.get_total_columns(),
            "tables_by_type": schema_info.get_tables_by_type(),
            "largest_tables": [
                {"name": t.name, "rows": t.row_count}
                for t in schema_info.get_largest_tables()
            ],
            "complexity_distribution": {},
            "column_type_distribution": {},
            "potential_fact_tables": [],
            "potential_dimension_tables": []
        }

        # Analyze complexity
        complexities = [table.estimate_complexity() for table in schema_info.tables]
        for complexity in set(complexities):
            analysis["complexity_distribution"][complexity] = complexities.count(complexity)

        # Analyze column types
        all_columns = []
        for table in schema_info.tables:
            all_columns.extend(table.columns)

        data_types = [col.data_type for col in all_columns]
        for data_type in set(data_types):
            analysis["column_type_distribution"][data_type] = data_types.count(data_type)

        # Identify potential fact and dimension tables
        for table in schema_info.tables:
            fk_count = len(table.get_foreign_key_columns())
            id_count = len(table.get_id_columns())

            if fk_count >= 2 and table.row_count and table.row_count > 1000:
                analysis["potential_fact_tables"].append({
                    "name": table.name,
                    "foreign_keys": fk_count,
                    "row_count": table.row_count
                })
            elif id_count == 1 and fk_count <= 1 and table.get_column_count() <= 20:
                analysis["potential_dimension_tables"].append({
                    "name": table.name,
                    "columns": table.get_column_count(),
                    "row_count": table.row_count
                })

        return analysis

    def _schema_info_to_dict(self, schema_info: SchemaInfo) -> Dict[str, Any]:
        """Convert SchemaInfo to dictionary for caching."""
        return {
            "database": schema_info.database,
            "schema_name": schema_info.schema_name,
            "discovered_at": schema_info.discovered_at.isoformat(),
            "discovery_duration_seconds": schema_info.discovery_duration_seconds,
            "tables": [
                {
                    "name": table.name,
                    "schema": table.schema,
                    "database": table.database,
                    "table_type": table.table_type,
                    "row_count": table.row_count,
                    "size_bytes": table.size_bytes,
                    "comment": table.comment,
                    "created_at": table.created_at.isoformat() if table.created_at else None,
                    "last_altered": table.last_altered.isoformat() if table.last_altered else None,
                    "columns": [asdict(col) for col in table.columns]
                }
                for table in schema_info.tables
            ]
        }

    def _dict_to_schema_info(self, data: Dict[str, Any]) -> SchemaInfo:
        """Convert dictionary back to SchemaInfo object."""
        tables = []

        for table_data in data.get("tables", []):
            columns = [ColumnInfo(**col_data) for col_data in table_data.get("columns", [])]

            table = TableInfo(
                name=table_data["name"],
                schema=table_data["schema"],
                database=table_data["database"],
                table_type=table_data.get("table_type", "TABLE"),
                row_count=table_data.get("row_count"),
                size_bytes=table_data.get("size_bytes"),
                comment=table_data.get("comment"),
                created_at=datetime.fromisoformat(table_data["created_at"]) if table_data.get("created_at") else None,
                last_altered=datetime.fromisoformat(table_data["last_altered"]) if table_data.get(
                    "last_altered") else None,
                columns=columns
            )
            tables.append(table)

        return SchemaInfo(
            database=data["database"],
            schema_name=data["schema_name"],
            discovered_at=datetime.fromisoformat(data["discovered_at"]),
            discovery_duration_seconds=data["discovery_duration_seconds"],
            tables=tables
        )


def inspect_schema(
        database: str = None,
        schema_name: str = None,
        force_refresh: bool = False,
        include_analysis: bool = True
) -> Tuple[SchemaInfo, Optional[Dict[str, Any]]]:
    """
    Convenience function to inspect schema and optionally analyze relationships.

    Returns:
        Tuple of (SchemaInfo, analysis_dict)
    """
    inspector = SchemaInspector()

    # Discover schema
    schema_info = inspector.discover_schema(
        database=database,
        schema_name=schema_name,
        force_refresh=force_refresh
    )

    # Analyze relationships if requested
    analysis = None
    if include_analysis:
        analysis = inspector.analyze_table_relationships(schema_info)

    return schema_info, analysis


# Testing and demonstration
if __name__ == "__main__":
    print("Testing Schema Inspector Tool")
    print("=" * 50)

    logger = get_logger("test_schema_inspector")

    try:
        # Test basic connection first
        print("🔍 Testing basic connection...")
        connector = get_connector()

        # Try a very simple query first
        simple_result = connector.execute_query("SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()")
        print(f"✅ Basic connection works!")
        print(f"   Current database: {simple_result.data[0]['CURRENT_DATABASE()']}")
        print(f"   Current schema: {simple_result.data[0]['CURRENT_SCHEMA()']}")

        # Test schema inspection
        print(f"\n🔍 Discovering schema structure...")

        inspector = SchemaInspector()

        # Get current database and schema from the connection
        current_db = simple_result.data[0]['CURRENT_DATABASE()']
        current_schema = simple_result.data[0]['CURRENT_SCHEMA()']

        print(f"   Using database: {current_db}")
        print(f"   Using schema: {current_schema}")

        schema_info = inspector.discover_schema(
            database=current_db,
            schema_name=current_schema,
            include_row_counts=True
        )

        print(f"✅ Schema discovery completed!")
        print(f"   Database: {schema_info.database}")
        print(f"   Schema: {schema_info.schema_name}")
        print(f"   Tables discovered: {schema_info.get_table_count()}")
        print(f"   Total columns: {schema_info.get_total_columns()}")
        print(f"   Discovery time: {schema_info.discovery_duration_seconds:.2f} seconds")

        # Show sample tables
        print(f"\n📋 Sample tables:")
        for table in schema_info.tables[:5]:  # Show first 5 tables
            row_info = f", {table.row_count:,} rows" if table.row_count else ""
            print(f"   - {table.name}: {table.get_column_count()} columns{row_info}")

        if len(schema_info.tables) > 5:
            print(f"   ... and {len(schema_info.tables) - 5} more tables")

        # Test relationship analysis
        print(f"\n🔗 Analyzing table relationships...")
        analysis = inspector.analyze_table_relationships(schema_info)

        print(f"   Complexity distribution:")
        for complexity, count in analysis["complexity_distribution"].items():
            print(f"      {complexity}: {count} tables")

        print(f"   Most common column types:")
        top_types = sorted(analysis["column_type_distribution"].items(), key=lambda x: x[1], reverse=True)[:5]
        for data_type, count in top_types:
            print(f"      {data_type}: {count} columns")

        if analysis["largest_tables"]:
            print(f"   Largest tables:")
            for table_info in analysis["largest_tables"][:3]:
                row_count = table_info.get('rows', 'unknown')
                print(f"      {table_info['name']}: {row_count} rows")

        # Test caching
        print(f"\n💾 Testing cache functionality...")

        # Second call should use cache
        start_time = time.time()
        cached_schema = inspector.discover_schema(
            database=current_db,
            schema_name=current_schema
        )
        cache_time = time.time() - start_time

        print(f"✅ Cache test completed in {cache_time:.3f} seconds")
        print(f"   Cache hit: {'Yes' if cache_time < 1.0 else 'No'}")

        print(f"\n✅ Schema inspector tool tested successfully!")
        print(f"   Features: Discovery, caching, analysis, relationships")
        print(f"   Integration: Database connector, state manager, logging")
        print(f"   Performance: {schema_info.discovery_duration_seconds:.2f}s discovery, {cache_time:.3f}s cache")

    except Exception as e:
        logger.error("Schema inspector test failed", error=str(e))
        print(f"❌ Test failed: {str(e)}")

        # Show configuration help
        config = get_config()
        print(f"\n🔧 Configuration check:")
        print(f"   Database: {config.snowflake.database}")
        print(f"   Schema: {config.snowflake.schema}")
        print(f"   Can connect: {'Yes' if config.can_connect_to_snowflake() else 'No'}")

        print(f"\n💡 The error might be related to:")
        print(f"   - Database/schema permissions")
        print(f"   - SQL parameter binding in the connector")
        print(f"   - Information schema access rights")
        print(f"   Try using your configured database: {config.snowflake.database}.{config.snowflake.schema}")