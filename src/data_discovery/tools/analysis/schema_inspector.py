# Production enhancements for Schema Inspector

from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from typing import Set


class SnowflakeOptimizedInspector(SchemaInspector):
    """Enhanced inspector with Snowflake-specific optimizations."""

    def __init__(self):
        super().__init__()
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

    def discover_schema_optimized(
            self,
            database: str = None,
            schema_name: str = None,
            parallel_discovery: bool = True,
            include_statistics: bool = True
    ) -> SchemaInfo:
        """Optimized schema discovery with parallel processing."""

        database = database or self.config.snowflake.database
        schema_name = schema_name or self.config.snowflake.schema

        start_time = time.time()

        # Single query to get all table and column info at once
        unified_query = """
        WITH table_info AS (
            SELECT 
                t.table_name,
                t.table_type,
                t.comment as table_comment,
                t.created,
                t.last_altered,
                t.row_count,
                t.bytes
            FROM information_schema.tables t
            WHERE t.table_schema = %(schema_name)s
              AND t.table_catalog = %(database)s
        ),
        column_info AS (
            SELECT 
                c.table_name,
                c.column_name,
                c.data_type,
                c.is_nullable,
                c.column_default,
                c.character_maximum_length,
                c.numeric_precision,
                c.numeric_scale,
                c.comment as column_comment,
                c.ordinal_position
            FROM information_schema.columns c
            WHERE c.table_schema = %(schema_name)s
              AND c.table_catalog = %(database)s
        )
        SELECT 
            ti.*,
            ci.column_name,
            ci.data_type,
            ci.is_nullable,
            ci.column_default,
            ci.character_maximum_length,
            ci.numeric_precision,
            ci.numeric_scale,
            ci.column_comment,
            ci.ordinal_position
        FROM table_info ti
        LEFT JOIN column_info ci ON ti.table_name = ci.table_name
        ORDER BY ti.table_name, ci.ordinal_position
        """

        try:
            result = self.connector.execute_query(unified_query, {
                'schema_name': schema_name,
                'database': database
            })

            # Process results into tables and columns
            tables_dict = {}

            for row in result.data:
                table_name = row['TABLE_NAME']

                # Create table if not exists
                if table_name not in tables_dict:
                    tables_dict[table_name] = TableInfo(
                        name=table_name,
                        schema=schema_name,
                        database=database,
                        table_type=row.get('TABLE_TYPE', 'TABLE'),
                        comment=row.get('TABLE_COMMENT'),
                        created_at=row.get('CREATED'),
                        last_altered=row.get('LAST_ALTERED'),
                        row_count=row.get('ROW_COUNT'),
                        size_bytes=row.get('BYTES'),
                        columns=[]
                    )

                # Add column if present
                if row['COLUMN_NAME']:
                    column = ColumnInfo(
                        name=row['COLUMN_NAME'],
                        data_type=row['DATA_TYPE'],
                        is_nullable=row['IS_NULLABLE'] == 'YES',
                        default_value=row.get('COLUMN_DEFAULT'),
                        character_maximum_length=row.get('CHARACTER_MAXIMUM_LENGTH'),
                        numeric_precision=row.get('NUMERIC_PRECISION'),
                        numeric_scale=row.get('NUMERIC_SCALE'),
                        comment=row.get('COLUMN_COMMENT')
                    )
                    tables_dict[table_name].columns.append(column)

            tables = list(tables_dict.values())

            # Parallel constraint discovery if enabled
            if parallel_discovery:
                self._discover_constraints_parallel(database, schema_name, tables)

            # Additional statistics if requested
            if include_statistics:
                self._gather_table_statistics_parallel(database, schema_name, tables)

            discovery_duration = time.time() - start_time

            schema_info = SchemaInfo(
                database=database,
                schema_name=schema_name,
                tables=tables,
                discovery_duration_seconds=discovery_duration
            )

            self.logger.info("Optimized schema discovery completed",
                             database=database,
                             schema=schema_name,
                             tables=len(tables),
                             duration_seconds=discovery_duration)

            return schema_info

        except Exception as e:
            raise SchemaInspectionError(
                f"Optimized schema discovery failed: {str(e)}",
                database=database
            )

    def _discover_constraints_parallel(self, database: str, schema_name: str, tables: List[TableInfo]):
        """Discover constraints using parallel queries."""

        # Batch tables for constraint discovery
        batch_size = 10
        table_batches = [tables[i:i + batch_size] for i in range(0, len(tables), batch_size)]

        def discover_batch_constraints(table_batch):
            table_names = [t.name for t in table_batch]

            # Primary keys
            pk_query = """
            SELECT 
                tc.table_name,
                kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema = kcu.table_schema
            WHERE tc.constraint_type = 'PRIMARY KEY'
              AND tc.table_schema = %(schema_name)s
              AND tc.table_catalog = %(database)s
              AND tc.table_name = ANY(%(table_names)s)
            """

            try:
                pk_result = self.connector.execute_query(pk_query, {
                    'schema_name': schema_name,
                    'database': database,
                    'table_names': table_names
                })

                # Update primary key info
                pk_map = {}
                for row in pk_result.data:
                    table_name = row['TABLE_NAME']
                    column_name = row['COLUMN_NAME']
                    if table_name not in pk_map:
                        pk_map[table_name] = []
                    pk_map[table_name].append(column_name)

                # Apply to tables
                for table in table_batch:
                    if table.name in pk_map:
                        pk_columns = pk_map[table.name]
                        for column in table.columns:
                            if column.name in pk_columns:
                                column.is_primary_key = True

                return True

            except Exception as e:
                self.logger.warning("Constraint discovery failed for batch", error=str(e))
                return False

        # Execute in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(discover_batch_constraints, batch) for batch in table_batches]

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    self.logger.error("Batch constraint discovery failed", error=str(e))

    def _gather_table_statistics_parallel(self, database: str, schema_name: str, tables: List[TableInfo]):
        """Gather detailed table statistics in parallel."""

        def get_table_stats(table: TableInfo):
            """Get statistics for a single table."""
            try:
                # Quick statistics query
                stats_query = f"""
                SELECT 
                    COUNT(*) as exact_row_count,
                    COUNT(DISTINCT *) as distinct_rows
                FROM "{database}"."{schema_name}"."{table.name}"
                LIMIT 1000000
                """

                result = self.connector.execute_query(stats_query, timeout_seconds=30)

                if result.data:
                    table.row_count = result.data[0]['EXACT_ROW_COUNT']
                    # Could add more statistics here

                return True

            except Exception as e:
                self.logger.debug("Statistics gathering failed for table",
                                  table=table.name, error=str(e))
                return False

        # Process smaller tables in parallel (avoid huge tables)
        small_tables = [t for t in tables if not t.row_count or t.row_count < 1000000]

        if small_tables:
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(get_table_stats, table) for table in small_tables[:20]]

                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        self.logger.debug("Table statistics failed", error=str(e))


class SchemaRelationshipAnalyzer:
    """Advanced relationship analysis for discovered schemas."""

    def __init__(self, schema_info: SchemaInfo):
        self.schema_info = schema_info
        self.logger = get_logger("relationship_analyzer")

    def analyze_naming_patterns(self) -> Dict[str, Any]:
        """Analyze naming patterns to infer relationships."""

        patterns = {
            "id_patterns": [],
            "date_patterns": [],
            "foreign_key_candidates": [],
            "junction_tables": [],
            "naming_conventions": {}
        }

        # Analyze ID column patterns
        for table in self.schema_info.tables:
            for column in table.columns:
                if column.is_likely_id_column():
                    patterns["id_patterns"].append({
                        "table": table.name,
                        "column": column.name,
                        "pattern": self._classify_id_pattern(column.name)
                    })

                if column.is_likely_date_column():
                    patterns["date_patterns"].append({
                        "table": table.name,
                        "column": column.name,
                        "data_type": column.data_type
                    })

        # Detect potential foreign keys by naming
        patterns["foreign_key_candidates"] = self._detect_fk_candidates()

        # Detect junction tables (many-to-many relationships)
        patterns["junction_tables"] = self._detect_junction_tables()

        return patterns

    def _classify_id_pattern(self, column_name: str) -> str:
        """Classify the type of ID pattern."""
        name_lower = column_name.lower()

        if name_lower == 'id':
            return 'primary_id'
        elif name_lower.endswith('_id'):
            return 'foreign_id'
        elif name_lower.endswith('_key'):
            return 'business_key'
        elif name_lower in ['uuid', 'guid']:
            return 'uuid'
        else:
            return 'other_id'

    def _detect_fk_candidates(self) -> List[Dict[str, str]]:
        """Detect potential foreign key relationships by naming."""
        candidates = []

        # Create a map of table names for quick lookup
        table_names = {table.name.lower(): table.name for table in self.schema_info.tables}

        for table in self.schema_info.tables:
            for column in table.columns:
                column_lower = column.name.lower()

                # Look for pattern: {table_name}_id
                if column_lower.endswith('_id') and not column.is_primary_key:
                    potential_table = column_lower[:-3]  # Remove '_id'

                    # Check if there's a table with this name
                    if potential_table in table_names:
                        candidates.append({
                            "source_table": table.name,
                            "source_column": column.name,
                            "target_table": table_names[potential_table],
                            "confidence": "high"
                        })

                    # Check for plural/singular variations
                    if potential_table.endswith('s') and potential_table[:-1] in table_names:
                        candidates.append({
                            "source_table": table.name,
                            "source_column": column.name,
                            "target_table": table_names[potential_table[:-1]],
                            "confidence": "medium"
                        })

        return candidates

    def _detect_junction_tables(self) -> List[Dict[str, Any]]:
        """Detect potential junction tables for many-to-many relationships."""
        junction_candidates = []

        for table in self.schema_info.tables:
            id_columns = table.get_id_columns()
            fk_candidates = [col for col in id_columns if not col.is_primary_key]

            # Junction table characteristics:
            # - Small number of columns (typically 2-4)
            # - Multiple foreign key columns
            # - Often has composite primary key
            if (len(table.columns) <= 4 and
                    len(fk_candidates) >= 2 and
                    not any(col.is_primary_key for col in table.columns)):
                junction_candidates.append({
                    "table": table.name,
                    "potential_relationships": [col.name for col in fk_candidates],
                    "column_count": len(table.columns),
                    "confidence": "medium"
                })

        return junction_candidates


class SchemaChangeDetector:
    """Advanced schema change detection and impact analysis."""

    def __init__(self, state_manager):
        self.state_manager = state_manager
        self.logger = get_logger("schema_change_detector")

    def detect_detailed_changes(
            self,
            current_schema: SchemaInfo,
            previous_schema_key: str
    ) -> List[Dict[str, Any]]:
        """Detect detailed schema changes with impact analysis."""

        # Load previous schema
        previous_data = self.state_manager.load_state(
            previous_schema_key,
            StateType.SCHEMA_METADATA
        )

        if not previous_data:
            return []

        previous_schema = self._dict_to_schema_info(previous_data)
        changes = []

        # Create lookup maps
        current_tables = {t.name: t for t in current_schema.tables}
        previous_tables = {t.name: t for t in previous_schema.tables}

        # Detect table changes
        for table_name in set(current_tables.keys()) | set(previous_tables.keys()):
            if table_name in current_tables and table_name not in previous_tables:
                changes.append({
                    "type": "table_added",
                    "table": table_name,
                    "impact": "low",
                    "description": f"New table '{table_name}' added"
                })

            elif table_name in previous_tables and table_name not in current_tables:
                changes.append({
                    "type": "table_removed",
                    "table": table_name,
                    "impact": "high",
                    "description": f"Table '{table_name}' removed"
                })

            elif table_name in both:
                # Detect column changes
                current_table = current_tables[table_name]
                previous_table = previous_tables[table_name]

                table_changes = self._detect_column_changes(current_table, previous_table)
                changes.extend(table_changes)

        return changes

    def _detect_column_changes(self, current: TableInfo, previous: TableInfo) -> List[Dict[str, Any]]:
        """Detect changes in table columns."""
        changes = []

        current_cols = {c.name: c for c in current.columns}
        previous_cols = {c.name: c for c in previous.columns}

        for col_name in set(current_cols.keys()) | set(previous_cols.keys()):
            if col_name in current_cols and col_name not in previous_cols:
                changes.append({
                    "type": "column_added",
                    "table": current.name,
                    "column": col_name,
                    "impact": "low",
                    "description": f"Column '{col_name}' added to '{current.name}'"
                })

            elif col_name in previous_cols and col_name not in current_cols:
                changes.append({
                    "type": "column_removed",
                    "table": current.name,
                    "column": col_name,
                    "impact": "medium",
                    "description": f"Column '{col_name}' removed from '{current.name}'"
                })

            elif col_name in both:
                current_col = current_cols[col_name]
                previous_col = previous_cols[col_name]

                if current_col.data_type != previous_col.data_type:
                    changes.append({
                        "type": "column_type_changed",
                        "table": current.name,
                        "column": col_name,
                        "impact": "high",
                        "old_type": previous_col.data_type,
                        "new_type": current_col.data_type,
                        "description": f"Column '{col_name}' type changed from {previous_col.data_type} to {current_col.data_type}"
                    })

        return changes


# Example usage of enhanced features
def example_enhanced_schema_discovery():
    """Example of using enhanced schema discovery features."""

    # Create optimized inspector
    inspector = SnowflakeOptimizedInspector()

    # Discover with optimizations
    schema_info = inspector.discover_schema_optimized(
        parallel_discovery=True,
        include_statistics=True
    )

    # Analyze relationships
    analyzer = SchemaRelationshipAnalyzer(schema_info)
    patterns = analyzer.analyze_naming_patterns()

    print(f"Discovered {len(schema_info.tables)} tables")
    print(f"Found {len(patterns['foreign_key_candidates'])} potential foreign keys")
    print(f"Detected {len(patterns['junction_tables'])} potential junction tables")

    return schema_info, patterns


if __name__ == "__main__":
    example_enhanced_schema_discovery()