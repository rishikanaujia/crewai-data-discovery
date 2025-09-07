# src/data_discovery/core/state_manager.py

"""
State management system for caching, checkpoints, and incremental updates.

Handles persistent state across agent runs, schema change detection,
and intelligent caching to avoid re-processing expensive operations.
"""

import os
import json
import pickle
import hashlib
import time
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import threading
from contextlib import contextmanager

# Core system imports
import sys

sys.path.append(str(Path(__file__).parent))

from core.config import get_config
from logging_config import get_logger
from exceptions import (
    DataDiscoveryException, CacheError, ErrorContext, ErrorSeverity
)


class StateType(Enum):
    """Types of state that can be managed."""
    SCHEMA_METADATA = "schema_metadata"
    QUERY_RESULTS = "query_results"
    AGENT_PROGRESS = "agent_progress"
    BUSINESS_QUESTIONS = "business_questions"
    DATA_QUALITY = "data_quality"
    GOVERNANCE_SCAN = "governance_scan"
    USER_PREFERENCES = "user_preferences"


class CacheStatus(Enum):
    """Status of cached data."""
    FRESH = "fresh"
    STALE = "stale"
    EXPIRED = "expired"
    INVALID = "invalid"


@dataclass
class StateMetadata:
    """Metadata for cached state."""
    state_id: str
    state_type: StateType
    created_at: datetime
    last_accessed: datetime
    last_modified: datetime
    ttl_hours: int
    dependencies: List[str] = field(default_factory=list)
    checksum: Optional[str] = None
    size_bytes: int = 0
    access_count: int = 0

    def is_expired(self) -> bool:
        """Check if state has expired based on TTL."""
        if self.ttl_hours <= 0:
            return False  # Never expires

        expiry_time = self.created_at + timedelta(hours=self.ttl_hours)
        return datetime.now() > expiry_time

    def get_age_hours(self) -> float:
        """Get age of state in hours."""
        age = datetime.now() - self.created_at
        return age.total_seconds() / 3600

    def update_access(self):
        """Update access tracking."""
        self.last_accessed = datetime.now()
        self.access_count += 1


@dataclass
class SchemaChangeEvent:
    """Represents a detected schema change."""
    change_type: str  # "table_added", "table_removed", "column_added", etc.
    database: str
    table: Optional[str] = None
    column: Optional[str] = None
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    detected_at: datetime = field(default_factory=datetime.now)

    def get_impact_score(self) -> int:
        """Get impact score for prioritizing change handling."""
        impact_scores = {
            "table_added": 3,
            "table_removed": 5,
            "column_added": 2,
            "column_removed": 4,
            "column_type_changed": 3,
            "constraint_added": 1,
            "constraint_removed": 2
        }
        return impact_scores.get(self.change_type, 1)


class StateManager:
    """Central state management system with caching and change detection."""

    def __init__(self, cache_directory: Optional[str] = None):
        self.config = get_config()
        self.logger = get_logger("state_manager")

        # Setup cache directory
        self.cache_dir = Path(cache_directory or self.config.cache.get_cache_directory())
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # State tracking
        self._metadata_cache: Dict[str, StateMetadata] = {}
        self._memory_cache: Dict[str, Any] = {}
        self._cache_lock = threading.RLock()

        # Change detection
        self._schema_signatures: Dict[str, str] = {}
        self._change_listeners: List[callable] = []

        # Configuration
        self.max_memory_cache_mb = 100
        self.cleanup_interval_hours = 24

        # Load existing metadata
        self._load_metadata()

        self.logger.info("State manager initialized",
                         cache_directory=str(self.cache_dir),
                         existing_states=len(self._metadata_cache))

    def _generate_state_id(self, key: str, state_type: StateType, **kwargs) -> str:
        """Generate unique state ID from key and parameters."""
        # Include type and kwargs in hash for uniqueness
        hash_input = f"{state_type.value}:{key}:{json.dumps(sorted(kwargs.items()), default=str)}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def _get_file_path(self, state_id: str, state_type: StateType) -> Path:
        """Get file path for storing state."""
        type_dir = self.cache_dir / state_type.value
        type_dir.mkdir(exist_ok=True)
        return type_dir / f"{state_id}.pkl"

    def _get_metadata_path(self) -> Path:
        """Get path for metadata storage."""
        return self.cache_dir / "metadata.json"

    def _calculate_checksum(self, data: Any) -> str:
        """Calculate checksum for data integrity."""
        serialized = pickle.dumps(data)
        return hashlib.md5(serialized).hexdigest()

    def _load_metadata(self):
        """Load metadata from disk."""
        metadata_path = self._get_metadata_path()

        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata_dict = json.load(f)

                # Convert to StateMetadata objects
                for state_id, meta_data in metadata_dict.items():
                    meta_data['created_at'] = datetime.fromisoformat(meta_data['created_at'])
                    meta_data['last_accessed'] = datetime.fromisoformat(meta_data['last_accessed'])
                    meta_data['last_modified'] = datetime.fromisoformat(meta_data['last_modified'])
                    meta_data['state_type'] = StateType(meta_data['state_type'])

                    self._metadata_cache[state_id] = StateMetadata(**meta_data)

                self.logger.info("Metadata loaded", count=len(self._metadata_cache))

            except Exception as e:
                self.logger.error("Failed to load metadata", error=str(e))
                self._metadata_cache = {}

    def _save_metadata(self):
        """Save metadata to disk."""
        try:
            metadata_dict = {}

            for state_id, metadata in self._metadata_cache.items():
                metadata_dict[state_id] = {
                    'state_id': metadata.state_id,
                    'state_type': metadata.state_type.value,
                    'created_at': metadata.created_at.isoformat(),
                    'last_accessed': metadata.last_accessed.isoformat(),
                    'last_modified': metadata.last_modified.isoformat(),
                    'ttl_hours': metadata.ttl_hours,
                    'dependencies': metadata.dependencies,
                    'checksum': metadata.checksum,
                    'size_bytes': metadata.size_bytes,
                    'access_count': metadata.access_count
                }

            metadata_path = self._get_metadata_path()
            with open(metadata_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2)

        except Exception as e:
            self.logger.error("Failed to save metadata", error=str(e))

    def save_state(
            self,
            key: str,
            data: Any,
            state_type: StateType,
            ttl_hours: int = 24,
            dependencies: Optional[List[str]] = None,
            **kwargs
    ) -> str:
        """Save state data with metadata."""
        with self._cache_lock:
            # Generate state ID
            state_id = self._generate_state_id(key, state_type, **kwargs)

            try:
                # Calculate checksum and size
                checksum = self._calculate_checksum(data)
                serialized_data = pickle.dumps(data)
                size_bytes = len(serialized_data)

                # Save to file
                file_path = self._get_file_path(state_id, state_type)
                with open(file_path, 'wb') as f:
                    f.write(serialized_data)

                # Create metadata
                now = datetime.now()
                metadata = StateMetadata(
                    state_id=state_id,
                    state_type=state_type,
                    created_at=now,
                    last_accessed=now,
                    last_modified=now,
                    ttl_hours=ttl_hours,
                    dependencies=dependencies or [],
                    checksum=checksum,
                    size_bytes=size_bytes
                )

                # Store metadata
                self._metadata_cache[state_id] = metadata

                # Add to memory cache if small enough
                if size_bytes < (5 * 1024 * 1024):  # 5MB limit for memory cache
                    self._memory_cache[state_id] = data

                # Save metadata to disk
                self._save_metadata()

                self.logger.info("State saved",
                                 state_id=state_id,
                                 state_type=state_type.value,
                                 size_bytes=size_bytes,
                                 ttl_hours=ttl_hours)

                return state_id

            except Exception as e:
                self.logger.error("Failed to save state",
                                  state_id=state_id,
                                  error=str(e))
                raise CacheError(f"Failed to save state: {str(e)}", cache_key=state_id)

    def load_state(self, key: str, state_type: StateType, **kwargs) -> Optional[Any]:
        """Load state data if available and valid."""
        with self._cache_lock:
            state_id = self._generate_state_id(key, state_type, **kwargs)

            # Check if state exists
            if state_id not in self._metadata_cache:
                return None

            metadata = self._metadata_cache[state_id]

            # Check if expired
            if metadata.is_expired():
                self.logger.debug("State expired", state_id=state_id, age_hours=metadata.get_age_hours())
                return None

            try:
                # Try memory cache first
                if state_id in self._memory_cache:
                    metadata.update_access()
                    self.logger.debug("State loaded from memory", state_id=state_id)
                    return self._memory_cache[state_id]

                # Load from file
                file_path = self._get_file_path(state_id, state_type)
                if not file_path.exists():
                    self.logger.warning("State file missing", state_id=state_id, file_path=str(file_path))
                    return None

                with open(file_path, 'rb') as f:
                    data = pickle.load(f)

                # Verify checksum
                if metadata.checksum:
                    current_checksum = self._calculate_checksum(data)
                    if current_checksum != metadata.checksum:
                        self.logger.error("State checksum mismatch", state_id=state_id)
                        return None

                # Update access tracking
                metadata.update_access()

                # Add to memory cache
                if metadata.size_bytes < (5 * 1024 * 1024):
                    self._memory_cache[state_id] = data

                self.logger.debug("State loaded from disk",
                                  state_id=state_id,
                                  age_hours=metadata.get_age_hours())

                return data

            except Exception as e:
                self.logger.error("Failed to load state",
                                  state_id=state_id,
                                  error=str(e))
                return None

    def invalidate_state(self, key: str, state_type: StateType, **kwargs):
        """Invalidate cached state."""
        with self._cache_lock:
            state_id = self._generate_state_id(key, state_type, **kwargs)

            # Remove from memory cache
            if state_id in self._memory_cache:
                del self._memory_cache[state_id]

            # Remove file
            if state_id in self._metadata_cache:
                metadata = self._metadata_cache[state_id]
                file_path = self._get_file_path(state_id, state_type)

                try:
                    if file_path.exists():
                        file_path.unlink()
                except Exception as e:
                    self.logger.error("Failed to delete state file", error=str(e))

                # Remove metadata
                del self._metadata_cache[state_id]
                self._save_metadata()

                self.logger.info("State invalidated", state_id=state_id)

    def get_state_status(self, key: str, state_type: StateType, **kwargs) -> CacheStatus:
        """Get status of cached state."""
        state_id = self._generate_state_id(key, state_type, **kwargs)

        if state_id not in self._metadata_cache:
            return CacheStatus.INVALID

        metadata = self._metadata_cache[state_id]

        if metadata.is_expired():
            return CacheStatus.EXPIRED

        # Check if stale (older than 50% of TTL)
        age_hours = metadata.get_age_hours()
        if age_hours > (metadata.ttl_hours * 0.5):
            return CacheStatus.STALE

        return CacheStatus.FRESH

    @contextmanager
    def checkpoint(self, checkpoint_name: str):
        """Context manager for creating checkpoints."""
        start_time = time.time()
        checkpoint_id = f"checkpoint_{checkpoint_name}_{int(start_time)}"

        self.logger.info("Checkpoint started", checkpoint_id=checkpoint_id)

        try:
            yield checkpoint_id

            duration = time.time() - start_time
            self.logger.info("Checkpoint completed",
                             checkpoint_id=checkpoint_id,
                             duration_seconds=duration)

        except Exception as e:
            duration = time.time() - start_time
            self.logger.error("Checkpoint failed",
                              checkpoint_id=checkpoint_id,
                              duration_seconds=duration,
                              error=str(e))
            raise

    def detect_schema_changes(self, database: str, current_schema: Dict[str, Any]) -> List[SchemaChangeEvent]:
        """Detect changes in database schema."""
        schema_key = f"schema_signature_{database}"
        current_signature = self._calculate_checksum(current_schema)

        changes = []

        # Check if we have a previous signature
        if schema_key in self._schema_signatures:
            previous_signature = self._schema_signatures[schema_key]

            if previous_signature != current_signature:
                # Schema has changed - detailed change detection would go here
                # For now, we'll create a generic change event
                changes.append(SchemaChangeEvent(
                    change_type="schema_modified",
                    database=database
                ))

                self.logger.info("Schema change detected",
                                 database=database,
                                 change_count=len(changes))

        # Update signature
        self._schema_signatures[schema_key] = current_signature

        return changes

    def cleanup_expired_states(self):
        """Clean up expired state files and metadata."""
        with self._cache_lock:
            expired_states = []

            for state_id, metadata in self._metadata_cache.items():
                if metadata.is_expired():
                    expired_states.append(state_id)

            for state_id in expired_states:
                metadata = self._metadata_cache[state_id]

                # Remove file
                file_path = self._get_file_path(state_id, metadata.state_type)
                try:
                    if file_path.exists():
                        file_path.unlink()
                except Exception as e:
                    self.logger.error("Failed to delete expired state file", error=str(e))

                # Remove from memory cache
                if state_id in self._memory_cache:
                    del self._memory_cache[state_id]

                # Remove metadata
                del self._metadata_cache[state_id]

            if expired_states:
                self._save_metadata()
                self.logger.info("Expired states cleaned up", count=len(expired_states))

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._cache_lock:
            total_size = sum(meta.size_bytes for meta in self._metadata_cache.values())

            by_type = {}
            for metadata in self._metadata_cache.values():
                state_type = metadata.state_type.value
                if state_type not in by_type:
                    by_type[state_type] = {"count": 0, "size_bytes": 0}
                by_type[state_type]["count"] += 1
                by_type[state_type]["size_bytes"] += metadata.size_bytes

            return {
                "total_states": len(self._metadata_cache),
                "memory_cached": len(self._memory_cache),
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "by_type": by_type,
                "cache_directory": str(self.cache_dir)
            }


# Global state manager instance
_state_manager: Optional[StateManager] = None


def get_state_manager() -> StateManager:
    """Get global state manager instance."""
    global _state_manager

    if _state_manager is None:
        _state_manager = StateManager()

    return _state_manager


# Testing and demonstration
if __name__ == "__main__":
    print("Testing State Manager System")
    print("=" * 50)

    # Test state manager creation
    logger = get_logger("test_state_manager")
    state_manager = get_state_manager()

    print("✅ State manager created")
    print(f"   Cache directory: {state_manager.cache_dir}")

    # Test saving and loading state
    print("\n🔄 Testing state save/load...")

    # Sample data for testing
    test_schema = {
        "tables": {
            "customers": {
                "columns": ["id", "name", "email", "created_at"],
                "row_count": 1000
            },
            "orders": {
                "columns": ["id", "customer_id", "amount", "order_date"],
                "row_count": 5000
            }
        },
        "discovered_at": datetime.now().isoformat()
    }

    # Save state
    schema_state_id = state_manager.save_state(
        key="ANALYTICS_DB",
        data=test_schema,
        state_type=StateType.SCHEMA_METADATA,
        ttl_hours=24,
        dependencies=["database_connection"]
    )

    print(f"✅ Schema state saved with ID: {schema_state_id}")

    # Load state back
    loaded_schema = state_manager.load_state(
        key="ANALYTICS_DB",
        state_type=StateType.SCHEMA_METADATA
    )

    if loaded_schema:
        print(f"✅ Schema state loaded successfully")
        print(f"   Tables: {len(loaded_schema['tables'])}")
        print(f"   Sample table: {list(loaded_schema['tables'].keys())[0]}")
    else:
        print("❌ Failed to load schema state")

    # Test state status
    print("\n📊 Testing state status...")
    status = state_manager.get_state_status("ANALYTICS_DB", StateType.SCHEMA_METADATA)
    print(f"   State status: {status.value}")

    # Test with checkpoint
    print("\n🚧 Testing checkpoint...")
    try:
        with state_manager.checkpoint("test_operation") as checkpoint_id:
            # Simulate some work
            time.sleep(0.1)

            # Save some progress
            progress_data = {
                "step": "schema_analysis",
                "progress_percent": 50,
                "tables_processed": 2,
                "total_tables": 4
            }

            state_manager.save_state(
                key=checkpoint_id,
                data=progress_data,
                state_type=StateType.AGENT_PROGRESS,
                ttl_hours=1
            )

        print("✅ Checkpoint completed successfully")

    except Exception as e:
        print(f"❌ Checkpoint failed: {e}")

    # Test schema change detection
    print("\n🔍 Testing schema change detection...")

    # First schema version
    schema_v1 = {"table1": {"columns": ["id", "name"]}}
    changes = state_manager.detect_schema_changes("TEST_DB", schema_v1)
    print(f"   Initial schema: {len(changes)} changes")

    # Modified schema
    schema_v2 = {"table1": {"columns": ["id", "name", "email"]}}
    changes = state_manager.detect_schema_changes("TEST_DB", schema_v2)
    print(f"   Modified schema: {len(changes)} changes detected")

    if changes:
        for change in changes:
            print(f"      - {change.change_type} in {change.database}")

    # Test cache statistics
    print("\n📈 Cache statistics:")
    stats = state_manager.get_cache_stats()
    print(f"   Total states: {stats['total_states']}")
    print(f"   Memory cached: {stats['memory_cached']}")
    print(f"   Total size: {stats['total_size_mb']:.2f} MB")
    print(f"   By type:")
    for state_type, type_stats in stats['by_type'].items():
        print(f"      {state_type}: {type_stats['count']} items, {type_stats['size_bytes'] / 1024:.1f} KB")

    # Test cleanup
    print("\n🧹 Testing cleanup...")
    state_manager.cleanup_expired_states()
    print("✅ Cleanup completed")

    print(f"\n✅ State manager system tested successfully!")
    print(f"   Features: Caching, checkpoints, change detection")
    print(f"   Integration: Config system, logging, exceptions")
    print(f"   Storage: File-based with metadata tracking")
    print("\n🚀 State manager is ready for agent use!")