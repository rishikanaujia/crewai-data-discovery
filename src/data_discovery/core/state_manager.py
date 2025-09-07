# Production enhancements for the state manager

import gzip
import lzma
from typing import Protocol
from concurrent.futures import ThreadPoolExecutor
import asyncio


class CompressionStrategy(Protocol):
    """Protocol for different compression strategies."""

    def compress(self, data: bytes) -> bytes: ...

    def decompress(self, data: bytes) -> bytes: ...

    def get_extension(self) -> str: ...


class GzipCompression:
    """Gzip compression strategy."""

    def compress(self, data: bytes) -> bytes:
        return gzip.compress(data)

    def decompress(self, data: bytes) -> bytes:
        return gzip.decompress(data)

    def get_extension(self) -> str:
        return ".gz"


class LzmaCompression:
    """LZMA compression strategy for better ratios."""

    def compress(self, data: bytes) -> bytes:
        return lzma.compress(data, preset=1)  # Fast compression

    def decompress(self, data: bytes) -> bytes:
        return lzma.decompress(data)

    def get_extension(self) -> str:
        return ".xz"


class StateTransactionManager:
    """Transaction-like support for complex state operations."""

    def __init__(self, state_manager: 'EnhancedStateManager'):
        self.state_manager = state_manager
        self.transaction_states: Dict[str, Any] = {}
        self.rollback_data: Dict[str, Any] = {}
        self.is_active = False

    def __enter__(self):
        self.is_active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Exception occurred, rollback
            self.rollback()
        else:
            # Success, commit
            self.commit()
        self.is_active = False

    def set_state(self, key: str, data: Any, state_type: StateType, **kwargs):
        """Set state within transaction."""
        if not self.is_active:
            raise ValueError("Transaction not active")

        # Store current state for rollback
        current_state = self.state_manager.load_state(key, state_type, **kwargs)
        if current_state is not None:
            self.rollback_data[f"{key}_{state_type.value}"] = current_state

        # Store new state in transaction
        self.transaction_states[f"{key}_{state_type.value}"] = (key, data, state_type, kwargs)

    def commit(self):
        """Commit all transaction states."""
        for transaction_key, (key, data, state_type, kwargs) in self.transaction_states.items():
            self.state_manager.save_state(key, data, state_type, **kwargs)
        self.transaction_states.clear()
        self.rollback_data.clear()

    def rollback(self):
        """Rollback transaction by restoring previous states."""
        for transaction_key, (key, data, state_type, kwargs) in self.transaction_states.items():
            if transaction_key in self.rollback_data:
                # Restore previous state
                previous_data = self.rollback_data[transaction_key]
                self.state_manager.save_state(key, previous_data, state_type, **kwargs)
        self.transaction_states.clear()
        self.rollback_data.clear()


class CacheWarmingStrategy:
    """Strategies for warming the cache proactively."""

    def __init__(self, state_manager: 'EnhancedStateManager'):
        self.state_manager = state_manager
        self.executor = ThreadPoolExecutor(max_workers=2)

    async def warm_schema_cache(self, databases: List[str]):
        """Warm schema cache for commonly accessed databases."""
        tasks = []
        for database in databases:
            task = asyncio.create_task(self._warm_database_schema(database))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful = sum(1 for r in results if not isinstance(r, Exception))
        self.state_manager.logger.info("Cache warming completed",
                                       databases_warmed=successful,
                                       total_databases=len(databases))

    async def _warm_database_schema(self, database: str):
        """Warm cache for a specific database."""
        # This would integrate with your database connector
        # to pre-fetch and cache schema information
        pass


class EnhancedStateManager(StateManager):
    """Enhanced state manager with compression, transactions, and advanced features."""

    def __init__(self, cache_directory: Optional[str] = None):
        super().__init__(cache_directory)

        # Enhanced features
        self.compression_strategy = GzipCompression()
        self.enable_compression = True
        self.compression_threshold_bytes = 1024  # Compress if larger than 1KB

        # Cache warming
        self.cache_warmer = CacheWarmingStrategy(self)

        # Advanced eviction policies
        self.max_cache_items = 10000
        self.eviction_policy = "lru"  # "lru", "lfu", "fifo"

        # Performance tracking
        self.performance_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "compression_saves_bytes": 0,
            "decompression_time_total": 0.0,
            "compression_time_total": 0.0
        }

    def _compress_data(self, data: bytes) -> Tuple[bytes, bool]:
        """Compress data if beneficial."""
        if not self.enable_compression or len(data) < self.compression_threshold_bytes:
            return data, False

        start_time = time.time()
        compressed_data = self.compression_strategy.compress(data)
        compression_time = time.time() - start_time

        self.performance_stats["compression_time_total"] += compression_time

        # Only use compression if it actually saves space
        if len(compressed_data) < len(data):
            space_saved = len(data) - len(compressed_data)
            self.performance_stats["compression_saves_bytes"] += space_saved
            return compressed_data, True

        return data, False

    def _decompress_data(self, data: bytes, is_compressed: bool) -> bytes:
        """Decompress data if needed."""
        if not is_compressed:
            return data

        start_time = time.time()
        decompressed_data = self.compression_strategy.decompress(data)
        decompression_time = time.time() - start_time

        self.performance_stats["decompression_time_total"] += decompression_time

        return decompressed_data

    def _get_file_path(self, state_id: str, state_type: StateType, is_compressed: bool = False) -> Path:
        """Get file path with compression extension if needed."""
        type_dir = self.cache_dir / state_type.value
        type_dir.mkdir(exist_ok=True)

        filename = f"{state_id}.pkl"
        if is_compressed:
            filename += self.compression_strategy.get_extension()

        return type_dir / filename

    def save_state(
            self,
            key: str,
            data: Any,
            state_type: StateType,
            ttl_hours: int = 24,
            dependencies: Optional[List[str]] = None,
            **kwargs
    ) -> str:
        """Enhanced save with compression and eviction."""
        with self._cache_lock:
            # Check if we need to evict items
            if len(self._metadata_cache) >= self.max_cache_items:
                self._evict_items()

            state_id = self._generate_state_id(key, state_type, **kwargs)

            try:
                # Serialize and optionally compress
                serialized_data = pickle.dumps(data)
                compressed_data, is_compressed = self._compress_data(serialized_data)

                # Calculate checksums
                checksum = self._calculate_checksum(data)

                # Save to file
                file_path = self._get_file_path(state_id, state_type, is_compressed)
                with open(file_path, 'wb') as f:
                    f.write(compressed_data)

                # Create enhanced metadata
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
                    size_bytes=len(compressed_data)
                )

                # Add compression info to metadata
                metadata.additional_info = {
                    "is_compressed": is_compressed,
                    "original_size_bytes": len(serialized_data),
                    "compression_ratio": len(compressed_data) / len(serialized_data) if is_compressed else 1.0
                }

                self._metadata_cache[state_id] = metadata

                # Memory cache for small items
                if len(compressed_data) < (5 * 1024 * 1024):
                    self._memory_cache[state_id] = data

                self._save_metadata()

                self.logger.info("Enhanced state saved",
                                 state_id=state_id,
                                 original_size=len(serialized_data),
                                 final_size=len(compressed_data),
                                 is_compressed=is_compressed)

                return state_id

            except Exception as e:
                self.logger.error("Failed to save enhanced state", error=str(e))
                raise CacheError(f"Failed to save state: {str(e)}", cache_key=state_id)

    def load_state(self, key: str, state_type: StateType, **kwargs) -> Optional[Any]:
        """Enhanced load with compression support and performance tracking."""
        with self._cache_lock:
            state_id = self._generate_state_id(key, state_type, **kwargs)

            if state_id not in self._metadata_cache:
                self.performance_stats["cache_misses"] += 1
                return None

            metadata = self._metadata_cache[state_id]

            if metadata.is_expired():
                self.performance_stats["cache_misses"] += 1
                return None

            try:
                # Check memory cache first
                if state_id in self._memory_cache:
                    self.performance_stats["cache_hits"] += 1
                    metadata.update_access()
                    return self._memory_cache[state_id]

                # Load from file
                is_compressed = metadata.additional_info.get("is_compressed", False)
                file_path = self._get_file_path(state_id, state_type, is_compressed)

                if not file_path.exists():
                    self.performance_stats["cache_misses"] += 1
                    return None

                with open(file_path, 'rb') as f:
                    file_data = f.read()

                # Decompress if needed
                serialized_data = self._decompress_data(file_data, is_compressed)
                data = pickle.load(io.BytesIO(serialized_data))

                # Verify checksum
                if metadata.checksum and self._calculate_checksum(data) != metadata.checksum:
                    self.logger.error("State checksum mismatch", state_id=state_id)
                    self.performance_stats["cache_misses"] += 1
                    return None

                self.performance_stats["cache_hits"] += 1
                metadata.update_access()

                # Add to memory cache
                if metadata.size_bytes < (5 * 1024 * 1024):
                    self._memory_cache[state_id] = data

                return data

            except Exception as e:
                self.logger.error("Failed to load enhanced state", error=str(e))
                self.performance_stats["cache_misses"] += 1
                return None

    def _evict_items(self):
        """Evict items based on configured policy."""
        if self.eviction_policy == "lru":
            self._evict_lru()
        elif self.eviction_policy == "lfu":
            self._evict_lfu()
        else:  # fifo
            self._evict_fifo()

    def _evict_lru(self):
        """Evict least recently used items."""
        # Sort by last_accessed, remove oldest 10%
        items_to_remove = int(len(self._metadata_cache) * 0.1)
        sorted_items = sorted(
            self._metadata_cache.items(),
            key=lambda x: x[1].last_accessed
        )

        for state_id, _ in sorted_items[:items_to_remove]:
            self.invalidate_state_by_id(state_id)

    def _evict_lfu(self):
        """Evict least frequently used items."""
        items_to_remove = int(len(self._metadata_cache) * 0.1)
        sorted_items = sorted(
            self._metadata_cache.items(),
            key=lambda x: x[1].access_count
        )

        for state_id, _ in sorted_items[:items_to_remove]:
            self.invalidate_state_by_id(state_id)

    def _evict_fifo(self):
        """Evict first in, first out."""
        items_to_remove = int(len(self._metadata_cache) * 0.1)
        sorted_items = sorted(
            self._metadata_cache.items(),
            key=lambda x: x[1].created_at
        )

        for state_id, _ in sorted_items[:items_to_remove]:
            self.invalidate_state_by_id(state_id)

    def invalidate_state_by_id(self, state_id: str):
        """Invalidate state by ID."""
        if state_id in self._metadata_cache:
            metadata = self._metadata_cache[state_id]

            # Remove files
            for is_compressed in [False, True]:
                file_path = self._get_file_path(state_id, metadata.state_type, is_compressed)
                if file_path.exists():
                    file_path.unlink()

            # Remove from caches
            if state_id in self._memory_cache:
                del self._memory_cache[state_id]

            del self._metadata_cache[state_id]

    def create_transaction(self) -> StateTransactionManager:
        """Create a new transaction for atomic state operations."""
        return StateTransactionManager(self)

    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get enhanced statistics including performance metrics."""
        base_stats = self.get_cache_stats()

        cache_hit_rate = 0.0
        total_requests = self.performance_stats["cache_hits"] + self.performance_stats["cache_misses"]
        if total_requests > 0:
            cache_hit_rate = self.performance_stats["cache_hits"] / total_requests

        return {
            **base_stats,
            "performance": {
                "cache_hit_rate": cache_hit_rate,
                "total_requests": total_requests,
                **self.performance_stats
            },
            "compression": {
                "enabled": self.enable_compression,
                "threshold_bytes": self.compression_threshold_bytes,
                "strategy": self.compression_strategy.__class__.__name__
            }
        }


# Example usage of enhanced features
def example_enhanced_usage():
    """Example of using enhanced state manager features."""

    # Create enhanced state manager
    state_manager = EnhancedStateManager()

    # Use transactions for atomic operations
    with state_manager.create_transaction() as txn:
        # Save related states atomically
        txn.set_state("schema_v1", {"tables": ["customers"]}, StateType.SCHEMA_METADATA)
        txn.set_state("questions_v1", [{"sql": "SELECT COUNT(*) FROM customers"}], StateType.BUSINESS_QUESTIONS)
        # Transaction commits automatically if no exceptions

    # Get enhanced statistics
    stats = state_manager.get_enhanced_stats()
    print(f"Cache hit rate: {stats['performance']['cache_hit_rate']:.2%}")
    print(f"Compression savings: {stats['performance']['compression_saves_bytes']} bytes")


if __name__ == "__main__":
    example_enhanced_usage()