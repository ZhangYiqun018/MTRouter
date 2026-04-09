"""
Tool result cache with SQLite storage.

Provides a global singleton cache for tool execution results with:
- Exact match caching (based on tool name + argument hash)
- LRU eviction when max size is reached
- Optional TTL expiration
- Persistent storage via SQLite
- Thread-safe operations
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import ToolResult


@dataclass
class CacheConfig:
    """Configuration for the tool cache."""

    enabled: bool = True
    max_size: int = 10000
    db_path: str | None = None  # e.g., ~/.miniagenticrouter/tool_cache.db
    ttl_seconds: int | None = None  # Cache expiration time, None = never expire


class GlobalToolCache:
    """
    Global singleton tool cache using SQLite storage.

    Features:
    - Thread-safe (SQLite built-in locking)
    - Incremental updates (no full rewrites needed)
    - LRU eviction based on accessed_at
    - Optional TTL expiration based on created_at
    - Queryable for debugging
    - Zero external dependencies (Python built-in sqlite3)
    """

    _instance: GlobalToolCache | None = None
    _lock: Lock = Lock()

    # SQL statements
    _CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS tool_cache (
            cache_key TEXT PRIMARY KEY,
            tool_name TEXT NOT NULL,
            output TEXT NOT NULL,
            returncode INTEGER NOT NULL,
            done INTEGER NOT NULL,
            error TEXT,
            metadata TEXT,
            created_at REAL NOT NULL,
            accessed_at REAL NOT NULL
        )
    """
    _CREATE_INDEX = """
        CREATE INDEX IF NOT EXISTS idx_accessed_at ON tool_cache(accessed_at)
    """
    _CREATE_TOOL_INDEX = """
        CREATE INDEX IF NOT EXISTS idx_tool_name ON tool_cache(tool_name)
    """

    @classmethod
    def get_instance(cls, config: CacheConfig | None = None) -> GlobalToolCache:
        """Get or create the singleton cache instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(config or CacheConfig())
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.close()
            cls._instance = None

    def __init__(self, config: CacheConfig) -> None:
        self.config = config
        self._stats = {"hits": 0, "misses": 0}
        self._conn: sqlite3.Connection | None = None
        self._db_lock = Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database."""
        if self.config.db_path:
            path = Path(self.config.db_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path), check_same_thread=False)
        else:
            # In-memory database (no persistence)
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)

        self._conn.execute(self._CREATE_TABLE)
        self._conn.execute(self._CREATE_INDEX)
        self._conn.execute(self._CREATE_TOOL_INDEX)
        self._conn.commit()

    @contextmanager
    def _transaction(self):
        """Transaction context manager."""
        with self._db_lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def get(self, key: str) -> ToolResult | None:
        """Get a cached result by key."""
        # Import here to avoid circular import
        from .base import ToolResult

        with self._db_lock:
            cursor = self._conn.execute(
                "SELECT output, returncode, done, error, metadata, created_at "
                "FROM tool_cache WHERE cache_key = ?",
                (key,),
            )
            row = cursor.fetchone()

        if row is None:
            self._stats["misses"] += 1
            return None

        output, returncode, done, error, metadata_json, created_at = row

        # Check TTL
        if self.config.ttl_seconds is not None:
            age = time.time() - created_at
            if age > self.config.ttl_seconds:
                self._delete(key)
                self._stats["misses"] += 1
                return None

        # Update access time (LRU)
        with self._db_lock:
            self._conn.execute(
                "UPDATE tool_cache SET accessed_at = ? WHERE cache_key = ?",
                (time.time(), key),
            )
            self._conn.commit()

        self._stats["hits"] += 1

        return ToolResult(
            output=output,
            returncode=returncode,
            done=bool(done),
            error=error,
            metadata=json.loads(metadata_json) if metadata_json else {},
        )

    def set(self, key: str, result: ToolResult, tool_name: str = "") -> None:
        """Set a cached result."""
        # Check if eviction is needed
        self._enforce_max_size()

        now = time.time()
        with self._transaction():
            self._conn.execute(
                """
                INSERT OR REPLACE INTO tool_cache
                (cache_key, tool_name, output, returncode, done, error, metadata, created_at, accessed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    tool_name,
                    result.output,
                    result.returncode,
                    int(result.done),
                    result.error,
                    json.dumps(result.metadata) if result.metadata else None,
                    now,
                    now,
                ),
            )

    def _delete(self, key: str) -> None:
        """Delete a cache entry."""
        with self._db_lock:
            self._conn.execute("DELETE FROM tool_cache WHERE cache_key = ?", (key,))
            self._conn.commit()

    def _enforce_max_size(self) -> None:
        """Perform LRU eviction to keep cache size within limits."""
        with self._db_lock:
            cursor = self._conn.execute("SELECT COUNT(*) FROM tool_cache")
            count = cursor.fetchone()[0]

            if count >= self.config.max_size:
                # Delete oldest 10% of entries
                delete_count = max(1, self.config.max_size // 10)
                self._conn.execute(
                    """
                    DELETE FROM tool_cache WHERE cache_key IN (
                        SELECT cache_key FROM tool_cache
                        ORDER BY accessed_at ASC
                        LIMIT ?
                    )
                    """,
                    (delete_count,),
                )
                self._conn.commit()

    def get_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        with self._db_lock:
            cursor = self._conn.execute("SELECT COUNT(*) FROM tool_cache")
            size = cursor.fetchone()[0]
        return {**self._stats, "size": size}

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._db_lock:
            self._conn.execute("DELETE FROM tool_cache")
            self._conn.commit()
        self._stats = {"hits": 0, "misses": 0}

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def vacuum(self) -> None:
        """Compress the database file."""
        with self._db_lock:
            self._conn.execute("VACUUM")

    def get_entries_by_tool(self, tool_name: str) -> list[dict[str, Any]]:
        """Query all cache entries for a specific tool (for debugging)."""
        with self._db_lock:
            cursor = self._conn.execute(
                "SELECT cache_key, output, returncode, created_at "
                "FROM tool_cache WHERE tool_name = ?",
                (tool_name,),
            )
            return [
                {
                    "key": row[0],
                    "output": row[1][:100] + "..." if len(row[1]) > 100 else row[1],
                    "returncode": row[2],
                    "created_at": row[3],
                }
                for row in cursor.fetchall()
            ]

    def delete_by_tool(self, tool_name: str) -> int:
        """Delete all cache entries for a specific tool. Returns count deleted."""
        with self._db_lock:
            cursor = self._conn.execute(
                "DELETE FROM tool_cache WHERE tool_name = ?", (tool_name,)
            )
            self._conn.commit()
            return cursor.rowcount


def compute_cache_key(
    tool_name: str,
    arguments: dict[str, Any],
    key_fields: list[str] | None = None,
) -> str:
    """
    Compute a cache key from tool name and arguments.

    Args:
        tool_name: Name of the tool
        arguments: Tool arguments
        key_fields: If specified, only use these fields for the key.
                   If None, use all arguments.

    Returns:
        Cache key string in format "tool_name:hash"
    """
    if key_fields is not None:
        key_data = {k: arguments.get(k) for k in key_fields}
    else:
        key_data = arguments

    normalized = json.dumps(key_data, sort_keys=True, default=str)
    hash_val = hashlib.sha256(normalized.encode()).hexdigest()[:16]
    return f"{tool_name}:{hash_val}"
