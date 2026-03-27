"""Emit log — records context text for each signal_id after emission.

T2: Replaces the removed trace field. Each system stores emit context
in its own SQLite database, keyed by signal_id for reverse lookup.
"""

from __future__ import annotations

import logging
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS emit_log (
    signal_id TEXT PRIMARY KEY,
    context TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""


async def ensure_emit_log_table(db_path: str | Path) -> None:
    """Create emit_log table if it doesn't exist."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute(_CREATE_TABLE_SQL)
        await db.commit()


async def insert_emit_log(
    db_path: str | Path,
    signal_id: str,
    context: str,
) -> None:
    """Record the context text for a given signal_id."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT OR IGNORE INTO emit_log (signal_id, context) VALUES (?, ?)",
            (signal_id, context),
        )
        await db.commit()


async def get_emit_log(
    db_path: str | Path,
    signal_id: str,
) -> str | None:
    """Look up the context text for a given signal_id."""
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "SELECT context FROM emit_log WHERE signal_id = ?",
            (signal_id,),
        )
        row = await cursor.fetchone()
        return row[0] if row else None
