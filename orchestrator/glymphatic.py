"""Glymphatic cleanup — infrastructure-level waste removal at sleep onset.

Rule-based, lightweight. No LLM or cognitive decisions.
"""

from __future__ import annotations

import glob
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)


async def glymphatic_cleanup(
    dialogue_db_path: str | Path,
    max_emit_log_age_days: int = 30,
    max_dialogue_log_age_days: int = 90,
    tmp_patterns: list[str] | None = None,
) -> dict:
    """Run glymphatic cleanup at sleep onset.

    1. Delete old emit_log entries (>30 days)
    2. Delete old dialogue_log entries (>90 days)
    3. Delete temporary files

    Returns summary dict.
    """
    summary = {
        "emit_log_deleted": 0,
        "dialogue_log_deleted": 0,
        "tmp_files_deleted": 0,
    }

    # 1. Prune old emit_log entries
    try:
        cutoff = (datetime.now() - timedelta(days=max_emit_log_age_days)).isoformat()
        async with aiosqlite.connect(dialogue_db_path) as db:
            # Check if emit_log table exists
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='emit_log'"
            )
            if await cursor.fetchone():
                cursor = await db.execute(
                    "DELETE FROM emit_log WHERE created_at < ?", (cutoff,)
                )
                summary["emit_log_deleted"] = cursor.rowcount
                await db.commit()
    except Exception as e:
        logger.warning("Glymphatic: emit_log cleanup failed: %s", e)

    # 2. Prune old dialogue_log entries
    try:
        cutoff = (datetime.now() - timedelta(days=max_dialogue_log_age_days)).isoformat()
        async with aiosqlite.connect(dialogue_db_path) as db:
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='dialogue_log'"
            )
            if await cursor.fetchone():
                cursor = await db.execute(
                    "DELETE FROM dialogue_log WHERE created_at < ?", (cutoff,)
                )
                summary["dialogue_log_deleted"] = cursor.rowcount
                await db.commit()
    except Exception as e:
        logger.warning("Glymphatic: dialogue_log cleanup failed: %s", e)

    # 3. Delete temporary files
    if tmp_patterns is None:
        tmp_patterns = [
            "data/*.tmp",
            "data/*.bak",
        ]
    for pattern in tmp_patterns:
        for fpath in glob.glob(pattern):
            try:
                os.remove(fpath)
                summary["tmp_files_deleted"] += 1
            except OSError as e:
                logger.debug("Glymphatic: Failed to remove %s: %s", fpath, e)

    logger.info(
        "Glymphatic cleanup: emit_log=%d, dialogue_log=%d, tmp=%d deleted",
        summary["emit_log_deleted"],
        summary["dialogue_log_deleted"],
        summary["tmp_files_deleted"],
    )
    return summary
