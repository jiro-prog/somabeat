"""Individuality experiment metrics collection.

Records measurements for experiment 3 (long-term observation):
- Dialogue sense results (signal categories and weights)
- Self-state mention detection in responses
- Immune event logging (conservative_mode activation)
- Fallback arrival rate (LLM metacognition vs programmatic override)

Storage: SQLite tables in the existing llamarcute_live.db.

Reference: individuality_taskflow.md T8
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

# Self-state mention keywords (from experiment 2, refined)
SELF_STATE_KEYWORDS = [
    "免疫", "慎重", "慎重モード", "ヘルスチェック",
    "学習した", "新しく学んだ", "忘却", "忘れた",
    "確信度", "自信",
    "困難", "難しかった", "苦手",
    "状態", "体調", "コンディション",
    "不安定", "回復",
]

# Signal category classification
SIGNAL_CATEGORIES = {
    "immune": lambda ws: ws.signal.origin.context == "immune",
    "dream": lambda ws: ws.signal.origin.context == "dream",
    "sleepyjean": lambda ws: ws.signal.origin.system == "sleepyjean",
    "difficulty": lambda ws: ws.signal.origin.context == "difficulty",
    "dialogue": lambda ws: ws.signal.origin.context == "dialogue",
}


async def init_metrics_db(db_path: str | Path) -> None:
    """Create metrics tables if they don't exist."""
    async with aiosqlite.connect(db_path) as db:
        # Per-dialogue sense results
        await db.execute("""
            CREATE TABLE IF NOT EXISTS metrics_sense (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                query_type TEXT NOT NULL,
                total_signals INTEGER NOT NULL,
                immune_count INTEGER NOT NULL DEFAULT 0,
                immune_weight_sum REAL NOT NULL DEFAULT 0.0,
                dream_count INTEGER NOT NULL DEFAULT 0,
                dream_weight_sum REAL NOT NULL DEFAULT 0.0,
                sleepyjean_count INTEGER NOT NULL DEFAULT 0,
                sleepyjean_weight_sum REAL NOT NULL DEFAULT 0.0,
                difficulty_count INTEGER NOT NULL DEFAULT 0,
                difficulty_weight_sum REAL NOT NULL DEFAULT 0.0,
                dialogue_count INTEGER NOT NULL DEFAULT 0,
                dialogue_weight_sum REAL NOT NULL DEFAULT 0.0
            )
        """)

        # Add dream columns to existing tables (idempotent migration)
        for col, col_type, default in [
            ("dream_count", "INTEGER", "0"),
            ("dream_weight_sum", "REAL", "0.0"),
        ]:
            try:
                await db.execute(
                    f"ALTER TABLE metrics_sense ADD COLUMN {col} {col_type} NOT NULL DEFAULT {default}"
                )
            except Exception:
                pass  # column already exists

        # Self-state mention detection per response
        await db.execute("""
            CREATE TABLE IF NOT EXISTS metrics_self_mention (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                user_input TEXT NOT NULL,
                response_length INTEGER NOT NULL,
                mention_count INTEGER NOT NULL,
                mentions TEXT NOT NULL DEFAULT '[]'
            )
        """)

        # Immune events
        await db.execute("""
            CREATE TABLE IF NOT EXISTS metrics_immune_event (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                event_type TEXT NOT NULL,
                severity TEXT,
                details TEXT
            )
        """)

        # Fallback arrival rate: records each self-improvement cycle
        await db.execute("""
            CREATE TABLE IF NOT EXISTS metrics_fallback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                conservative_mode_active INTEGER NOT NULL,
                max_changes_before_override INTEGER NOT NULL,
                max_changes_after_override INTEGER NOT NULL,
                fallback_triggered INTEGER NOT NULL
            )
        """)

        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_sense_created
            ON metrics_sense(created_at)
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_mention_created
            ON metrics_self_mention(created_at)
        """)
        await db.commit()

    logger.info("Metrics tables initialized in %s", db_path)


async def record_sense_result(
    db_path: str | Path,
    query_type: str,
    weighted_signals: list,
) -> None:
    """Record signal category breakdown from a sense() result.

    Args:
        query_type: "self_awareness" or "dialogue_context"
        weighted_signals: list of WeightedSignal from FieldReading
    """
    counts = {cat: 0 for cat in SIGNAL_CATEGORIES}
    weights = {cat: 0.0 for cat in SIGNAL_CATEGORIES}

    for ws in weighted_signals:
        for cat, predicate in SIGNAL_CATEGORIES.items():
            if predicate(ws):
                counts[cat] += 1
                weights[cat] += ws.effective_weight
                break  # first match only

    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """INSERT INTO metrics_sense
               (query_type, total_signals,
                immune_count, immune_weight_sum,
                dream_count, dream_weight_sum,
                sleepyjean_count, sleepyjean_weight_sum,
                difficulty_count, difficulty_weight_sum,
                dialogue_count, dialogue_weight_sum)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                query_type, len(weighted_signals),
                counts["immune"], weights["immune"],
                counts["dream"], weights["dream"],
                counts["sleepyjean"], weights["sleepyjean"],
                counts["difficulty"], weights["difficulty"],
                counts["dialogue"], weights["dialogue"],
            ),
        )
        await db.commit()


def detect_self_mentions(text: str) -> list[str]:
    """Detect self-state keywords in response text."""
    return [kw for kw in SELF_STATE_KEYWORDS if kw in text]


async def record_self_mention(
    db_path: str | Path,
    user_input: str,
    response: str,
) -> int:
    """Record self-state mention detection for a response.

    Returns the number of mentions detected.
    """
    import json
    mentions = detect_self_mentions(response)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """INSERT INTO metrics_self_mention
               (user_input, response_length, mention_count, mentions)
               VALUES (?, ?, ?, ?)""",
            (user_input[:200], len(response), len(mentions), json.dumps(mentions, ensure_ascii=False)),
        )
        await db.commit()
    return len(mentions)


async def record_immune_event(
    db_path: str | Path,
    event_type: str,
    severity: str | None = None,
    details: str | None = None,
) -> None:
    """Record an immune system event.

    event_type: "conservative_mode_activated", "conservative_mode_expired",
                "health_check_warning", "health_check_critical",
                "personality_rollback"
    """
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """INSERT INTO metrics_immune_event
               (event_type, severity, details)
               VALUES (?, ?, ?)""",
            (event_type, severity, details),
        )
        await db.commit()
    logger.info("Immune event recorded: %s (%s)", event_type, severity or "")


async def record_fallback(
    db_path: str | Path,
    conservative_mode_active: bool,
    max_changes_before: int,
    max_changes_after: int,
) -> None:
    """Record a self-improvement cycle's fallback status.

    - conservative_mode_active: whether immune system had conservative_mode signal
    - max_changes_before: what determine_max_changes() returned (before immune override)
    - max_changes_after: actual max_changes used (after immune override if any)
    - fallback_triggered: True if immune override was needed (before != after)
    """
    fallback = 1 if max_changes_before != max_changes_after else 0
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """INSERT INTO metrics_fallback
               (conservative_mode_active, max_changes_before_override,
                max_changes_after_override, fallback_triggered)
               VALUES (?, ?, ?, ?)""",
            (int(conservative_mode_active), max_changes_before, max_changes_after, fallback),
        )
        await db.commit()


# --- Query helpers for experiment 3 analysis ---

async def get_sense_summary(
    db_path: str | Path,
    days: int = 7,
) -> dict:
    """Get aggregated sense metrics for the last N days."""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT
                COUNT(*) as total_queries,
                AVG(immune_count) as avg_immune_count,
                AVG(immune_weight_sum) as avg_immune_weight,
                AVG(dream_count) as avg_dream_count,
                AVG(dream_weight_sum) as avg_dream_weight,
                AVG(sleepyjean_count) as avg_sj_count,
                AVG(sleepyjean_weight_sum) as avg_sj_weight,
                AVG(difficulty_count) as avg_diff_count,
                AVG(difficulty_weight_sum) as avg_diff_weight,
                AVG(total_signals) as avg_total_signals
               FROM metrics_sense
               WHERE created_at >= datetime('now', 'localtime', ?)""",
            (f"-{days} days",),
        )
        row = await cursor.fetchone()
        return dict(row) if row else {}


async def get_self_mention_rate(
    db_path: str | Path,
    days: int = 7,
) -> dict:
    """Get self-state mention rate for the last N days."""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT
                COUNT(*) as total_responses,
                SUM(CASE WHEN mention_count > 0 THEN 1 ELSE 0 END) as responses_with_mentions,
                AVG(mention_count) as avg_mentions,
                AVG(response_length) as avg_response_length
               FROM metrics_self_mention
               WHERE created_at >= datetime('now', 'localtime', ?)""",
            (f"-{days} days",),
        )
        row = await cursor.fetchone()
        return dict(row) if row else {}


async def get_fallback_rate(
    db_path: str | Path,
    days: int = 30,
) -> dict:
    """Get fallback arrival rate for the last N days."""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT
                COUNT(*) as total_cycles,
                SUM(conservative_mode_active) as conservative_mode_cycles,
                SUM(fallback_triggered) as fallback_triggered_cycles,
                CASE WHEN SUM(conservative_mode_active) > 0
                     THEN CAST(SUM(fallback_triggered) AS REAL) / SUM(conservative_mode_active)
                     ELSE NULL END as fallback_rate
               FROM metrics_fallback
               WHERE created_at >= datetime('now', 'localtime', ?)""",
            (f"-{days} days",),
        )
        row = await cursor.fetchone()
        return dict(row) if row else {}
