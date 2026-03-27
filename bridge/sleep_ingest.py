"""Sleep ingest bridge — transfers data from llamarcute-live to SleepyJean.

Runs at the start of the sleep cycle, before SleepyJean's night cycle.

Reference: llamarcute_live_design.md section 6.3, phase1_taskflow.md T7
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import date, timedelta
from pathlib import Path

import aiosqlite
import numpy as np

from shared_state.backends.chromadb_backend import ChromaDBField
from shared_state.emit_log import get_emit_log
from shared_state.encoder import E5SmallEncoder

logger = logging.getLogger(__name__)


async def sleep_ingest(
    field: ChromaDBField,
    encoder: E5SmallEncoder,
    llamarcute_db_path: str | Path,
    sleepyjean_db_path: str | Path,
    knowledge_index_path: str | Path,
    knowledge_index_backup_path: str | Path,
    difficulty_threshold: float = 0.5,
) -> dict:
    """Transfer today's dialogue and difficulty signals to SleepyJean.

    Returns a summary dict for logging.
    """
    summary = {
        "dialogue_transferred": 0,
        "homework_added": 0,
        "knowledge_snapshot_saved": False,
    }

    # 1. Transfer dialogue logs from llamarcute-live SQLite → SleepyJean SQLite
    try:
        transferred = await _transfer_dialogue_logs(
            llamarcute_db_path, sleepyjean_db_path,
        )
        summary["dialogue_transferred"] = transferred
        logger.info("sleep_ingest: Transferred %d dialogue entries", transferred)
    except Exception as e:
        logger.error("sleep_ingest: Dialogue transfer failed: %s", e)

    # 2. Sense difficulty signals from field → SleepyJean homework (with dialogue context)
    try:
        homework_count = await _ingest_difficulty_signals(
            field, encoder, sleepyjean_db_path, llamarcute_db_path,
        )
        summary["homework_added"] = homework_count
        logger.info("sleep_ingest: Added %d homework entries from difficulty signals", homework_count)
    except Exception as e:
        logger.error("sleep_ingest: Difficulty signal ingestion failed: %s", e)

    # 3. Snapshot knowledge_index.json for wake_export diff
    try:
        ki_path = Path(knowledge_index_path)
        backup_path = Path(knowledge_index_backup_path)
        if ki_path.exists():
            shutil.copy2(ki_path, backup_path)
            summary["knowledge_snapshot_saved"] = True
            logger.info("sleep_ingest: Saved knowledge_index snapshot")
        else:
            logger.warning("sleep_ingest: knowledge_index.json not found at %s", ki_path)
    except Exception as e:
        logger.error("sleep_ingest: Knowledge snapshot failed: %s", e)

    return summary


async def _transfer_dialogue_logs(
    llamarcute_db_path: str | Path,
    sleepyjean_db_path: str | Path,
) -> int:
    """Read today's logs from llamarcute-live and insert into SleepyJean's conversations."""
    today = date.today().isoformat()

    async with aiosqlite.connect(llamarcute_db_path) as src_db:
        src_db.row_factory = aiosqlite.Row
        cursor = await src_db.execute(
            """SELECT role, content, created_at FROM dialogue_log
               WHERE date(created_at) = ? ORDER BY id ASC""",
            (today,),
        )
        rows = await cursor.fetchall()

    if not rows:
        return 0

    async with aiosqlite.connect(sleepyjean_db_path) as dst_db:
        for row in rows:
            await dst_db.execute(
                """INSERT INTO conversations
                   (channel_id, user_id, role, content, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                ("integrated_system", "integrated_user", row["role"], row["content"], row["created_at"]),
            )
        await dst_db.commit()

    return len(rows)


async def _ingest_difficulty_signals(
    field: ChromaDBField,
    encoder: E5SmallEncoder,
    sleepyjean_db_path: str | Path,
    llamarcute_db_path: str | Path = "",
    time_horizon_hours: int = 25,
    max_homework: int = 10,
) -> int:
    """Collect difficulty signals from the field and add as homework.

    Uses snapshot() + context filter instead of sense() to avoid
    knowledge_update signals (norm=2.0) crowding out difficulty signals
    in the cosine-similarity ranking. Same pattern as the rotation_score fix.

    Two layers of dedup:
    1. time_horizon (range defence): only signals emitted since last cycle (~25h)
    2. duplicate check (entry defence): skip if same theme already queued/learned
    """
    from datetime import datetime

    now = datetime.now()
    horizon = timedelta(hours=time_horizon_hours)

    snap = await field.snapshot()
    difficulty_signals = sorted(
        [
            s for s in snap.signals
            if s.origin.context == "difficulty"
            and (now - s.emitted_at) < horizon
        ],
        key=lambda s: s.emitted_at,
        reverse=True,
    )[:max_homework]

    # Ensure context column exists (migration for existing DBs)
    async with aiosqlite.connect(sleepyjean_db_path) as db:
        try:
            await db.execute("ALTER TABLE homework ADD COLUMN context TEXT NOT NULL DEFAULT ''")
            await db.commit()
        except Exception:
            pass  # column already exists

    homework_added = 0
    async with aiosqlite.connect(sleepyjean_db_path) as db:
        for signal in difficulty_signals:
            # signal_idでllamarcute-liveのemit_logを逆引き
            log_entry = await get_emit_log(llamarcute_db_path, signal.signal_id)
            theme = log_entry if log_entry else f"difficulty:{signal.signal_id}"

            # Duplicate check: skip if same theme already exists (queued or learned)
            cursor = await db.execute(
                "SELECT COUNT(*) FROM homework WHERE theme = ? AND status IN ('queued', 'learned')",
                (theme,),
            )
            count = (await cursor.fetchone())[0]
            if count > 0:
                logger.debug("sleep_ingest: Skipping duplicate homework: %s", theme[:60])
                continue

            # Get dialogue context around the difficulty signal
            context = await _get_dialogue_context(
                llamarcute_db_path, signal.emitted_at, n_turns=5,
            )

            await db.execute(
                "INSERT INTO homework (theme, context) VALUES (?, ?)",
                (theme, context),
            )
            homework_added += 1
        if homework_added:
            await db.commit()

    return homework_added


async def _get_dialogue_context(
    llamarcute_db_path: str | Path,
    around_time,
    n_turns: int = 5,
) -> str:
    """difficulty信号の前後の対話ログを取得し、ローカル8Bで要約する。"""
    if not llamarcute_db_path:
        return ""

    try:
        async with aiosqlite.connect(llamarcute_db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT role, content FROM dialogue_log
                   WHERE created_at <= ?
                   ORDER BY id DESC LIMIT ?""",
                (around_time.isoformat(), n_turns),
            )
            rows = await cursor.fetchall()

        if not rows:
            return ""

        # 古い順に並び替え
        turns = [f"{r['role']}: {r['content'][:200]}" for r in reversed(rows)]
        turns_text = "\n".join(turns)

        # ローカル8Bで要約
        from llamarcute_live.ollama_client import chat
        summary, _ = await chat(
            system_prompt="100字以内で対話の要約を書いてください。",
            user_message=turns_text,
            timeout_sec=30,
        )
        return summary[:200] if summary else ""

    except Exception as e:
        logger.debug("Dialogue context extraction failed: %s", e)
        return ""
