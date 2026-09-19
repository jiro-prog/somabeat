"""FieldObserver implementation for logging and debugging.

Reference: shared_field_design.md section 7, phase1_taskflow.md T5
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median

import aiosqlite
import numpy as np

from shared_state.interface import FieldPerception, PurgeResult, Signal

logger = logging.getLogger(__name__)


@dataclass
class ObserverLogEntry:
    timestamp: datetime
    event_type: str  # "emit" | "perceive" | "purge"
    summary: str


class LoggingObserver:
    """FieldObserver that logs operations and keeps a buffer for /field command."""

    def __init__(self, buffer_size: int = 50) -> None:
        self._buffer: deque[ObserverLogEntry] = deque(maxlen=buffer_size)

    def on_emit(self, signal: Signal) -> None:
        norm = float(np.linalg.norm(signal.embedding))
        summary = (
            f"EMIT [{signal.origin.system}/{signal.origin.context}] "
            f"id={signal.signal_id[:8]} norm={norm:.3f}"
        )
        logger.info(summary)
        self._buffer.append(ObserverLogEntry(
            timestamp=datetime.now(), event_type="emit", summary=summary,
        ))

    def on_perceive(self, perception: FieldPerception) -> None:
        top_strength = perception.signals[0].strength if perception.signals else 0.0
        summary = (
            f"PERCEIVE → {len(perception.signals)} signals, "
            f"top_strength={top_strength:.4f}"
        )
        logger.info(summary)
        self._buffer.append(ObserverLogEntry(
            timestamp=datetime.now(), event_type="perceive", summary=summary,
        ))

    def on_purge(self, result: PurgeResult) -> None:
        summary = (
            f"PURGE removed={result.purged_count}, "
            f"remaining={result.remaining_count}"
        )
        logger.info(summary)
        self._buffer.append(ObserverLogEntry(
            timestamp=datetime.now(), event_type="purge", summary=summary,
        ))

    def get_recent_logs(self, n: int = 20) -> list[ObserverLogEntry]:
        """Return the most recent N log entries."""
        entries = list(self._buffer)
        return entries[-n:]

    def format_logs(self, n: int = 20) -> str:
        """Format recent logs for display."""
        entries = self.get_recent_logs(n)
        if not entries:
            return "(no field activity yet)"
        lines = []
        for e in entries:
            ts = e.timestamp.strftime("%H:%M:%S")
            lines.append(f"[{ts}] {e.summary}")
        return "\n".join(lines)


class FieldSnapshotLogger(LoggingObserver):
    """Extends LoggingObserver with SQLite snapshot recording for M1 measurement."""

    def __init__(self, db_path: str | Path, buffer_size: int = 50) -> None:
        super().__init__(buffer_size=buffer_size)
        self._db_path = Path(db_path)
        self._last_perception: FieldPerception | None = None

    def on_perceive(self, perception: FieldPerception) -> None:
        super().on_perceive(perception)
        self._last_perception = perception

    async def record_snapshot(
        self,
        trigger_context: str,
        dialogue_log_id: int | None = None,
        response_time_ms: int | None = None,
        strength_exponent: float | None = None,
    ) -> None:
        """Compute metrics from the last perception and INSERT into field_snapshots."""
        perception = self._last_perception
        if perception is None:
            return

        signals = perception.signals
        now = perception.perceived_at

        signal_count = len(signals)

        if signal_count == 0:
            await self._insert(
                timestamp=now,
                trigger_context=trigger_context,
                signal_count=0,
                signal_count_by_origin=None,
                strength_max=None,
                strength_median=None,
                strength_std=None,
                raw_norm_mean=None,
                oldest_signal_age_hours=None,
                dialogue_log_id=dialogue_log_id,
                response_time_ms=response_time_ms,
                strength_exponent=strength_exponent,
            )
            return

        # signal_count_by_origin
        origin_counts: dict[str, int] = {}
        for ps in signals:
            ctx = ps.signal.origin.context
            origin_counts[ctx] = origin_counts.get(ctx, 0) + 1

        strengths = [ps.strength for ps in signals]
        raw_norms = [float(np.linalg.norm(ps.signal.embedding)) for ps in signals]

        # oldest signal age
        oldest_emitted = min(ps.signal.emitted_at for ps in signals)
        age_hours = (now - oldest_emitted).total_seconds() / 3600.0

        await self._insert(
            timestamp=now,
            trigger_context=trigger_context,
            signal_count=signal_count,
            signal_count_by_origin=json.dumps(origin_counts),
            strength_max=max(strengths),
            strength_median=median(strengths),
            strength_std=float(np.std(strengths)),
            raw_norm_mean=float(np.mean(raw_norms)),
            oldest_signal_age_hours=age_hours,
            dialogue_log_id=dialogue_log_id,
            response_time_ms=response_time_ms,
            strength_exponent=strength_exponent,
        )

    async def _insert(
        self,
        timestamp: datetime,
        trigger_context: str,
        signal_count: int,
        signal_count_by_origin: str | None,
        strength_max: float | None,
        strength_median: float | None,
        strength_std: float | None,
        raw_norm_mean: float | None,
        oldest_signal_age_hours: float | None,
        dialogue_log_id: int | None,
        response_time_ms: int | None,
        strength_exponent: float | None = None,
    ) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """INSERT INTO field_snapshots (
                    timestamp, trigger_context, signal_count,
                    signal_count_by_origin, strength_max, strength_median,
                    strength_std, raw_norm_mean, oldest_signal_age_hours,
                    dialogue_log_id, response_time_ms, strength_exponent,
                    baseline_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'v2')""",
                (
                    timestamp.isoformat(),
                    trigger_context,
                    signal_count,
                    signal_count_by_origin,
                    strength_max,
                    strength_median,
                    strength_std,
                    raw_norm_mean,
                    oldest_signal_age_hours,
                    dialogue_log_id,
                    response_time_ms,
                    strength_exponent,
                ),
            )
            await db.commit()
