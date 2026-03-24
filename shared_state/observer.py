"""FieldObserver implementation for logging and debugging.

Reference: shared_field_design.md section 7, phase1_taskflow.md T5
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from shared_state.interface import FieldReading, PurgeResult, Signal

logger = logging.getLogger(__name__)


@dataclass
class ObserverLogEntry:
    timestamp: datetime
    event_type: str  # "emit" | "sense" | "purge"
    summary: str


class LoggingObserver:
    """FieldObserver that logs operations and keeps a buffer for /field command."""

    def __init__(self, buffer_size: int = 50) -> None:
        self._buffer: deque[ObserverLogEntry] = deque(maxlen=buffer_size)

    def on_emit(self, signal: Signal) -> None:
        norm = float(np.linalg.norm(signal.embedding))
        summary = (
            f"EMIT [{signal.origin.system}/{signal.origin.context}] "
            f"norm={norm:.3f} trace=\"{signal.trace[:80]}\""
        )
        logger.info(summary)
        self._buffer.append(ObserverLogEntry(
            timestamp=datetime.now(), event_type="emit", summary=summary,
        ))

    def on_sense(self, reading: FieldReading, query_text: str | None) -> None:
        top_weight = reading.signals[0].effective_weight if reading.signals else 0.0
        query_desc = f"\"{query_text[:50]}\"" if query_text else "(embedding)"
        summary = (
            f"SENSE {query_desc} → {len(reading.signals)} signals, "
            f"top_weight={top_weight:.4f}"
        )
        logger.info(summary)
        self._buffer.append(ObserverLogEntry(
            timestamp=datetime.now(), event_type="sense", summary=summary,
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
