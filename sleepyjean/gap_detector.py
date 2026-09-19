"""Knowledge gap detection for SleepyJean.

Records recall failures and emits recurring gaps to the field at sleep time.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from shared_state.interface import Signal, SignalOrigin

logger = logging.getLogger(__name__)

_GAP_ORIGIN = SignalOrigin(system="sleepyjean", context="gap")


class GapDetector:
    """Detects knowledge gaps from recall failures."""

    def __init__(self, field, field_encoder, config: dict) -> None:
        self.field = field
        self.field_encoder = field_encoder
        self.min_repeat_count = config.get("min_repeat_count", 2)
        self.similarity_threshold = config.get("similarity_threshold", 0.7)
        self._failures: list[dict] = []  # {query_text, timestamp, best_similarity, embedding}

    def record_failure(self, query_text: str, best_similarity: float) -> None:
        """Record a recall failure. Called by RecallEngine."""
        embedding = self.field_encoder.encode(query_text)
        self._failures.append({
            "query_text": query_text,
            "timestamp": time.time(),
            "best_similarity": best_similarity,
            "embedding": embedding,
        })

    async def emit_gaps(self) -> list[str]:
        """Emit recurring knowledge gaps to the field.

        Groups failures by topic similarity, emits topics that failed
        min_repeat_count or more times.

        Called at the start of sleep cycle.
        """
        if not self._failures:
            return []

        # Group failures by topic similarity
        groups: list[list[dict]] = []
        for failure in self._failures:
            placed = False
            for group in groups:
                rep = group[0]
                sim = self._cosine_similarity(failure["embedding"], rep["embedding"])
                if sim >= self.similarity_threshold:
                    group.append(failure)
                    placed = True
                    break
            if not placed:
                groups.append([failure])

        # Emit groups that meet min_repeat_count
        signal_ids = []
        for group in groups:
            if len(group) < self.min_repeat_count:
                continue

            # Use average embedding of the group
            avg_emb = np.mean(
                [f["embedding"] for f in group], axis=0,
            ).astype(np.float32)

            signal = Signal.create(embedding=avg_emb, origin=_GAP_ORIGIN)
            await self.field.emit(signal)
            signal_ids.append(signal.signal_id)
            logger.info("Gap emitted: %d failures for topic (sample: %s)",
                        len(group), group[0]["query_text"][:50])

        return signal_ids

    def clear(self) -> None:
        """Clear accumulated failures. Called after emit_gaps."""
        self._failures.clear()

    @staticmethod
    def _cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
