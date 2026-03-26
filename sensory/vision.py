"""SensoryVision — visual perception module for the shared field.

Processes image frames, applies a thalamic filter (novelty gate) to suppress
redundant inputs, and emits visual signals into the shared field.

The thalamic filter uses cosine distance between consecutive frames to
determine if the visual input has changed enough to warrant emission.

Reference: docs/design/instructions_vision_phase_c.md C-3
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from shared_state.interface import FieldEncoder, SharedField, Signal, SignalOrigin

logger = logging.getLogger(__name__)


class SensoryVision:
    """Visual sensory module that emits image embeddings into the shared field.

    The thalamic filter suppresses emissions when consecutive frames are
    too similar (cosine distance < threshold), preventing field pollution
    from static or near-static visual input.
    """

    def __init__(
        self,
        encoder: FieldEncoder,
        field: SharedField,
        thalamus_threshold: float = 0.1,
    ) -> None:
        self._encoder = encoder
        self._field = field
        self._thalamus_threshold = thalamus_threshold
        self._last_embedding: NDArray[np.float32] | None = None

    async def process_frame(self, image: NDArray[np.uint8]) -> bool:
        """Process an image frame. Returns True if emitted, False if suppressed."""
        embedding = self._encoder.encode_image(image)

        # Thalamic filter: suppress if too similar to previous frame
        if self._last_embedding is not None:
            norm_a = np.linalg.norm(embedding)
            norm_b = np.linalg.norm(self._last_embedding)
            if norm_a > 0 and norm_b > 0:
                cosine_dist = 1.0 - np.dot(embedding, self._last_embedding) / (norm_a * norm_b)
                if cosine_dist < self._thalamus_threshold:
                    logger.debug("Thalamic filter: suppressed (cosine_dist=%.4f < %.4f)", cosine_dist, self._thalamus_threshold)
                    return False

        # Emit visual signal
        signal = Signal.create(
            embedding=embedding,
            origin=SignalOrigin(system="sensory:vision", context="frame"),
            trace="[視覚] 画面キャプチャの視覚信号",
        )
        await self._field.emit(signal)
        self._last_embedding = embedding
        logger.debug("Visual signal emitted: %s", signal.signal_id)
        return True

    def clear(self) -> None:
        """Reset state (called on sleep entry)."""
        self._last_embedding = None
        logger.info("SensoryVision state cleared")
