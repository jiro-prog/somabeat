"""FieldEncoder implementation using intfloat/multilingual-e5-small.

Reference: llamarcute_live_design.md section 7.3, phase1_taskflow.md T4
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class E5SmallEncoder:
    """FieldEncoder using multilingual-e5-small (384 dimensions).

    e5 models use asymmetric prefixes:
    - "query: " for search queries (sense)
    - "passage: " for documents (emit)

    The prefix handling is internal to encode_for_emit / encode_for_sense.
    The generic encode() uses "passage: " prefix by default.
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small") -> None:
        logger.info("Loading encoder model: %s", model_name)
        self._model = SentenceTransformer(model_name, device="cpu")
        self._dimensionality = 384

    def encode(self, text: str) -> NDArray[np.float32]:
        """Encode text to a non-normalized embedding vector."""
        return self.encode_for_emit(text)

    def encode_for_emit(self, text: str) -> NDArray[np.float32]:
        """Encode text for emission (passage prefix)."""
        vec = self._model.encode(
            f"passage: {text}", normalize_embeddings=False,
        )
        return np.array(vec, dtype=np.float32)

    def encode_for_sense(self, text: str) -> NDArray[np.float32]:
        """Encode text for sensing/querying (query prefix)."""
        vec = self._model.encode(
            f"query: {text}", normalize_embeddings=False,
        )
        return np.array(vec, dtype=np.float32)

    def encode_image(self, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        """E5SmallEncoder does not support image encoding."""
        raise NotImplementedError(
            "E5SmallEncoder does not support image encoding. "
            "Use MultimodalFieldEncoder."
        )

    @property
    def dimensionality(self) -> int:
        return self._dimensionality
