"""Shared test fixtures for the integrated-system test suite.

Note: conftest.py is auto-loaded by pytest. Fixtures defined here are
available to all test files in this directory without explicit import.
For shared helper functions, define them in tests/helpers/ instead.
"""

import numpy as np
import pytest


class SeededMockEncoder:
    """Deterministic mock encoder using seeded random for reproducibility.

    Each unique text input produces a deterministic 384-dim vector.
    Replaces unseeded np.random.randn() patterns across tests.
    """

    dimensionality = 384

    def _vec(self, text: str) -> np.ndarray:
        seed = hash(text) % (2**31)
        rng = np.random.RandomState(seed)
        return rng.randn(384).astype(np.float32)

    def encode(self, text: str) -> np.ndarray:
        return self._vec(text)

    def encode_for_emit(self, text: str) -> np.ndarray:
        return self._vec(text)

    def encode_for_sense(self, text: str) -> np.ndarray:
        return self._vec(text)


@pytest.fixture
def seeded_encoder():
    """Provide a deterministic mock encoder for tests."""
    return SeededMockEncoder()
