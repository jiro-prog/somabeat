"""Shared fixtures for SleepyJean tests."""

import os
import uuid
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from sleepyjean.memory_store import MemoryStore

DIM = 384


def rand_emb() -> np.ndarray:
    """Generate a random embedding vector."""
    return np.random.randn(DIM).astype(np.float32)


def cluster_embs(center: np.ndarray, n: int = 3, noise: float = 0.05) -> list[np.ndarray]:
    """Generate n embeddings near a center point."""
    return [
        (center + np.random.randn(DIM).astype(np.float32) * noise)
        for _ in range(n)
    ]


def make_field():
    """Create a mock SharedField."""
    field = AsyncMock()
    field.emit = AsyncMock()
    return field


def make_encoder(return_emb=None):
    """Create a mock FieldEncoder."""
    encoder = MagicMock()
    if return_emb is not None:
        encoder.encode = MagicMock(return_value=return_emb)
    else:
        encoder.encode = MagicMock(side_effect=lambda text: rand_emb())
    return encoder


@pytest.fixture
def store(tmp_path):
    """Create a fresh MemoryStore with unique collection per test."""
    uid = uuid.uuid4().hex[:8]
    cfg = {
        "chromadb_collection": f"test_{uid}",
        "sqlite_path": os.path.join(str(tmp_path), f"test_{uid}.db"),
    }
    s = MemoryStore(cfg)
    yield s
    s.close()
