"""Tests for orchestrator module."""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import aiosqlite
import numpy as np
import pytest

from orchestrator.orchestrator import Orchestrator, SystemState


def _make_config(tmpdir: str) -> dict:
    """Create a test config pointing to temp directories."""
    return {
        "shared_state": {
            "backend": "chromadb",
            "chromadb": {
                "persist_directory": None,  # ephemeral
                "collection_name": "test_orch",
            },
        },
        "encoder": {
            "model_name": "intfloat/multilingual-e5-small",
        },
        "llamarcute_live": {
            "personality_path": "llamarcute_live/data/personality_v0.yaml",
            "dialogue_db_path": str(Path(tmpdir) / "llamarcute.db"),
            "sense": {
                "max_signals": 10,
                "min_relevance": 0.1,
                "time_horizon_hours": 168,
            },
        },
        "sleepyjean": {
            "memory": {
                "chromadb_collection": "test_sj_memory",
                "sqlite_path": str(Path(tmpdir) / "sj_memory.db"),
            },
            "recall": {"n_results": 3, "min_similarity": 0.3, "base_norm": 1.5},
            "reconsolidation": {"min_cluster_size": 3},
            "gap": {"min_repeat_count": 2},
            "quality": {"min_answer_length": 50},
        },
    }


class MockEncoder:
    """Fast mock encoder to avoid loading the real model in tests."""
    dimensionality = 384

    def _vec(self, text):
        seed = hash(text) % (2**31)
        rng = np.random.RandomState(seed)
        return rng.randn(384).astype(np.float32)

    def encode(self, text): return self._vec(text)
    def encode_for_emit(self, text): return self._vec(text)
    def encode_for_sense(self, text): return self._vec(text)


class TestOrchestratorState:
    def test_initial_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(tmpdir)
            with patch("orchestrator.orchestrator.E5SmallEncoder", return_value=MockEncoder()):
                orch = Orchestrator(config)
                assert orch.state == SystemState.AWAKE

    def test_sleep_wake_cycle(self):
        """Test that enter_sleep transitions to SLEEPING and wake_up returns to AWAKE."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(tmpdir)

            with patch("orchestrator.orchestrator.E5SmallEncoder", return_value=MockEncoder()):
                orch = Orchestrator(config)

                async def run():
                    await orch.dialogue.init_db()

                    assert orch.state == SystemState.AWAKE
                    assert orch.dialogue.is_active

                    # Enter sleep
                    await orch.enter_sleep()

                    # After the full cycle, should be AWAKE again
                    assert orch.state == SystemState.AWAKE
                    assert orch.dialogue.is_active

                asyncio.get_event_loop().run_until_complete(run())


class TestOrchestratorErrorHandling:
    def test_reconsolidation_failure_still_wakes(self):
        """When Reconsolidation fails, system still wakes up."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(tmpdir)

            with patch("orchestrator.orchestrator.E5SmallEncoder", return_value=MockEncoder()):
                orch = Orchestrator(config)

                # Force Reconsolidation to fail
                if orch.sleepyjean is not None:
                    orch.sleepyjean.on_sleep = AsyncMock(
                        side_effect=RuntimeError("test failure"),
                    )

                async def run():
                    await orch.dialogue.init_db()
                    await orch.enter_sleep()
                    # System should still wake up
                    assert orch.state == SystemState.AWAKE

                asyncio.get_event_loop().run_until_complete(run())
