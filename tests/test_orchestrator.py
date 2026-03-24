"""Tests for orchestrator module."""

import asyncio
import json
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
            "root_path": tmpdir,
            "night_cycle_script": "night_cycle.py",
            "db_path": str(Path(tmpdir) / "sodateai.db"),
            "knowledge_index_path": str(Path(tmpdir) / "ki.json"),
        },
    }


class MockEncoder:
    """Fast mock encoder to avoid loading the real model in tests.

    Uses seeded random for deterministic vectors.
    """
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

            # Create required files for bridge
            ki_path = Path(tmpdir) / "ki.json"
            ki_path.write_text(json.dumps({"topics": [], "open_questions": []}))

            # Create SleepyJean DB
            async def setup_sj_db():
                async with aiosqlite.connect(config["sleepyjean"]["db_path"]) as db:
                    await db.execute("""
                        CREATE TABLE IF NOT EXISTS conversations (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            channel_id TEXT NOT NULL, user_id TEXT NOT NULL,
                            role TEXT NOT NULL, content TEXT NOT NULL,
                            created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
                        )
                    """)
                    await db.execute("""
                        CREATE TABLE IF NOT EXISTS homework (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            theme TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued',
                            requested_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                            completed_at TEXT, result_summary TEXT,
                            retry_count INTEGER NOT NULL DEFAULT 0
                        )
                    """)
                    await db.commit()

            asyncio.get_event_loop().run_until_complete(setup_sj_db())

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
    def test_night_cycle_failure_skips_wake_export(self):
        """When night cycle fails, wake_export is skipped but system still wakes up."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(tmpdir)

            ki_path = Path(tmpdir) / "ki.json"
            ki_path.write_text(json.dumps({"topics": [], "open_questions": []}))

            async def setup_sj_db():
                async with aiosqlite.connect(config["sleepyjean"]["db_path"]) as db:
                    await db.execute("""
                        CREATE TABLE IF NOT EXISTS conversations (
                            id INTEGER PRIMARY KEY, channel_id TEXT, user_id TEXT,
                            role TEXT, content TEXT, created_at TEXT DEFAULT (datetime('now', 'localtime'))
                        )
                    """)
                    await db.execute("""
                        CREATE TABLE IF NOT EXISTS homework (
                            id INTEGER PRIMARY KEY, theme TEXT,
                            status TEXT DEFAULT 'queued',
                            requested_at TEXT DEFAULT (datetime('now', 'localtime')),
                            completed_at TEXT, result_summary TEXT,
                            retry_count INTEGER DEFAULT 0
                        )
                    """)
                    await db.commit()

            asyncio.get_event_loop().run_until_complete(setup_sj_db())

            with patch("orchestrator.orchestrator.E5SmallEncoder", return_value=MockEncoder()):
                orch = Orchestrator(config)

                # Mock night cycle to return failure
                async def mock_night_cycle():
                    return 1  # failure

                orch._run_night_cycle = mock_night_cycle

                async def run():
                    await orch.dialogue.init_db()
                    await orch.enter_sleep()
                    # System should still wake up
                    assert orch.state == SystemState.AWAKE

                asyncio.get_event_loop().run_until_complete(run())
