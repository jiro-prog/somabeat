"""Tests for FieldSnapshotLogger (M1-1: field measurement)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta

import aiosqlite
import numpy as np
import pytest

from shared_state.interface import FieldPerception, PerceivedSignal, Signal, SignalOrigin
from shared_state.observer import FieldSnapshotLogger


def _make_signal(
    context: str = "dialogue",
    norm: float = 1.0,
    age_hours: float = 0.0,
) -> Signal:
    rng = np.random.RandomState(hash(context) % (2**31))
    vec = rng.randn(384).astype(np.float32)
    vec = vec / np.linalg.norm(vec) * norm
    return Signal.create(
        embedding=vec,
        origin=SignalOrigin(system="llamarcute_live", context=context),
    )


def _make_perceived(
    context: str = "dialogue",
    norm: float = 1.0,
    decay: float = 0.9,
    age_hours: float = 1.0,
) -> PerceivedSignal:
    rng = np.random.RandomState(hash(context) % (2**31))
    vec = rng.randn(384).astype(np.float32)
    vec = vec / np.linalg.norm(vec) * norm
    sig = Signal(
        signal_id=uuid.uuid4().hex,
        embedding=vec,
        emitted_at=datetime.now() - timedelta(hours=age_hours),
        origin=SignalOrigin(system="llamarcute_live", context=context),
    )
    return PerceivedSignal(
        signal=sig,
        decay_factor=decay,
        strength=decay * norm,
    )


def _make_perception(perceived_signals: list[PerceivedSignal]) -> FieldPerception:
    return FieldPerception(
        signals=perceived_signals,
        perceived_at=datetime.now(),
    )


@pytest.fixture
async def db_path(tmp_path):
    path = tmp_path / "test.db"
    async with aiosqlite.connect(path) as db:
        await db.execute("""
            CREATE TABLE field_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                trigger_context TEXT NOT NULL,
                signal_count INTEGER NOT NULL,
                signal_count_by_origin TEXT,
                strength_max REAL,
                strength_median REAL,
                strength_std REAL,
                raw_norm_mean REAL,
                oldest_signal_age_hours REAL,
                dialogue_log_id INTEGER,
                response_time_ms INTEGER,
                strength_exponent REAL,
                baseline_version TEXT NOT NULL DEFAULT 'v2'
            )
        """)
        await db.commit()
    return path


class TestFieldSnapshotLogger:
    @pytest.mark.asyncio
    async def test_record_with_signals(self, db_path):
        logger = FieldSnapshotLogger(db_path)

        ps1 = _make_perceived(context="dialogue", norm=2.0, decay=0.9, age_hours=3.0)
        ps2 = _make_perceived(context="difficulty", norm=4.0, decay=0.8, age_hours=1.0)
        ps3 = _make_perceived(context="dialogue", norm=1.5, decay=0.7, age_hours=5.0)
        perception = _make_perception([ps1, ps2, ps3])
        logger.on_perceive(perception)

        await logger.record_snapshot(
            trigger_context="dialogue",
            dialogue_log_id=42,
            response_time_ms=1500,
        )

        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM field_snapshots")
            row = await cursor.fetchone()

        assert row is not None
        assert row["signal_count"] == 3
        assert row["trigger_context"] == "dialogue"
        assert row["dialogue_log_id"] == 42
        assert row["response_time_ms"] == 1500
        assert row["baseline_version"] == "v2"

        origin = json.loads(row["signal_count_by_origin"])
        assert origin["dialogue"] == 2
        assert origin["difficulty"] == 1

        # strength_max should be max of [0.9*2.0, 0.8*4.0, 0.7*1.5] = 3.2
        assert abs(row["strength_max"] - 3.2) < 1e-6

        # oldest_signal_age_hours should be ~5.0
        assert row["oldest_signal_age_hours"] > 4.5

    @pytest.mark.asyncio
    async def test_record_empty_perception(self, db_path):
        logger = FieldSnapshotLogger(db_path)
        perception = _make_perception([])
        logger.on_perceive(perception)

        await logger.record_snapshot(trigger_context="dialogue")

        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM field_snapshots")
            row = await cursor.fetchone()

        assert row is not None
        assert row["signal_count"] == 0
        assert row["strength_max"] is None
        assert row["signal_count_by_origin"] is None

    @pytest.mark.asyncio
    async def test_record_without_perceive_is_noop(self, db_path):
        logger = FieldSnapshotLogger(db_path)
        await logger.record_snapshot(trigger_context="dialogue")

        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM field_snapshots")
            count = (await cursor.fetchone())[0]

        assert count == 0

    @pytest.mark.asyncio
    async def test_inherits_logging_observer(self, db_path):
        logger = FieldSnapshotLogger(db_path)
        ps = _make_perceived()
        perception = _make_perception([ps])
        logger.on_perceive(perception)

        # LoggingObserver behavior should still work
        logs = logger.get_recent_logs(10)
        assert len(logs) == 1
        assert "PERCEIVE" in logs[0].summary

    @pytest.mark.asyncio
    async def test_signal_count_by_origin_json_parseable(self, db_path):
        logger = FieldSnapshotLogger(db_path)
        signals = [
            _make_perceived(context="dialogue"),
            _make_perceived(context="dialogue"),
            _make_perceived(context="difficulty"),
            _make_perceived(context="knowledge_update"),
        ]
        perception = _make_perception(signals)
        logger.on_perceive(perception)
        await logger.record_snapshot(trigger_context="dialogue")

        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(
                "SELECT signal_count_by_origin FROM field_snapshots"
            )
            row = await cursor.fetchone()

        parsed = json.loads(row[0])
        assert isinstance(parsed, dict)
        assert parsed["dialogue"] == 2
        assert parsed["difficulty"] == 1
        assert parsed["knowledge_update"] == 1

    @pytest.mark.asyncio
    async def test_manual_trigger_context(self, db_path):
        logger = FieldSnapshotLogger(db_path)
        perception = _make_perception([_make_perceived()])
        logger.on_perceive(perception)

        await logger.record_snapshot(
            trigger_context="manual",
            dialogue_log_id=None,
            response_time_ms=None,
        )

        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM field_snapshots")
            row = await cursor.fetchone()

        assert row["trigger_context"] == "manual"
        assert row["dialogue_log_id"] is None
        assert row["response_time_ms"] is None
