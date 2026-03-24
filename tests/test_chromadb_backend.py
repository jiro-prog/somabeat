"""Tests for ChromaDB backend."""

import asyncio
from datetime import timedelta

import numpy as np
import pytest

from shared_state.backends.chromadb_backend import ChromaDBField
from shared_state.interface import (
    ExponentialDecay,
    PurgeCriteria,
    SenseParams,
    Signal,
    SignalOrigin,
)
from shared_state.observer import LoggingObserver


def _vec(norm: float = 1.0, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(384).astype(np.float32)
    return v / np.linalg.norm(v) * norm


def _signal(trace: str, norm: float = 1.0, context: str = "test", seed: int = 42) -> Signal:
    return Signal.create(
        embedding=_vec(norm=norm, seed=seed),
        origin=SignalOrigin(system="test", context=context),
        trace=trace,
    )


@pytest.fixture
def field():
    import uuid
    observer = LoggingObserver()
    return ChromaDBField(
        persist_directory=None,
        collection_name=f"test_{uuid.uuid4().hex[:8]}",
        observer=observer,
    )


class TestEmitAndSense:
    def test_emit_then_sense(self, field):
        s = _signal("hello world", seed=1)

        async def run():
            await field.emit(s)
            reading = await field.sense(s.embedding)
            assert len(reading.signals) == 1
            assert reading.signals[0].signal.trace == "hello world"

        asyncio.get_event_loop().run_until_complete(run())

    def test_sense_empty_field(self, field):
        async def run():
            reading = await field.sense(_vec())
            assert len(reading.signals) == 0

        asyncio.get_event_loop().run_until_complete(run())

    def test_norm_affects_effective_weight(self, field):
        s_low = _signal("low norm", norm=1.0, seed=10)
        s_high = _signal("high norm", norm=3.0, seed=10)

        async def run():
            await field.emit(s_low)
            await field.emit(s_high)
            reading = await field.sense(s_high.embedding)
            weights = {ws.signal.trace: ws.effective_weight for ws in reading.signals}
            assert weights["high norm"] > weights["low norm"]

        asyncio.get_event_loop().run_until_complete(run())

    def test_max_signals_respected(self, field):
        async def run():
            for i in range(20):
                await field.emit(_signal(f"sig_{i}", seed=i))

            params = SenseParams(max_signals=5)
            reading = await field.sense(_vec(seed=0), params=params)
            assert len(reading.signals) <= 5

        asyncio.get_event_loop().run_until_complete(run())


class TestPurge:
    def test_purge_by_context(self, field):
        async def run():
            await field.emit(_signal("keep", context="dialogue", seed=1))
            await field.emit(_signal("remove", context="difficulty", seed=2))

            result = await field.purge(PurgeCriteria(origin_context="difficulty"))
            assert result.purged_count == 1
            assert result.remaining_count == 1

        asyncio.get_event_loop().run_until_complete(run())

    def test_purge_empty_field(self, field):
        async def run():
            result = await field.purge(PurgeCriteria())
            assert result.purged_count == 0

        asyncio.get_event_loop().run_until_complete(run())


class TestSnapshot:
    def test_snapshot_returns_all(self, field):
        async def run():
            for i in range(5):
                await field.emit(_signal(f"snap_{i}", seed=i))
            snap = await field.snapshot()
            assert snap.total_count == 5

        asyncio.get_event_loop().run_until_complete(run())


class TestSignalExtra:
    def test_extra_roundtrip_via_sense(self, field):
        """Signal.extra should survive emit → sense roundtrip."""
        s = Signal.create(
            embedding=_vec(seed=10),
            origin=SignalOrigin(system="test", context="rotation_task"),
            trace="Q&A: test question",
            extra={"instruction": "test question", "output": "test answer"},
        )

        async def run():
            await field.emit(s)
            reading = await field.sense(s.embedding, SenseParams(max_signals=10, min_relevance=0.0))
            assert len(reading.signals) == 1
            sig = reading.signals[0].signal
            assert sig.extra["instruction"] == "test question"
            assert sig.extra["output"] == "test answer"

        asyncio.get_event_loop().run_until_complete(run())

    def test_extra_roundtrip_via_snapshot(self, field):
        """Signal.extra should survive emit → snapshot roundtrip."""
        s = Signal.create(
            embedding=_vec(seed=11),
            origin=SignalOrigin(system="test", context="test"),
            trace="with extra",
            extra={"key": "value"},
        )

        async def run():
            await field.emit(s)
            snap = await field.snapshot()
            assert snap.total_count == 1
            assert snap.signals[0].extra["key"] == "value"

        asyncio.get_event_loop().run_until_complete(run())

    def test_no_extra_returns_empty_dict(self, field):
        """Signals without extra should have empty dict."""
        s = _signal("no extra", seed=12)

        async def run():
            await field.emit(s)
            reading = await field.sense(s.embedding, SenseParams(max_signals=10, min_relevance=0.0))
            assert len(reading.signals) == 1
            assert reading.signals[0].signal.extra == {}

        asyncio.get_event_loop().run_until_complete(run())


class TestObserver:
    def test_observer_logs_emit(self, field):
        async def run():
            await field.emit(_signal("observed"))

        asyncio.get_event_loop().run_until_complete(run())
        logs = field._observer.get_recent_logs()
        assert len(logs) == 1
        assert "EMIT" in logs[0].summary

    def test_observer_logs_sense(self, field):
        async def run():
            await field.emit(_signal("x", seed=1))
            await field.sense(_vec(seed=1))

        asyncio.get_event_loop().run_until_complete(run())
        logs = field._observer.get_recent_logs()
        assert any("SENSE" in e.summary for e in logs)
