"""Tests for ChromaDB backend."""

import asyncio
from datetime import timedelta

import numpy as np
import pytest

from shared_state.backends.chromadb_backend import ChromaDBField
from shared_state.interface import (
    ExponentialDecay,
    PerceiveParams,
    PurgeCriteria,
    Signal,
    SignalOrigin,
)
from shared_state.observer import LoggingObserver


def _vec(norm: float = 1.0, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(384).astype(np.float32)
    return v / np.linalg.norm(v) * norm


def _signal(label: str = "test", norm: float = 1.0, context: str = "test", seed: int = 42) -> Signal:
    return Signal.create(
        embedding=_vec(norm=norm, seed=seed),
        origin=SignalOrigin(system="test", context=context),
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


class TestEmitAndPerceive:
    def test_emit_then_perceive(self, field):
        s = _signal("hello world", seed=1)

        async def run():
            await field.emit(s)
            perception = await field.perceive(PerceiveParams(min_strength=0.0))
            assert len(perception.signals) == 1
            assert perception.signals[0].signal.signal_id == s.signal_id

        asyncio.get_event_loop().run_until_complete(run())

    def test_perceive_empty_field(self, field):
        async def run():
            perception = await field.perceive()
            assert len(perception.signals) == 0

        asyncio.get_event_loop().run_until_complete(run())

    def test_norm_affects_strength(self, field):
        s_low = _signal("low norm", norm=1.0, seed=10)
        s_high = _signal("high norm", norm=3.0, seed=10)

        async def run():
            await field.emit(s_low)
            await field.emit(s_high)
            perception = await field.perceive(PerceiveParams(min_strength=0.0))
            strengths = {ps.signal.signal_id: ps.strength for ps in perception.signals}
            assert strengths[s_high.signal_id] > strengths[s_low.signal_id]

        asyncio.get_event_loop().run_until_complete(run())

    def test_max_signals_respected(self, field):
        async def run():
            for i in range(20):
                await field.emit(_signal(f"sig_{i}", seed=i))

            params = PerceiveParams(max_signals=5, min_strength=0.0)
            perception = await field.perceive(params)
            assert len(perception.signals) <= 5

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
    def test_extra_roundtrip_via_perceive(self, field):
        """Signal.extra should survive emit → perceive roundtrip."""
        s = Signal.create(
            embedding=_vec(seed=10),
            origin=SignalOrigin(system="test", context="rotation_task"),
            extra={"instruction": "test question", "output": "test answer"},
        )

        async def run():
            await field.emit(s)
            perception = await field.perceive(PerceiveParams(max_signals=10, min_strength=0.0))
            assert len(perception.signals) == 1
            sig = perception.signals[0].signal
            assert sig.extra["instruction"] == "test question"
            assert sig.extra["output"] == "test answer"

        asyncio.get_event_loop().run_until_complete(run())

    def test_extra_roundtrip_via_snapshot(self, field):
        """Signal.extra should survive emit → snapshot roundtrip."""
        s = Signal.create(
            embedding=_vec(seed=11),
            origin=SignalOrigin(system="test", context="test"),
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
            perception = await field.perceive(PerceiveParams(max_signals=10, min_strength=0.0))
            assert len(perception.signals) == 1
            assert perception.signals[0].signal.extra == {}

        asyncio.get_event_loop().run_until_complete(run())


class TestObserver:
    def test_observer_logs_emit(self, field):
        async def run():
            await field.emit(_signal("observed"))

        asyncio.get_event_loop().run_until_complete(run())
        logs = field._observer.get_recent_logs()
        assert len(logs) == 1
        assert "EMIT" in logs[0].summary

    def test_observer_logs_perceive(self, field):
        async def run():
            await field.emit(_signal("x", seed=1))
            await field.perceive()

        asyncio.get_event_loop().run_until_complete(run())
        logs = field._observer.get_recent_logs()
        assert any("PERCEIVE" in e.summary for e in logs)
