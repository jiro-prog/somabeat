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


class TestStrengthExponent:
    def test_exponent_1_matches_linear(self, field):
        """α=1.0 should produce the same strength as the old linear formula."""
        s = _signal("exp_test", norm=25.0, seed=7)

        async def run():
            await field.emit(s)
            linear = await field.perceive(
                PerceiveParams(min_strength=0.0, strength_exponent=1.0)
            )
            assert len(linear.signals) == 1
            # With α=1.0: strength = decay * norm^1.0 = decay * norm
            ps = linear.signals[0]
            expected = ps.decay_factor * 25.0
            assert abs(ps.strength - expected) < 1e-4

        asyncio.get_event_loop().run_until_complete(run())

    def test_exponent_half_compresses_range(self, field):
        """α=0.5 should compress the strength dynamic range."""
        s_low = _signal("low", norm=4.0, seed=10)
        s_high = _signal("high", norm=36.0, seed=11)

        async def run():
            await field.emit(s_low)
            await field.emit(s_high)

            # Linear (α=1.0): ratio = 36/4 = 9x
            linear = await field.perceive(
                PerceiveParams(min_strength=0.0, strength_exponent=1.0)
            )
            lin_strengths = {
                ps.signal.signal_id: ps.strength for ps in linear.signals
            }

            # Compressed (α=0.5): ratio = sqrt(36)/sqrt(4) = 6/2 = 3x
            compressed = await field.perceive(
                PerceiveParams(min_strength=0.0, strength_exponent=0.5)
            )
            comp_strengths = {
                ps.signal.signal_id: ps.strength for ps in compressed.signals
            }

            lin_ratio = lin_strengths[s_high.signal_id] / lin_strengths[s_low.signal_id]
            comp_ratio = comp_strengths[s_high.signal_id] / comp_strengths[s_low.signal_id]

            assert lin_ratio > comp_ratio
            assert abs(lin_ratio - 9.0) < 0.5
            assert abs(comp_ratio - 3.0) < 0.5

        asyncio.get_event_loop().run_until_complete(run())
