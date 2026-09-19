"""Tests for shared_state interface definitions."""

from datetime import datetime, timedelta

import numpy as np

from shared_state.interface import (
    ExponentialDecay,
    FieldReading,
    FieldSnapshot,
    PerceiveParams,
    PurgeCriteria,
    PurgeResult,
    Signal,
    SignalOrigin,
    WeightedSignal,
)


def _make_signal(label: str = "test", norm: float = 1.0) -> Signal:
    seed = hash(label) % (2**31)
    rng = np.random.RandomState(seed)
    vec = rng.randn(384).astype(np.float32)
    vec = vec / np.linalg.norm(vec) * norm
    return Signal.create(
        embedding=vec,
        origin=SignalOrigin(system="test", context="unit_test"),
    )


class TestExponentialDecay:
    def test_zero_elapsed(self):
        decay = ExponentialDecay(half_life_hours=24.0)
        assert decay(timedelta(0)) == 1.0

    def test_half_life(self):
        decay = ExponentialDecay(half_life_hours=24.0)
        result = decay(timedelta(hours=24))
        assert abs(result - 0.5) < 1e-6

    def test_double_half_life(self):
        decay = ExponentialDecay(half_life_hours=24.0)
        result = decay(timedelta(hours=48))
        assert abs(result - 0.25) < 1e-6

    def test_negative_elapsed_returns_1(self):
        decay = ExponentialDecay(half_life_hours=24.0)
        assert decay(timedelta(hours=-1)) == 1.0

    def test_monotonically_decreasing(self):
        decay = ExponentialDecay(half_life_hours=12.0)
        values = [decay(timedelta(hours=h)) for h in range(0, 100, 10)]
        for i in range(len(values) - 1):
            assert values[i] > values[i + 1]


class TestSignal:
    def test_create_generates_unique_ids(self):
        s1 = _make_signal("a")
        s2 = _make_signal("b")
        assert s1.signal_id != s2.signal_id

    def test_create_sets_timestamp(self):
        before = datetime.now()
        s = _make_signal()
        after = datetime.now()
        assert before <= s.emitted_at <= after

    def test_embedding_not_normalized(self):
        s = _make_signal(norm=2.5)
        actual_norm = float(np.linalg.norm(s.embedding))
        assert abs(actual_norm - 2.5) < 0.01


class TestWeightedSignal:
    def test_effective_weight_calculation(self):
        s = _make_signal(norm=2.0)
        ws = WeightedSignal(
            signal=s,
            relevance=0.8,
            decay_factor=0.5,
            effective_weight=0.8 * 0.5 * float(np.linalg.norm(s.embedding)),
        )
        expected = 0.8 * 0.5 * 2.0
        assert abs(ws.effective_weight - expected) < 0.01


class TestPerceiveParams:
    def test_defaults(self):
        p = PerceiveParams()
        assert p.decay_fn is None
        assert p.max_signals == 50
        assert p.min_strength == 0.1
        assert p.time_horizon is None


class TestDataclasses:
    def test_field_reading(self):
        reading = FieldReading(
            signals=[],
            observed_at=datetime.now(),
            query_embedding=np.zeros(384, dtype=np.float32),
        )
        assert len(reading.signals) == 0

    def test_purge_criteria(self):
        c = PurgeCriteria(older_than=timedelta(days=7))
        assert c.older_than == timedelta(days=7)
        assert c.max_signals_to_purge is None

    def test_purge_result(self):
        r = PurgeResult(purged_count=5, remaining_count=95)
        assert r.purged_count == 5

    def test_field_snapshot(self):
        snap = FieldSnapshot(signals=[], taken_at=datetime.now(), total_count=0)
        assert snap.total_count == 0
