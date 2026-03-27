"""Tests for individuality experiment 3 metrics (T8)."""

import asyncio
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from shared_state.interface import Signal, SignalOrigin, WeightedSignal
from llamarcute_live.metrics import (
    init_metrics_db,
    record_sense_result,
    record_self_mention,
    record_immune_event,
    record_fallback,
    detect_self_mentions,
    get_sense_summary,
    get_self_mention_rate,
    get_fallback_rate,
)


def _make_weighted_signal(system: str, context: str, label: str, weight: float) -> WeightedSignal:
    return WeightedSignal(
        signal=Signal.create(
            embedding=np.zeros(384, dtype=np.float32),
            origin=SignalOrigin(system=system, context=context),
        ),
        relevance=0.5,
        decay_factor=1.0,
        effective_weight=weight,
    )


@pytest.fixture
def db_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_metrics.db"
        asyncio.get_event_loop().run_until_complete(init_metrics_db(path))
        yield path


class TestInitMetricsDb:
    def test_creates_tables(self, db_path):
        import aiosqlite

        async def run():
            async with aiosqlite.connect(db_path) as db:
                cursor = await db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                )
                tables = [row[0] for row in await cursor.fetchall()]
                assert "metrics_sense" in tables
                assert "metrics_self_mention" in tables
                assert "metrics_immune_event" in tables
                assert "metrics_fallback" in tables

        asyncio.get_event_loop().run_until_complete(run())

    def test_idempotent(self, db_path):
        """Calling init twice should not error."""
        asyncio.get_event_loop().run_until_complete(init_metrics_db(db_path))


class TestRecordSenseResult:
    def test_records_signal_categories(self, db_path):
        signals = [
            _make_weighted_signal("llamarcute_live", "immune", "conservative_mode", 2.0),
            _make_weighted_signal("sleepyjean", "knowledge_update", "learned asyncio", 1.5),
            _make_weighted_signal("sleepyjean", "knowledge_update", "forgot HTTP cache", 1.0),
            _make_weighted_signal("llamarcute_live", "difficulty", "quantum was hard", 2.5),
            _make_weighted_signal("llamarcute_live", "dialogue", "talked about sorting", 1.0),
        ]

        async def run():
            await record_sense_result(db_path, "self_awareness", signals)
            summary = await get_sense_summary(db_path, days=1)
            assert summary["total_queries"] == 1
            assert summary["avg_immune_count"] == 1
            assert summary["avg_sj_count"] == 2
            assert summary["avg_diff_count"] == 1
            assert summary["avg_total_signals"] == 5

        asyncio.get_event_loop().run_until_complete(run())

    def test_empty_signals(self, db_path):
        async def run():
            await record_sense_result(db_path, "self_awareness", [])
            summary = await get_sense_summary(db_path, days=1)
            assert summary["total_queries"] == 1
            assert summary["avg_total_signals"] == 0

        asyncio.get_event_loop().run_until_complete(run())


class TestDetectSelfMentions:
    def test_detects_keywords(self):
        text = "免疫系が慎重モードを発動しています。最近学習した内容も含めて状態を確認中。"
        mentions = detect_self_mentions(text)
        assert "免疫" in mentions
        assert "慎重" in mentions
        assert "状態" in mentions

    def test_no_mentions(self):
        text = "Pythonでループを書く方法を説明します。"
        mentions = detect_self_mentions(text)
        assert len(mentions) == 0


class TestRecordSelfMention:
    def test_records_mention(self, db_path):
        async def run():
            count = await record_self_mention(
                db_path,
                "調子どう？",
                "免疫系が慎重モードに入っていて、少し不安定な状態です",
            )
            assert count >= 3  # 免疫, 慎重, 不安定, 状態

            rate = await get_self_mention_rate(db_path, days=1)
            assert rate["total_responses"] == 1
            assert rate["responses_with_mentions"] == 1

        asyncio.get_event_loop().run_until_complete(run())

    def test_records_no_mention(self, db_path):
        async def run():
            count = await record_self_mention(
                db_path,
                "Pythonについて教えて",
                "Pythonは汎用プログラミング言語です",
            )
            assert count == 0

            rate = await get_self_mention_rate(db_path, days=1)
            assert rate["responses_with_mentions"] == 0

        asyncio.get_event_loop().run_until_complete(run())


class TestRecordImmuneEvent:
    def test_records_event(self, db_path):
        async def run():
            await record_immune_event(
                db_path,
                "health_check_warning",
                severity="WARNING",
                details="fitness 0.75 < 0.80",
            )
            import aiosqlite
            async with aiosqlite.connect(db_path) as db:
                cursor = await db.execute("SELECT COUNT(*) FROM metrics_immune_event")
                count = (await cursor.fetchone())[0]
                assert count == 1

        asyncio.get_event_loop().run_until_complete(run())


class TestRecordFallback:
    def test_no_fallback(self, db_path):
        """When conservative_mode is not active, no fallback."""
        async def run():
            await record_fallback(db_path, False, 2, 2)
            rate = await get_fallback_rate(db_path)
            assert rate["total_cycles"] == 1
            assert rate["conservative_mode_cycles"] == 0
            assert rate["fallback_triggered_cycles"] == 0

        asyncio.get_event_loop().run_until_complete(run())

    def test_fallback_triggered(self, db_path):
        """conservative_mode active and LLM chose max_changes > 1 → fallback triggered."""
        async def run():
            await record_fallback(db_path, True, 3, 1)  # difficulty said 3, immune forced 1
            rate = await get_fallback_rate(db_path)
            assert rate["total_cycles"] == 1
            assert rate["conservative_mode_cycles"] == 1
            assert rate["fallback_triggered_cycles"] == 1
            assert rate["fallback_rate"] == 1.0

        asyncio.get_event_loop().run_until_complete(run())

    def test_no_fallback_needed(self, db_path):
        """conservative_mode active but LLM already chose 1 → no fallback."""
        async def run():
            await record_fallback(db_path, True, 1, 1)  # difficulty said 1, immune agrees
            rate = await get_fallback_rate(db_path)
            assert rate["conservative_mode_cycles"] == 1
            assert rate["fallback_triggered_cycles"] == 0
            assert rate["fallback_rate"] == 0.0

        asyncio.get_event_loop().run_until_complete(run())


class TestDreamSignalCategory:
    """T3: dream signal category is correctly classified."""

    def test_dream_signal_classified_as_dream(self, db_path):
        """Dream signals (context='dream') should be counted under dream, not sleepyjean."""
        signals = [
            _make_weighted_signal("sleepyjean", "dream", "夢の接続: 量子×AI", 1.5),
            _make_weighted_signal("sleepyjean", "dream", "夢の洞察: 意識の境界", 1.0),
            _make_weighted_signal("sleepyjean", "knowledge_update", "learned topic", 2.0),
            _make_weighted_signal("llamarcute_live", "dialogue", "user talked", 1.0),
        ]

        async def run():
            await record_sense_result(db_path, "self_awareness", signals)
            summary = await get_sense_summary(db_path, days=1)
            assert summary["avg_dream_count"] == 2, "dream signals should be counted"
            assert summary["avg_sj_count"] == 1, "only non-dream sleepyjean should be counted"

        asyncio.get_event_loop().run_until_complete(run())

    def test_no_dream_signals(self, db_path):
        """When there are no dream signals, dream count should be 0."""
        signals = [
            _make_weighted_signal("llamarcute_live", "immune", "check", 1.0),
            _make_weighted_signal("sleepyjean", "knowledge_update", "learned", 1.0),
        ]

        async def run():
            await record_sense_result(db_path, "dialogue_context", signals)
            summary = await get_sense_summary(db_path, days=1)
            assert summary["avg_dream_count"] == 0

        asyncio.get_event_loop().run_until_complete(run())


class TestMaxChangesResult:
    def test_result_dataclass(self):
        from llamarcute_live.self_improve import MaxChangesResult
        r = MaxChangesResult(max_changes=1, conservative_mode_active=True, difficulty_based_value=3)
        assert r.max_changes == 1
        assert r.conservative_mode_active is True
        assert r.difficulty_based_value == 3
