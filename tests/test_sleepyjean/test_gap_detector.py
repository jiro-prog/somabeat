"""Tests for sleepyjean.gap_detector.GapDetector."""

import numpy as np
import pytest

from sleepyjean.gap_detector import GapDetector
from tests.test_sleepyjean.conftest import make_field, rand_emb, DIM


def _make_detector(config_overrides=None):
    field = make_field()
    from unittest.mock import MagicMock
    encoder = MagicMock()
    config = {"min_repeat_count": 2, "similarity_threshold": 0.7}
    if config_overrides:
        config.update(config_overrides)
    return GapDetector(field, encoder, config), field, encoder


class TestGapDetector:
    def test_record_failure(self):
        """record_failureで記録される"""
        detector, field, encoder = _make_detector()
        encoder.encode.return_value = rand_emb()

        detector.record_failure("test query", best_similarity=0.1)
        assert len(detector._failures) == 1
        assert detector._failures[0]["query_text"] == "test query"

    @pytest.mark.asyncio
    async def test_repeated_topic_emitted(self):
        """複数回の同一トピック失敗がemitされる"""
        detector, field, encoder = _make_detector()
        same_emb = rand_emb()
        encoder.encode.return_value = same_emb

        detector.record_failure("topic A", best_similarity=0.1)
        detector.record_failure("topic A again", best_similarity=0.05)

        ids = await detector.emit_gaps()
        assert len(ids) == 1
        field.emit.assert_called_once()

    @pytest.mark.asyncio
    async def test_single_failure_not_emitted(self):
        """1回限りの失敗はemitされない"""
        detector, field, encoder = _make_detector()
        encoder.encode.side_effect = lambda text: rand_emb()

        detector.record_failure("unique topic", best_similarity=0.1)

        ids = await detector.emit_gaps()
        assert len(ids) == 0
        field.emit.assert_not_called()

    def test_clear(self):
        """clear後は空になる"""
        detector, field, encoder = _make_detector()
        encoder.encode.return_value = rand_emb()

        detector.record_failure("test", best_similarity=0.1)
        assert len(detector._failures) == 1
        detector.clear()
        assert len(detector._failures) == 0

    @pytest.mark.asyncio
    async def test_empty_emit_gaps(self):
        """失敗が0件のときemit_gapsは空リストを返す"""
        detector, field, encoder = _make_detector()
        ids = await detector.emit_gaps()
        assert ids == []
        field.emit.assert_not_called()
