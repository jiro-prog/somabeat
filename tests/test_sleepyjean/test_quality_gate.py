"""Tests for sleepyjean.quality_gate.QualityGate."""

import numpy as np
from unittest.mock import MagicMock

from sleepyjean.quality_gate import QualityGate
from tests.test_sleepyjean.conftest import make_encoder, rand_emb, DIM


def _make_gate(config_overrides=None):
    encoder = MagicMock()
    config = {"min_answer_length": 50, "min_relevance": 0.2}
    if config_overrides:
        config.update(config_overrides)
    return QualityGate(encoder, config), encoder


class TestQualityGate:
    def test_normal_qa_passes(self):
        """正常なQ&AがPASS"""
        gate, encoder = _make_gate()
        emb = rand_emb()
        encoder.encode.return_value = emb

        result = gate.check(
            "日本の首都はどこですか？",
            "日本の首都は東京です。東京は関東地方に位置する大都市で、政治・経済・文化の中心地です。人口は約1400万人で、日本最大の都市です。",
        )
        assert result.passed is True

    def test_empty_answer_fails(self):
        """空回答がFAIL"""
        gate, encoder = _make_gate()
        result = gate.check("質問です", "")
        assert result.passed is False
        assert "too short" in result.reason

    def test_short_answer_fails(self):
        """極端に短い回答がFAIL"""
        gate, encoder = _make_gate()
        result = gate.check("質問です", "はい")
        assert result.passed is False
        assert "too short" in result.reason

    def test_irrelevant_answer_fails(self):
        """無関係な回答がFAIL"""
        gate, encoder = _make_gate()

        call_count = [0]
        def diverse_encode(text):
            call_count[0] += 1
            if call_count[0] == 1:
                return np.array([1.0] * DIM, dtype=np.float32)
            else:
                return np.array([-1.0] * DIM, dtype=np.float32)
        encoder.encode.side_effect = diverse_encode

        result = gate.check(
            "日本の首都はどこですか？",
            "今日の天気はとても良いですね。散歩に行きたいと思います。青い空が広がっています。鳥が飛んでいます。風が気持ちいいです。",
        )
        assert result.passed is False
        assert "relevance" in result.reason.lower()

    def test_language_check(self):
        """言語チェックが機能する"""
        gate, encoder = _make_gate()
        emb = rand_emb()
        encoder.encode.return_value = emb

        result = gate.check(
            "質問です",
            "This is a long enough answer that contains no Japanese characters at all and should fail the language check.",
        )
        assert result.passed is False
        assert "Japanese" in result.reason
