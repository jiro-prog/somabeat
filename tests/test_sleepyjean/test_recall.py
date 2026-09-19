"""Tests for sleepyjean.recall.RecallEngine (KG-based).

8 tests per instructions_kg_ollama_removal.md T-1.
"""

from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

from sleepyjean.knowledge_graph import KnowledgeGraph, Triple
from sleepyjean.recall import RecallEngine, RecallResult
from tests.test_sleepyjean.conftest import make_encoder, make_field, rand_emb


def _make_kg_with_triples(tmp_path, triples=None):
    kg = KnowledgeGraph(sqlite_path=str(tmp_path / "test_recall_kg.db"))
    if triples is None:
        triples = [
            Triple("フォレトス", "タイプ", "むし・はがね", "ep1", datetime.now()),
            Triple("フォレトス", "覚える技", "だいばくはつ", "ep1", datetime.now()),
            Triple("ピカチュウ", "タイプ", "でんき", "ep2", datetime.now()),
        ]
    kg.add_triples(triples)
    return kg


class TestRecallEngine:
    def test_keyword_extraction_returns_nouns(self):
        """キーワード抽出が名詞・固有名詞を返す"""
        engine = RecallEngine(
            MagicMock(), MagicMock(), MagicMock(), {},
        )
        keywords = engine._extract_keywords("フォレトスのタイプを教えて")
        # Should contain noun-like words, not particles
        assert len(keywords) >= 1
        # particles like "の", "を" should be excluded
        assert "の" not in keywords
        assert "を" not in keywords

    def test_kg_hit_returns_triples_text(self, tmp_path):
        """KGヒットありの場合、triples_textが返される"""
        kg = _make_kg_with_triples(tmp_path)
        engine = RecallEngine(
            MagicMock(), make_field(), make_encoder(), {},
            knowledge_graph=kg,
        )
        result = engine.recall("フォレトスについて教えて")
        assert result.source == "kg"
        assert result.triples_text is not None
        assert "あなたの記憶:" in result.triples_text
        assert "フォレトス" in result.triples_text

    def test_kg_miss_falls_back_to_embedding(self, tmp_path, store):
        """KGヒットなしの場合、MemoryStoreフォールバック"""
        kg = _make_kg_with_triples(tmp_path)
        emb = rand_emb()
        store.add_episode(emb)

        engine = RecallEngine(
            store, make_field(), make_encoder(return_emb=emb),
            {"min_similarity": 0.0},
            knowledge_graph=kg,
        )
        # Query something not in KG
        result = engine.recall("天気はどうですか")
        assert result.source == "embedding"

    def test_both_miss_returns_none(self, tmp_path, store):
        """KGもMemoryStoreもヒットなしの場合、source='none'"""
        kg = KnowledgeGraph(sqlite_path=str(tmp_path / "empty_kg.db"))
        engine = RecallEngine(
            store, make_field(), make_encoder(),
            {"min_similarity": 0.99},
            knowledge_graph=kg,
        )
        result = engine.recall("全然関係ない話題")
        assert result.source == "none"
        assert result.triples_text is None

    def test_format_triples_generates_text(self, tmp_path):
        """_format_triplesが正しいテキストを生成する"""
        kg = _make_kg_with_triples(tmp_path)
        engine = RecallEngine(
            MagicMock(), make_field(), make_encoder(), {},
            knowledge_graph=kg,
        )
        triples = [
            Triple("フォレトス", "タイプ", "むし・はがね", "ep1", datetime.now()),
        ]
        text = engine._format_triples(triples)
        assert text.startswith("あなたの記憶:")
        assert "フォレトス" in text
        assert "むし・はがね" in text

    def test_query_increments_access_count(self, tmp_path):
        """KG queryでaccess_countがインクリメントされる"""
        kg = _make_kg_with_triples(tmp_path)
        engine = RecallEngine(
            MagicMock(), make_field(), make_encoder(), {},
            knowledge_graph=kg,
        )
        assert kg.graph.nodes["フォレトス"]["access_count"] == 0
        engine.recall("フォレトスについて")
        assert kg.graph.nodes["フォレトス"]["access_count"] >= 1

    def test_empty_input_no_error(self):
        """空文字入力でエラーにならない"""
        engine = RecallEngine(
            MagicMock(), make_field(), make_encoder(), {},
        )
        result = engine.recall("")
        assert isinstance(result, RecallResult)
        assert result.source == "none"

    def test_japanese_morphological_analysis(self):
        """形態素解析が日本語テキストで動作する"""
        engine = RecallEngine(
            MagicMock(), MagicMock(), MagicMock(), {},
        )
        keywords = engine._extract_keywords("東京タワーに行きたい")
        assert len(keywords) >= 1
        # Should extract nouns like "東京" or "タワー"
        found_noun = any(len(kw) >= 2 for kw in keywords)
        assert found_noun
