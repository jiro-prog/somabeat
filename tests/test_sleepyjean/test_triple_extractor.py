"""Tests for TripleExtractor — dialogue log to triple extraction.

6 tests per instructions_kg_ollama_removal.md K-2.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from sleepyjean.triple_extractor import TripleExtractor


def _make_mock_llm(response: str | None = "主語 | 関係 | 目的語"):
    llm = MagicMock()
    llm.generate_bare = AsyncMock(return_value=(response, 1.0))
    return llm


def _make_logs(*pairs):
    """Make dialogue logs from (user, assistant) pairs."""
    logs = []
    for user, asst in pairs:
        logs.append({"role": "user", "content": user})
        logs.append({"role": "assistant", "content": asst})
    return logs


@pytest.fixture
def config():
    return {"chunk_size": 2, "max_new_tokens": 512, "temperature": 0.3}


class TestTripleExtractor:

    def test_extracts_triples_from_factual_dialogue(self, config):
        """Factual dialogue produces triples."""
        llm = _make_mock_llm("フォレトス | タイプ | むし・はがね\nフォレトス | 覚える技 | だいばくはつ")
        extractor = TripleExtractor(llm, config)
        logs = _make_logs(
            ("フォレトスって何タイプ？", "フォレトスはむし・はがねタイプだよ"),
        )
        triples = asyncio.get_event_loop().run_until_complete(extractor.extract(logs))
        assert len(triples) == 2
        assert triples[0].subject == "フォレトス"
        assert triples[0].relation == "タイプ"

    def test_empty_result_for_chitchat(self, config):
        """Chitchat-only dialogue returns empty list."""
        llm = _make_mock_llm("なし")
        extractor = TripleExtractor(llm, config)
        logs = _make_logs(("おはよう", "おはよう！今日もいい天気だね"))
        triples = asyncio.get_event_loop().run_until_complete(extractor.extract(logs))
        assert triples == []

    def test_parse_failure_skipped(self, config):
        """Lines that don't match format are skipped."""
        llm = _make_mock_llm("フォレトス | タイプ | むし\nbad line here\n| also | bad |")
        extractor = TripleExtractor(llm, config)
        logs = _make_logs(("test", "test"))
        triples = asyncio.get_event_loop().run_until_complete(extractor.extract(logs))
        assert len(triples) == 1
        assert triples[0].subject == "フォレトス"

    def test_multiple_chunks_processed(self, config):
        """Multiple chunks are processed correctly."""
        config["chunk_size"] = 1  # 1 pair per chunk
        call_count = 0

        async def mock_generate_bare(**kwargs):
            nonlocal call_count
            call_count += 1
            return f"entity{call_count} | rel | obj{call_count}", 1.0

        llm = MagicMock()
        llm.generate_bare = mock_generate_bare
        extractor = TripleExtractor(llm, config)
        logs = _make_logs(("q1", "a1"), ("q2", "a2"), ("q3", "a3"))
        triples = asyncio.get_event_loop().run_until_complete(extractor.extract(logs))
        assert call_count == 3
        assert len(triples) == 3

    def test_source_episode_id_assigned(self, config):
        """Each triple has a source_episode_id."""
        llm = _make_mock_llm("A | rel | B")
        extractor = TripleExtractor(llm, config)
        logs = _make_logs(("test", "test"))
        triples = asyncio.get_event_loop().run_until_complete(extractor.extract(logs))
        assert len(triples) == 1
        assert triples[0].source_episode_id.startswith("chunk_0_")

    def test_system_prompt_prohibits_hallucination(self, config):
        """system_promptにハルシネーション禁止ルールが含まれる"""
        from sleepyjean.triple_extractor import EXTRACTION_SYSTEM_PROMPT
        assert "あなた自身の知識で補完・推測してはいけません" in EXTRACTION_SYSTEM_PROMPT

    def test_llm_returns_none_no_error(self, config):
        """LLM returning None does not raise an error."""
        llm = _make_mock_llm(None)
        extractor = TripleExtractor(llm, config)
        logs = _make_logs(("test", "test"))
        triples = asyncio.get_event_loop().run_until_complete(extractor.extract(logs))
        assert triples == []
