"""Tests for SleepyJean integration with DialogueManager (Phase 3)."""

import os
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from llamarcute_live.dialogue import DialogueManager
from sleepyjean.recall import RecallResult
from tests.test_sleepyjean.conftest import rand_emb, DIM


def _make_dm(tmp_path, sleepyjean=None):
    """Create a minimal DialogueManager for testing."""
    personality = MagicMock()
    personality.to_prompt_section.return_value = "test prompt"

    field = AsyncMock()
    field.perceive = AsyncMock(return_value=MagicMock(signals=[]))
    field.emit = AsyncMock()

    encoder = MagicMock()
    encoder.encode = MagicMock(return_value=rand_emb())

    return DialogueManager(
        personality=personality,
        field=field,
        encoder=encoder,
        observer=MagicMock(),
        db_path=os.path.join(str(tmp_path), "test_dm.db"),
        sleepyjean=sleepyjean,
    )


class TestDialogueManagerSleepyJeanIntegration:
    @pytest.mark.asyncio
    async def test_none_sleepyjean_graceful(self, tmp_path):
        """SleepyJean=Noneの場合、Recallがスキップされ正常動作する"""
        dm = _make_dm(tmp_path, sleepyjean=None)
        await dm.init_db()
        dm.generate_response = AsyncMock(return_value="test response")

        result = await dm.process_input("テスト入力")
        assert result == "test response"

    @pytest.mark.asyncio
    async def test_sleepyjean_recall_called(self, tmp_path):
        """SleepyJeanありの場合、on_user_inputが呼ばれる"""
        mock_sj = MagicMock()
        mock_sj.on_user_input = MagicMock(
            return_value=RecallResult(triples_text="あなたの記憶:\n- テスト", source="kg"),
        )

        dm = _make_dm(tmp_path, sleepyjean=mock_sj)
        await dm.init_db()
        dm.generate_response = AsyncMock(return_value="test response")

        await dm.process_input("テスト入力")
        mock_sj.on_user_input.assert_called_once_with("テスト入力")

    @pytest.mark.asyncio
    async def test_sleepyjean_failure_nonfatal(self, tmp_path):
        """SleepyJeanのRecallが失敗しても対話は続行される"""
        mock_sj = MagicMock()
        mock_sj.on_user_input = MagicMock(side_effect=RuntimeError("recall error"))

        dm = _make_dm(tmp_path, sleepyjean=mock_sj)
        await dm.init_db()
        dm.generate_response = AsyncMock(return_value="test response")

        result = await dm.process_input("テスト入力")
        assert result == "test response"
