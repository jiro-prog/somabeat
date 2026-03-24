"""Tests for llamarcute-live conversation buffer (working memory).

Tests:
- T1: Buffer accumulation and rotation
- T2: Prompt injection of buffer content
- T3: Buffer clearing on sleep
- T4: Token budget safety
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from shared_state.backends.chromadb_backend import ChromaDBField
from shared_state.interface import SenseParams
from shared_state.observer import LoggingObserver

from llamarcute_live.dialogue import DialogueManager
from llamarcute_live.personality import Personality


class MockEncoder:
    dimensionality = 384

    def _vec(self, text):
        h = hash(text) % (2**31)
        rng = np.random.RandomState(h)
        return rng.randn(384).astype(np.float32)

    def encode(self, text):
        return self._vec(text)

    def encode_for_emit(self, text):
        return self._vec(text)

    def encode_for_sense(self, text):
        return self._vec(text)


@pytest.fixture
def dialogue_manager(tmp_path):
    """Create a DialogueManager with mock field and encoder."""
    personality = Personality.load("llamarcute_live/data/personality_v0.yaml")
    encoder = MockEncoder()
    field = ChromaDBField(persist_directory=None, collection_name="test_buf")
    observer = LoggingObserver()

    dm = DialogueManager(
        personality=personality,
        field=field,
        encoder=encoder,
        observer=observer,
        db_path=str(tmp_path / "test.db"),
        use_llm=False,
        max_conversation_history=6,  # 3往復分
    )
    return dm


# ══════════════════════════════════════════════════════════════════════════════
# T1: バッファの蓄積とローテーション
# ══════════════════════════════════════════════════════════════════════════════

class TestBufferAccumulation:
    def test_initial_buffer_empty(self, dialogue_manager):
        """初期状態でバッファが空。"""
        assert dialogue_manager._history == []

    @pytest.mark.asyncio
    async def test_buffer_grows_with_dialogue(self, dialogue_manager):
        """対話ごとにバッファが蓄積される。"""
        await dialogue_manager.init_db()
        await dialogue_manager.process_input("こんにちは")
        assert len(dialogue_manager._history) == 2  # user + assistant

    @pytest.mark.asyncio
    async def test_buffer_rotates_at_max(self, dialogue_manager):
        """max_historyを超えたら古いターンが削除される。"""
        await dialogue_manager.init_db()
        # 4回対話 = 8エントリ、max=6なので2エントリ削除
        for i in range(4):
            await dialogue_manager.process_input(f"メッセージ{i}")

        assert len(dialogue_manager._history) == 6
        # 最初の対話（メッセージ0）が消えている
        assert "メッセージ0" not in dialogue_manager._history[0]["content"]
        # 最後の対話（メッセージ3）は残っている
        assert any("メッセージ3" in h["content"] for h in dialogue_manager._history)


# ══════════════════════════════════════════════════════════════════════════════
# T2: プロンプトへのバッファ注入
# ══════════════════════════════════════════════════════════════════════════════

class TestPromptInjection:
    def test_empty_buffer_no_section(self, dialogue_manager):
        """バッファ空のとき「直近の会話」セクションがない。"""
        prompt = dialogue_manager.build_prompt("テスト", [], [])
        assert "直近の会話" not in prompt

    def test_buffer_included_in_prompt(self, dialogue_manager):
        """バッファがあるとき「直近の会話」セクションが含まれる。"""
        dialogue_manager._history = [
            {"role": "user", "content": "量子力学って何？"},
            {"role": "assistant", "content": "量子力学はミクロな世界の物理学だよ。"},
        ]
        prompt = dialogue_manager.build_prompt("もっと教えて", [], [])
        assert "## 直近の会話" in prompt
        assert "ユーザー: 量子力学って何？" in prompt
        assert "あなた: 量子力学はミクロな世界の物理学だよ。" in prompt

    def test_section_order(self, dialogue_manager):
        """不変制約→行動規範→自己認識→記憶→直近の会話の順序が正しい。"""
        dialogue_manager._history = [
            {"role": "user", "content": "テスト"},
            {"role": "assistant", "content": "応答"},
        ]
        prompt = dialogue_manager.build_prompt(
            "入力", ["自己認識1"], ["記憶1"],
        )
        # セクションヘッダ（## 付き）の出現順を確認
        idx_constraint = prompt.index("## 不変制約")
        idx_rules = prompt.index("## 行動規範")
        idx_self = prompt.index("## 自己認識")
        idx_memory = prompt.index("## 最近の記憶")
        idx_history = prompt.index("## 直近の会話")
        assert idx_constraint < idx_rules < idx_self < idx_memory < idx_history


# ══════════════════════════════════════════════════════════════════════════════
# T3: バッファクリア
# ══════════════════════════════════════════════════════════════════════════════

class TestBufferClear:
    def test_clear_history(self, dialogue_manager):
        """clear_history()でバッファが空になる。"""
        dialogue_manager._history = [
            {"role": "user", "content": "テスト"},
            {"role": "assistant", "content": "応答"},
        ]
        dialogue_manager.clear_history()
        assert dialogue_manager._history == []

    @pytest.mark.asyncio
    async def test_clean_after_sleep(self, dialogue_manager):
        """入眠後にバッファがクリアされる（stop + clear_history のシミュレーション）。"""
        await dialogue_manager.init_db()
        await dialogue_manager.process_input("対話1")
        assert len(dialogue_manager._history) == 2

        # 入眠シーケンスのシミュレーション
        dialogue_manager.stop()
        dialogue_manager.clear_history()

        assert dialogue_manager._history == []

        # 覚醒後
        dialogue_manager.resume()
        prompt = dialogue_manager.build_prompt("覚醒後の最初の対話", [], [])
        assert "直近の会話" not in prompt


# ══════════════════════════════════════════════════════════════════════════════
# T4: トークン数の安全確認
# ══════════════════════════════════════════════════════════════════════════════

class TestTokenBudget:
    def test_max_prompt_within_budget(self, dialogue_manager):
        """最大バッファ + 最大sense結果でプロンプトが32kトークン以内。"""
        # 最大バッファ（6エントリ、各200文字）
        dialogue_manager._history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": "あ" * 200}
            for i in range(6)
        ]
        # 最大sense結果（各10件、各200文字）
        self_traces = [f"自己認識信号{'あ' * 200}" for _ in range(10)]
        ctx_traces = [f"コンテキスト信号{'あ' * 200}" for _ in range(10)]

        prompt = dialogue_manager.build_prompt("テスト入力", self_traces, ctx_traces)

        # 日本語は1文字≒1.5-2トークン。8000文字なら~16000トークン（32kの半分以下）
        assert len(prompt) < 10000, f"Prompt too long: {len(prompt)} chars"
