"""Tests for llamarcute_live module."""

import asyncio
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest

from llamarcute_live.personality import Personality


class TestPersonality:
    def test_load_seed(self):
        p = Personality.load("llamarcute_live/data/personality_v0.yaml")
        # Personality may have evolved from seed (T28 mutated v0 → v3)
        assert p.id is not None
        assert p.version >= 0

    def test_rule_count(self):
        p = Personality.load("llamarcute_live/data/personality_v0.yaml")
        total = sum(len(v) for v in p.rules.values())
        # Rule count should be within valid bounds (RULE_COUNT_MIN=5, RULE_COUNT_MAX=15)
        assert 5 <= total <= 15

    def test_to_prompt_section(self):
        p = Personality.load("llamarcute_live/data/personality_v0.yaml")
        prompt = p.to_prompt_section()
        assert "reasoning" in prompt
        assert "identity" in prompt

    def test_clone(self):
        p = Personality.load("llamarcute_live/data/personality_v0.yaml")
        c = p.clone()
        assert c.id == p.id
        assert c is not p

    def test_validation_fails_with_too_few_rules(self):
        data = {
            "id": "test",
            "version": 0,
            "updated_at": None,
            "previous_version": None,
            "behavioral_rules": {
                "reasoning": [{"rule": "x", "added_ver": 0, "modified_ver": None}],
            },
        }
        with pytest.raises(ValueError, match="below minimum"):
            Personality(data)


class _SeededMockEncoder:
    """Deterministic mock encoder using seeded random for reproducibility."""
    dimensionality = 384

    def _vec(self, text):
        seed = hash(text) % (2**31)
        rng = np.random.RandomState(seed)
        return rng.randn(384).astype(np.float32)

    def encode(self, text):
        return self._vec(text)

    def encode_for_emit(self, text):
        return self._vec(text)

    def encode_for_sense(self, text):
        return self._vec(text)


def _make_dialogue_manager(tmpdir):
    """Create an isolated DialogueManager with ephemeral DB and field."""
    from shared_state.backends.chromadb_backend import ChromaDBField
    from shared_state.observer import LoggingObserver
    from llamarcute_live.dialogue import DialogueManager

    observer = LoggingObserver()
    field = ChromaDBField(
        persist_directory=None,
        collection_name=f"test_{uuid.uuid4().hex[:8]}",
        observer=observer,
    )
    p = Personality.load("llamarcute_live/data/personality_v0.yaml")
    dm = DialogueManager(
        personality=p, field=field, encoder=_SeededMockEncoder(),
        observer=observer, db_path=Path(tmpdir) / "test.db",
    )
    return dm, field


class TestDialogueManager:
    def test_init_db_and_save_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dm, _ = _make_dialogue_manager(tmpdir)

            async def run():
                await dm.init_db()
                await dm.save_log("user", "hello")
                await dm.save_log("assistant", "hi there")
                logs = await dm.get_today_logs()
                assert len(logs) == 2

            asyncio.get_event_loop().run_until_complete(run())

    def test_process_input_mock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dm, field = _make_dialogue_manager(tmpdir)

            async def run():
                await dm.init_db()
                response = await dm.process_input("テスト入力")
                assert "[mock]" in response
                snap = await field.snapshot()
                assert snap.total_count >= 1

            asyncio.get_event_loop().run_until_complete(run())

    def test_build_prompt_has_immutable_constraints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dm, _ = _make_dialogue_manager(tmpdir)
            prompt = dm.build_prompt()
            assert "不変制約" in prompt
            assert "日本語で応答すること" in prompt
            assert "400文字以内" in prompt

    def test_build_prompt_immutable_before_behavioral(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dm, _ = _make_dialogue_manager(tmpdir)
            prompt = dm.build_prompt()
            immutable_pos = prompt.index("## 不変制約")
            behavioral_pos = prompt.index("## 行動規範")
            assert immutable_pos < behavioral_pos

    def test_stop_and_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dm, _ = _make_dialogue_manager(tmpdir)
            assert dm.is_active
            dm.stop()
            assert not dm.is_active
            dm.resume()
            assert dm.is_active
