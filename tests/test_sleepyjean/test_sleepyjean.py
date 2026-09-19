"""Tests for sleepyjean.sleepyjean.SleepyJean integration class."""

import os
import uuid
from unittest.mock import AsyncMock

import pytest

from sleepyjean.sleepyjean import SleepyJean
from tests.test_sleepyjean.conftest import make_field, make_encoder


def _make_config(tmp_path):
    uid = uuid.uuid4().hex[:8]
    return {
        "sleepyjean": {
            "memory": {
                "chromadb_collection": f"test_sj_{uid}",
                "sqlite_path": os.path.join(str(tmp_path), f"test_sj_{uid}.db"),
            },
            "kg": {
                "sqlite_path": os.path.join(str(tmp_path), f"test_kg_{uid}.db"),
                "entity_resolution": {"embedding_threshold": 0.85, "levenshtein_threshold": 0.8},
                "prune": {"min_pagerank": 0.001, "min_access_count": 0, "protect_recent_days": 7},
            },
            "extraction": {"chunk_size": 3, "max_new_tokens": 512, "temperature": 0.3},
            "recall": {"n_results": 3, "min_similarity": 0.3, "base_norm": 1.5},
            "reconsolidation": {
                "prune": {"min_pagerank": 0.001, "min_access_count": 0, "protect_recent_days": 7},
            },
            "gap": {"min_repeat_count": 2, "similarity_threshold": 0.7},
            "quality": {"min_answer_length": 50, "min_relevance": 0.2},
        }
    }


class TestSleepyJean:
    def test_init_creates_all_modules(self, tmp_path):
        """初期化が全モジュールを生成する"""
        sj = SleepyJean(_make_config(tmp_path), make_field(), make_encoder())
        assert sj.memory_store is not None
        assert sj.knowledge_graph is not None
        assert sj.recall is not None
        assert sj.reconsolidation is not None
        assert sj.gap_detector is not None
        assert sj.quality_gate is not None
        sj.memory_store.close()

    def test_on_user_input_calls_recall(self, tmp_path):
        """on_user_inputがRecallEngineを呼ぶ"""
        from sleepyjean.recall import RecallResult
        sj = SleepyJean(_make_config(tmp_path), make_field(), make_encoder())
        result = sj.on_user_input("テスト入力")
        assert isinstance(result, RecallResult)
        sj.memory_store.close()

    @pytest.mark.asyncio
    async def test_on_sleep_runs_gap_and_reconsolidation(self, tmp_path):
        """on_sleepがGapDetector.emit_gaps + Reconsolidationを実行する"""
        sj = SleepyJean(_make_config(tmp_path), make_field(), make_encoder())
        sj.gap_detector.record_failure("test gap", best_similarity=0.1)
        assert len(sj.gap_detector._failures) == 1

        result = await sj.on_sleep([])
        assert len(sj.gap_detector._failures) == 0
        assert result is not None
        sj.memory_store.close()

    def test_config_passed_to_modules(self, tmp_path):
        """configが正しく各モジュールに渡される"""
        sj = SleepyJean(_make_config(tmp_path), make_field(), make_encoder())
        assert sj.recall.n_results == 3
        assert sj.recall.min_similarity == 0.3
        assert sj.recall.base_norm == 1.5
        assert sj.reconsolidation._prune_min_pagerank == 0.001
        assert sj.gap_detector.min_repeat_count == 2
        assert sj.quality_gate.min_answer_length == 50
        sj.memory_store.close()
