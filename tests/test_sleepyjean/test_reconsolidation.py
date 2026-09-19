"""Tests for sleepyjean.reconsolidation.Reconsolidation (KG-based).

8 tests per instructions_kg_ollama_removal.md K-3.
"""

import os
import uuid
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from sleepyjean.knowledge_graph import KnowledgeGraph, Triple
from sleepyjean.memory_store import MemoryStore
from sleepyjean.reconsolidation import Reconsolidation, ConsolidationResult
from sleepyjean.triple_extractor import TripleExtractor
from tests.test_sleepyjean.conftest import make_field, rand_emb, DIM


def _make_recon(tmp_path, extract_result=None, config_overrides=None):
    """Create a Reconsolidation with mocked components."""
    uid = uuid.uuid4().hex[:8]
    store_cfg = {
        "chromadb_collection": f"test_recon_{uid}",
        "sqlite_path": os.path.join(str(tmp_path), f"test_recon_{uid}.db"),
    }
    store = MemoryStore(store_cfg)
    field = make_field()
    encoder = MagicMock()
    encoder.encode = MagicMock(side_effect=lambda text: rand_emb())

    kg = KnowledgeGraph(
        sqlite_path=os.path.join(str(tmp_path), f"test_kg_{uid}.db"),
    )

    # Mock triple extractor
    extractor = MagicMock(spec=TripleExtractor)
    if extract_result is None:
        from datetime import datetime
        extract_result = [
            Triple("フォレトス", "タイプ", "むし・はがね", "ep1", datetime.now()),
            Triple("フォレトス", "覚える技", "だいばくはつ", "ep1", datetime.now()),
        ]
    extractor.extract = AsyncMock(return_value=extract_result)

    config = {
        "prune": {
            "min_pagerank": 0.001,
            "min_access_count": 0,
            "protect_recent_days": 7,
        },
    }
    if config_overrides:
        config.update(config_overrides)

    recon = Reconsolidation(kg, store, field, extractor, encoder, config)
    return recon, store, field, encoder, kg


class TestReconsolidation:
    @pytest.mark.asyncio
    async def test_triples_extracted_and_stored_in_kg(self, tmp_path):
        """Dialogue logs → triples extracted and stored in KG."""
        recon, store, field, encoder, kg = _make_recon(tmp_path)
        logs = [{"content": "テスト対話", "role": "user"}]
        result = await recon.consolidate(logs)

        assert result.triple_count == 2
        assert kg.graph.number_of_nodes() == 3  # フォレトス, むし・はがね, だいばくはつ
        assert kg.graph.number_of_edges() == 2
        store.close()

    @pytest.mark.asyncio
    async def test_diff_detects_new_nodes(self, tmp_path):
        """New nodes are detected in diff."""
        recon, store, field, encoder, kg = _make_recon(tmp_path)
        logs = [{"content": "test", "role": "user"}]
        result = await recon.consolidate(logs)

        # First run: all nodes are new
        assert result.new_node_count == 3

        # Second run with same triples: no new nodes
        result2 = await recon.consolidate(logs)
        assert result2.new_node_count == 0
        store.close()

    @pytest.mark.asyncio
    async def test_new_nodes_emitted_to_field(self, tmp_path):
        """New nodes emit knowledge_update signals to the field."""
        recon, store, field, encoder, kg = _make_recon(tmp_path)
        logs = [{"content": "test"}]
        result = await recon.consolidate(logs)

        assert result.new_node_count > 0
        assert field.emit.call_count >= result.new_node_count
        # Check signal origin
        for call in field.emit.call_args_list:
            signal = call[0][0]
            assert signal.origin.context in ("knowledge_update", "dream", "forgetting")
        store.close()

    @pytest.mark.asyncio
    async def test_merge_emitted_to_field(self, tmp_path):
        """Subgraph merges emit dream signals."""
        from datetime import datetime

        # First cycle: two disconnected subgraphs
        triples_1 = [
            Triple("A", "links", "B", "ep1", datetime.now()),
            Triple("C", "links", "D", "ep1", datetime.now()),
        ]
        recon, store, field, encoder, kg = _make_recon(tmp_path, extract_result=triples_1)
        await recon.consolidate([{"content": "first"}])

        # Second cycle: bridge connects the two subgraphs
        triples_2 = [
            Triple("B", "links", "C", "ep2", datetime.now()),
        ]
        recon.triple_extractor.extract = AsyncMock(return_value=triples_2)
        result = await recon.consolidate([{"content": "second"}])

        assert result.merge_count > 0
        # Check a dream signal was emitted
        dream_emitted = any(
            call[0][0].origin.context == "dream"
            for call in field.emit.call_args_list
        )
        assert dream_emitted
        store.close()

    @pytest.mark.asyncio
    async def test_prune_emitted_to_field(self, tmp_path):
        """Pruned nodes emit forgetting signals."""
        from datetime import datetime, timedelta

        recon, store, field, encoder, kg = _make_recon(
            tmp_path,
            extract_result=[Triple("new", "rel", "node", "ep1", datetime.now())],
            config_overrides={"prune": {"min_pagerank": 1.0, "min_access_count": 0, "protect_recent_days": 0}},
        )
        # Add old low-importance node directly
        old_date = (datetime.now() - timedelta(days=30)).isoformat()
        kg.graph.add_node("old_orphan", created_at=old_date, access_count=0)
        recon._previous_snapshot = kg.snapshot()

        result = await recon.consolidate([{"content": "test"}])
        assert result.forget_count > 0
        store.close()

    @pytest.mark.asyncio
    async def test_consolidation_result_stats(self, tmp_path):
        """ConsolidationResult has correct statistics."""
        recon, store, field, encoder, kg = _make_recon(tmp_path)
        logs = [{"content": f"test {i}"} for i in range(3)]
        result = await recon.consolidate(logs)

        assert result.episode_count == 3
        assert result.triple_count == 2
        assert isinstance(result.signal_ids, list)
        assert isinstance(result.new_node_count, int)
        store.close()

    @pytest.mark.asyncio
    async def test_empty_dialogue_logs_no_error(self, tmp_path):
        """Empty dialogue logs don't cause errors."""
        recon, store, field, encoder, kg = _make_recon(tmp_path, extract_result=[])
        result = await recon.consolidate([])
        assert result.episode_count == 0
        assert result.triple_count == 0
        store.close()

    @pytest.mark.asyncio
    async def test_zero_triples_no_error(self, tmp_path):
        """Zero triples extracted (chitchat day) doesn't error."""
        recon, store, field, encoder, kg = _make_recon(tmp_path, extract_result=[])
        result = await recon.consolidate([{"content": "おはよう"}])
        assert result.triple_count == 0
        assert result.episode_count == 1
        store.close()
