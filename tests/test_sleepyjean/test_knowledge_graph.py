"""Tests for KnowledgeGraph — NetworkX + SQLite persistence.

10 tests per instructions_kg_ollama_removal.md K-1.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from sleepyjean.knowledge_graph import AddResult, GraphDiff, KnowledgeGraph, Triple


def _make_triple(subj: str, rel: str, obj: str, episode_id: str = "ep1") -> Triple:
    return Triple(
        subject=subj, relation=rel, object=obj,
        source_episode_id=episode_id, created_at=datetime.now(),
    )


@pytest.fixture
def kg(tmp_path):
    return KnowledgeGraph(sqlite_path=str(tmp_path / "test_kg.db"))


class TestKnowledgeGraph:

    def test_add_triples_creates_nodes_and_edges(self, kg):
        """add_triples creates nodes and edges in the graph."""
        triples = [
            _make_triple("フォレトス", "タイプ", "むし・はがね"),
            _make_triple("フォレトス", "覚える技", "だいばくはつ"),
        ]
        result = kg.add_triples(triples)
        assert result.added == 2
        assert kg.graph.number_of_nodes() == 3  # フォレトス, むし・はがね, だいばくはつ
        assert kg.graph.number_of_edges() == 2

    def test_add_triples_entity_resolution_exact(self, kg):
        """Exact-match entity resolution: same name → same node."""
        triples = [
            _make_triple("フォレトス", "タイプ", "むし・はがね"),
            _make_triple("フォレトス", "防御力", "高い"),
        ]
        result = kg.add_triples(triples)
        assert kg.graph.number_of_nodes() == 3  # フォレトス shared, not duplicated

    def test_query_returns_triples_by_keyword(self, kg):
        """query returns triples reachable from keyword-matched nodes."""
        kg.add_triples([
            _make_triple("フォレトス", "タイプ", "むし・はがね"),
            _make_triple("フォレトス", "覚える技", "だいばくはつ"),
            _make_triple("ピカチュウ", "タイプ", "でんき"),
        ])
        results = kg.query(["フォレトス"], max_hops=1)
        subjects = {t.subject for t in results}
        assert "フォレトス" in subjects
        # ピカチュウ should NOT be reachable
        assert "ピカチュウ" not in subjects
        assert len(results) >= 2

    def test_query_increments_access_count(self, kg):
        """query increments access_count on matched nodes."""
        kg.add_triples([_make_triple("テスト", "関係", "対象")])
        assert kg.graph.nodes["テスト"]["access_count"] == 0
        kg.query(["テスト"])
        assert kg.graph.nodes["テスト"]["access_count"] == 1
        kg.query(["テスト"])
        assert kg.graph.nodes["テスト"]["access_count"] == 2

    def test_query_empty_result(self, kg):
        """query returns empty list when no nodes match."""
        kg.add_triples([_make_triple("A", "rel", "B")])
        results = kg.query(["存在しないキーワード"])
        assert results == []

    def test_pagerank_computed(self, kg):
        """pagerank returns scores for all nodes."""
        kg.add_triples([
            _make_triple("A", "links", "B"),
            _make_triple("B", "links", "C"),
            _make_triple("C", "links", "A"),
        ])
        pr = kg.pagerank()
        assert len(pr) == 3
        assert all(v > 0 for v in pr.values())

    def test_diff_detects_changes(self, kg):
        """diff detects new nodes, new edges, and lost nodes."""
        kg.add_triples([_make_triple("A", "rel", "B")])
        snap1 = kg.snapshot()

        kg.add_triples([_make_triple("C", "rel", "D")])
        kg.graph.remove_node("A")  # simulate loss

        diff = kg.diff(snap1)
        assert "C" in diff.new_nodes
        assert "D" in diff.new_nodes
        assert "A" in diff.lost_nodes

    def test_prune_removes_low_importance_nodes(self, kg):
        """prune removes nodes with low pagerank and zero access."""
        # Create a graph where 'orphan' has low importance
        kg.add_triples([
            _make_triple("hub", "links", "spoke1"),
            _make_triple("hub", "links", "spoke2"),
            _make_triple("spoke1", "links", "hub"),
        ])
        # Add an isolated node with old creation date
        old_date = (datetime.now() - timedelta(days=30)).isoformat()
        kg.graph.add_node("orphan", created_at=old_date, access_count=0)

        pruned = kg.prune(min_pagerank=0.1, min_access_count=0, protect_recent_days=7)
        assert "orphan" in pruned
        assert "hub" not in pruned

    def test_save_load_roundtrip(self, kg):
        """save then load reproduces the same graph."""
        kg.add_triples([
            _make_triple("フォレトス", "タイプ", "むし・はがね"),
            _make_triple("フォレトス", "覚える技", "だいばくはつ"),
        ])
        kg.save()

        # Load into new KG instance
        kg2 = KnowledgeGraph(sqlite_path=kg._sqlite_path)
        kg2.load()

        assert kg2.graph.number_of_nodes() == kg.graph.number_of_nodes()
        assert kg2.graph.number_of_edges() == kg.graph.number_of_edges()
        assert set(kg2.graph.nodes) == set(kg.graph.nodes)

    def test_empty_graph_operations(self, kg):
        """All operations work on an empty graph without error."""
        assert kg.query(["test"]) == []
        assert kg.pagerank() == {}
        assert kg.diff({"nodes": [], "edges": []}).new_nodes == []
        assert kg.prune() == []
        assert kg.snapshot() == {"nodes": [], "edges": []}
        assert kg.stats() == {"node_count": 0, "edge_count": 0, "connected_components": 0}
        kg.save()  # should not error
        kg.load()  # should not error
