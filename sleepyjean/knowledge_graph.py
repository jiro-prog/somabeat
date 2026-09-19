"""Knowledge Graph for SleepyJean — NetworkX + SQLite persistence.

Stores factual triples (subject, relation, object) extracted from dialogue.
Supports entity resolution, N-hop query, PageRank, diff, and pruning.

Reference: instructions_kg_ollama_removal.md K-1
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import networkx as nx
import numpy as np
from Levenshtein import ratio as levenshtein_ratio

logger = logging.getLogger(__name__)


@dataclass
class Triple:
    subject: str
    relation: str
    object: str
    source_episode_id: str
    created_at: datetime
    access_count: int = 0


@dataclass
class AddResult:
    added: int = 0
    merged: int = 0  # entities resolved to existing nodes
    skipped: int = 0


@dataclass
class GraphDiff:
    new_nodes: list[str] = field(default_factory=list)
    new_edges: list[tuple] = field(default_factory=list)
    merged_subgraphs: list[tuple[set, set]] = field(default_factory=list)
    lost_nodes: list[str] = field(default_factory=list)


class KnowledgeGraph:
    """SleepyJean's knowledge graph. NetworkX DiGraph + SQLite persistence."""

    def __init__(
        self,
        sqlite_path: str,
        encoder=None,
        embedding_threshold: float = 0.85,
        levenshtein_threshold: float = 0.8,
    ) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self._sqlite_path = sqlite_path
        self._encoder = encoder
        self._embedding_threshold = embedding_threshold
        self._levenshtein_threshold = levenshtein_threshold

    def add_triples(self, triples: list[Triple]) -> AddResult:
        """Add triples to the graph with entity resolution.

        Entity resolution:
        - Exact match: connect to existing node
        - Near match (Levenshtein + embedding): merge if above thresholds
        - Otherwise: create new node (conservative)
        """
        result = AddResult()

        for triple in triples:
            resolved_subj = self._resolve_entity(triple.subject)
            resolved_obj = self._resolve_entity(triple.object)

            if resolved_subj != triple.subject or resolved_obj != triple.object:
                result.merged += 1

            # Add/update nodes
            for name in (resolved_subj, resolved_obj):
                if not self.graph.has_node(name):
                    self.graph.add_node(
                        name,
                        created_at=triple.created_at.isoformat(),
                        access_count=0,
                    )

            # Add edge
            self.graph.add_edge(
                resolved_subj,
                resolved_obj,
                relation=triple.relation,
                source_episode_id=triple.source_episode_id,
                created_at=triple.created_at.isoformat(),
                access_count=0,
            )
            result.added += 1

        return result

    def _resolve_entity(self, name: str) -> str:
        """Resolve entity name to existing node if similar enough."""
        if self.graph.has_node(name):
            return name

        best_match = None
        best_score = 0.0

        for existing in self.graph.nodes:
            # Levenshtein similarity
            lev_score = levenshtein_ratio(name, existing)
            if lev_score < self._levenshtein_threshold:
                continue

            # Embedding similarity (if encoder available)
            if self._encoder is not None:
                try:
                    emb_a = self._encoder.encode(name)
                    emb_b = self._encoder.encode(existing)
                    cos_sim = float(
                        np.dot(emb_a, emb_b)
                        / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-10)
                    )
                    if cos_sim < self._embedding_threshold:
                        continue
                    combined = (lev_score + cos_sim) / 2
                except Exception:
                    combined = lev_score
            else:
                combined = lev_score

            if combined > best_score:
                best_score = combined
                best_match = existing

        if best_match is not None:
            logger.debug("Entity resolved: '%s' → '%s' (score=%.2f)", name, best_match, best_score)
            return best_match

        return name

    def query(self, keywords: list[str], max_hops: int = 2) -> list[Triple]:
        """Query triples by keyword, traversing N hops from matched nodes.

        1. Partial match keywords against node names
        2. BFS up to max_hops from matched nodes
        3. Collect and return triples sorted by access_count
        4. Increment access_count on matched nodes
        """
        if not keywords or not self.graph.nodes:
            return []

        # Find matching nodes
        matched_nodes: set[str] = set()
        for kw in keywords:
            kw_lower = kw.lower()
            for node in self.graph.nodes:
                if kw_lower in node.lower():
                    matched_nodes.add(node)

        if not matched_nodes:
            return []

        # BFS to collect reachable nodes within max_hops
        reachable: set[str] = set()
        for start in matched_nodes:
            visited = {start}
            frontier = {start}
            for _ in range(max_hops):
                next_frontier: set[str] = set()
                for node in frontier:
                    for neighbor in set(self.graph.successors(node)) | set(self.graph.predecessors(node)):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            next_frontier.add(neighbor)
                frontier = next_frontier
            reachable |= visited

        # Collect triples from edges within reachable subgraph
        triples: list[Triple] = []
        for u, v, data in self.graph.edges(data=True):
            if u in reachable and v in reachable:
                triples.append(Triple(
                    subject=u,
                    relation=data.get("relation", ""),
                    object=v,
                    source_episode_id=data.get("source_episode_id", ""),
                    created_at=datetime.fromisoformat(data["created_at"])
                    if "created_at" in data else datetime.now(),
                    access_count=data.get("access_count", 0),
                ))

        # Sort by access_count descending
        triples.sort(key=lambda t: t.access_count, reverse=True)

        # Increment access_count on matched nodes
        for node in matched_nodes:
            self.graph.nodes[node]["access_count"] = (
                self.graph.nodes[node].get("access_count", 0) + 1
            )

        return triples

    def pagerank(self) -> dict[str, float]:
        """Compute PageRank for all nodes."""
        if not self.graph.nodes:
            return {}
        try:
            return nx.pagerank(self.graph)
        except nx.NetworkXError:
            return {n: 0.0 for n in self.graph.nodes}

    def diff(self, previous_snapshot: dict) -> GraphDiff:
        """Compute diff between current graph and a previous snapshot."""
        prev_nodes = set(previous_snapshot.get("nodes", []))
        prev_edges = set(tuple(e) for e in previous_snapshot.get("edges", []))
        curr_nodes = set(self.graph.nodes)
        curr_edges = set((u, v) for u, v in self.graph.edges())

        new_nodes = list(curr_nodes - prev_nodes)
        lost_nodes = list(prev_nodes - curr_nodes)
        new_edges = list(curr_edges - prev_edges)

        # Detect merged subgraphs: previous disconnected components now connected
        merged_subgraphs: list[tuple[set, set]] = []
        if prev_nodes and curr_nodes:
            prev_graph = nx.DiGraph()
            prev_graph.add_nodes_from(prev_nodes)
            prev_graph.add_edges_from(prev_edges)

            prev_components = list(nx.weakly_connected_components(prev_graph))
            curr_undirected = self.graph.to_undirected()
            for curr_comp in nx.connected_components(curr_undirected):
                # Check if this component spans multiple previous components
                overlapping_prev = [
                    pc for pc in prev_components if pc & curr_comp
                ]
                if len(overlapping_prev) > 1:
                    old_sets = frozenset().union(*overlapping_prev)
                    merged_subgraphs.append((old_sets, curr_comp))

        return GraphDiff(
            new_nodes=new_nodes,
            new_edges=new_edges,
            merged_subgraphs=merged_subgraphs,
            lost_nodes=lost_nodes,
        )

    def prune(
        self,
        min_pagerank: float = 0.001,
        min_access_count: int = 0,
        protect_recent_days: int = 7,
    ) -> list[str]:
        """Remove low-importance, unreferenced nodes.

        A node is pruned if:
        - PageRank < min_pagerank
        - access_count <= min_access_count
        - Not created within protect_recent_days
        """
        if not self.graph.nodes:
            return []

        pr = self.pagerank()
        now = datetime.now()
        protect_cutoff = now - timedelta(days=protect_recent_days)
        pruned: list[str] = []

        for node in list(self.graph.nodes):
            node_data = self.graph.nodes[node]
            node_pr = pr.get(node, 0.0)
            node_access = node_data.get("access_count", 0)

            # Check recency protection
            created_str = node_data.get("created_at", "")
            if created_str:
                try:
                    created = datetime.fromisoformat(created_str)
                    if created > protect_cutoff:
                        continue
                except ValueError:
                    pass

            if node_pr < min_pagerank and node_access <= min_access_count:
                pruned.append(node)

        for node in pruned:
            self.graph.remove_node(node)

        if pruned:
            logger.info("KG pruned %d nodes", len(pruned))

        return pruned

    def snapshot(self) -> dict:
        """Current graph snapshot for diff()."""
        return {
            "nodes": list(self.graph.nodes),
            "edges": [(u, v) for u, v in self.graph.edges()],
        }

    def stats(self) -> dict:
        """Graph statistics."""
        undirected = self.graph.to_undirected() if self.graph.nodes else nx.Graph()
        return {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "connected_components": nx.number_connected_components(undirected)
            if self.graph.nodes else 0,
        }

    def save(self) -> None:
        """Persist graph to SQLite."""
        Path(self._sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self._sqlite_path)
        try:
            db.execute("""
                CREATE TABLE IF NOT EXISTS kg_nodes (
                    name TEXT PRIMARY KEY,
                    created_at TEXT,
                    access_count INTEGER DEFAULT 0,
                    pagerank_score REAL DEFAULT 0.0
                )
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS kg_edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    object TEXT NOT NULL,
                    source_episode_id TEXT,
                    created_at TEXT
                )
            """)

            # Clear and re-insert (simple strategy for now)
            db.execute("DELETE FROM kg_nodes")
            db.execute("DELETE FROM kg_edges")

            pr = self.pagerank()
            for node, data in self.graph.nodes(data=True):
                db.execute(
                    "INSERT INTO kg_nodes (name, created_at, access_count, pagerank_score) VALUES (?, ?, ?, ?)",
                    (node, data.get("created_at", ""), data.get("access_count", 0), pr.get(node, 0.0)),
                )

            for u, v, data in self.graph.edges(data=True):
                db.execute(
                    "INSERT INTO kg_edges (subject, relation, object, source_episode_id, created_at) VALUES (?, ?, ?, ?, ?)",
                    (u, data.get("relation", ""), v, data.get("source_episode_id", ""), data.get("created_at", "")),
                )

            db.commit()
            logger.info("KG saved: %d nodes, %d edges", self.graph.number_of_nodes(), self.graph.number_of_edges())
        finally:
            db.close()

    def load(self) -> None:
        """Restore graph from SQLite."""
        if not Path(self._sqlite_path).exists():
            return

        db = sqlite3.connect(self._sqlite_path)
        db.row_factory = sqlite3.Row
        try:
            # Check tables exist
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "kg_nodes" not in tables or "kg_edges" not in tables:
                return

            self.graph.clear()

            for row in db.execute("SELECT * FROM kg_nodes"):
                self.graph.add_node(
                    row["name"],
                    created_at=row["created_at"] or "",
                    access_count=row["access_count"] or 0,
                )

            for row in db.execute("SELECT * FROM kg_edges"):
                self.graph.add_edge(
                    row["subject"],
                    row["object"],
                    relation=row["relation"],
                    source_episode_id=row["source_episode_id"] or "",
                    created_at=row["created_at"] or "",
                    access_count=0,
                )

            logger.info("KG loaded: %d nodes, %d edges", self.graph.number_of_nodes(), self.graph.number_of_edges())
        finally:
            db.close()
