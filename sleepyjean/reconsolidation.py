"""Sleep-time memory reconsolidation for SleepyJean — KG-based.

Extracts triples from dialogue logs via LLM, stores in KnowledgeGraph,
detects structural changes (new nodes, merges, forgetting), and emits
results to the field.

Replaces the previous HDBSCAN clustering approach.

Reference: instructions_kg_ollama_removal.md K-3
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field as dc_field
from typing import TYPE_CHECKING

import numpy as np

from shared_state.interface import Signal, SignalOrigin

if TYPE_CHECKING:
    from sleepyjean.knowledge_graph import KnowledgeGraph
    from sleepyjean.memory_store import MemoryStore
    from sleepyjean.triple_extractor import TripleExtractor

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationResult:
    """Result of a consolidation run."""
    triple_count: int = 0
    new_node_count: int = 0
    merge_count: int = 0
    forget_count: int = 0
    signal_ids: list[str] = dc_field(default_factory=list)
    episode_count: int = 0
    # Legacy fields for backward compat with bridge/sleep_ingest
    new_cluster_count: int = 0


class Reconsolidation:
    """Sleep-time memory reconsolidation — KG-based."""

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        memory_store: MemoryStore,
        field,
        triple_extractor: TripleExtractor,
        field_encoder,
        config: dict,
    ) -> None:
        self.kg = knowledge_graph
        self.memory_store = memory_store  # fallback for embedding search
        self.field = field
        self.triple_extractor = triple_extractor
        self.field_encoder = field_encoder

        # Prune config
        prune_cfg = config.get("prune", {})
        self._prune_min_pagerank = prune_cfg.get("min_pagerank", 0.001)
        self._prune_min_access = prune_cfg.get("min_access_count", 0)
        self._prune_protect_days = prune_cfg.get("protect_recent_days", 7)

        # Load KG from SQLite
        self.kg.load()

        # Load previous snapshot for diff
        self._previous_snapshot = self._load_snapshot()

    async def consolidate(self, dialogue_logs: list[dict]) -> ConsolidationResult:
        """Execute memory reconsolidation.

        1. Store dialogue logs in MemoryStore (fallback for embedding search)
        2. Extract triples via LLM (generate_bare)
        3. Add triples to KG
        4. Diff against previous snapshot
        5. Emit signals for changes
        6. Prune low-importance nodes
        7. Save KG + snapshot
        """
        result = ConsolidationResult()
        result.episode_count = len(dialogue_logs)

        # 1. Store in MemoryStore (fallback)
        for ep in dialogue_logs:
            content = ep.get("content", "")
            if not content or not content.strip():
                continue
            try:
                emb = self.field_encoder.encode(content)
                self.memory_store.add_episode(emb, {
                    "source": ep.get("source", "dialogue"),
                    "confidence": ep.get("confidence", 1.0),
                })
            except Exception as e:
                logger.debug("MemoryStore add failed (non-fatal): %s", e)

        # 2. Extract triples via LLM (skip if no extractor)
        if self.triple_extractor is None:
            logger.info("Reconsolidation: no triple extractor (LLM unavailable), skipping extraction")
            self.kg.save()
            self._save_snapshot(self.kg.snapshot())
            return result

        triples = await self.triple_extractor.extract(dialogue_logs)
        result.triple_count = len(triples)

        if not triples:
            logger.info("Reconsolidation: no triples extracted (chitchat day?)")
            self.kg.save()
            self._save_snapshot(self.kg.snapshot())
            return result

        # 3. Add to KG
        add_result = self.kg.add_triples(triples)
        logger.info(
            "KG add: %d added, %d merged entities, %d skipped",
            add_result.added, add_result.merged, add_result.skipped,
        )

        # 4. Diff against previous snapshot
        diff = self.kg.diff(self._previous_snapshot or {"nodes": [], "edges": []})

        # 5. Emit signals
        signal_ids: list[str] = []

        # New nodes → knowledge_update
        if diff.new_nodes:
            for node_name in diff.new_nodes:
                try:
                    emb = self.field_encoder.encode(node_name)
                    signal = Signal.create(
                        embedding=emb,
                        origin=SignalOrigin(system="sleepyjean", context="knowledge_update"),
                    )
                    await self.field.emit(signal)
                    signal_ids.append(signal.signal_id)
                except Exception as e:
                    logger.debug("Emit knowledge_update failed: %s", e)
            result.new_node_count = len(diff.new_nodes)
            result.new_cluster_count = len(diff.new_nodes)  # legacy compat

        # Merged subgraphs → dream
        if diff.merged_subgraphs:
            for old_sets, new_set in diff.merged_subgraphs:
                try:
                    # Encode the merged subgraph summary
                    summary = "統合: " + ", ".join(list(new_set)[:5])
                    emb = self.field_encoder.encode(summary)
                    signal = Signal.create(
                        embedding=emb,
                        origin=SignalOrigin(system="sleepyjean", context="dream"),
                    )
                    await self.field.emit(signal)
                    signal_ids.append(signal.signal_id)
                except Exception as e:
                    logger.debug("Emit dream failed: %s", e)
            result.merge_count = len(diff.merged_subgraphs)

        # 6. Prune
        pruned = self.kg.prune(
            min_pagerank=self._prune_min_pagerank,
            min_access_count=self._prune_min_access,
            protect_recent_days=self._prune_protect_days,
        )
        if pruned:
            result.forget_count = len(pruned)
            try:
                summary = "忘却: " + ", ".join(pruned[:5])
                emb = self.field_encoder.encode(summary)
                signal = Signal.create(
                    embedding=emb,
                    origin=SignalOrigin(system="sleepyjean", context="forgetting"),
                )
                await self.field.emit(signal)
                signal_ids.append(signal.signal_id)
            except Exception as e:
                logger.debug("Emit forgetting failed: %s", e)

        result.signal_ids = signal_ids

        # 7. Save KG + snapshot
        self.kg.save()
        current_snapshot = self.kg.snapshot()
        self._save_snapshot(current_snapshot)
        self._previous_snapshot = current_snapshot

        stats = self.kg.stats()
        logger.info(
            "Reconsolidation: %d episodes, %d triples, "
            "%d new nodes, %d merged, %d forgotten "
            "(KG: %d nodes, %d edges, %d components)",
            result.episode_count, result.triple_count,
            result.new_node_count, result.merge_count, result.forget_count,
            stats["node_count"], stats["edge_count"], stats["connected_components"],
        )
        return result

    def _load_snapshot(self) -> dict | None:
        """Load previous KG snapshot from SQLite."""
        try:
            db = self.memory_store._db
            self._ensure_state_table(db)
            row = db.execute(
                "SELECT data FROM reconsolidation_state WHERE key = 'kg_snapshot'"
            ).fetchone()
            if row:
                return json.loads(row["data"])
        except Exception as e:
            logger.debug("Loading KG snapshot failed: %s", e)
        return None

    def _save_snapshot(self, snapshot: dict) -> None:
        """Persist KG snapshot to SQLite."""
        try:
            db = self.memory_store._db
            self._ensure_state_table(db)
            db.execute(
                """INSERT OR REPLACE INTO reconsolidation_state (key, data)
                   VALUES ('kg_snapshot', ?)""",
                (json.dumps(snapshot),),
            )
            db.commit()
        except Exception as e:
            logger.debug("Saving KG snapshot failed: %s", e)

    def _ensure_state_table(self, db) -> None:
        db.execute("""
            CREATE TABLE IF NOT EXISTS reconsolidation_state (
                key TEXT PRIMARY KEY,
                data TEXT NOT NULL
            )
        """)
        db.commit()
