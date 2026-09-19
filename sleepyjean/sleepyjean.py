"""SleepyJean: hippocampal module.

Integrates: MemoryStore, KnowledgeGraph, TripleExtractor, RecallEngine,
Reconsolidation, GapDetector, QualityGate.

LLM (FieldAwareLLM) is used only during sleep for triple extraction via
generate_bare. Awake-time Recall is non-LLM (KG graph traversal).

Reference: instructions_kg_ollama_removal.md K-4
"""

from __future__ import annotations

import logging

from sleepyjean.gap_detector import GapDetector
from sleepyjean.knowledge_graph import KnowledgeGraph
from sleepyjean.memory_store import MemoryStore
from sleepyjean.quality_gate import QualityGate
from sleepyjean.recall import RecallEngine, RecallResult
from sleepyjean.reconsolidation import ConsolidationResult, Reconsolidation
from sleepyjean.triple_extractor import TripleExtractor

logger = logging.getLogger(__name__)


class SleepyJean:
    """Hippocampal module. LLM used only during sleep (triple extraction)."""

    def __init__(self, config: dict, field, field_encoder, llm=None) -> None:
        sj_cfg = config.get("sleepyjean", {})

        self.memory_store = MemoryStore(sj_cfg.get("memory", {}))

        # KnowledgeGraph
        kg_cfg = sj_cfg.get("kg", {})
        er_cfg = kg_cfg.get("entity_resolution", {})
        self.knowledge_graph = KnowledgeGraph(
            sqlite_path=kg_cfg.get("sqlite_path", "data/sleepyjean_kg.db"),
            encoder=field_encoder,
            embedding_threshold=er_cfg.get("embedding_threshold", 0.85),
            levenshtein_threshold=er_cfg.get("levenshtein_threshold", 0.8),
        )

        # TripleExtractor (uses LLM during sleep only)
        self.triple_extractor = TripleExtractor(
            llm, sj_cfg.get("extraction", {}),
        ) if llm is not None else None

        self.gap_detector = GapDetector(
            field, field_encoder, sj_cfg.get("gap", {}),
        )
        self.recall = RecallEngine(
            self.memory_store, field, field_encoder,
            sj_cfg.get("recall", {}),
            gap_detector=self.gap_detector,
            knowledge_graph=self.knowledge_graph,
        )
        self.reconsolidation = Reconsolidation(
            self.knowledge_graph,
            self.memory_store,
            field,
            self.triple_extractor,
            field_encoder,
            sj_cfg.get("reconsolidation", {}),
        )
        self.quality_gate = QualityGate(
            field_encoder, sj_cfg.get("quality", {}),
        )

    # --- Awake phase (called on every dialogue turn) ---

    def on_user_input(self, user_input: str) -> RecallResult:
        """Recall support. Called by DialogueManager."""
        return self.recall.recall(user_input)

    # --- Sleep phase ---

    async def on_sleep(self, dialogue_logs: list[dict]) -> ConsolidationResult:
        """Sleep-time memory reconsolidation. Called by orchestrator."""
        # 1. Emit accumulated knowledge gaps
        await self.gap_detector.emit_gaps()
        self.gap_detector.clear()

        # 2. Run reconsolidation (KG-based, with triple extraction)
        result = await self.reconsolidation.consolidate(dialogue_logs)
        return result
