"""Awake-time recall engine for SleepyJean — KG-based.

Searches KnowledgeGraph via keyword extraction (fugashi/MeCab),
returns triples as text for prompt injection. Falls back to
MemoryStore embedding search when KG has no hits.

Reference: instructions_kg_ollama_removal.md T-1
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field as dc_field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from shared_state.interface import Signal, SignalOrigin

if TYPE_CHECKING:
    from sleepyjean.knowledge_graph import KnowledgeGraph, Triple
    from sleepyjean.memory_store import MemoryStore

logger = logging.getLogger(__name__)

_RECALL_ORIGIN = SignalOrigin(system="sleepyjean", context="recall")

# POS tags to extract as keywords (UniDic)
_KEYWORD_POS = frozenset({
    "名詞-固有名詞",
    "名詞-普通名詞",
    "形容詞-一般",
})


@dataclass
class RecallResult:
    """Result of a recall operation."""
    triples_text: str | None = None
    signal_ids: list[str] = dc_field(default_factory=list)
    source: str = "none"  # "kg" | "embedding" | "none"


class RecallEngine:
    """Awake-time recall. KG graph traversal, non-LLM."""

    def __init__(
        self,
        memory_store: MemoryStore,
        field,
        field_encoder,
        config: dict,
        gap_detector=None,
        knowledge_graph: KnowledgeGraph | None = None,
    ):
        self.memory_store = memory_store
        self.field = field
        self.field_encoder = field_encoder
        self.gap_detector = gap_detector
        self.kg = knowledge_graph
        self.n_results = config.get("n_results", 3)
        self.min_similarity = config.get("min_similarity", 0.3)
        self.base_norm = config.get("base_norm", 1.5)
        self.max_hops = config.get("max_hops", 2)

        # Lazy-init fugashi tagger
        self._tagger = None

    def _get_tagger(self):
        if self._tagger is None:
            import fugashi
            self._tagger = fugashi.Tagger()
        return self._tagger

    def recall(self, user_input: str) -> RecallResult:
        """Recall relevant knowledge from KG or MemoryStore.

        1. Extract keywords via morphological analysis
        2. Query KG for triples
        3. If KG hits: format as text
        4. If no KG hits: fallback to MemoryStore embedding search
        """
        if not user_input or not user_input.strip():
            return RecallResult()

        # 1. Try KG path
        if self.kg is not None and self.kg.graph.number_of_nodes() > 0:
            keywords = self._extract_keywords(user_input)
            if keywords:
                triples = self.kg.query(keywords, max_hops=self.max_hops)
                if triples:
                    text = self._format_triples(triples)
                    logger.info(
                        "Recall (KG): %d triples for keywords=%s",
                        len(triples), keywords[:5],
                    )
                    return RecallResult(triples_text=text, source="kg")

        # 2. Fallback: MemoryStore embedding search
        return self._fallback_embedding(user_input)

    async def recall_async(self, user_input: str) -> RecallResult:
        """Async wrapper for recall (for backward compat with await calls)."""
        return self.recall(user_input)

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords (nouns, proper nouns) via fugashi/MeCab."""
        tagger = self._get_tagger()
        words = tagger(text)
        keywords = []
        for w in words:
            # UniDic POS: "名詞,固有名詞,人名,一般" etc
            pos = w.pos if hasattr(w, 'pos') else ""
            # Check top-2 POS levels
            pos_parts = pos.split(",")
            if len(pos_parts) >= 2:
                pos_key = f"{pos_parts[0]}-{pos_parts[1]}"
                if pos_key in _KEYWORD_POS:
                    surface = w.surface
                    if len(surface) >= 2:  # skip single-char particles
                        keywords.append(surface)
        return keywords

    def _format_triples(self, triples: list[Triple]) -> str:
        """Format triples as prompt-injectable text."""
        lines = ["あなたの記憶:"]
        seen = set()
        for t in triples[:self.n_results * 3]:  # cap to avoid huge prompts
            line = f"- {t.subject}は{t.relation}が{t.object}"
            if line not in seen:
                seen.add(line)
                lines.append(line)
        return "\n".join(lines)

    def _fallback_embedding(self, user_input: str) -> RecallResult:
        """Fallback: embedding-based recall via MemoryStore."""
        try:
            query_emb = self.field_encoder.encode(user_input)
            results = self.memory_store.search(query_emb, n_results=self.n_results)

            if not results:
                if self.gap_detector:
                    self.gap_detector.record_failure(user_input, best_similarity=0.0)
                return RecallResult()

            best_similarity = 1.0 - results[0]["distance"]
            good_results = [
                r for r in results if (1.0 - r["distance"]) >= self.min_similarity
            ]

            if not good_results:
                if self.gap_detector:
                    self.gap_detector.record_failure(user_input, best_similarity=best_similarity)
                return RecallResult()

            logger.info(
                "Recall (embedding fallback): %d/%d memories (best_sim=%.3f)",
                len(good_results), len(results), best_similarity,
            )
            return RecallResult(source="embedding")

        except Exception as e:
            logger.warning("Recall embedding fallback failed: %s", e)
            return RecallResult()
