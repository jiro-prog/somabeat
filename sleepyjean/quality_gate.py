"""Q&A quality gate for SleepyJean.

Validates quality of Q&A pairs from the sensory system before ingestion.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Match CJK Unified Ideographs, Hiragana, Katakana
_JP_PATTERN = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]')


@dataclass(frozen=True)
class QualityResult:
    """Result of a quality check."""
    passed: bool
    reason: str | None = None


class QualityGate:
    """Quality gate for Q&A data from the sensory system."""

    def __init__(self, field_encoder, config: dict) -> None:
        self.field_encoder = field_encoder
        self.min_answer_length = config.get("min_answer_length", 50)
        self.min_relevance = config.get("min_relevance", 0.2)

    def check(self, question: str, answer: str) -> QualityResult:
        """Check quality of a Q&A pair.

        Checks:
        - Answer length >= min_answer_length
        - Question-answer embedding similarity >= min_relevance
        - Answer contains Japanese characters
        """
        # Length check
        if len(answer.strip()) < self.min_answer_length:
            return QualityResult(
                passed=False,
                reason=f"Answer too short ({len(answer.strip())} < {self.min_answer_length})",
            )

        # Language check
        if not _JP_PATTERN.search(answer):
            return QualityResult(
                passed=False,
                reason="Answer does not contain Japanese characters",
            )

        # Relevance check
        q_emb = self.field_encoder.encode(question)
        a_emb = self.field_encoder.encode(answer)

        norm_q = np.linalg.norm(q_emb)
        norm_a = np.linalg.norm(a_emb)
        if norm_q < 1e-8 or norm_a < 1e-8:
            return QualityResult(passed=False, reason="Zero-norm embedding")

        similarity = float(np.dot(q_emb, a_emb) / (norm_q * norm_a))
        if similarity < self.min_relevance:
            return QualityResult(
                passed=False,
                reason=f"Low relevance ({similarity:.3f} < {self.min_relevance})",
            )

        return QualityResult(passed=True)
