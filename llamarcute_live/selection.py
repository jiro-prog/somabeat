"""Selection engine — integrates fitness and cuteness scores, selects winner.

Adapted from llamarcute/src/selection.py for the integrated system.

Reference: llamarcute_live_design.md section 5.4, phase1.5_phase2_taskflow.md T21
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np

from shared_state.backends.chromadb_backend import ChromaDBField
from shared_state.emit_log import insert_emit_log
from shared_state.encoder import E5SmallEncoder
from shared_state.interface import Signal, SignalOrigin

from llamarcute_live.personality import Personality

logger = logging.getLogger(__name__)

SELF_IMPROVE_ORIGIN = SignalOrigin(system="llamarcute_live", context="self_improvement")


class ScoredCandidate(NamedTuple):
    id: str
    fitness_z: float
    cuteness_z: float
    integrated: float


def z_score_normalize(values: list[float]) -> list[float]:
    """Normalize values to z-scores. Returns zeros if no variance."""
    n = len(values)
    if n == 0:
        return []
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    if variance < 1e-10:
        return [0.0] * n
    std = variance ** 0.5
    return [(v - mean) / std for v in values]


def integrate_scores(
    fitness_scores: dict[str, float],
    cuteness_scores: dict[str, float],
    fitness_weight: float = 0.75,
    cuteness_weight: float = 0.25,
) -> list[ScoredCandidate]:
    """Z-score normalize both dimensions, compute weighted sum, sort descending."""
    ids = list(fitness_scores.keys())

    fit_vals = [fitness_scores[i] for i in ids]
    cute_vals = [cuteness_scores.get(i, 0.0) for i in ids]

    fit_z = z_score_normalize(fit_vals)
    cute_z = z_score_normalize(cute_vals)

    scored = []
    for idx, cid in enumerate(ids):
        integrated = fitness_weight * fit_z[idx] + cuteness_weight * cute_z[idx]
        scored.append(ScoredCandidate(
            id=cid,
            fitness_z=fit_z[idx],
            cuteness_z=cute_z[idx],
            integrated=integrated,
        ))

    scored.sort(key=lambda s: s.integrated, reverse=True)
    return scored


async def select_and_update(
    current_personality: Personality,
    candidates: list[Personality],
    fitness_scores: dict[str, float],
    cuteness_scores: dict[str, float],
    personality_path: str | Path,
    field: ChromaDBField,
    encoder: E5SmallEncoder,
    fitness_weight: float = 0.75,
    cuteness_weight: float = 0.25,
    db_path: str | Path | None = None,
) -> dict:
    """Select the best candidate and update personality if a mutation wins.

    Returns:
        {
            "winner_id": str,
            "is_current": bool,
            "scored": list[ScoredCandidate],
            "personality_updated": bool,
        }
    """
    # Build fitness score map from results
    fitness_map = {}
    for cid, result in fitness_scores.items():
        if isinstance(result, dict):
            fitness_map[cid] = result.get("combined_fitness", 0.0)
        else:
            fitness_map[cid] = float(result)

    scored = integrate_scores(fitness_map, cuteness_scores, fitness_weight, cuteness_weight)

    winner = scored[0]
    is_current = winner.id == current_personality.id
    personality_updated = False

    logger.info("=== Self-improvement scoring ===")
    for s in scored:
        marker = " ★" if s.id == winner.id else ""
        logger.info(
            "  %s: fitness_z=%.3f cuteness_z=%.3f integrated=%.3f%s",
            s.id, s.fitness_z, s.cuteness_z, s.integrated, marker,
        )

    old_ver = current_personality.version
    old_count = current_personality.rule_count
    old_tokens = current_personality.approx_tokens

    if is_current:
        logger.info("Current personality is the best. No changes.")
        logger.info(
            "[SELF_IMPROVE] v%d→v%d: rules %d→%d, tokens %d→%d, changes: no update (current best)",
            old_ver, old_ver, old_count, old_count, old_tokens, old_tokens,
        )
        emit_text = (
            f"自己改善: 現行人格が最良。変更なし。"
            f" ルール数 {old_count}, fitness {winner.integrated:.2f}"
        )
    else:
        # Find the winning candidate
        winner_personality = None
        for c in candidates:
            if c.id == winner.id:
                winner_personality = c
                break

        if winner_personality is not None:
            new_ver = winner_personality.version
            new_count = winner_personality.rule_count
            new_tokens = winner_personality.approx_tokens

            # Summarize change types
            add_count = modify_count = delete_count = 0
            # Count changes by comparing rule sets
            # Use a simple heuristic: diff in rule count + structure
            diff = new_count - old_count
            if diff > 0:
                add_count = diff
            elif diff < 0:
                delete_count = -diff

            winner_personality.save(personality_path)
            personality_updated = True
            logger.info(
                "Personality updated to %s (v%d)",
                winner.id, new_ver,
            )
            logger.info(
                "[SELF_IMPROVE] v%d→v%d: rules %d→%d, tokens %d→%d, changes: +%d add, %d modify, -%d delete",
                old_ver, new_ver, old_count, new_count,
                old_tokens, new_tokens,
                add_count, modify_count, delete_count,
            )
            emit_text = (
                f"自己改善: v{old_ver}→v{new_ver}, "
                f"ルール数 {old_count}→{new_count}, "
                f"fitness {winner.integrated:.2f}, "
                f"cuteness {winner.cuteness_z:.1f}"
            )
        else:
            logger.error("Winner %s not found in candidates", winner.id)
            emit_text = f"自己改善: 勝者 {winner.id} が候補に見つからず。変更なし。"

    # Emit result to the shared field
    embedding = encoder.encode_for_emit(emit_text)
    signal = Signal.create(
        embedding=embedding,
        origin=SELF_IMPROVE_ORIGIN,
    )
    await field.emit(signal)
    if db_path:
        await insert_emit_log(db_path, signal.signal_id, emit_text)

    return {
        "winner_id": winner.id,
        "is_current": is_current,
        "scored": scored,
        "personality_updated": personality_updated,
    }
