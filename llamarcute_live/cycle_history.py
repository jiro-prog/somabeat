"""Cycle history tracker — persists self-improvement cycle results for trend detection.

Reference: Phase 3 immune system design, T32
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MAX_ENTRIES = 20


def load_history(path: Path) -> list[dict]:
    """Load cycle history from JSON file. Returns empty list if file doesn't exist."""
    if not path.exists():
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load cycle history from %s: %s", path, e)
        return []


def append_cycle(
    path: Path,
    cycle_result: dict,
    max_entries: int = DEFAULT_MAX_ENTRIES,
) -> None:
    """Append a cycle result to history, keeping only the last max_entries."""
    history = load_history(path)
    history.append(cycle_result)

    # Trim to max_entries
    if len(history) > max_entries:
        history = history[-max_entries:]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Cycle history: %d entries saved to %s", len(history), path)


def get_consecutive_retains(history: list[dict]) -> int:
    """Count how many most recent consecutive cycles retained the current personality.

    Walks backward through history counting entries where
    is_current_winner is True (personality was not updated).
    """
    count = 0
    for entry in reversed(history):
        if entry.get("is_current_winner", False):
            count += 1
        else:
            break
    return count
