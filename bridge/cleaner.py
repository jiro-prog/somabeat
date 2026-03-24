"""Excretory system — cleans up stale data from SleepyJean's data stores.

Bio analogy: Glymphatic system — washes out metabolic waste from the brain
during sleep, without modifying the brain's internal logic.

Operates on SleepyJean's data stores (ChromaDB, LoRA adapters) as external
maintenance, not modifying SleepyJean's own code.

Executed during the orchestrator's purge step, after SleepyJean's night cycle.

Reference: Phase 4 taskflow (T35, T36)
"""

from __future__ import annotations

import logging
import shutil
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import chromadb

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# T35: ChromaDB RAG entry cleanup
# ---------------------------------------------------------------------------


def clean_chromadb_rag(
    chromadb_path: str,
    collection_name: str = "sleepyjean_knowledge",
    max_generations_per_topic: int = 1,
    max_age_days: int = 90,
) -> dict:
    """Remove stale RAG entries from SleepyJean's ChromaDB.

    Two-phase cleanup:
      Phase 1 — Duplicate removal: For each topic, keep only the latest
          N generations (date-based). A "generation" is all entries for a
          topic on a given date.
      Phase 2 — Age fallback: Among remaining entries, remove any older
          than max_age_days.

    Args:
        chromadb_path: Path to SleepyJean's ChromaDB persistent directory.
        collection_name: ChromaDB collection name.
        max_generations_per_topic: Number of date-generations to keep per topic.
        max_age_days: Entries older than this are removed (after dedup).

    Returns:
        {"duplicate_removed": int, "aged_removed": int, "retained": int}
    """
    client = chromadb.PersistentClient(path=chromadb_path)

    try:
        collection = client.get_collection(collection_name)
    except Exception:
        logger.info("[CLEANER] Collection '%s' not found, skipping", collection_name)
        return {"duplicate_removed": 0, "aged_removed": 0, "retained": 0}

    all_entries = collection.get(include=["metadatas"])
    ids = all_entries["ids"]
    metadatas = all_entries["metadatas"]

    if not ids:
        logger.info("[CLEANER] ChromaDB collection '%s' is empty", collection_name)
        return {"duplicate_removed": 0, "aged_removed": 0, "retained": 0}

    # Phase 1: Remove older generations of the same topic
    duplicate_ids = _find_duplicate_entries(ids, metadatas, max_generations_per_topic)

    # Phase 2: Remove aged entries from remaining
    remaining_set = set(ids) - set(duplicate_ids)
    remaining_ids = [i for i in ids if i in remaining_set]
    remaining_metas = [m for i, m in zip(ids, metadatas) if i in remaining_set]
    aged_ids = _find_aged_entries(remaining_ids, remaining_metas, max_age_days)

    # Execute deletion
    all_remove = duplicate_ids + aged_ids
    if all_remove:
        # ChromaDB delete has batch size limits; chunk if needed
        for i in range(0, len(all_remove), 5000):
            collection.delete(ids=all_remove[i : i + 5000])

    result = {
        "duplicate_removed": len(duplicate_ids),
        "aged_removed": len(aged_ids),
        "retained": len(ids) - len(all_remove),
    }

    logger.info(
        "[CLEANER] ChromaDB: duplicate_removed=%d, aged_removed=%d, retained=%d",
        result["duplicate_removed"],
        result["aged_removed"],
        result["retained"],
    )
    return result


def _find_duplicate_entries(
    ids: list[str],
    metadatas: list[dict],
    max_generations: int,
) -> list[str]:
    """Return IDs of entries in older generations of the same topic.

    A "generation" is defined by the date field in metadata.
    For each topic, sort dates descending and mark all entries
    beyond the latest max_generations dates for removal.
    """
    # topic -> {date -> [ids]}
    topic_dates: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for doc_id, meta in zip(ids, metadatas):
        topic = meta.get("topic", "unknown")
        date = meta.get("date", "1970-01-01")
        topic_dates[topic][date].append(doc_id)

    remove_ids: list[str] = []
    for topic, dates in topic_dates.items():
        sorted_dates = sorted(dates.keys(), reverse=True)  # newest first
        for old_date in sorted_dates[max_generations:]:
            remove_ids.extend(dates[old_date])

    return remove_ids


def _find_aged_entries(
    ids: list[str],
    metadatas: list[dict],
    max_age_days: int,
) -> list[str]:
    """Return IDs of entries older than max_age_days."""
    cutoff = (datetime.now() - timedelta(days=max_age_days)).strftime("%Y-%m-%d")
    remove_ids: list[str] = []
    for doc_id, meta in zip(ids, metadatas):
        date = meta.get("date", "9999-99-99")
        if date < cutoff:
            remove_ids.append(doc_id)
    return remove_ids


# ---------------------------------------------------------------------------
# T36: LoRA adapter cleanup
# ---------------------------------------------------------------------------


def clean_lora_adapters(
    adapter_dir: str,
    max_keep: int = 3,
) -> dict:
    """Remove old LoRA adapters, keeping the latest max_keep.

    The current_lora symlink target is always protected, even if it
    falls outside the max_keep newest directories.

    Args:
        adapter_dir: Path to the LoRA adapter directory (contains lora_{date}/ dirs).
        max_keep: Number of most recent adapters to keep.

    Returns:
        {"removed": int, "retained": int, "removed_dirs": list[str]}
    """
    adapter_path = Path(adapter_dir)
    if not adapter_path.exists():
        logger.info("[CLEANER] LoRA adapter dir does not exist: %s", adapter_dir)
        return {"removed": 0, "retained": 0, "removed_dirs": []}

    # List lora_{date} directories, sorted newest first (date in name = lexicographic order)
    lora_dirs = sorted(
        [d for d in adapter_path.iterdir() if d.is_dir() and d.name.startswith("lora_")],
        key=lambda d: d.name,
        reverse=True,
    )

    if len(lora_dirs) <= max_keep:
        logger.info(
            "[CLEANER] LoRA: %d adapters <= max_keep=%d, nothing to remove",
            len(lora_dirs),
            max_keep,
        )
        return {"removed": 0, "retained": len(lora_dirs), "removed_dirs": []}

    # Protect current_lora symlink target
    current_link = adapter_path / "current_lora"
    protected: Path | None = None
    if current_link.exists() and current_link.is_symlink():
        protected = current_link.resolve()

    # Keep newest max_keep + current_lora target
    keep_dirs = set(d.resolve() for d in lora_dirs[:max_keep])
    if protected:
        keep_dirs.add(protected)

    removed: list[str] = []
    for d in lora_dirs:
        if d.resolve() not in keep_dirs:
            shutil.rmtree(d)
            removed.append(d.name)
            logger.info("[CLEANER] Removed LoRA adapter: %s", d.name)

    result = {
        "removed": len(removed),
        "retained": len(lora_dirs) - len(removed),
        "removed_dirs": removed,
    }

    logger.info(
        "[CLEANER] LoRA: removed=%d, retained=%d",
        result["removed"],
        result["retained"],
    )
    return result
